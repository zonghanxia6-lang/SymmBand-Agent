"""Build a reproducible 500-2000-candidate multifidelity screening queue."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from emergent_particles import lookup_emergent_particles


def build_funnel(
    catalog_jsonl: Path,
    novelty_csv: Path,
    predictions_csv: Path,
    output_csv: Path,
    budget: int = 1000,
) -> list[dict]:
    if not 500 <= budget <= 2000:
        raise ValueError("The publication funnel budget must be between 500 and 2000")
    with novelty_csv.open(encoding="utf-8-sig", newline="") as handle:
        novelty = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    with predictions_csv.open(encoding="utf-8-sig", newline="") as handle:
        predictions = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        cid = candidate["candidate_id"]
        if not (
            candidate.get("stoichiometry_valid")
            and candidate.get("target_spacegroup_match")
            and candidate.get("is_unique")
        ):
            continue
        if str(novelty.get(cid, {}).get("is_novel", "")).lower() != "true":
            continue
        prediction = predictions.get(cid, {})
        stability = float(prediction.get("stability_probability", 0.5))
        topology = float(prediction.get("topology_probability", 0.5))
        uncertainty = max(
            float(prediction.get("stability_uncertainty", 0.0)),
            float(prediction.get("topology_uncertainty", 0.0)),
        )
        # Geometric mean prevents one objective from compensating for a near-zero second objective.
        score = math.sqrt(max(stability, 1e-9) * max(topology, 1e-9)) + 0.05 * uncertainty
        rows.append(
            {
                "candidate_id": cid,
                "reduced_formula": candidate.get("reduced_formula"),
                "spacegroup": candidate.get("actual_spacegroup"),
                "stability_probability": stability,
                "topology_probability": topology,
                "uncertainty_bonus": uncertainty,
                "multiobjective_score": score,
                "novelty_reference": novelty_csv.stem,
                "novelty_status": novelty.get(cid, {}).get("novelty_status", "novel"),
                "selected_file": candidate.get("selected_file"),
            }
        )
    rows.sort(key=lambda row: (-row["multiobjective_score"], row["candidate_id"]))
    selected = rows[:budget]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
        row["stage_1"] = "valid+SG-retained+StructureMatcher-novel"
        row["stage_2"] = "queued: ML surrogate + MACE geometry review"
        row["stage_3"] = "queued: low-cost DFT relax/static"
        row["stage_4"] = "queued: SOC band + IRVSP emergent-particle analysis"
        row["stage_5"] = "queued: phonons for final hits"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) if selected else []
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        if selected:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "requested_budget": budget,
                "selected_count": len(selected),
                "budget_shortfall": budget - len(selected),
                "novelty_file": str(novelty_csv.resolve()),
                "selection_requirements": [
                    "strict_stoichiometry",
                    "target_spacegroup_retained",
                    "internally_unique_by_StructureMatcher",
                    "externally_novel_within_covered_reference_formula",
                ],
                "scientific_scope": (
                    "The novelty claim inherits the scope of the supplied reference snapshot. "
                    "For MP20 input this is not a complete Materials Project novelty claim."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected


def build_prenovelty_funnel(
    catalog_jsonl: Path,
    predictions_csv: Path,
    output_csv: Path,
    budget: int = 500,
    per_formula_spacegroup_cap: int = 20,
) -> list[dict]:
    """Rank a particle-aware queue while external StructureMatcher novelty is pending."""
    if not 500 <= budget <= 2000:
        raise ValueError("The publication funnel budget must be between 500 and 2000")
    with predictions_csv.open(encoding="utf-8-sig", newline="") as handle:
        predictions = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    rows: list[dict] = []
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        cid = candidate["candidate_id"]
        if not (
            candidate.get("condition_valid")
            and candidate.get("target_spacegroup_match")
            and candidate.get("is_unique")
        ):
            continue
        prediction = predictions.get(cid)
        if not prediction or prediction.get("prediction_error"):
            continue
        spacegroup = candidate.get("actual_spacegroup")
        if spacegroup is None:
            continue
        lookup = lookup_emergent_particles(int(spacegroup), soc=True)
        accidental = sorted({particle.abbreviation for particle in lookup.accidental})
        if not accidental:
            continue
        path_map = {
            particle.abbreviation: sorted({path.path for path in (particle.paths or [])})
            for particle in lookup.accidental
        }
        stability = float(prediction.get("stability_probability", 0.5))
        topology = float(prediction.get("topology_probability", 0.5))
        uncertainty = max(
            float(prediction.get("stability_uncertainty", 0.0)),
            float(prediction.get("topology_uncertainty", 0.0)),
        )
        score = math.sqrt(max(stability, 1e-9) * max(topology, 1e-9)) + 0.05 * uncertainty
        rows.append(
            {
                "candidate_id": cid,
                "selection_tier": (
                    "A_strict_stoichiometry"
                    if candidate.get("stoichiometry_valid")
                    else "B_element_set_only"
                ),
                "target_formula": candidate.get("target_formula"),
                "reduced_formula": candidate.get("reduced_formula"),
                "actual_spacegroup": int(spacegroup),
                "compatible_particles": ";".join(accidental),
                "particle_path_map": json.dumps(path_map, ensure_ascii=False, sort_keys=True),
                "stability_proxy_probability": stability,
                "topology_probability": topology,
                "uncertainty_bonus": uncertainty,
                "multiobjective_score": score,
                "novelty_status": "not_evaluated",
                "selected_file": candidate.get("selected_file"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["selection_tier"] != "A_strict_stoichiometry",
            -row["multiobjective_score"],
            row["candidate_id"],
        )
    )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    group_counts: Counter[tuple[str, int]] = Counter()
    for enforce_cap in (True, False):
        for row in rows:
            if len(selected) >= budget:
                break
            if row["candidate_id"] in selected_ids:
                continue
            key = (row["reduced_formula"] or "", row["actual_spacegroup"])
            if enforce_cap and group_counts[key] >= per_formula_spacegroup_cap:
                continue
            row = dict(row)
            row["diversity_cap_relaxed"] = not enforce_cap
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            group_counts[key] += 1
        if len(selected) >= budget:
            break

    particle_counts: Counter[str] = Counter()
    for rank, row in enumerate(selected, 1):
        compatible = row["compatible_particles"].split(";")
        assigned = min(compatible, key=lambda particle: (particle_counts[particle], particle))
        particle_counts[assigned] += 1
        row["rank"] = rank
        row["portfolio_particle"] = assigned
        row["stage_1"] = "passed: valid+target-SG-retained+internal-StructureMatcher-unique"
        row["stage_2"] = "pending: Materials Project/external StructureMatcher novelty"
        row["stage_3"] = "queued: DFT relax/static and thermodynamic labels"
        row["stage_4"] = "queued: SOC band+IRVSP particle-path validation"
        row["stage_5"] = "queued: phonons for final hits"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        fields = list(selected[0])
        with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary = {
        "requested_budget": budget,
        "selected_count": len(selected),
        "tier_counts": dict(sorted(Counter(row["selection_tier"] for row in selected).items())),
        "portfolio_particle_counts": dict(sorted(particle_counts.items())),
        "novelty_status": "not_evaluated",
        "scientific_scope": (
            "This is a pre-novelty prioritization queue. It must not be reported as a novel-material "
            "set until StructureMatcher comparison against frozen external database snapshots is complete."
        ),
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return selected
