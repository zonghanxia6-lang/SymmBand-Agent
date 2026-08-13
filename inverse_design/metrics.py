"""Benchmark metrics with explicit denominators and fixed-budget acceleration."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _csv_by_id(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle)}


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, center - spread), 6), round(min(1.0, center + spread), 6)]


def _metric(successes: int, total: int) -> dict:
    return {
        "successes": successes,
        "denominator": total,
        "rate": round(successes / total, 6) if total else None,
        "wilson_95_ci": _wilson(successes, total),
    }


def evaluate_benchmark(
    catalog_jsonl: Path,
    assignments_csv: Path,
    output_json: Path,
    novelty_csv: Path | None = None,
    labels_csv: Path | None = None,
    stability_threshold_ev_per_atom: float = 0.1,
    random_baseline_arm: str = "random_spacegroup",
) -> dict:
    assignments = _csv_by_id(assignments_csv)
    novelty = _csv_by_id(novelty_csv)
    labels = _csv_by_id(labels_csv)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assignment = assignments.get(row["candidate_id"])
        if assignment and assignment.get("arm"):
            row["assignment"] = assignment
            row["novelty"] = novelty.get(row["candidate_id"], {})
            row["label"] = labels.get(row["candidate_id"], {})
            grouped[assignment["arm"]].append(row)

    arms: dict[str, dict] = {}
    for arm, rows in sorted(grouped.items()):
        total = len(rows)
        valid = [row for row in rows if row.get("valid_structure")]
        condition_valid = [row for row in rows if row.get("condition_valid")]
        stoichiometry_valid = [row for row in rows if row.get("stoichiometry_valid")]
        sg_hits = [row for row in rows if row.get("target_spacegroup_match")]
        unique = [row for row in valid if row.get("is_unique") is True]
        novelty_evaluated = [
            row for row in valid if _bool(row["novelty"].get("is_novel")) is not None
        ]
        novel = [row for row in novelty_evaluated if _bool(row["novelty"].get("is_novel"))]
        stability_evaluated = []
        stable = []
        topology_evaluated = []
        topology_hits = []
        multiobjective_hits = []
        for row in rows:
            label = row["label"]
            ehull_text = str(label.get("energy_above_hull_ev_per_atom", "")).strip()
            phonon = _bool(label.get("phonon_stable"))
            if ehull_text or phonon is not None:
                stability_evaluated.append(row)
                stable_by_hull = bool(ehull_text) and float(ehull_text) <= stability_threshold_ev_per_atom
                if stable_by_hull or phonon is True:
                    stable.append(row)
            if _bool(label.get("topology_evaluated")) is True:
                topology_evaluated.append(row)
                target_particle = str(row["assignment"].get("target_particle", "")).strip()
                observed_particles = {
                    item.strip() for item in str(label.get("topology_particles", "")).split(";") if item.strip()
                }
                target_hit = not target_particle or target_particle in observed_particles
                if _bool(label.get("topology_hit")) is True and target_hit:
                    topology_hits.append(row)
            if (
                row.get("valid_structure")
                and row.get("target_spacegroup_match")
                and _bool(row["novelty"].get("is_novel")) is True
                and row in stable
                and row in topology_hits
            ):
                multiobjective_hits.append(row)
        arms[arm] = {
            "generated_count": total,
            "validity": _metric(len(valid), total),
            "composition_condition_retention": _metric(len(condition_valid), total),
            "stoichiometry_condition_retention": _metric(len(stoichiometry_valid), total),
            "target_spacegroup_retention": _metric(len(sg_hits), total),
            "within_valid_unique_rate": _metric(len(unique), len(valid)),
            "novelty": _metric(len(novel), len(novelty_evaluated)),
            "stability": _metric(len(stable), len(stability_evaluated)),
            "dft_topology_hit": _metric(len(topology_hits), len(topology_evaluated)),
            "end_to_end_multiobjective_hit": _metric(len(multiobjective_hits), total),
            "multiobjective_candidate_ids": [row["candidate_id"] for row in multiobjective_hits],
        }

    baseline_rate = (
        arms.get(random_baseline_arm, {})
        .get("end_to_end_multiobjective_hit", {})
        .get("rate")
    )
    for arm, result in arms.items():
        rate = result["end_to_end_multiobjective_hit"]["rate"]
        result["discovery_acceleration_vs_random"] = (
            round(rate / baseline_rate, 6) if rate is not None and baseline_rate else None
        )
    report = {
        "metric_contract": {
            "validity": "valid structures / all generated candidates",
            "composition_condition_retention": "structurally valid candidates retaining target elements / all generated",
            "stoichiometry_condition_retention": "structurally valid candidates retaining reduced target formula / all generated",
            "target_spacegroup_retention": "strict-symprec SG matches / all generated candidates",
            "novelty": "StructureMatcher non-matches / candidates evaluated against frozen references",
            "stability": f"Ehull <= {stability_threshold_ev_per_atom} eV/atom or phonon stable / evaluated",
            "dft_topology_hit": "target emergent-particle hits / SOC DFT topology evaluated",
            "end_to_end_multiobjective_hit": "valid, SG-retained, novel, stable, topology-hit / all generated",
            "acceleration": "end-to-end hit rate / random-spacegroup baseline hit rate at equal budget",
        },
        "arms": arms,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def create_assignment_template(catalog_jsonl: Path, output_csv: Path) -> Path:
    fields = ["candidate_id", "arm", "seed", "target_particle", "budget_order", "notes"]
    rows = []
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "arm": "",
                "seed": "",
                "target_particle": "",
                "budget_order": "",
                "notes": "legacy data are not assigned to an ablation arm automatically",
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
