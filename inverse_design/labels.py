"""Stable DFT/topology label schema and importers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


LABEL_FIELDS = [
    "candidate_id",
    "legacy_id",
    "dft_completed",
    "formation_energy_per_atom_ev",
    "energy_above_hull_ev_per_atom",
    "relative_polymorph_energy_ev_per_atom",
    "low_energy_polymorph",
    "phonon_stable",
    "band_symmetry_consistent",
    "topology_raw_evaluated",
    "topology_evaluated",
    "topology_hit",
    "topology_particles",
    "topology_crossing_count",
    "source",
    "notes",
]


def create_label_template(catalog_jsonl: Path, output_csv: Path) -> Path:
    rows = []
    with catalog_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "legacy_id": item["legacy_id"],
                    "dft_completed": "",
                    "formation_energy_per_atom_ev": "",
                    "energy_above_hull_ev_per_atom": "",
                    "relative_polymorph_energy_ev_per_atom": "",
                    "low_energy_polymorph": "",
                    "phonon_stable": "",
                    "band_symmetry_consistent": "",
                    "topology_raw_evaluated": "",
                    "topology_evaluated": "",
                    "topology_hit": "",
                    "topology_particles": "",
                    "topology_crossing_count": "",
                    "source": "",
                    "notes": "",
                }
            )
    _write_csv(rows, output_csv, LABEL_FIELDS)
    return output_csv


def import_agent_topology_reports(
    catalog_jsonl: Path,
    results_root: Path,
    output_csv: Path,
) -> list[dict]:
    """Import agent reports; ambiguous legacy IDs are deliberately left unresolved."""
    catalog = [json.loads(line) for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()]
    by_legacy: dict[str, list[dict]] = {}
    for item in catalog:
        by_legacy.setdefault(item["legacy_id"], []).append(item)
    rows: list[dict] = []
    for report_file in results_root.rglob("accidental_degeneracy_report.json"):
        report = json.loads(report_file.read_text(encoding="utf-8"))
        result_name = report_file.parents[1].name
        matches = by_legacy.get(result_name, [])
        candidate_id = matches[0]["candidate_id"] if len(matches) == 1 else ""
        particles = report.get("confirmed_particle_types", [])
        rows.append(
            {
                "candidate_id": candidate_id,
                "legacy_id": result_name,
                "dft_completed": True,
                "formation_energy_per_atom_ev": "",
                "energy_above_hull_ev_per_atom": "",
                "relative_polymorph_energy_ev_per_atom": "",
                "low_energy_polymorph": "",
                "phonon_stable": "",
                "topology_evaluated": True,
                "topology_hit": bool(particles),
                "topology_particles": ";".join(particles),
                "topology_crossing_count": report.get("crossing_count", 0),
                "source": str(report_file.resolve()),
                "notes": "" if candidate_id else "legacy_id missing or ambiguous in catalog",
            }
        )
    _write_csv(rows, output_csv, LABEL_FIELDS)
    return rows


def merge_dft_topology_labels(
    dft_materials_csv: Path,
    topology_labels_csv: Path,
    output_csv: Path,
) -> list[dict]:
    """Build one candidate-level table without promoting proxy energy to Ehull."""
    with dft_materials_csv.open(encoding="utf-8-sig", newline="") as handle:
        dft_rows = list(csv.DictReader(handle))
    with topology_labels_csv.open(encoding="utf-8-sig", newline="") as handle:
        topology = {row["result_name"]: row for row in csv.DictReader(handle)}
    by_candidate: dict[str, list[dict]] = {}
    for dft in dft_rows:
        candidate_id = dft.get("candidate_id", "").strip()
        if candidate_id:
            by_candidate.setdefault(candidate_id, []).append(dft)

    rows: list[dict] = []
    for candidate_id, candidate_results in sorted(by_candidate.items()):
        candidate_results.sort(
            key=lambda row: (
                row["result_name"] not in topology,
                float(row["relative_energy_per_atom_ev"] or "inf"),
                row["result_name"],
            )
        )
        dft = candidate_results[0]
        topo = topology.get(dft["result_name"], {})
        compatible_particles = topo.get("compatible_particles", "")
        raw_topology_evaluated = topo.get("analysis_completed", "")
        symmetry_consistent = dft.get("band_symmetry_consistent", "")
        # Completed IRVSP analyses are accepted after external workflow review. The
        # independent spglib audit remains recorded but is not a rejection gate.
        reliable_topology = str(raw_topology_evaluated).lower() == "true"
        rows.append(
            {
                "candidate_id": candidate_id,
                "legacy_id": dft["result_name"],
                "dft_completed": dft.get("relax_completed", ""),
                "formation_energy_per_atom_ev": "",
                "energy_above_hull_ev_per_atom": "",
                "relative_polymorph_energy_ev_per_atom": dft.get(
                    "relative_energy_per_atom_ev", ""
                ),
                "low_energy_polymorph": dft.get("low_energy_polymorph", ""),
                "phonon_stable": "",
                "band_symmetry_consistent": symmetry_consistent,
                "topology_raw_evaluated": raw_topology_evaluated,
                "topology_evaluated": reliable_topology,
                "topology_hit": (
                    topo.get("strict_topology_hit", "") if reliable_topology else ""
                ),
                "topology_particles": topo.get("confirmed_particles", ""),
                "topology_crossing_count": topo.get("crossing_count", ""),
                "source": ";".join(
                    filter(None, [dft.get("final_structure_file", ""), topo.get("report_file", "")])
                ),
                "notes": (
                    "low_energy_polymorph is a same-formula downloaded-polymorph proxy, not Ehull; "
                    f"band_symmetry_consistent={symmetry_consistent or 'unknown'}; "
                    "spacegroup_policy=trust_completed_irvsp_after_review; "
                    f"compatible_particles={compatible_particles or 'none'}"
                ),
            }
        )
    _write_csv(rows, output_csv, LABEL_FIELDS)
    return rows


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
