"""Extract reproducible DFT workflow labels from compact calculation archives."""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


_WRONG_SG = re.compile(r"_wrong_sg\d+$", re.IGNORECASE)
_INCAR_VALUE = re.compile(r"^\s*(?P<key>[A-Z0-9_]+)\s*=\s*(?P<value>[^!#\n]+)", re.MULTILINE)
_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)")
_SIGMA_ZERO = re.compile(r"energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)")
_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
_ELAPSED = re.compile(r"Elapsed time \(sec\):\s*([-+0-9.Ee]+)")


@dataclass
class DFTJobRecord:
    result_name: str
    candidate_id: str | None
    mapping_status: str
    mapping_method: str | None
    job_directory: str
    stage: str
    primary_for_stage: bool
    calculation_complete: bool
    ionic_converged: bool | None
    electronic_converged: bool | None
    atom_count: int | None
    reduced_formula: str | None
    total_energy_ev: float | None
    sigma_zero_energy_ev: float | None
    energy_per_atom_ev: float | None
    elapsed_seconds: float | None
    final_structure_file: str | None
    error: str | None = None


@dataclass
class DFTMaterialRecord:
    result_name: str
    candidate_id: str | None
    mapping_status: str
    mapping_method: str | None
    workflow_run_count: int
    relax_job_count: int
    band_job_count: int
    relax_completed: bool
    relax_converged: bool | None
    static_soc_completed: bool
    band_soc_completed: bool
    irvsp_completed: bool
    target_spacegroup: int | None
    dft_spacegroup_hint: int | None
    relaxed_spacegroup_strict: int | None
    relaxed_spacegroup_standard: int | None
    band_symmetry_consistent: bool | None
    target_spacegroup_retained_strict: bool | None
    symmetry_tolerance_sensitive: bool | None
    reduced_formula: str | None
    atom_count: int | None
    relax_energy_per_atom_ev: float | None
    static_soc_energy_per_atom_ev: float | None
    relative_energy_per_atom_ev: float | None = None
    low_energy_polymorph: bool | None = None
    final_structure_file: str | None = None
    primary_band_job_directory: str | None = None


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _first_existing(directory: Path, *names: str) -> Path | None:
    return next((directory / name for name in names if (directory / name).is_file()), None)


def _is_vasp_job_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("job_"):
        return True
    return any(
        (path / name).is_file()
        for name in ("INCAR", "INCAR.gz", "OUTCAR", "OUTCAR.gz", "vasprun.xml", "vasprun.xml.gz")
    )


def _incar_values(job_dir: Path) -> dict[str, str]:
    path = _first_existing(job_dir, "INCAR.gz", "INCAR")
    if path is None:
        return {}
    return {
        match.group("key").upper(): match.group("value").strip().upper()
        for match in _INCAR_VALUE.finditer(_read_text(path))
    }


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip(".").upper() in {"T", "TRUE", "1", "YES"}


def classify_stage(job_dir: Path) -> str:
    values = _incar_values(job_dir)
    try:
        nsw = int(float(values.get("NSW", "0")))
    except ValueError:
        nsw = 0
    lsorbit = _as_bool(values.get("LSORBIT"))
    if nsw > 0:
        return "relax"
    if (job_dir / "outir").is_file() or (lsorbit and values.get("ICHARG") == "11"):
        return "band_soc"
    if lsorbit:
        return "static_soc"
    if values.get("ISYM") == "2":
        return "one_step_symmetry"
    return "static_nosoc"


def _structure_from_job(job_dir: Path, final: bool) -> tuple[Structure | None, Path | None]:
    names = (
        ("CONTCAR.gz", "CONTCAR", "POSCAR.gz", "POSCAR")
        if final
        else ("POSCAR.gz", "POSCAR", "CONTCAR.gz", "CONTCAR")
    )
    path = _first_existing(job_dir, *names)
    if path is None:
        return None, None
    try:
        return Structure.from_file(path), path
    except Exception:
        return None, path


def _parse_job(result_name: str, job_dir: Path) -> DFTJobRecord:
    stage = classify_stage(job_dir)
    outcar_path = _first_existing(job_dir, "OUTCAR.gz", "OUTCAR")
    if outcar_path is None:
        return DFTJobRecord(
            result_name=result_name,
            candidate_id=None,
            mapping_status="unmapped",
            mapping_method=None,
            job_directory=str(job_dir.resolve()),
            stage=stage,
            primary_for_stage=False,
            calculation_complete=False,
            ionic_converged=None,
            electronic_converged=None,
            atom_count=None,
            reduced_formula=None,
            total_energy_ev=None,
            sigma_zero_energy_ev=None,
            energy_per_atom_ev=None,
            elapsed_seconds=None,
            final_structure_file=None,
            error="OUTCAR missing",
        )
    try:
        text = _read_text(outcar_path)
        energies = _TOTEN.findall(text)
        sigma_zero = _SIGMA_ZERO.findall(text)
        elapsed = _ELAPSED.findall(text)
        nions = _NIONS.findall(text)
        structure, structure_path = _structure_from_job(job_dir, final=True)
        atom_count = len(structure) if structure is not None else (int(nions[-1]) if nions else None)
        total_energy = float(energies[-1]) if energies else None
        sigma_energy = float(sigma_zero[-1]) if sigma_zero else None
        energy_for_normalization = sigma_energy if sigma_energy is not None else total_energy
        completed = bool(
            energies
            and (
                "General timing and accounting informations for this job" in text
                or "Voluntary context switches" in text
                or elapsed
            )
        )
        electronic_converged = (
            "aborting loop because EDIFF is reached" in text
            or "EDIFF is reached" in text
        )
        ionic_converged = (
            "reached required accuracy" in text if stage == "relax" else None
        )
        return DFTJobRecord(
            result_name=result_name,
            candidate_id=None,
            mapping_status="unmapped",
            mapping_method=None,
            job_directory=str(job_dir.resolve()),
            stage=stage,
            primary_for_stage=False,
            calculation_complete=completed,
            ionic_converged=ionic_converged,
            electronic_converged=electronic_converged,
            atom_count=atom_count,
            reduced_formula=(structure.composition.reduced_formula if structure else None),
            total_energy_ev=total_energy,
            sigma_zero_energy_ev=sigma_energy,
            energy_per_atom_ev=(
                energy_for_normalization / atom_count
                if energy_for_normalization is not None and atom_count
                else None
            ),
            elapsed_seconds=float(elapsed[-1]) if elapsed else None,
            final_structure_file=str(structure_path.resolve()) if structure_path else None,
        )
    except Exception as exc:
        return DFTJobRecord(
            result_name=result_name,
            candidate_id=None,
            mapping_status="unmapped",
            mapping_method=None,
            job_directory=str(job_dir.resolve()),
            stage=stage,
            primary_for_stage=False,
            calculation_complete=False,
            ionic_converged=None,
            electronic_converged=None,
            atom_count=None,
            reduced_formula=None,
            total_energy_ev=None,
            sigma_zero_energy_ev=None,
            energy_per_atom_ev=None,
            elapsed_seconds=None,
            final_structure_file=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_catalog(catalog_jsonl: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = [json.loads(line) for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()]
    by_legacy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_legacy[row["legacy_id"]].append(row)
    return rows, by_legacy


def _identity_structures(result_dir: Path, jobs: list[DFTJobRecord]) -> list[Structure]:
    """Return initial structures first; relaxed CONTCARs are not identity records."""
    structures: list[Structure] = []
    relax_dirs = [Path(job.job_directory) for job in jobs if job.stage == "relax"]
    other_dirs = [Path(job.job_directory) for job in jobs if job.stage != "relax"]
    paths: list[Path] = []
    for directory in relax_dirs + other_dirs:
        path = _first_existing(directory, "POSCAR.gz", "POSCAR")
        if path is not None:
            paths.append(path)
    paths.extend(
        path
        for path in sorted(result_dir.rglob("POSCAR*"))
        if ".orig" not in path.name and path not in paths
    )
    for path in paths:
        try:
            structures.append(Structure.from_file(path))
        except Exception:
            continue
    return structures


def _map_result(
    result_dir: Path,
    jobs: list[DFTJobRecord],
    catalog_rows: list[dict],
    by_legacy: dict[str, list[dict]],
) -> tuple[str | None, str, str | None]:
    legacy = _WRONG_SG.sub("", result_dir.name)
    direct = by_legacy.get(legacy, [])
    if len(direct) == 1:
        return direct[0]["candidate_id"], "mapped", "unique_legacy_id"

    identity_structures = _identity_structures(result_dir, jobs)
    if not identity_structures:
        return None, "unmapped", "no_readable_structure"
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True)
    matches_by_id: dict[str, dict] = {}
    for structure in identity_structures:
        formula = structure.composition.reduced_formula
        atom_count = len(structure)
        candidates = [
            row
            for row in catalog_rows
            if row.get("parse_valid")
            and row.get("reduced_formula") == formula
            and row.get("atom_count") == atom_count
        ]
        for row in candidates:
            try:
                if matcher.fit(structure, Structure.from_file(row["selected_file"])):
                    matches_by_id[row["candidate_id"]] = row
            except Exception:
                continue
    matches = list(matches_by_id.values())
    if len(matches) == 1:
        return matches[0]["candidate_id"], "mapped", "structure_matcher"
    if len(matches) > 1:
        # A direct legacy candidate wins only after structural confirmation.
        direct_ids = {row["candidate_id"] for row in direct}
        confirmed_direct = [row for row in matches if row["candidate_id"] in direct_ids]
        if len(confirmed_direct) == 1:
            return confirmed_direct[0]["candidate_id"], "mapped", "legacy_plus_structure_matcher"
        return None, "ambiguous", f"{len(matches)}_structure_matches"
    return None, "unmapped", "no_structure_match"


def _spacegroup_hint(result_name: str, primary_band: DFTJobRecord | None) -> int | None:
    if primary_band:
        little_group = next(Path(primary_band.job_directory).glob("Littlegroup_*.cht"), None)
        if little_group:
            match = re.search(r"(\d{1,3})", little_group.stem)
            if match:
                return int(match.group(1))
    wrong = re.search(r"_wrong_sg(\d+)$", result_name, re.IGNORECASE)
    if wrong:
        return int(wrong.group(1))
    match = re.search(r"_sg(\d+)", result_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _target_spacegroup(result_name: str) -> int | None:
    match = re.search(r"_sg(\d+)", result_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _relaxed_spacegroups(primary_relax: DFTJobRecord | None) -> tuple[int | None, int | None]:
    if primary_relax is None or not primary_relax.final_structure_file:
        return None, None
    try:
        structure = Structure.from_file(primary_relax.final_structure_file)
        strict = SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_number()
        standard = SpacegroupAnalyzer(structure, symprec=0.05).get_space_group_number()
        return strict, standard
    except Exception:
        return None, None


def extract_dft_results(
    results_root: Path,
    catalog_jsonl: Path,
    output_dir: Path,
    low_energy_threshold_ev_per_atom: float = 0.1,
) -> tuple[list[DFTMaterialRecord], list[DFTJobRecord]]:
    catalog_rows, by_legacy = _load_catalog(catalog_jsonl)
    all_jobs: list[DFTJobRecord] = []
    materials: list[DFTMaterialRecord] = []
    for result_dir in sorted(path for path in results_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        jobs = [
            _parse_job(result_dir.name, job_dir)
            for job_dir in sorted(
                path for path in result_dir.iterdir() if _is_vasp_job_directory(path)
            )
        ]
        candidate_id, mapping_status, mapping_method = _map_result(
            result_dir, jobs, catalog_rows, by_legacy
        )
        grouped: dict[str, list[DFTJobRecord]] = defaultdict(list)
        for job in jobs:
            job.candidate_id = candidate_id
            job.mapping_status = mapping_status
            job.mapping_method = mapping_method
            grouped[job.stage].append(job)
        for stage_jobs in grouped.values():
            complete = [job for job in stage_jobs if job.calculation_complete]
            primary = max(complete or stage_jobs, key=lambda job: job.job_directory)
            primary.primary_for_stage = True
        all_jobs.extend(jobs)
        primary_relax = next(
            (job for job in grouped.get("relax", []) if job.primary_for_stage), None
        )
        primary_static = next(
            (job for job in grouped.get("static_soc", []) if job.primary_for_stage), None
        )
        primary_band = next(
            (job for job in grouped.get("band_soc", []) if job.primary_for_stage), None
        )
        source = primary_static or primary_relax or primary_band
        target_spacegroup = _target_spacegroup(result_dir.name)
        dft_spacegroup_hint = _spacegroup_hint(result_dir.name, primary_band)
        relaxed_sg_strict, relaxed_sg_standard = _relaxed_spacegroups(primary_relax)
        materials.append(
            DFTMaterialRecord(
                result_name=result_dir.name,
                candidate_id=candidate_id,
                mapping_status=mapping_status,
                mapping_method=mapping_method,
                workflow_run_count=max(len(grouped.get("relax", [])), len(grouped.get("band_soc", [])), 0),
                relax_job_count=len(grouped.get("relax", [])),
                band_job_count=len(grouped.get("band_soc", [])),
                relax_completed=bool(primary_relax and primary_relax.calculation_complete),
                relax_converged=(primary_relax.ionic_converged if primary_relax else None),
                static_soc_completed=bool(primary_static and primary_static.calculation_complete),
                band_soc_completed=bool(primary_band and primary_band.calculation_complete),
                irvsp_completed=bool(primary_band and (Path(primary_band.job_directory) / "outir").is_file()),
                target_spacegroup=target_spacegroup,
                dft_spacegroup_hint=dft_spacegroup_hint,
                relaxed_spacegroup_strict=relaxed_sg_strict,
                relaxed_spacegroup_standard=relaxed_sg_standard,
                band_symmetry_consistent=(
                    relaxed_sg_strict == dft_spacegroup_hint
                    if relaxed_sg_strict is not None and dft_spacegroup_hint is not None
                    else None
                ),
                target_spacegroup_retained_strict=(
                    relaxed_sg_strict == target_spacegroup
                    if relaxed_sg_strict is not None and target_spacegroup is not None
                    else None
                ),
                symmetry_tolerance_sensitive=(
                    relaxed_sg_strict != relaxed_sg_standard
                    if relaxed_sg_strict is not None and relaxed_sg_standard is not None
                    else None
                ),
                reduced_formula=source.reduced_formula if source else None,
                atom_count=source.atom_count if source else None,
                relax_energy_per_atom_ev=(primary_relax.energy_per_atom_ev if primary_relax else None),
                static_soc_energy_per_atom_ev=(primary_static.energy_per_atom_ev if primary_static else None),
                final_structure_file=(primary_relax.final_structure_file if primary_relax else None),
                primary_band_job_directory=(primary_band.job_directory if primary_band else None),
            )
        )

    # Relative polymorph energies are valid only within the same reduced composition.
    def label_energy(item: DFTMaterialRecord) -> float | None:
        if item.static_soc_completed and item.static_soc_energy_per_atom_ev is not None:
            return item.static_soc_energy_per_atom_ev
        if (
            item.relax_completed
            and item.relax_converged is True
            and item.relax_energy_per_atom_ev is not None
        ):
            return item.relax_energy_per_atom_ev
        return None

    formula_minimum: dict[str, float] = {}
    for item in materials:
        energy = label_energy(item)
        if item.reduced_formula and energy is not None and math.isfinite(energy):
            formula_minimum[item.reduced_formula] = min(
                formula_minimum.get(item.reduced_formula, energy), energy
            )
    for item in materials:
        energy = label_energy(item)
        if item.reduced_formula in formula_minimum and energy is not None:
            item.relative_energy_per_atom_ev = round(
                energy - formula_minimum[item.reduced_formula], 8
            )
            item.low_energy_polymorph = (
                item.relative_energy_per_atom_ev <= low_energy_threshold_ev_per_atom
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dft_materials.csv", materials, DFTMaterialRecord)
    _write_csv(output_dir / "dft_jobs.csv", all_jobs, DFTJobRecord)
    _write_dft_exclusions(output_dir / "excluded_dft_results.csv", materials)
    summary = {
        "material_count": len(materials),
        "job_count": len(all_jobs),
        "mapped_count": sum(item.mapping_status == "mapped" for item in materials),
        "ambiguous_count": sum(item.mapping_status == "ambiguous" for item in materials),
        "unmapped_count": sum(item.mapping_status == "unmapped" for item in materials),
        "relax_completed_count": sum(item.relax_completed for item in materials),
        "relax_converged_count": sum(item.relax_converged is True for item in materials),
        "static_soc_completed_count": sum(item.static_soc_completed for item in materials),
        "band_soc_completed_count": sum(item.band_soc_completed for item in materials),
        "irvsp_completed_count": sum(item.irvsp_completed for item in materials),
        "relaxed_spacegroup_labeled_count": sum(
            item.relaxed_spacegroup_strict is not None for item in materials
        ),
        "band_symmetry_consistent_count": sum(
            item.band_symmetry_consistent is True for item in materials
        ),
        "band_symmetry_inconsistent_count": sum(
            item.band_symmetry_consistent is False for item in materials
        ),
        "target_spacegroup_retained_strict_count": sum(
            item.target_spacegroup_retained_strict is True for item in materials
        ),
        "symmetry_tolerance_sensitive_count": sum(
            item.symmetry_tolerance_sensitive is True for item in materials
        ),
        "relative_energy_labeled_count": sum(item.relative_energy_per_atom_ev is not None for item in materials),
        "low_energy_polymorph_count": sum(item.low_energy_polymorph is True for item in materials),
        "scientific_scope": (
            "relative_energy_per_atom is referenced only to the lowest downloaded polymorph "
            "of the same reduced formula. It is not formation energy or energy above hull. "
            "Relaxed symmetry is independently determined at symprec 0.01 and 0.05 angstrom."
        ),
    }
    (output_dir / "dft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return materials, all_jobs


def _write_csv(path: Path, rows: list[object], schema: type) -> None:
    names = [field.name for field in fields(schema)]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_dft_exclusions(path: Path, materials: list[DFTMaterialRecord]) -> None:
    rows = []
    for item in materials:
        stability_reason = ""
        topology_reason = ""
        if item.mapping_status != "mapped":
            stability_reason = f"candidate_{item.mapping_status}"
            topology_reason = f"candidate_{item.mapping_status}"
        elif item.relative_energy_per_atom_ev is None:
            stability_reason = "no_completed_converged_energy_label"
        if not topology_reason:
            if not item.band_soc_completed:
                topology_reason = "band_soc_incomplete"
            elif not item.irvsp_completed:
                topology_reason = "irvsp_output_missing"
        if stability_reason or topology_reason:
            rows.append(
                {
                    "result_name": item.result_name,
                    "candidate_id": item.candidate_id or "",
                    "mapping_status": item.mapping_status,
                    "excluded_from_stability": bool(stability_reason),
                    "stability_exclusion_reason": stability_reason,
                    "excluded_from_topology": bool(topology_reason),
                    "topology_exclusion_reason": topology_reason,
                    "spacegroup_review_policy": "trust_completed_irvsp_after_review",
                }
            )
    fields = [
        "result_name",
        "candidate_id",
        "mapping_status",
        "excluded_from_stability",
        "stability_exclusion_reason",
        "excluded_from_topology",
        "topology_exclusion_reason",
        "spacegroup_review_policy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
