"""Non-destructive cataloging and symmetry reclassification of generated crystals."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


_NAMED_CANDIDATE = re.compile(
    r"^gen_(?P<formula>[A-Za-z][A-Za-z0-9]*?)(?:_(?P<batch>\d+))?_sg"
    r"(?P<sg>\d+)_(?P<index>\d+)$"
)
_TASK_CANDIDATE = re.compile(r"^gen_task(?P<index>\d+)_sg(?P<sg>\d+)(?:_|$)")
_RELAX_LINE = re.compile(
    r"(?:FIRE|BFGS|LBFGS):\s*(?P<step>\d+)\s+\S+\s+"
    r"(?P<energy>[-+0-9.eE]+)\s+(?P<fmax>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class CatalogConfig:
    source_root: Path
    output_root: Path
    curated_root: Path | None = None
    symprec_values: tuple[float, ...] = (0.01, 0.05, 0.1)
    angle_tolerance: float = 5.0
    min_distance_angstrom: float = 0.6
    min_volume_per_atom: float = 1.0
    max_volume_per_atom: float = 500.0
    relax_fmax_threshold: float = 0.1
    divergence_energy_ev: float = 1.0e6
    divergence_fmax_ev_per_angstrom: float = 1.0e5
    materialize: bool = True
    deduplicate: bool = True
    offset: int = 0
    skip_indices: tuple[int, ...] = ()
    limit: int | None = None


@dataclass
class CandidateRecord:
    candidate_id: str
    legacy_id: str
    source_collection: str
    source_file: str
    selected_file: str
    selected_stage: str
    target_formula: str | None
    target_spacegroup: int
    actual_formula: str | None = None
    reduced_formula: str | None = None
    anonymous_formula: str | None = None
    atom_count: int | None = None
    element_count: int | None = None
    element_set_match: bool | None = None
    formula_match: bool | None = None
    parse_valid: bool = False
    geometry_valid: bool = False
    symmetry_valid: bool = False
    valid_structure: bool = False
    condition_valid: bool = False
    stoichiometry_valid: bool = False
    validation_reason: str = "not_analyzed"
    min_distance_angstrom: float | None = None
    volume_per_atom: float | None = None
    density_g_cm3: float | None = None
    actual_spacegroup: int | None = None
    actual_spacegroup_symbol: str | None = None
    sg_symprec_001: int | None = None
    sg_symprec_005: int | None = None
    sg_symprec_010: int | None = None
    symmetry_tolerance_sensitive: bool | None = None
    target_spacegroup_match: bool | None = None
    relax_log: str | None = None
    relax_final_energy_ev: float | None = None
    relax_energy_per_atom_ev: float | None = None
    relax_final_fmax: float | None = None
    relax_step: int | None = None
    relax_converged: bool | None = None
    relax_diverged: bool | None = None
    duplicate_of: str | None = None
    is_unique: bool | None = None
    classification: str = "unclassified"
    canonical_name: str | None = None
    curated_file: str | None = None
    error: str | None = None


def _upgrade_record(record: CandidateRecord) -> CandidateRecord:
    """Derive fields added after an older shard was written."""
    record.condition_valid = bool(
        record.valid_structure and record.element_set_match is not False
    )
    record.stoichiometry_valid = bool(
        record.condition_valid and record.formula_match is not False
    )
    return record


def discover_candidates(
    source_root: Path,
) -> list[tuple[Path, str, str, str | None, int]]:
    """Return raw generation files, never derivative ``*_std.cif`` files."""
    candidates: list[tuple[Path, str, str, str | None, int]] = []
    for path in sorted(source_root.rglob("gen_*.cif")):
        if path.name.endswith("_std.cif"):
            continue
        source_collection = path.parent.relative_to(source_root).as_posix()
        collection_id = re.sub(r"[^A-Za-z0-9.-]", "_", source_collection)
        match = _NAMED_CANDIDATE.match(path.stem)
        if match:
            legacy_id = path.stem.removeprefix("gen_")
            candidate_id = f"{collection_id}__{legacy_id}"
            candidates.append(
                (path, candidate_id, legacy_id, match.group("formula"), int(match.group("sg")))
            )
            continue
        match = _TASK_CANDIDATE.match(path.stem)
        if match:
            candidate_id = f"{collection_id}__{path.stem}"
            candidates.append((path, candidate_id, path.stem, None, int(match.group("sg"))))
    return candidates


def _select_structure(raw_file: Path, legacy_id: str) -> tuple[Path, str]:
    options = (
        (raw_file.with_name(f"{legacy_id}_finerelax.cif"), "finerelax"),
        (raw_file.with_name(f"{legacy_id}_relaxed.cif"), "relaxed"),
        (raw_file.with_name(f"{legacy_id}_prerelax.cif"), "prerelax"),
        (raw_file.with_name(f"{raw_file.stem}_std.cif"), "standardized_generation"),
        (raw_file, "generated"),
    )
    return next((item for item in options if item[0].is_file()), (raw_file, "generated"))


def _target_composition(formula: str | None) -> Composition | None:
    if not formula:
        return None
    try:
        if formula.lower() == "carbon":
            formula = "C"
        return Composition(formula)
    except Exception:
        return None


def _minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        radius = max(structure.lattice.abc)
        distances = np.asarray(structure.get_neighbor_list(radius)[3], dtype=float)
        nonzero = distances[distances > 1.0e-8]
        return float(np.min(nonzero)) if nonzero.size else math.inf
    distances = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _parse_relax_log(
    selected_file: Path,
    atom_count: int,
    config: CatalogConfig,
) -> dict[str, object]:
    log_file = selected_file.with_suffix(".log")
    if not log_file.is_file():
        return {}
    last_match = None
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RELAX_LINE.search(line)
        if match:
            last_match = match
    if last_match is None:
        return {"relax_log": str(log_file), "relax_converged": False}
    energy = float(last_match.group("energy"))
    fmax = float(last_match.group("fmax"))
    diverged = (
        not math.isfinite(energy)
        or not math.isfinite(fmax)
        or abs(energy) >= config.divergence_energy_ev
        or abs(fmax) >= config.divergence_fmax_ev_per_angstrom
    )
    return {
        "relax_log": str(log_file),
        "relax_final_energy_ev": energy,
        "relax_energy_per_atom_ev": energy / atom_count,
        "relax_final_fmax": fmax,
        "relax_step": int(last_match.group("step")),
        "relax_converged": not diverged and fmax <= config.relax_fmax_threshold,
        "relax_diverged": diverged,
    }


def _structure_hash(structure: Structure) -> str:
    sorted_structure = structure.get_sorted_structure()
    payload = {
        "lattice": np.round(sorted_structure.lattice.matrix, 6).tolist(),
        "sites": [
            [site.species_string, *np.round(site.frac_coords % 1.0, 6).tolist()]
            for site in sorted_structure
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _safe_formula(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "_", formula)


def _classification(record: CandidateRecord) -> str:
    if not record.parse_valid:
        return "parse_failed"
    if not record.symmetry_valid:
        return "symmetry_failed"
    if not record.geometry_valid or record.relax_diverged:
        return "invalid_geometry"
    if record.element_set_match is False:
        return "composition_changed"
    if record.target_spacegroup_match:
        return "target_spacegroup_retained"
    return "symmetry_changed"


def _analyze_candidate(
    raw_file: Path,
    candidate_id: str,
    legacy_id: str,
    target_formula: str | None,
    target_spacegroup: int,
    config: CatalogConfig,
) -> tuple[CandidateRecord, Structure | None]:
    selected_file, selected_stage = _select_structure(raw_file, legacy_id)
    record = CandidateRecord(
        candidate_id=candidate_id,
        legacy_id=legacy_id,
        source_collection=raw_file.parent.name,
        source_file=str(raw_file),
        selected_file=str(selected_file),
        selected_stage=selected_stage,
        target_formula=target_formula,
        target_spacegroup=target_spacegroup,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(selected_file)
        record.parse_valid = True
        record.actual_formula = structure.composition.formula.replace(" ", "")
        record.reduced_formula = structure.composition.reduced_formula
        record.anonymous_formula = structure.composition.anonymized_formula
        record.atom_count = len(structure)
        record.element_count = len(structure.composition.element_composition)
        target = _target_composition(target_formula)
        if target is not None:
            actual_elements = {element.symbol for element in structure.composition.elements}
            target_elements = {element.symbol for element in target.elements}
            record.element_set_match = actual_elements == target_elements
            record.formula_match = structure.composition.reduced_composition.almost_equals(
                target.reduced_composition
            )

        finite = bool(
            np.all(np.isfinite(structure.lattice.matrix))
            and np.all(np.isfinite(structure.frac_coords))
        )
        record.min_distance_angstrom = round(_minimum_distance(structure), 8)
        record.volume_per_atom = round(structure.volume / len(structure), 8)
        record.density_g_cm3 = round(float(structure.density), 8)
        record.geometry_valid = (
            finite
            and structure.volume > 0
            and record.min_distance_angstrom >= config.min_distance_angstrom
            and config.min_volume_per_atom
            <= record.volume_per_atom
            <= config.max_volume_per_atom
        )
        record.validation_reason = "ok" if record.geometry_valid else "geometry_threshold"

        symmetry_numbers: list[int] = []
        symbols: list[str] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for symprec in config.symprec_values:
                analyzer = SpacegroupAnalyzer(
                    structure,
                    symprec=symprec,
                    angle_tolerance=config.angle_tolerance,
                )
                symmetry_numbers.append(analyzer.get_space_group_number())
                symbols.append(analyzer.get_space_group_symbol())
        record.actual_spacegroup = symmetry_numbers[0]
        record.actual_spacegroup_symbol = symbols[0]
        record.symmetry_valid = True
        padded = symmetry_numbers + [None] * (3 - len(symmetry_numbers))
        record.sg_symprec_001, record.sg_symprec_005, record.sg_symprec_010 = padded[:3]
        record.symmetry_tolerance_sensitive = len(set(symmetry_numbers)) > 1
        record.target_spacegroup_match = record.actual_spacegroup == target_spacegroup

        relax_values = _parse_relax_log(selected_file, len(structure), config)
        for key, value in relax_values.items():
            setattr(record, key, value)
        record.valid_structure = bool(
            record.geometry_valid
            and record.symmetry_valid
            and record.relax_diverged is not True
        )
        record.condition_valid = bool(
            record.valid_structure and record.element_set_match is not False
        )
        record.stoichiometry_valid = bool(
            record.condition_valid and record.formula_match is not False
        )
        record.classification = _classification(record)

        structure_hash = _structure_hash(structure)
        formula = _safe_formula(record.reduced_formula or "unknown")
        sg = record.actual_spacegroup or 0
        record.canonical_name = f"{formula}_sg{sg:03d}_{structure_hash}.cif"
        return record, structure
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.validation_reason = "parse_or_symmetry_error"
        record.classification = "parse_failed" if not record.parse_valid else "symmetry_failed"
        return record, None


def _materialize_structure(
    record: CandidateRecord,
    structure: Structure,
    curated_root: Path,
) -> None:
    sg_folder = f"sg{(record.actual_spacegroup or 0):03d}"
    formula_folder = _safe_formula(record.reduced_formula or "unknown")
    destination = curated_root / record.classification / sg_folder / formula_folder
    destination.mkdir(parents=True, exist_ok=True)
    output_file = destination / str(record.canonical_name)
    CifWriter(structure, symprec=None).write_file(output_file)
    record.curated_file = str(output_file)


def _deduplicate(
    records: list[CandidateRecord],
    structures: dict[str, Structure],
) -> None:
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
    grouped: dict[tuple[str, int, int], list[CandidateRecord]] = defaultdict(list)
    for record in records:
        if record.condition_valid and record.candidate_id in structures:
            grouped[
                (
                    record.reduced_formula or "",
                    record.actual_spacegroup or 0,
                    record.atom_count or 0,
                )
            ].append(record)
        else:
            record.is_unique = None

    for group_number, group in enumerate(grouped.values(), 1):
        representatives: list[CandidateRecord] = []
        exact_representatives: dict[str, CandidateRecord] = {}
        for record in group:
            exact_key = record.canonical_name or ""
            exact_duplicate = exact_representatives.get(exact_key)
            if exact_duplicate is not None:
                record.is_unique = False
                record.duplicate_of = exact_duplicate.candidate_id
                continue
            structure = structures[record.candidate_id]
            duplicate = None
            for representative in representatives:
                if matcher.fit(structure, structures[representative.candidate_id]):
                    duplicate = representative
                    break
            if duplicate is None:
                record.is_unique = True
                representatives.append(record)
                exact_representatives[exact_key] = record
            else:
                record.is_unique = False
                record.duplicate_of = duplicate.candidate_id
        if group_number % 100 == 0 or group_number == len(grouped):
            print(f"Deduplicated {group_number}/{len(grouped)} structure groups", flush=True)


def write_records(records: Iterable[CandidateRecord], output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    csv_path = output_root / "structure_catalog.csv"
    jsonl_path = output_root / "structure_catalog.jsonl"
    fieldnames = [field.name for field in fields(CandidateRecord)]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def _write_summary(records: list[CandidateRecord], output_root: Path, config: CatalogConfig) -> Path:
    count = len(records)

    def rate(predicate) -> float | None:
        return round(sum(1 for record in records if predicate(record)) / count, 6) if count else None

    summary = {
        "source_root": str(config.source_root.resolve()),
        "candidate_count": count,
        "parse_valid_count": sum(record.parse_valid for record in records),
        "geometry_valid_count": sum(record.geometry_valid for record in records),
        "symmetry_valid_count": sum(record.symmetry_valid for record in records),
        "valid_structure_count": sum(record.valid_structure for record in records),
        "condition_valid_count": sum(record.condition_valid for record in records),
        "stoichiometry_valid_count": sum(record.stoichiometry_valid for record in records),
        "element_set_match_count": sum(record.element_set_match is True for record in records),
        "formula_match_count": sum(record.formula_match is True for record in records),
        "target_spacegroup_match_count": sum(
            record.target_spacegroup_match is True for record in records
        ),
        "symmetry_tolerance_sensitive_count": sum(
            record.symmetry_tolerance_sensitive is True for record in records
        ),
        "relax_diverged_count": sum(record.relax_diverged is True for record in records),
        "relax_converged_count": sum(record.relax_converged is True for record in records),
        "unique_count": sum(record.is_unique is True for record in records),
        "rates": {
            "parse_valid": rate(lambda record: record.parse_valid),
            "geometry_valid": rate(lambda record: record.geometry_valid),
            "symmetry_valid": rate(lambda record: record.symmetry_valid),
            "valid_structure": rate(lambda record: record.valid_structure),
            "condition_valid": rate(lambda record: record.condition_valid),
            "stoichiometry_valid": rate(lambda record: record.stoichiometry_valid),
            "internal_uniqueness_given_condition_valid": (
                round(
                    sum(record.is_unique is True for record in records)
                    / sum(record.condition_valid for record in records),
                    6,
                )
                if any(record.condition_valid for record in records)
                else None
            ),
            "element_set_match": rate(lambda record: record.element_set_match is True),
            "formula_match": rate(lambda record: record.formula_match is True),
            "target_spacegroup_retention": rate(
                lambda record: record.target_spacegroup_match is True
            ),
            "symmetry_tolerance_sensitive": rate(
                lambda record: record.symmetry_tolerance_sensitive is True
            ),
            "relax_diverged": rate(lambda record: record.relax_diverged is True),
        },
        "classifications": dict(
            sorted(
                {
                    name: sum(record.classification == name for record in records)
                    for name in {record.classification for record in records}
                }.items()
            )
        ),
        "important_note": (
            "Rates are data-quality audit results, not DFT stability or topology metrics. "
            "MACE potential energies are not formation energies or energies above hull."
        ),
    }
    path = output_root / "catalog_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_group_summary(records, output_root / "catalog_by_target_spacegroup.csv", "target_spacegroup")
    _write_group_summary(records, output_root / "catalog_by_collection.csv", "source_collection")
    return path


def _write_group_summary(
    records: list[CandidateRecord],
    output_file: Path,
    group_field: str,
) -> None:
    grouped: dict[object, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, group_field)].append(record)
    rows = []
    for group, items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        total = len(items)
        rows.append(
            {
                group_field: group,
                "candidate_count": total,
                "parse_valid_rate": sum(item.parse_valid for item in items) / total,
                "geometry_valid_rate": sum(item.geometry_valid for item in items) / total,
                "element_set_match_rate": sum(item.element_set_match is True for item in items) / total,
                "formula_match_rate": sum(item.formula_match is True for item in items) / total,
                "target_spacegroup_retention_rate": sum(
                    item.target_spacegroup_match is True for item in items
                )
                / total,
                "symmetry_tolerance_sensitive_rate": sum(
                    item.symmetry_tolerance_sensitive is True for item in items
                )
                / total,
                "relax_diverged_rate": sum(item.relax_diverged is True for item in items) / total,
                "valid_structure_rate": sum(item.valid_structure for item in items) / total,
                "condition_valid_rate": sum(item.condition_valid for item in items) / total,
                "stoichiometry_valid_rate": sum(item.stoichiometry_valid for item in items) / total,
            }
        )
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [group_field])
        writer.writeheader()
        writer.writerows(rows)


def build_catalog(config: CatalogConfig) -> list[CandidateRecord]:
    source_root = config.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Structure source directory does not exist: {source_root}")
    discovered = discover_candidates(source_root)
    discovered = discovered[config.offset :]
    if config.limit is not None:
        discovered = discovered[: config.limit]

    records: list[CandidateRecord] = []
    structures: dict[str, Structure] = {}
    for number, (raw_file, candidate_id, legacy_id, target_formula, target_sg) in enumerate(
        discovered, 1
    ):
        global_index = config.offset + number - 1
        if global_index in config.skip_indices:
            selected_file, selected_stage = _select_structure(raw_file, legacy_id)
            record = CandidateRecord(
                candidate_id=candidate_id,
                legacy_id=legacy_id,
                source_collection=raw_file.parent.name,
                source_file=str(raw_file),
                selected_file=str(selected_file),
                selected_stage=selected_stage,
                target_formula=target_formula,
                target_spacegroup=target_sg,
                validation_reason="native_spglib_failure",
                classification="symmetry_failed",
                error="Excluded after isolated process reproduced a native spglib failure",
            )
            structure = None
        else:
            record, structure = _analyze_candidate(
                raw_file, candidate_id, legacy_id, target_formula, target_sg, config
            )
        records.append(record)
        if structure is not None:
            structures[candidate_id] = structure
            if config.materialize and config.curated_root is not None:
                _materialize_structure(record, structure, config.curated_root.resolve())
        if number % 250 == 0 or number == len(discovered):
            print(f"Cataloged {number}/{len(discovered)} candidates", flush=True)

    # Persist the expensive parse/symmetry pass before optional matching.
    write_records(records, config.output_root.resolve())
    _write_summary(records, config.output_root.resolve(), config)
    if config.deduplicate:
        _deduplicate(records, structures)
    write_records(records, config.output_root.resolve())
    _write_summary(records, config.output_root.resolve(), config)
    provenance = config.output_root.resolve() / "catalog_config.json"
    payload = asdict(config)
    payload["source_root"] = str(config.source_root)
    payload["output_root"] = str(config.output_root)
    payload["curated_root"] = str(config.curated_root) if config.curated_root else None
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return records


def merge_catalog_shards(shard_roots: Iterable[Path], output_root: Path) -> list[CandidateRecord]:
    """Merge independently generated catalog shards and regenerate summaries."""
    records_by_id: dict[str, CandidateRecord] = {}
    source_root: Path | None = None
    for shard_root in sorted(shard_roots):
        catalog_file = shard_root / "structure_catalog.jsonl"
        if not catalog_file.is_file():
            continue
        config_file = shard_root / "catalog_config.json"
        if config_file.is_file() and source_root is None:
            source_root = Path(json.loads(config_file.read_text(encoding="utf-8"))["source_root"])
        for line in catalog_file.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            record = CandidateRecord(**data)
            records_by_id[data["candidate_id"]] = _upgrade_record(record)
    records = sorted(records_by_id.values(), key=lambda record: record.candidate_id)
    output_root = output_root.resolve()
    write_records(records, output_root)
    summary_config = CatalogConfig(
        source_root=source_root or Path("unknown"),
        output_root=output_root,
        materialize=False,
        deduplicate=False,
    )
    _write_summary(records, output_root, summary_config)
    return records


def deduplicate_catalog(catalog_jsonl: Path, output_root: Path) -> list[CandidateRecord]:
    """Deduplicate a completed catalog without repeating symmetry analysis."""
    records = [
        _upgrade_record(CandidateRecord(**json.loads(line)))
        for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    structures: dict[str, Structure] = {}
    for record in records:
        if record.condition_valid:
            try:
                structures[record.candidate_id] = Structure.from_file(record.selected_file)
            except Exception:
                record.valid_structure = False
                record.validation_reason = "reload_failed_during_deduplication"
    _deduplicate(records, structures)
    output_root = output_root.resolve()
    write_records(records, output_root)
    config = CatalogConfig(
        source_root=Path(records[0].source_file).parents[1] if records else Path("unknown"),
        output_root=output_root,
        materialize=False,
        deduplicate=True,
    )
    _write_summary(records, output_root, config)
    return records


def materialize_catalog(catalog_jsonl: Path, curated_root: Path) -> int:
    """Write canonical copies from a completed catalog without recomputing symmetry."""
    count = 0
    mappings: list[dict[str, object]] = []
    for number, line in enumerate(catalog_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        record = _upgrade_record(CandidateRecord(**json.loads(line)))
        if not record.parse_valid or not record.symmetry_valid or not record.canonical_name:
            continue
        try:
            structure = Structure.from_file(record.selected_file)
            _materialize_structure(record, structure, curated_root.resolve())
            count += 1
            mappings.append(
                {
                    "candidate_id": record.candidate_id,
                    "legacy_id": record.legacy_id,
                    "source_file": record.source_file,
                    "selected_file": record.selected_file,
                    "canonical_file": record.curated_file,
                    "classification": record.classification,
                    "reduced_formula": record.reduced_formula,
                    "actual_spacegroup": record.actual_spacegroup,
                }
            )
        except Exception:
            continue
        if number % 500 == 0:
            print(f"Materialized canonical CIFs for {number} catalog rows", flush=True)
    mapping_file = curated_root.resolve() / "canonical_mapping.csv"
    with mapping_file.open("w", encoding="utf-8-sig", newline="") as handle:
        if mappings:
            writer = csv.DictWriter(handle, fieldnames=list(mappings[0]))
            writer.writeheader()
            writer.writerows(mappings)
    return count


def copy_catalog_artifacts(source: Path, destination: Path) -> None:
    """Copy the small catalog reports without copying the curated CIF archive."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("structure_catalog.csv", "structure_catalog.jsonl", "catalog_summary.json"):
        shutil.copy2(source / name, destination / name)
