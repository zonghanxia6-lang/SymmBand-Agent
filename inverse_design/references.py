"""Reference-database snapshots and StructureMatcher novelty evaluation."""

from __future__ import annotations

import csv
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_materials_project_snapshot(
    catalog_jsonl: Path,
    output_jsonl: Path,
    api_key: str | None = None,
    batch_size: int = 50,
) -> int:
    """Download all MP structures sharing a reduced formula with catalog candidates."""
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency with: pip install mp-api") from exc

    key = api_key or os.getenv("MP_API_KEY")
    if not key:
        raise RuntimeError("Set MP_API_KEY or pass --api-key to download an MP snapshot")
    formulas = sorted(
        {
            row["reduced_formula"]
            for row in _read_jsonl(catalog_jsonl)
            if row.get("valid_structure") and row.get("reduced_formula")
        }
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        with MPRester(key, mute_progress_bars=True) as mpr:
            database_version = str(mpr.db_version)
            for start in range(0, len(formulas), batch_size):
                batch = formulas[start : start + batch_size]
                docs = mpr.materials.summary.search(
                    formula=batch,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "structure",
                        "formation_energy_per_atom",
                        "energy_above_hull",
                        "symmetry",
                    ],
                )
                for doc in docs:
                    material_id = str(doc.material_id)
                    if material_id in seen:
                        continue
                    seen.add(material_id)
                    structure = doc.structure
                    row = {
                        "database": "Materials Project",
                        "database_version": database_version,
                        "material_id": material_id,
                        "formula": doc.formula_pretty,
                        "reduced_formula": structure.composition.reduced_formula,
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "energy_above_hull": doc.energy_above_hull,
                        "structure": structure.as_dict(),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                print(
                    f"Downloaded MP references for {min(start + batch_size, len(formulas))}/"
                    f"{len(formulas)} formulas",
                    flush=True,
                )
    metadata = output_jsonl.with_suffix(".metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "database": "Materials Project",
                "database_version": database_version,
                "formula_count": len(formulas),
                "structure_count": count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return count


def import_reference_structures(paths: Iterable[Path], output_jsonl: Path, database: str) -> int:
    """Create a reference snapshot from local CIF/POSCAR files."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for root in paths:
            files = [root] if root.is_file() else list(root.rglob("*"))
            for path in files:
                if not path.is_file() or path.suffix.lower() not in {".cif", ".vasp", ".poscar"}:
                    continue
                try:
                    structure = Structure.from_file(path)
                except Exception:
                    continue
                row = {
                    "database": database,
                    "database_version": "local_snapshot",
                    "material_id": f"{database}:{path.name}",
                    "formula": structure.composition.formula,
                    "reduced_formula": structure.composition.reduced_formula,
                    "source_file": str(path.resolve()),
                    "structure": structure.as_dict(),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def import_mp20_snapshot(mp20_root: Path, output_jsonl: Path) -> dict:
    """Convert the local SymmCD MP20 CSV splits into a frozen structure snapshot."""
    split_files = [mp20_root / f"{split}.csv" for split in ("train", "val", "test")]
    missing = [str(path) for path in split_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MP20 split files: {', '.join(missing)}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    parse_failed = 0
    duplicate_material_ids = 0
    seen_ids: set[str] = set()
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as output:
        for split, split_file in zip(("train", "val", "test"), split_files):
            split_count = 0
            with split_file.open(encoding="utf-8", newline="") as handle:
                for source_row in csv.DictReader(handle):
                    material_id = source_row.get("material_id", "").strip()
                    if material_id in seen_ids:
                        duplicate_material_ids += 1
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            structure = Structure.from_str(source_row["cif"], fmt="cif")
                    except Exception:
                        parse_failed += 1
                        continue
                    seen_ids.add(material_id)
                    row = {
                        "database": "MP20",
                        "database_version": "local_symmcd_mp20_snapshot",
                        "split": split,
                        "material_id": material_id,
                        "formula": source_row.get("pretty_formula") or structure.composition.formula,
                        "reduced_formula": structure.composition.reduced_formula,
                        "formation_energy_per_atom": _optional_float(
                            source_row.get("formation_energy_per_atom")
                        ),
                        "energy_above_hull": _optional_float(source_row.get("e_above_hull")),
                        "spacegroup_number": _optional_int(source_row.get("spacegroup.number")),
                        "structure": structure.as_dict(),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    split_count += 1
            counts[split] = split_count
    metadata = {
        "database": "MP20",
        "database_version": "local_symmcd_mp20_snapshot",
        "source_root": str(mp20_root.resolve()),
        "split_counts": counts,
        "structure_count": sum(counts.values()),
        "parse_failed_count": parse_failed,
        "duplicate_material_id_count": duplicate_material_ids,
        "scientific_scope": (
            "This snapshot supports novelty claims relative to the MP20 dataset only, "
            "not the complete current Materials Project database."
        ),
    }
    output_jsonl.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if str(value or "").strip() else None
    except ValueError:
        return None


def _optional_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if str(value or "").strip() else None
    except ValueError:
        return None


def evaluate_novelty(
    catalog_jsonl: Path,
    reference_jsonl: list[Path],
    output_csv: Path,
) -> list[dict]:
    """Use formula prefiltering followed by full StructureMatcher comparison."""
    catalog = _read_jsonl(catalog_jsonl)
    relevant_formulas = {
        row["reduced_formula"]
        for row in catalog
        if row.get("valid_structure") and row.get("reduced_formula")
    }
    references: dict[str, list[tuple[dict, Structure]]] = defaultdict(list)
    for snapshot in reference_jsonl:
        for row in _read_jsonl(snapshot):
            if row.get("reduced_formula") not in relevant_formulas:
                continue
            try:
                references[row["reduced_formula"]].append((row, Structure.from_dict(row["structure"])))
            except Exception:
                continue
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
    results: list[dict] = []
    for number, row in enumerate(catalog, 1):
        if not row.get("valid_structure"):
            result = {
                "candidate_id": row["candidate_id"],
                "novelty_status": "not_evaluated_invalid",
                "is_novel": "",
                "mp20_training_set_novel": "",
                "novelty_type": "not_evaluated_invalid",
                "composition_novel": "",
                "structure_novel": "",
                "global_novelty_evaluated": False,
                "reference_formula_covered": "",
                "matched_reference_ids": "",
                "reference_candidate_count": 0,
            }
        else:
            candidates = references.get(row.get("reduced_formula", ""), [])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                structure = Structure.from_file(row["selected_file"])
            matches = [ref["material_id"] for ref, ref_structure in candidates if matcher.fit(structure, ref_structure)]
            result = {
                "candidate_id": row["candidate_id"],
                "novelty_status": (
                    "matched"
                    if matches else ("structure_novel" if candidates else "composition_novel")
                ),
                "is_novel": not matches,
                "mp20_training_set_novel": not matches,
                "novelty_type": (
                    "known_match" if matches else ("structure_novel" if candidates else "composition_novel")
                ),
                "composition_novel": not candidates,
                "structure_novel": bool(candidates and not matches),
                "global_novelty_evaluated": False,
                "reference_formula_covered": bool(candidates),
                "matched_reference_ids": ";".join(matches),
                "reference_candidate_count": len(candidates),
            }
        results.append(result)
        if number % 250 == 0:
            print(f"Evaluated novelty for {number} candidates", flush=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    return results
