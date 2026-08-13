"""Random-space-group baseline and post-generation particle compatibility."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

from emergent_particles import lookup_emergent_particles


def generate_random_spacegroup_baseline(
    targets_json: Path,
    output_dir: Path,
    seed: int,
    attempts_per_target: int = 10,
) -> list[dict]:
    """Generate pyxtal structures with the exact registered formula and SG targets."""
    from pyxtal import pyxtal

    targets = json.loads(targets_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for target_number, target in enumerate(targets):
        formula = str(target["formula"])
        spacegroup = int(target["spacegroup"])
        count = int(target.get("count", 1))
        composition = Composition(formula)
        species = [element.symbol for element in composition.elements]
        amounts = [int(round(composition[element])) for element in composition.elements]
        for sample_number in range(1, count + 1):
            generated = None
            error = "generation failed"
            sample_seed = int(rng.integers(0, 2**31 - 1))
            for attempt in range(1, attempts_per_target + 1):
                crystal = pyxtal()
                try:
                    crystal.from_random(
                        dim=3,
                        group=spacegroup,
                        species=species,
                        numIons=amounts,
                        seed=sample_seed + attempt - 1,
                    )
                    generated = crystal.to_pymatgen()
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            candidate_id = f"random_sg__{formula}_sg{spacegroup}_{target_number:04d}_{sample_number:04d}"
            output_file = output_dir / f"{candidate_id}.cif"
            if generated is not None:
                generated.to(filename=output_file)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "arm": "random_spacegroup",
                    "formula": formula,
                    "spacegroup": spacegroup,
                    "seed": sample_seed,
                    "generated": generated is not None,
                    "output_file": str(output_file) if generated is not None else "",
                    "error": "" if generated is not None else error,
                }
            )
    manifest = output_dir / "random_baseline_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def particle_compatible_candidates(
    catalog_jsonl: Path,
    target_particle: str,
    output_csv: Path,
    soc: bool = True,
    require_strict_stoichiometry: bool = True,
) -> list[dict]:
    """Select structures whose recalculated SG permits the requested accidental particle."""
    rows = []
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        valid = (
            candidate.get("stoichiometry_valid")
            if require_strict_stoichiometry
            else candidate.get("condition_valid")
        )
        if not valid or not candidate.get("is_unique"):
            continue
        lookup = lookup_emergent_particles(int(candidate["actual_spacegroup"]), soc=soc)
        compatible = {particle.abbreviation: particle for particle in lookup.accidental}
        particle = compatible.get(target_particle)
        if particle is None:
            continue
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "reduced_formula": candidate["reduced_formula"],
                "actual_spacegroup": candidate["actual_spacegroup"],
                "target_particle": target_particle,
                "soc": soc,
                "compatible_paths": ";".join(path.path for path in (particle.paths or [])),
                "selected_file": candidate["selected_file"],
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def create_matched_random_targets(catalog_jsonl: Path, output_json: Path, per_pair: int) -> list[dict]:
    """Freeze the exact formula/SG distribution used by the random baseline."""
    counts: Counter[tuple[str, int]] = Counter()
    for line in catalog_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row.get("target_formula")
            and row.get("stoichiometry_valid")
            and row.get("target_spacegroup_match")
        ):
            counts[(row["target_formula"], int(row["target_spacegroup"]))] += 1
    targets = [
        {"formula": formula, "spacegroup": spacegroup, "count": min(count, per_pair)}
        for (formula, spacegroup), count in sorted(counts.items())
    ]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    return targets
