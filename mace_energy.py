"""Single-point MACE energy calculations for user-provided structures."""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ase.io import read


_MACE_CACHE: dict[tuple[str, str], Any] = {}
_MACE_CACHE_LOCK = threading.Lock()
_MACE_CALCULATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class MaceEnergyConfig:
    structure_filename: str
    input_directory: Path
    output_root: Path
    mace_model: str
    mace_device: str


@dataclass(frozen=True)
class MaceEnergyResult:
    structure_filename: str
    structure_path: str
    formula: str
    atom_count: int
    total_energy_ev: float
    energy_per_atom_ev: float
    calculation_type: str
    mace_model: str
    mace_device: str
    output_directory: str
    report_file: str


def load_mace_calculator(model: str, device: str):
    """Load and cache one MACE calculator per model/device pair.

    MACE is imported here instead of at module import time.  Its training utilities
    can indirectly import torchvision, whose image extension is irrelevant to path
    validation and calculator-injected tests.  Other ML dependencies used by the main
    CLI may still import torchvision, so the environment's native libraries must also
    remain binary-compatible.
    """
    from mace.calculators import mace_mp

    cache_key = (str(model), device)
    with _MACE_CACHE_LOCK:
        if cache_key not in _MACE_CACHE:
            _MACE_CACHE[cache_key] = mace_mp(
                model=model,
                default_dtype="float64",
                device=device,
            )
    return _MACE_CACHE[cache_key]


def _structure_format(path: Path) -> str:
    lower_name = path.name.lower()
    if path.suffix.lower() == ".cif":
        return "cif"
    if (
        path.suffix.lower() in {".vasp", ".poscar"}
        or lower_name == "poscar"
        or lower_name.startswith("poscar_")
        or lower_name.startswith("poscar-")
    ):
        return "vasp"
    raise ValueError(
        "unsupported structure format; use .cif, .vasp, .poscar, POSCAR, or POSCAR_*"
    )


def resolve_structure_file(input_directory: Path, structure_filename: str) -> Path:
    """Resolve a direct child of the configured input directory without path traversal."""
    requested = structure_filename.strip()
    if not requested:
        raise ValueError("structure filename cannot be empty")
    if Path(requested).name != requested or Path(requested).is_absolute():
        raise ValueError("structure filename must not contain a directory path")

    root = input_directory.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"structure input directory not found: {root}")

    candidate = root / requested
    if not candidate.is_file():
        matches = [path for path in root.iterdir() if path.is_file() and path.name.lower() == requested.lower()]
        if len(matches) == 1:
            candidate = matches[0]
        else:
            available = sorted(
                path.name
                for path in root.iterdir()
                if path.is_file() and path.name != "README.md"
            )
            suffix = f" Available files: {', '.join(available)}" if available else ""
            raise FileNotFoundError(f"structure file not found in {root}: {requested}.{suffix}")

    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("structure file must remain inside the configured input directory")
    _structure_format(resolved)
    return resolved


def calculate_mace_energy(
    config: MaceEnergyConfig,
    *,
    calculator: Any | None = None,
) -> MaceEnergyResult:
    """Calculate an unrelaxed, single-point MACE potential energy."""
    structure_path = resolve_structure_file(config.input_directory, config.structure_filename)
    atoms = read(structure_path, index=0, format=_structure_format(structure_path))
    atom_count = len(atoms)
    if atom_count < 1:
        raise ValueError(f"structure contains no atoms: {structure_path}")

    mace_calculator = (
        calculator
        if calculator is not None
        else load_mace_calculator(config.mace_model, config.mace_device)
    )
    atoms.calc = mace_calculator
    with _MACE_CALCULATION_LOCK:
        total_energy = float(atoms.get_potential_energy())

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", structure_path.stem).strip("._") or "structure"
    output_directory = config.output_root.expanduser().resolve() / f"{safe_stem}_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)
    report_path = output_directory / "mace_energy.json"

    result = MaceEnergyResult(
        structure_filename=structure_path.name,
        structure_path=str(structure_path),
        formula=atoms.get_chemical_formula(mode="hill", empirical=False),
        atom_count=atom_count,
        total_energy_ev=total_energy,
        energy_per_atom_ev=total_energy / atom_count,
        calculation_type="MACE single-point potential energy (unrelaxed)",
        mace_model=str(config.mace_model),
        mace_device=config.mace_device,
        output_directory=str(output_directory),
        report_file=str(report_path),
    )
    report_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
