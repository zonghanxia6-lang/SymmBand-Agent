"""Run the band-analysis workflow for an explicit list of structure files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def _task_name(path: Path, index: int) -> str:
    name = path.stem
    if name.upper().startswith("POSCAR_"):
        name = name[7:]
    elif name.upper() == "POSCAR":
        name = "structure"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
    return f"{index:03d}_{name or 'structure'}"


def check_environment() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "python": sys.executable,
        "project_root": str(PROJECT_ROOT),
    }
    errors: list[str] = []
    for module_name in ("atomate2", "jobflow", "pymatgen"):
        try:
            module = __import__(module_name)
            checks[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            checks[module_name] = None
            errors.append(f"{module_name}: {exc}")

    import shutil

    checks["vasp_std"] = shutil.which("vasp_std")
    checks["irvsp"] = shutil.which("irvsp")
    if checks["irvsp"] is None:
        errors.append("irvsp: executable not found on PATH")
    checks["ready"] = not errors
    checks["errors"] = errors
    return checks


def run_band_calculations(structure_paths: list[Path], output_root: Path) -> dict[str, Any]:
    from jobflow import SETTINGS, run_locally
    from pymatgen.core import Structure

    from workflow_builder import build_degeneracy_flow

    output_root = output_root.resolve()
    calculations_dir = output_root / "calculations"
    bands_dir = output_root / "bands"
    calculations_dir.mkdir(parents=True, exist_ok=True)
    bands_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, source_path in enumerate(structure_paths, start=1):
        source_path = source_path.resolve()
        task_name = _task_name(source_path, index)
        task_dir = calculations_dir / task_name
        image_path = bands_dir / f"band_{task_name}.png"
        item: dict[str, Any] = {
            "source_structure": str(source_path),
            "task_name": task_name,
            "calculation_directory": str(task_dir),
            "band_image": str(image_path),
            "status": "failed",
            "error": None,
        }

        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"structure file not found: {source_path}")

            structure = Structure.from_file(source_path)
            flow = build_degeneracy_flow(structure, task_name, str(bands_dir))

            docs_dir = task_dir / "docs"
            data_dir = task_dir / "data"
            docs_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            SETTINGS.JOB_STORE.docs_store.paths = [str(docs_dir / "doc.json")]
            SETTINGS.JOB_STORE.additional_stores["data"].paths = [str(data_dir / "data.json")]

            run_locally(flow, create_folders=True, root_dir=str(task_dir))
            if not image_path.is_file():
                raise RuntimeError("workflow finished without producing the expected band image")
            item["status"] = "completed"
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    completed_images = [
        item["band_image"] for item in results if item["status"] == "completed"
    ]
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(structure_paths),
        "completed_count": len(completed_images),
        "failed_count": len(structure_paths) - len(completed_images),
        "output_root": str(output_root),
        "bands_directory": str(bands_dir),
        "band_images": completed_images,
        "results": results,
    }


def _read_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_paths = payload.get("structure_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("manifest must contain a non-empty structure_paths list")
    return [Path(str(item)).expanduser() for item in raw_paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent adapter for the band-analysis workflow")
    parser.add_argument("--check", action="store_true", help="Check Python dependencies and commands")
    parser.add_argument("--manifest", type=Path, help="JSON file containing structure_paths")
    parser.add_argument("--output-root", type=Path, help="Directory for calculations and band images")
    parser.add_argument("--report", type=Path, help="JSON result report path")
    args = parser.parse_args()

    if args.check:
        result = check_environment()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 2

    if args.manifest is None or args.output_root is None or args.report is None:
        parser.error("--manifest, --output-root, and --report are required unless --check is used")

    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_band_calculations(_read_manifest(args.manifest), args.output_root)
    except Exception as exc:
        result = {
            "requested_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "output_root": str(args.output_root.resolve()),
            "bands_directory": str((args.output_root / "bands").resolve()),
            "band_images": [],
            "results": [],
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }

    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result.get("fatal_error") and result["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
