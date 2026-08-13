#!/usr/bin/env python3
"""Validate the unified SymmCD, Pydantic AI, and atomate2 environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SYMMCD_ROOT = SCRIPT_DIR.parent


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _add_source_paths(pydantic_root: Path) -> None:
    for path in (
        SYMMCD_ROOT,
        pydantic_root / "pydantic_ai_slim",
        pydantic_root / "pydantic_graph",
    ):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def validate(python_only: bool, load_models: bool) -> tuple[dict, bool]:
    pydantic_root = Path(
        os.getenv("PYDANTIC_AI_SOURCE", SYMMCD_ROOT / "vendor" / "pydantic-ai")
    ).expanduser().resolve()
    band_root = Path(
        os.getenv(
            "BAND_ANALYZER_ROOT",
            SYMMCD_ROOT / "band_analysis",
        )
    ).expanduser().resolve()
    _add_source_paths(pydantic_root)

    report: dict = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "symmcd_root": str(SYMMCD_ROOT),
        "pydantic_ai_root": str(pydantic_root),
        "band_analyzer_root": str(band_root),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    def ok(name: str, detail: object = True) -> None:
        report["checks"][name] = {"ok": True, "detail": detail}

    def fail(name: str, detail: str) -> None:
        report["checks"][name] = {"ok": False, "detail": detail}
        report["errors"].append(f"{name}: {detail}")

    if sys.version_info[:2] == (3, 11):
        ok("python_3_11", report["python_version"])
    else:
        fail("python_3_11", f"expected Python 3.11, found {report['python_version']}")

    modules = {
        "torch": "torch",
        "torch_geometric": "torch-geometric",
        "torch_scatter": "torch-scatter",
        "pytorch_lightning": "pytorch-lightning",
        "mace": "mace-torch",
        "ase": "ase",
        "pymatgen": "pymatgen",
        "spglib": "spglib",
        "pyxtal": "pyxtal",
        "atomate2": "atomate2",
        "jobflow": "jobflow",
        "pydantic": "pydantic",
        "openai": "openai",
    }
    for module_name, distribution in modules.items():
        try:
            importlib.import_module(module_name)
            ok(f"import_{module_name}", _version(distribution))
        except Exception as exc:
            fail(f"import_{module_name}", f"{type(exc).__name__}: {exc}")

    try:
        import torch

        ok(
            "torch_runtime",
            {
                "version": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )
    except Exception:
        pass

    checkpoint = SYMMCD_ROOT / "epoch699.ckpt"
    if checkpoint.is_file():
        ok("checkpoint", {"path": str(checkpoint), "bytes": checkpoint.stat().st_size})
    else:
        fail("checkpoint", f"not found: {checkpoint}")

    mace_model_setting = os.getenv("MACE_MODEL")
    local_mace_model = SYMMCD_ROOT / "macemodel" / "2023-12-03-mace-128-L1_epoch-199.model"
    if mace_model_setting and Path(mace_model_setting).expanduser().is_file():
        ok("mace_model", str(Path(mace_model_setting).expanduser().resolve()))
    elif local_mace_model.is_file():
        ok("mace_model", str(local_mace_model))
        report["warnings"].append(
            "MACE_MODEL is not a local file; set it to the reported local model on offline nodes"
        )
    else:
        fail("mace_model", "no local MACE model found")

    structure_input_dir = Path(
        os.getenv("STRUCTURE_INPUT_DIR", str(SYMMCD_ROOT / "input_structures"))
    ).expanduser()
    if structure_input_dir.is_dir():
        ok("structure_input_dir", str(structure_input_dir.resolve()))
    else:
        fail("structure_input_dir", f"not found: {structure_input_dir}")

    energy_output_root = Path(
        os.getenv(
            "ENERGY_OUTPUT_ROOT",
            str(SYMMCD_ROOT / "calculation_results" / "mace_energy"),
        )
    ).expanduser()
    energy_output_root.mkdir(parents=True, exist_ok=True)
    ok("energy_output_root", str(energy_output_root.resolve()))

    if (pydantic_root / "pydantic_ai_slim" / "pydantic_ai").is_dir():
        ok("pydantic_ai_source", str(pydantic_root))
    else:
        fail("pydantic_ai_source", f"invalid source checkout: {pydantic_root}")

    metadata_shim = SYMMCD_ROOT / ".local_packages" / "pydantic_ai_slim-0.0.0+local.dist-info" / "METADATA"
    if metadata_shim.is_file():
        ok("pydantic_ai_metadata", str(metadata_shim))
    else:
        fail(
            "pydantic_ai_metadata",
            "missing .local_packages metadata shim; copy hidden files from the symmcd directory",
        )

    if (band_root / "agent_runner.py").is_file():
        ok("band_analyzer_source", str(band_root))
    else:
        fail("band_analyzer_source", f"agent_runner.py not found under {band_root}")

    try:
        import structure_agent  # noqa: F401

        ok("import_structure_agent")
    except Exception as exc:
        fail("import_structure_agent", f"{type(exc).__name__}: {exc}")

    if band_root.is_dir():
        sys.path.insert(0, str(band_root))
        try:
            import workflow_builder

            workflow_builder.get_makers()
            ok("import_band_workflow", "workflow imports and makers initialize")
        except Exception as exc:
            fail("import_band_workflow", f"{type(exc).__name__}: {exc}")

    if load_models and checkpoint.is_file():
        try:
            import torch
            from workflow_sym import WorkflowConfig, _load_generation_model, _load_mace_calculator

            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = WorkflowConfig(
                formula="BN",
                spacegroup_number=194,
                num_samples=1,
                checkpoint_path=checkpoint,
                output_root=SYMMCD_ROOT / "generated_structures",
                mace_model=mace_model_setting or str(local_mace_model),
                mace_device=device,
            )
            _load_generation_model(config, device)
            _load_mace_calculator(config)
            ok("load_models", {"device": device})
        except Exception as exc:
            fail("load_models", f"{type(exc).__name__}: {exc}")

    if not python_only:
        atomate_config = os.getenv("ATOMATE2_CONFIG_FILE")
        if atomate_config and Path(atomate_config).expanduser().is_file():
            ok("atomate2_config", str(Path(atomate_config).expanduser().resolve()))
        else:
            fail("atomate2_config", "ATOMATE2_CONFIG_FILE does not point to a file")

        jobflow_config = os.getenv("JOBFLOW_CONFIG_FILE")
        if jobflow_config and Path(jobflow_config).expanduser().is_file():
            ok("jobflow_config", str(Path(jobflow_config).expanduser().resolve()))
        else:
            fail("jobflow_config", "JOBFLOW_CONFIG_FILE does not point to a file")

        potcar_root = os.getenv("PMG_VASP_PSP_DIR")
        if potcar_root and Path(potcar_root).expanduser().is_dir():
            ok("potcar_root", str(Path(potcar_root).expanduser().resolve()))
        else:
            fail("potcar_root", "PMG_VASP_PSP_DIR does not point to a directory")

        for executable in ("irvsp", "srun"):
            path = shutil.which(executable)
            if path:
                ok(f"executable_{executable}", path)
            else:
                fail(f"executable_{executable}", "not found on PATH")

        if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
            ok("llm_api_key", "configured")
        else:
            fail("llm_api_key", "LLM_API_KEY or OPENAI_API_KEY is not set")

    report["ready"] = not report["errors"]
    return report, report["ready"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Skip VASP, POTCAR, scheduler, IRVSP, and API checks.",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Also load the 700 MB checkpoint and local MACE model.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    args = parser.parse_args()

    report, ready = validate(args.python_only, args.load_models)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name, check in report["checks"].items():
            status = "OK" if check["ok"] else "FAIL"
            print(f"[{status}] {name}: {check['detail']}")
        for warning in report["warnings"]:
            print(f"[WARN] {warning}")
        print(f"\nEnvironment ready: {ready}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
