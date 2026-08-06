"""Subprocess bridge from the SymmCD agent to the VASP band workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYZER_ROOT = PROJECT_ROOT / "band_analysis"


@dataclass(frozen=True)
class BandWorkflowConfig:
    structure_paths: list[Path]
    output_root: Path
    analyzer_root: Path = DEFAULT_ANALYZER_ROOT
    python_executable: Path = Path(sys.executable)
    timeout_seconds: int = 0

    def validate(self) -> None:
        if not self.structure_paths:
            raise ValueError("at least one POSCAR is required for band analysis")
        missing = [str(path) for path in self.structure_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"structure files not found: {', '.join(missing)}")
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"band Python executable not found: {self.python_executable}")
        runner = self.analyzer_root / "agent_runner.py"
        if not runner.is_file():
            raise FileNotFoundError(f"band workflow adapter not found: {runner}")
        if self.timeout_seconds < 0:
            raise ValueError("band timeout cannot be negative")


@dataclass(frozen=True)
class BandWorkflowResult:
    requested_count: int
    completed_count: int
    failed_count: int
    output_root: str
    bands_directory: str
    band_images: list[str]
    report_file: str
    log_file: str
    failures: list[str]


def check_band_environment(analyzer_root: Path, python_executable: Path) -> dict:
    runner = analyzer_root.resolve() / "agent_runner.py"
    python_executable = python_executable.resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"band workflow adapter not found: {runner}")
    if not python_executable.is_file():
        raise FileNotFoundError(f"band Python executable not found: {python_executable}")

    completed = subprocess.run(
        [str(python_executable), str(runner), "--check"],
        cwd=str(analyzer_root.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"invalid band environment check output: {details}") from exc
    if completed.returncode != 0 or not payload.get("ready"):
        details = "; ".join(payload.get("errors", [])) or "unknown dependency error"
        raise RuntimeError(f"band environment is not ready: {details}")
    return payload


def run_band_workflow(config: BandWorkflowConfig) -> BandWorkflowResult:
    config.validate()
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "structures_manifest.json"
    report_path = output_root / "band_report.json"
    log_path = output_root / "band_workflow.log"

    manifest_path.write_text(
        json.dumps(
            {"structure_paths": [str(path.resolve()) for path in config.structure_paths]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    command = [
        str(config.python_executable),
        str(config.analyzer_root / "agent_runner.py"),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--report",
        str(report_path),
    ]

    timeout = config.timeout_seconds or None
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=str(config.analyzer_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"band workflow exceeded the {config.timeout_seconds}-second timeout; see {log_path}"
            ) from exc

    if not report_path.is_file():
        raise RuntimeError(
            f"band workflow exited with code {completed.returncode} without a report; see {log_path}"
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("fatal_error"):
        raise RuntimeError(f"{payload['fatal_error']}; see {log_path}")

    failures = [
        f"{item.get('source_structure')}: {item.get('error')}"
        for item in payload.get("results", [])
        if item.get("status") != "completed"
    ]
    return BandWorkflowResult(
        requested_count=int(payload.get("requested_count", len(config.structure_paths))),
        completed_count=int(payload.get("completed_count", 0)),
        failed_count=int(payload.get("failed_count", len(failures))),
        output_root=str(output_root),
        bands_directory=str(Path(payload.get("bands_directory", output_root / "bands")).resolve()),
        band_images=[str(Path(path).resolve()) for path in payload.get("band_images", [])],
        report_file=str(report_path),
        log_file=str(log_path),
        failures=failures,
    )


def config_from_environment(structure_paths: list[str], output_root: Path) -> BandWorkflowConfig:
    analyzer_root = Path(os.getenv("BAND_ANALYZER_ROOT", str(DEFAULT_ANALYZER_ROOT))).expanduser()
    python_executable = Path(os.getenv("BAND_PYTHON", sys.executable)).expanduser()
    timeout_seconds = int(os.getenv("BAND_TIMEOUT_SECONDS", "0"))
    return BandWorkflowConfig(
        structure_paths=[Path(path) for path in structure_paths],
        output_root=output_root,
        analyzer_root=analyzer_root,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
