"""Fail-fast audit for files and secrets required before publishing the repository."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    ".env.agent.example",
    ".gitattributes",
    ".gitignore",
    ".local_packages/pydantic_ai_slim-0.0.0+local.dist-info/METADATA",
    "README.md",
    "WORKFLOW.md",
    "RELEASE_MANIFEST.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "structure_agent.py",
    "workflow_sym.py",
    "mace_energy.py",
    "band_workflow.py",
    "band_result_analysis.py",
    "calculation_results/README.md",
    "emergent_particles.py",
    "inverse_design_cli.py",
    "epoch699.ckpt",
    "macemodel/2023-12-03-mace-128-L1_epoch-199.model",
    "emergent particles/emergent_particles_index.json",
    "research_models/particles/particle_dp_surrogate.joblib",
    "cluster_environment/environment.yml",
    "cluster_environment/install.sh",
    "cluster_environment/validate_environment.py",
    "vendor/pydantic-ai/LICENSE",
    "topological-material-discovery-workflow-v3-agent-integrated.svg",
    "symmband-agent-capabilities.svg",
)

FORBIDDEN_TRACKED_PATTERNS = (
    re.compile(r"(^|/)\.env\.agent$"),
    re.compile(r"(^|/)POTCAR", re.IGNORECASE),
    re.compile(r"(^|/)calculation_results/(?!README\.md$)"),
    re.compile(r"(^|/)research_data/"),
    re.compile(r"^emergent particles/.*\.pdf$", re.IGNORECASE),
)

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?im)^LLM_API_KEY=(?!replace-with-|your-|[\"']?\$|$)\S+"),
    re.compile(r"(?im)^OPENAI_API_KEY=(?!replace-with-|your-|[\"']?\$|$)\S+"),
)

TEXT_SUFFIXES = {
    ".cfg", ".csv", ".example", ".ini", ".json", ".md", ".py", ".sh",
    ".toml", ".txt", ".yaml", ".yml", ".ps1", ".svg",
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    tracked = [line for line in git("ls-files").splitlines() if line]
    for relative in tracked:
        if any(pattern.search(relative) for pattern in FORBIDDEN_TRACKED_PATTERNS):
            errors.append(f"forbidden tracked file: {relative}")
        # Vendored upstream source is immutable and may contain documented fake key
        # examples; its integrity is governed by the retained upstream license.
        if relative.startswith("vendor/"):
            continue
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible secret in tracked file: {relative}")
                break

    attributes = git("check-attr", "filter", "--", "epoch699.ckpt",
                     "macemodel/2023-12-03-mace-128-L1_epoch-199.model")
    if attributes.count("filter: lfs") != 2:
        errors.append("checkpoint and MACE model must both use Git LFS")

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Release audit passed: {len(REQUIRED)} required files, {len(tracked)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
