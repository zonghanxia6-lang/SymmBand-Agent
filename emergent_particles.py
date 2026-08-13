"""Token-efficient lookup for emergent particles by crystallographic space group."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "emergent particles" / "emergent_particles_index.json"


@dataclass(frozen=True)
class HighSymmetryPath:
    line_label: str
    path: str
    source_pdf_page: int


@dataclass(frozen=True)
class EmergentParticle:
    abbreviation: str
    name: str
    paths: list[HighSymmetryPath] | None = None


@dataclass(frozen=True)
class EmergentParticleLookupResult:
    spacegroup_number: int
    soc: bool
    essential: list[EmergentParticle]
    accidental: list[EmergentParticle]
    all_particles: list[EmergentParticle]
    source_title: str
    source_file: str
    source_table: str
    source_pdf_page: int
    path_source_section: str


@lru_cache(maxsize=4)
def _load_index(index_path: str) -> dict[str, object]:
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"Emergent-particle index not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def lookup_emergent_particles(
    spacegroup_number: int,
    soc: bool,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> EmergentParticleLookupResult:
    if not 1 <= spacegroup_number <= 230:
        raise ValueError("spacegroup_number must be between 1 and 230")

    index = _load_index(str(index_path.resolve()))
    mode = "with_soc" if soc else "without_soc"
    record = index["space_groups"][str(spacegroup_number)][mode]
    essential = [EmergentParticle(**item) for item in record["essential"]]
    accidental_paths = record.get("accidental_paths", {})
    accidental = [
        EmergentParticle(
            **item,
            paths=[
                HighSymmetryPath(**path_record)
                for path_record in accidental_paths.get(item["abbreviation"], [])
            ],
        )
        for item in record["accidental"]
    ]

    unique: dict[str, EmergentParticle] = {}
    for particle in essential + accidental:
        existing = unique.get(particle.abbreviation)
        if existing is None or (existing.paths is None and particle.paths):
            unique[particle.abbreviation] = particle

    source = index["source"]
    table = source["tables"][mode]
    return EmergentParticleLookupResult(
        spacegroup_number=spacegroup_number,
        soc=soc,
        essential=essential,
        accidental=accidental,
        all_particles=list(unique.values()),
        source_title=source["title"],
        source_file=source["filename"],
        source_table=table["table"],
        source_pdf_page=table["pdf_page"],
        path_source_section=source["path_sections"][mode],
    )
