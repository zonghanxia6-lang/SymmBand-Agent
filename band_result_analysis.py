"""Reproducible accidental-degeneracy analysis for completed band results."""

from __future__ import annotations

import gzip
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from emergent_particles import DEFAULT_INDEX_PATH, EmergentParticle, lookup_emergent_particles
from physics_evidence import (
    PhysicsEvidenceAssessment,
    build_band_evidence_assessment,
    crossing_diagnostics,
    validate_analysis_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "calculation_results"
DEFAULT_ANALYZER_ROOT = PROJECT_ROOT / "band_analysis"


@dataclass(frozen=True)
class ParticlePathCandidate:
    abbreviation: str
    name: str
    line_label: str
    path: str
    source_pdf_page: int


@dataclass(frozen=True)
class AccidentalCrossing:
    crossing_number: int
    branch: str
    high_symmetry_path: str
    line_label: str | None
    k_point_interval: tuple[int, int]
    k_point_coordinates: tuple[list[float], list[float]]
    band_indices: tuple[int, int]
    irreps_swapped: tuple[str, str]
    energy_relative_to_fermi_ev: float
    energy_reference: str
    minimum_gap_ev: float | None
    neighboring_gaps_ev: tuple[float, float] | None
    gap_to_tolerance_ratio: float | None
    threshold_margin: str
    fermi_proximity: str
    classification: str
    candidates: list[ParticlePathCandidate]


@dataclass(frozen=True)
class PathMatchSummary:
    high_symmetry_path: str
    crossing_count: int
    classification: str
    candidates: list[ParticlePathCandidate]


@dataclass(frozen=True)
class EncyclopediaPathComparison:
    """One encyclopedia particle allowed on an observed high-symmetry path."""

    high_symmetry_path: str
    line_label: str
    crossing_count: int
    abbreviation: str
    name: str
    classification: str
    source_pdf_page: int


@dataclass(frozen=True)
class BandAccidentalDegeneracyReport:
    result_name: str
    result_directory: str
    band_job_directory: str
    band_image: str | None
    spacegroup_number: int
    spacegroup_symbol: str
    soc: bool
    detector: str
    energy_window_ev: float
    gap_tolerance_ev: float
    crossing_count: int
    electronic_converged: bool | None
    line_path_kpoint_count: int | None
    path_unique_particle_types: list[str]
    confirmed_particle_types: list[str]
    path_compatible_particle_types: list[str]
    path_summaries: list[PathMatchSummary]
    encyclopedia_path_table: list[EncyclopediaPathComparison]
    crossings: list[AccidentalCrossing]
    source_title: str
    source_file: str
    source_table: str
    source_table_pdf_page: int
    source_path_section: str
    report_file: str
    scientific_scope: str
    evidence_assessment: PhysicsEvidenceAssessment


def _normalized_name(value: str) -> str:
    value = re.sub(r"(?i)results?|analysis|calculation", "", value)
    value = value.replace("结果", "")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def find_result_directory(result_name: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Resolve a conversational material/result name without silently choosing an ambiguous run."""
    root = results_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Calculation-results directory not found: {root}")

    explicit = Path(result_name).expanduser()
    if explicit.is_absolute() and explicit.is_dir():
        return explicit.resolve()
    relative = (root / explicit).resolve()
    if relative.is_dir() and relative.parent == root:
        return relative

    query = _normalized_name(result_name)
    if not query:
        raise ValueError("result_name must identify a material or result directory")

    ranked: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate.name.lower() == "mace_energy":
            continue
        name = _normalized_name(candidate.name)
        if name == query:
            score = 100
        elif name.startswith(query):
            score = 80
        elif query in name:
            score = 60
        else:
            continue
        ranked.append((score, candidate.resolve()))

    if not ranked:
        available = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir()))
        raise FileNotFoundError(
            f"No calculation result matched {result_name!r} under {root}. Available: {available or 'none'}"
        )
    best_score = max(score for score, _ in ranked)
    best = [path for score, path in ranked if score == best_score]
    if len(best) > 1:
        choices = ", ".join(path.name for path in best)
        raise ValueError(f"Result name {result_name!r} is ambiguous; use one of: {choices}")
    return best[0]


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _locate_band_job(result_dir: Path) -> Path:
    candidates: list[Path] = []
    for outir in result_dir.rglob("outir"):
        job_dir = outir.parent
        if (job_dir / "vasprun.xml.gz").is_file() or (job_dir / "vasprun.xml").is_file():
            candidates.append(job_dir.resolve())
    if not candidates:
        raise FileNotFoundError(
            f"No completed IRVSP band job (outir + vasprun.xml[.gz]) found under {result_dir}"
        )
    if len(candidates) > 1:
        choices = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple IRVSP band jobs found under {result_dir}: {choices}")
    return candidates[0]


def _read_spacegroup_number(job_dir: Path) -> int:
    outir = _read_text(job_dir / "outir")
    match = re.search(r"\birvsp\s+-sg\s+(\d{1,3})\b", outir, flags=re.IGNORECASE)
    if not match:
        little_group = next(job_dir.glob("Littlegroup_*.cht"), None)
        if little_group:
            match = re.search(r"(\d{1,3})", little_group.stem)
    if not match:
        raise ValueError(f"Could not determine the space-group number from {job_dir}")
    value = int(match.group(1))
    if not 1 <= value <= 230:
        raise ValueError(f"Invalid space-group number {value} in {job_dir}")
    return value


def _read_soc(job_dir: Path) -> bool:
    incar = job_dir / "INCAR"
    if not incar.is_file():
        incar = job_dir / "INCAR.gz"
    if not incar.is_file():
        raise FileNotFoundError(f"INCAR[.gz] not found in band job: {job_dir}")
    text = _read_text(incar)
    match = re.search(r"(?im)^\s*LSORBIT\s*=\s*([^\s#;]+)", text)
    if not match:
        return False
    return match.group(1).strip(".").lower() in {"true", "t", "1", "yes"}


def _spacegroup_symbol(spacegroup_number: int) -> str:
    from pymatgen.symmetry.groups import SpaceGroup

    return SpaceGroup.from_int_number(spacegroup_number).symbol


def _load_analyzer(analyzer_root: Path, xml_path: Path) -> Any:
    root = str(analyzer_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    analyzer_module = importlib.import_module("degeneracy_analyzer")
    analyzer = analyzer_module.DegeneracyAnalyzer(str(xml_path))
    if not analyzer.valid or analyzer.bs is None:
        raise RuntimeError(f"Could not parse line-mode band structure: {xml_path}")
    return analyzer


def _canonical_path(value: str) -> str:
    value = value.replace("\\Gamma", "Γ").replace("Gamma", "Γ")
    return re.sub(r"[^A-Za-zΑ-Ωα-ωΓ]+", "", value).upper()


def _branch_for_crossing(branches: list[dict[str, Any]], crossing: dict[str, Any]) -> dict[str, Any]:
    minimum_knum = sum(crossing["k_interval"]) // 2
    global_index = minimum_knum - 1
    for branch in branches:
        if branch["start_index"] <= global_index <= branch["end_index"]:
            return branch
    raise ValueError(f"Crossing k-point {minimum_knum} is outside all band branches")


def match_accidental_particles(
    branch_name: str,
    accidental_particles: list[EmergentParticle],
) -> list[ParticlePathCandidate]:
    """Return encyclopedia entries whose indexed path equals the plotted branch."""
    branch_path = _canonical_path(branch_name)
    reversed_path = branch_path[::-1]
    matches: list[ParticlePathCandidate] = []
    for particle in accidental_particles:
        for path in particle.paths or []:
            indexed_path = _canonical_path(path.path)
            if indexed_path not in {branch_path, reversed_path}:
                continue
            matches.append(
                ParticlePathCandidate(
                    abbreviation=particle.abbreviation,
                    name=particle.name,
                    line_label=path.line_label,
                    path=path.path,
                    source_pdf_page=path.source_pdf_page,
                )
            )
    return matches


def _find_band_image(result_dir: Path) -> Path | None:
    images = sorted(result_dir.rglob("band_*.png"))
    return images[0].resolve() if len(images) == 1 else None


def _ensure_band_image(
    result_dir: Path,
    analyzer: Any,
    raw_crossings: list[dict[str, Any]],
    analyzer_root: Path,
    image_path: Path | None = None,
) -> Path:
    if image_path is None:
        existing = _find_band_image(result_dir)
        if existing is not None:
            return existing
    root = str(analyzer_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    plotter_module = importlib.import_module("irrep_plotter")
    image_path = image_path or (
        result_dir / "agent_analysis" / f"band_{result_dir.name}_annotated.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    plotter_module.plot_irrep_crossings(
        bs=analyzer.bs,
        crossings=raw_crossings,
        output_filename=str(image_path),
        material_info=result_dir.name,
        y_lim=[-1.0, 1.0],
    )
    if not image_path.is_file():
        raise RuntimeError(f"Band plotter did not create the expected image: {image_path}")
    return image_path.resolve()


def analyze_band_result(
    result_name: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    emergent_index: Path = DEFAULT_INDEX_PATH,
    analyzer_root: Path = DEFAULT_ANALYZER_ROOT,
    energy_window_ev: float = 1.0,
    gap_tolerance_ev: float = 0.03,
    band_job_directory: Path | None = None,
    report_path: Path | None = None,
    image_path: Path | None = None,
    generate_image: bool = True,
) -> BandAccidentalDegeneracyReport:
    validate_analysis_parameters(energy_window_ev, gap_tolerance_ev)
    result_dir = find_result_directory(result_name, results_root)
    if band_job_directory is None:
        job_dir = _locate_band_job(result_dir)
    else:
        job_dir = band_job_directory.resolve()
        if job_dir.parent != result_dir or not (job_dir / "outir").is_file():
            raise ValueError(f"Invalid band job for {result_dir.name}: {job_dir}")
    spacegroup_number = _read_spacegroup_number(job_dir)
    soc = _read_soc(job_dir)
    encyclopedia = lookup_emergent_particles(spacegroup_number, soc, emergent_index)

    xml_path = job_dir / "vasprun.xml.gz"
    if not xml_path.is_file():
        xml_path = job_dir / "vasprun.xml"
    analyzer = _load_analyzer(analyzer_root, xml_path)
    raw_crossings = analyzer.find_crossings_by_irreps(
        str(job_dir / "outir"),
        e_window=energy_window_ev,
        gap_tol=gap_tolerance_ev,
    )
    band_image = (
        _ensure_band_image(
            result_dir,
            analyzer,
            raw_crossings,
            analyzer_root,
            image_path=image_path,
        )
        if generate_image
        else None
    )

    crossings: list[AccidentalCrossing] = []
    for number, raw in enumerate(raw_crossings, start=1):
        diagnostics = crossing_diagnostics(
            raw,
            energy_window_ev=energy_window_ev,
            gap_tolerance_ev=gap_tolerance_ev,
        )
        branch = _branch_for_crossing(analyzer.bs.branches, raw)
        candidates = match_accidental_particles(branch["name"], encyclopedia.accidental)
        if len(candidates) == 1:
            classification = "confirmed_by_unique_path"
        elif candidates:
            classification = "path_compatible_ambiguous"
        else:
            classification = "not_indexed_for_this_path"
        crossings.append(
            AccidentalCrossing(
                crossing_number=number,
                branch=branch["name"],
                high_symmetry_path=_canonical_path(branch["name"]),
                line_label=candidates[0].line_label if candidates else None,
                k_point_interval=tuple(raw["k_interval"]),
                k_point_coordinates=(list(raw["k1_coords"]), list(raw["k2_coords"])),
                band_indices=tuple(raw["band_indices"]),
                irreps_swapped=tuple(raw["irreps_swapped"]),
                energy_relative_to_fermi_ev=round(float(raw["energy_approx"]), 6),
                energy_reference=diagnostics["energy_reference"],
                minimum_gap_ev=diagnostics["minimum_gap_ev"],
                neighboring_gaps_ev=diagnostics["neighboring_gaps_ev"],
                gap_to_tolerance_ratio=diagnostics["gap_to_tolerance_ratio"],
                threshold_margin=diagnostics["threshold_margin"],
                fermi_proximity=diagnostics["fermi_proximity"],
                classification=classification,
                candidates=candidates,
            )
        )

    summaries: list[PathMatchSummary] = []
    for path_name in dict.fromkeys(item.high_symmetry_path for item in crossings):
        grouped = [item for item in crossings if item.high_symmetry_path == path_name]
        summaries.append(
            PathMatchSummary(
                high_symmetry_path=path_name,
                crossing_count=len(grouped),
                classification=grouped[0].classification,
                candidates=grouped[0].candidates,
            )
        )

    confirmed = sorted(
        {item.candidates[0].abbreviation for item in crossings if len(item.candidates) == 1}
    )
    compatible = sorted(
        {candidate.abbreviation for item in crossings for candidate in item.candidates}
    )
    encyclopedia_path_table = [
        EncyclopediaPathComparison(
            high_symmetry_path=summary.high_symmetry_path,
            line_label=candidate.line_label,
            crossing_count=summary.crossing_count,
            abbreviation=candidate.abbreviation,
            name=candidate.name,
            classification=summary.classification,
            source_pdf_page=candidate.source_pdf_page,
        )
        for summary in summaries
        for candidate in summary.candidates
    ]
    report_path = report_path or (
        result_dir / "agent_analysis" / "accidental_degeneracy_report.json"
    )
    run = getattr(analyzer, "run", None)
    electronic_converged = getattr(run, "converged_electronic", None)
    electronic_converged = (
        bool(electronic_converged) if electronic_converged is not None else None
    )
    kpoints = getattr(analyzer.bs, "kpoints", None)
    line_path_kpoint_count = len(kpoints) if kpoints is not None else None
    evidence_assessment = build_band_evidence_assessment(
        crossings,
        electronic_converged=electronic_converged,
        kpoint_count=line_path_kpoint_count,
    )
    report = BandAccidentalDegeneracyReport(
        result_name=result_dir.name,
        result_directory=str(result_dir),
        band_job_directory=str(job_dir),
        band_image=str(band_image) if band_image else None,
        spacegroup_number=spacegroup_number,
        spacegroup_symbol=_spacegroup_symbol(spacegroup_number),
        soc=soc,
        detector="local gap minimum < tolerance plus adjacent-band IRVSP irrep exchange",
        energy_window_ev=energy_window_ev,
        gap_tolerance_ev=gap_tolerance_ev,
        crossing_count=len(crossings),
        electronic_converged=electronic_converged,
        line_path_kpoint_count=line_path_kpoint_count,
        path_unique_particle_types=confirmed,
        confirmed_particle_types=confirmed,
        path_compatible_particle_types=compatible,
        path_summaries=summaries,
        encyclopedia_path_table=encyclopedia_path_table,
        crossings=crossings,
        source_title=encyclopedia.source_title,
        source_file=encyclopedia.source_file,
        source_table=encyclopedia.source_table,
        source_table_pdf_page=encyclopedia.source_pdf_page,
        source_path_section=encyclopedia.path_source_section,
        report_file=str(report_path.resolve()),
        scientific_scope=(
            "A unique path match identifies only an encyclopedia path taxonomy. Every detector hit "
            "remains a symmetry-supported candidate: a one-dimensional path cannot establish exact "
            "gap closure, point/line dimensionality, topological charge, or a bulk topological phase."
        ),
        evidence_assessment=evidence_assessment,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
