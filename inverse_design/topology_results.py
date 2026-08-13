"""Batch accidental-degeneracy extraction and candidate-label integration."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TopologyRecord:
    result_name: str
    candidate_id: str | None
    band_job_directory: str | None
    analysis_completed: bool
    spacegroup_number: int | None
    soc: bool | None
    crossing_count: int | None
    strict_topology_hit: bool | None
    compatible_topology_hit: bool | None
    confirmed_particles: str
    compatible_particles: str
    high_symmetry_paths: str
    report_file: str | None
    band_image: str | None
    error: str | None = None


def _analyze_one(payload: dict) -> dict:
    try:
        from band_result_analysis import analyze_band_result

        result_dir = Path(payload["results_root"]) / payload["result_name"]
        job_dir = Path(payload["band_job_directory"])
        analysis_dir = result_dir / "agent_analysis"
        report_path = analysis_dir / "accidental_degeneracy_report.json"
        image_path = analysis_dir / f"band_{payload['result_name']}_annotated.png"
        report = analyze_band_result(
            payload["result_name"],
            results_root=Path(payload["results_root"]),
            band_job_directory=job_dir,
            report_path=report_path,
            image_path=image_path,
            generate_image=bool(payload.get("generate_images", False)),
        )
        paths = sorted({item.high_symmetry_path for item in report.crossings})
        return asdict(
            TopologyRecord(
                result_name=payload["result_name"],
                candidate_id=payload.get("candidate_id") or None,
                band_job_directory=str(job_dir),
                analysis_completed=True,
                spacegroup_number=report.spacegroup_number,
                soc=report.soc,
                crossing_count=report.crossing_count,
                strict_topology_hit=bool(report.confirmed_particle_types),
                compatible_topology_hit=bool(report.path_compatible_particle_types),
                confirmed_particles=";".join(report.confirmed_particle_types),
                compatible_particles=";".join(report.path_compatible_particle_types),
                high_symmetry_paths=";".join(paths),
                report_file=str(report_path.resolve()),
                band_image=(str(image_path.resolve()) if image_path.is_file() else None),
            )
        )
    except Exception as exc:
        return asdict(
            TopologyRecord(
                result_name=payload["result_name"],
                candidate_id=payload.get("candidate_id") or None,
                band_job_directory=payload.get("band_job_directory"),
                analysis_completed=False,
                spacegroup_number=None,
                soc=None,
                crossing_count=None,
                strict_topology_hit=None,
                compatible_topology_hit=None,
                confirmed_particles="",
                compatible_particles="",
                high_symmetry_paths="",
                report_file=None,
                band_image=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        )


def analyze_topology_batch(
    results_root: Path,
    dft_materials_csv: Path,
    output_dir: Path,
    workers: int = 4,
    generate_images: bool = False,
    resume: bool = True,
) -> list[TopologyRecord]:
    with dft_materials_csv.open(encoding="utf-8-sig", newline="") as handle:
        materials = list(csv.DictReader(handle))
    payloads = [
        {
            "results_root": str(results_root.resolve()),
            "result_name": row["result_name"],
            "candidate_id": row.get("candidate_id", ""),
            "band_job_directory": row["primary_band_job_directory"],
            "generate_images": generate_images,
        }
        for row in materials
        if row.get("primary_band_job_directory") and row.get("irvsp_completed", "").lower() == "true"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "topology_checkpoint.jsonl"
    raw_results: list[dict] = []
    completed_names: set[str] = set()
    if resume and checkpoint_file.is_file():
        for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                raw_results.append(row)
                completed_names.add(row["result_name"])
    payloads = [payload for payload in payloads if payload["result_name"] not in completed_names]

    def checkpoint(row: dict) -> None:
        with checkpoint_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if workers <= 1:
        for number, payload in enumerate(payloads, 1):
            row = _analyze_one(payload)
            raw_results.append(row)
            checkpoint(row)
            print(f"Analyzed topology {number}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_analyze_one, payload): payload for payload in payloads}
            for number, future in enumerate(as_completed(futures), 1):
                row = future.result()
                raw_results.append(row)
                checkpoint(row)
                if number % 10 == 0 or number == len(futures):
                    print(f"Analyzed topology {number}/{len(futures)}", flush=True)
    records = [TopologyRecord(**row) for row in sorted(raw_results, key=lambda row: row["result_name"])]
    output_csv = output_dir / "topology_labels.csv"
    names = [field.name for field in fields(TopologyRecord)]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    particle_counts: dict[str, int] = {}
    for record in records:
        for particle in filter(None, record.confirmed_particles.split(";")):
            particle_counts[particle] = particle_counts.get(particle, 0) + 1
    completed = [record for record in records if record.analysis_completed]
    summary = {
        "requested_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "crossing_positive_count": sum((record.crossing_count or 0) > 0 for record in completed),
        "strict_topology_hit_count": sum(record.strict_topology_hit is True for record in completed),
        "compatible_topology_hit_count": sum(
            record.compatible_topology_hit is True for record in completed
        ),
        "confirmed_particle_material_counts": dict(sorted(particle_counts.items())),
        "scientific_scope": (
            "strict hits require a crossing on a path uniquely indexed to one particle type. "
            "Compatible hits may remain ambiguous between point, line, and line-net particles."
        ),
    }
    (output_dir / "topology_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records
