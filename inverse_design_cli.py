"""Command-line entry point for the inverse-design research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_design.catalog import (
    CatalogConfig,
    build_catalog,
    deduplicate_catalog,
    materialize_catalog,
    merge_catalog_shards,
)
from inverse_design.baselines import (
    create_matched_random_targets,
    generate_random_spacegroup_baseline,
    particle_compatible_candidates,
)
from inverse_design.funnel import build_funnel, build_prenovelty_funnel
from inverse_design.dft_results import extract_dft_results
from inverse_design.topology_results import analyze_topology_batch
from inverse_design.labels import (
    create_label_template,
    import_agent_topology_reports,
    merge_dft_topology_labels,
)
from inverse_design.metrics import create_assignment_template, evaluate_benchmark
from inverse_design.references import (
    download_materials_project_snapshot,
    evaluate_novelty,
    import_mp20_snapshot,
    import_reference_structures,
)
from inverse_design.surrogate import (
    predict_surrogates,
    train_particle_surrogate,
    train_surrogate,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symmband-research",
        description="Reproducible emergent-particle inverse-design experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="reclassify and catalog generated structures")
    catalog.add_argument("--source", type=_path, required=True)
    catalog.add_argument("--output", type=_path, default=PROJECT_ROOT / "research_data" / "catalog")
    catalog.add_argument("--curated", type=_path)
    catalog.add_argument("--limit", type=int)
    catalog.add_argument("--offset", type=int, default=0)
    catalog.add_argument("--skip-index", type=int, action="append", default=[])
    catalog.add_argument("--no-materialize", action="store_true")
    catalog.add_argument("--no-deduplicate", action="store_true")
    catalog.add_argument("--min-distance", type=float, default=0.6)
    catalog.add_argument("--fmax-threshold", type=float, default=0.1)

    merge = commands.add_parser("merge-catalog", help="merge recoverable catalog shards")
    merge.add_argument("--shards", type=_path, required=True)
    merge.add_argument("--output", type=_path, required=True)

    dedup = commands.add_parser("deduplicate", help="deduplicate a completed catalog")
    dedup.add_argument("--catalog", type=_path, required=True)
    dedup.add_argument("--output", type=_path, required=True)

    materialize = commands.add_parser("materialize", help="write canonical CIF copies from a catalog")
    materialize.add_argument("--catalog", type=_path, required=True)
    materialize.add_argument("--curated", type=_path, required=True)

    random_targets = commands.add_parser("random-targets", help="freeze formula/SG matched targets")
    random_targets.add_argument("--catalog", type=_path, required=True)
    random_targets.add_argument("--output", type=_path, required=True)
    random_targets.add_argument("--per-pair", type=int, default=20)

    random_generate = commands.add_parser("random-generate", help="generate the pyxtal random baseline")
    random_generate.add_argument("--targets", type=_path, required=True)
    random_generate.add_argument("--output-dir", type=_path, required=True)
    random_generate.add_argument("--seed", type=int, required=True)

    particle = commands.add_parser("particle-select", help="select SG/particle-compatible structures")
    particle.add_argument("--catalog", type=_path, required=True)
    particle.add_argument("--particle", required=True)
    particle.add_argument("--output", type=_path, required=True)
    particle.add_argument("--without-soc", action="store_true")

    dft = commands.add_parser("extract-dft", help="extract DFT stage, energy, mapping, and convergence labels")
    dft.add_argument("--results", type=_path, required=True)
    dft.add_argument("--catalog", type=_path, required=True)
    dft.add_argument("--output-dir", type=_path, required=True)
    dft.add_argument("--low-energy-threshold", type=float, default=0.1)

    topology = commands.add_parser("analyze-topology-batch", help="analyze all primary SOC/IRVSP band jobs")
    topology.add_argument("--results", type=_path, required=True)
    topology.add_argument("--dft-materials", type=_path, required=True)
    topology.add_argument("--output-dir", type=_path, required=True)
    topology.add_argument("--workers", type=int, default=4)
    topology.add_argument("--generate-images", action="store_true")
    topology.add_argument("--no-resume", action="store_true")

    merge_labels = commands.add_parser(
        "merge-dft-labels", help="merge DFT proxy and topology labels by candidate ID"
    )
    merge_labels.add_argument("--dft-materials", type=_path, required=True)
    merge_labels.add_argument("--topology", type=_path, required=True)
    merge_labels.add_argument("--output", type=_path, required=True)

    mp = commands.add_parser("download-mp", help="freeze a Materials Project reference snapshot")
    mp.add_argument("--catalog", type=_path, required=True)
    mp.add_argument("--output", type=_path, required=True)
    mp.add_argument("--api-key", help="prefer MP_API_KEY so the key is not written to shell history")

    local = commands.add_parser("import-references", help="import local reference CIF/POSCAR files")
    local.add_argument("paths", type=_path, nargs="+")
    local.add_argument("--database", required=True)
    local.add_argument("--output", type=_path, required=True)

    mp20 = commands.add_parser("import-mp20", help="freeze local SymmCD MP20 CSV splits")
    mp20.add_argument("--root", type=_path, required=True)
    mp20.add_argument("--output", type=_path, required=True)

    novelty = commands.add_parser("novelty", help="run StructureMatcher against frozen snapshots")
    novelty.add_argument("--catalog", type=_path, required=True)
    novelty.add_argument("--reference", type=_path, action="append", required=True)
    novelty.add_argument("--output", type=_path, required=True)

    template = commands.add_parser("templates", help="create label and ablation-assignment templates")
    template.add_argument("--catalog", type=_path, required=True)
    template.add_argument("--output-dir", type=_path, required=True)

    labels = commands.add_parser("import-topology", help="import agent band-analysis reports")
    labels.add_argument("--catalog", type=_path, required=True)
    labels.add_argument("--results", type=_path, required=True)
    labels.add_argument("--output", type=_path, required=True)

    train = commands.add_parser("train-surrogate", help="train a leakage-aware RF baseline")
    train.add_argument("--catalog", type=_path, required=True)
    train.add_argument("--labels", type=_path, required=True)
    train.add_argument("--output-dir", type=_path, required=True)
    train.add_argument(
        "--task", choices=("stability", "stability_proxy", "topology"), required=True
    )
    train.add_argument("--minimum-labeled", type=int, default=30)

    particle_train = commands.add_parser(
        "train-particle-surrogate",
        help="train a symmetry-compatible DP or DNL classifier for frozen-model guidance",
    )
    particle_train.add_argument("--catalog", type=_path, required=True)
    particle_train.add_argument("--labels", type=_path, required=True)
    particle_train.add_argument("--output-dir", type=_path, required=True)
    particle_train.add_argument("--particle", choices=("DP", "DNL"), required=True)
    particle_train.add_argument("--spacegroup", type=int)
    particle_train.add_argument("--minimum-labeled", type=int, default=30)

    smc = commands.add_parser(
        "smc-generate",
        help="run TDS-inspired SMC guidance without updating the SymmCD checkpoint",
    )
    smc.add_argument("--formula", required=True)
    smc.add_argument("--spacegroup", type=int, required=True)
    smc.add_argument("--model", type=_path, required=True)
    smc.add_argument("--output-dir", type=_path, required=True)
    smc.add_argument("--checkpoint", type=_path, default=PROJECT_ROOT / "epoch699.ckpt")
    smc.add_argument("--particles", type=int, default=32)
    smc.add_argument("--diffusion-steps", type=int, default=500)
    smc.add_argument("--resample-interval", type=int, default=50)
    smc.add_argument("--guidance-start-fraction", type=float, default=0.5)
    smc.add_argument("--alpha", type=float, default=3.0)
    smc.add_argument("--ess-threshold", type=float, default=0.8)
    smc.add_argument("--seed", type=int, default=20260813)
    smc.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smc.add_argument("--allow-unvalidated-surrogate", action="store_true")

    predict = commands.add_parser("predict", help="score candidates with trained surrogates")
    predict.add_argument("--catalog", type=_path, required=True)
    predict.add_argument("--models", type=_path, required=True)
    predict.add_argument("--output", type=_path, required=True)

    funnel = commands.add_parser("funnel", help="rank a fixed 500-2000 candidate DFT budget")
    funnel.add_argument("--catalog", type=_path, required=True)
    funnel.add_argument("--novelty", type=_path, required=True)
    funnel.add_argument("--predictions", type=_path, required=True)
    funnel.add_argument("--output", type=_path, required=True)
    funnel.add_argument("--budget", type=int, default=1000)

    pre_funnel = commands.add_parser(
        "pre-funnel", help="rank a particle-aware queue before external novelty evaluation"
    )
    pre_funnel.add_argument("--catalog", type=_path, required=True)
    pre_funnel.add_argument("--predictions", type=_path, required=True)
    pre_funnel.add_argument("--output", type=_path, required=True)
    pre_funnel.add_argument("--budget", type=int, default=500)
    pre_funnel.add_argument("--per-formula-spacegroup-cap", type=int, default=20)

    benchmark = commands.add_parser("benchmark", help="calculate ablation and acceleration metrics")
    benchmark.add_argument("--catalog", type=_path, required=True)
    benchmark.add_argument("--assignments", type=_path, required=True)
    benchmark.add_argument("--novelty", type=_path)
    benchmark.add_argument("--labels", type=_path)
    benchmark.add_argument("--output", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "catalog":
        records = build_catalog(
            CatalogConfig(
                source_root=args.source,
                output_root=args.output,
                curated_root=args.curated,
                materialize=not args.no_materialize and args.curated is not None,
                deduplicate=not args.no_deduplicate,
                offset=args.offset,
                skip_indices=tuple(args.skip_index),
                limit=args.limit,
                min_distance_angstrom=args.min_distance,
                relax_fmax_threshold=args.fmax_threshold,
            )
        )
        print(json.dumps({"cataloged": len(records), "output": str(args.output)}, ensure_ascii=False))
    elif args.command == "merge-catalog":
        records = merge_catalog_shards(
            [path for path in args.shards.iterdir() if path.is_dir()], args.output
        )
        print(json.dumps({"merged": len(records), "output": str(args.output)}))
    elif args.command == "deduplicate":
        records = deduplicate_catalog(args.catalog, args.output)
        print(json.dumps({"deduplicated": len(records), "output": str(args.output)}))
    elif args.command == "materialize":
        print(json.dumps({"materialized": materialize_catalog(args.catalog, args.curated)}))
    elif args.command == "random-targets":
        targets = create_matched_random_targets(args.catalog, args.output, args.per_pair)
        print(json.dumps({"target_pairs": len(targets), "output": str(args.output)}))
    elif args.command == "random-generate":
        rows = generate_random_spacegroup_baseline(args.targets, args.output_dir, args.seed)
        print(json.dumps({"attempted": len(rows), "generated": sum(row["generated"] for row in rows)}))
    elif args.command == "particle-select":
        rows = particle_compatible_candidates(
            args.catalog, args.particle, args.output, soc=not args.without_soc
        )
        print(json.dumps({"compatible": len(rows), "output": str(args.output)}))
    elif args.command == "extract-dft":
        materials, jobs = extract_dft_results(
            args.results,
            args.catalog,
            args.output_dir,
            low_energy_threshold_ev_per_atom=args.low_energy_threshold,
        )
        print(json.dumps({"materials": len(materials), "jobs": len(jobs), "output": str(args.output_dir)}))
    elif args.command == "analyze-topology-batch":
        rows = analyze_topology_batch(
            args.results,
            args.dft_materials,
            args.output_dir,
            workers=args.workers,
            generate_images=args.generate_images,
            resume=not args.no_resume,
        )
        print(json.dumps({"analyzed": len(rows), "output": str(args.output_dir)}))
    elif args.command == "merge-dft-labels":
        rows = merge_dft_topology_labels(
            args.dft_materials, args.topology, args.output
        )
        print(json.dumps({"candidate_labels": len(rows), "output": str(args.output)}))
    elif args.command == "download-mp":
        print(download_materials_project_snapshot(args.catalog, args.output, args.api_key))
    elif args.command == "import-references":
        print(import_reference_structures(args.paths, args.output, args.database))
    elif args.command == "import-mp20":
        print(json.dumps(import_mp20_snapshot(args.root, args.output), ensure_ascii=False, indent=2))
    elif args.command == "novelty":
        print(len(evaluate_novelty(args.catalog, args.reference, args.output)))
    elif args.command == "templates":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        create_label_template(args.catalog, args.output_dir / "dft_labels.csv")
        create_assignment_template(args.catalog, args.output_dir / "experiment_assignments.csv")
        print(args.output_dir)
    elif args.command == "import-topology":
        print(len(import_agent_topology_reports(args.catalog, args.results, args.output)))
    elif args.command == "train-surrogate":
        print(
            json.dumps(
                train_surrogate(
                    args.catalog,
                    args.labels,
                    args.output_dir,
                    args.task,
                    minimum_labeled=args.minimum_labeled,
                ),
                indent=2,
            )
        )
    elif args.command == "train-particle-surrogate":
        print(
            json.dumps(
                train_particle_surrogate(
                    args.catalog,
                    args.labels,
                    args.output_dir,
                    particle=args.particle,
                    spacegroup_number=args.spacegroup,
                    minimum_labeled=args.minimum_labeled,
                ),
                indent=2,
            )
        )
    elif args.command == "smc-generate":
        from inverse_design.smc import SMCConfig, run_smc_generation

        print(
            json.dumps(
                run_smc_generation(
                    SMCConfig(
                        formula=args.formula,
                        spacegroup_number=args.spacegroup,
                        model_path=args.model,
                        output_dir=args.output_dir,
                        checkpoint_path=args.checkpoint,
                        particles=args.particles,
                        diffusion_steps=args.diffusion_steps,
                        resample_interval=args.resample_interval,
                        guidance_start_fraction=args.guidance_start_fraction,
                        alpha=args.alpha,
                        ess_threshold=args.ess_threshold,
                        seed=args.seed,
                        device=args.device,
                        allow_unvalidated_surrogate=args.allow_unvalidated_surrogate,
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "predict":
        print(predict_surrogates(args.catalog, args.models, args.output))
    elif args.command == "funnel":
        print(len(build_funnel(args.catalog, args.novelty, args.predictions, args.output, args.budget)))
    elif args.command == "pre-funnel":
        rows = build_prenovelty_funnel(
            args.catalog,
            args.predictions,
            args.output,
            args.budget,
            per_formula_spacegroup_cap=args.per_formula_spacegroup_cap,
        )
        print(json.dumps({"selected": len(rows), "output": str(args.output)}))
    elif args.command == "benchmark":
        report = evaluate_benchmark(
            args.catalog,
            args.assignments,
            args.output,
            novelty_csv=args.novelty,
            labels_csv=args.labels,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
