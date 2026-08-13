# Emergent-Particle Inverse-Design Study

This directory defines a reproducible study without modifying the SymmCD model. The
first publication claim to test is whether symmetry/particle compatibility and learned
multi-objective ranking improve DFT-confirmed discoveries at a fixed budget.

## Data Contract

- `STRUCTURE` is immutable raw data. Folder and file names are treated only as intended
  formula/space-group metadata.
- `structure_catalog.jsonl` stores recalculated symmetry at `symprec` 0.01, 0.05, and
  0.10 Angstrom, composition retention, geometry checks, relaxation status, and a
  globally unique candidate ID.
- Curated CIFs are copies named from recalculated formula, space group, and a structure
  hash. Originals are never moved or overwritten.
- Novelty is determined by `pymatgen.analysis.structure_matcher.StructureMatcher` against
  versioned local database snapshots. Formula matching is only a prefilter.
- MACE potential energy is not formation energy and cannot independently provide Ehull.
  Formation energy/Ehull labels must come from compatible DFT entries and a phase diagram.
- Stability and topology rates use only evaluated structures as denominators. The primary
  end-to-end hit rate uses all generated candidates as its denominator.

## Experimental Arms

Run every arm with identical composition distributions, target space groups, sample
counts, seeds, and DFT budgets. Do not assign historical samples retrospectively to a
condition that was not active during generation.

1. `random_spacegroup`: random/pyxtal structures under the shared composition and SG budget.
2. `symmcd_no_topology`: original SymmCD without SG/particle selection.
3. `symmcd_spacegroup`: original SymmCD with SG conditioning only.
4. `symmcd_spacegroup_particle`: SG generation plus emergent-particle/path compatibility.
5. `multiobjective`: particle compatibility plus novelty and learned stability/topology ranking.

The requested four-condition ablation is arms 2-5. Arm 1 is the independent random-screening
baseline required to report discovery acceleration.

## Commands

```bash
symmband-research catalog \
  --source /path/to/STRUCTURE \
  --output research_data/catalog \
  --curated /path/to/STRUCTURE_CURATED

export MP_API_KEY='your-key'
symmband-research download-mp \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --output research_data/references/materials_project.jsonl

symmband-research novelty \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --reference research_data/references/materials_project.jsonl \
  --output research_data/novelty.csv

# Local MP20 train/validation/test snapshot
symmband-research import-mp20 \
  --root /path/to/data/mp_20 \
  --output research_data/references/mp20.jsonl

symmband-research novelty \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --reference research_data/references/mp20.jsonl \
  --output research_data/novelty_release/mp20_novelty.csv

symmband-research templates \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --output-dir research_data/experiments

symmband-research random-targets \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --output research_data/experiments/random_targets.json \
  --per-pair 20

symmband-research random-generate \
  --targets research_data/experiments/random_targets.json \
  --output-dir research_data/random_baseline \
  --seed 20260812

symmband-research particle-select \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --particle DP \
  --output research_data/experiments/dp_compatible.csv
```

After importing at least 30 labels with both classes represented across multiple chemical
systems, train leakage-aware baselines:

```bash
symmband-research train-surrogate --task stability \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --labels research_data/experiments/dft_labels.csv \
  --output-dir research_models

symmband-research train-surrogate --task topology \
  --catalog research_data/catalog/structure_catalog.jsonl \
  --labels research_data/experiments/dft_labels.csv \
  --output-dir research_models
```

Then score candidates and freeze a 500-2000 candidate queue. The queue records all funnel
stages, but later DFT and phonon jobs must update the label table rather than editing the
catalog.

## Frozen-Checkpoint DP Guidance

The current 97-positive/137-hard-negative DP dataset passes the group-held-out
enrichment gate for TDS-inspired SMC guidance. Train and run it without updating
`epoch699.ckpt`:

```bash
symmband-research train-particle-surrogate \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --labels research_data/dft_release/candidate_labels.csv \
  --output-dir research_models/particles --particle DP

symmband-research smc-generate \
  --formula BiTe --spacegroup 194 \
  --model research_models/particles/particle_dp_surrogate.joblib \
  --output-dir research_data/smc_runs/dp_sg194/BiTe/alpha3/seed20260813 \
  --particles 64 --diffusion-steps 500 --resample-interval 25 \
  --guidance-start-fraction 0.5 --alpha 3 --ess-threshold 0.8 \
  --seed 20260813 --device cuda
```

Use `--alpha 0` with the same seed and budget for the paired SG-only baseline.
The complete frozen protocol and current OOF evidence are in
`research/TDS_SMC_DP_PROTOCOL.md`.

## Downloaded DFT Results

The current compact archive contains 250 material directories. Rebuild the DFT and
particle labels without trusting directory names as final symmetry labels:

```bash
symmband-research extract-dft \
  --results /path/to/calculation_results \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --output-dir research_data/dft_release

symmband-research analyze-topology-batch \
  --results /path/to/calculation_results \
  --dft-materials research_data/dft_release/dft_materials.csv \
  --output-dir research_data/topology_release \
  --workers 1

symmband-research merge-dft-labels \
  --dft-materials research_data/dft_release/dft_materials.csv \
  --topology research_data/topology_release/topology_labels.csv \
  --output research_data/dft_release/candidate_labels.csv
```

The extractor independently recalculates relaxed symmetry at `symprec=0.01` and `0.05`
Angstrom. A topology result is eligible for surrogate training only when the strict relaxed
space group agrees with the space group used by the band/IRVSP workflow.

Current retrospective candidate-level results are 231 reliable topology screens, 133 strict
path hits, and 90 joint strict-topology plus low-relative-polymorph-energy hits. These are
screening labels, not confirmed topological invariants or thermodynamic stability claims.

Train the current proxy baselines and create the explicitly provisional 500-candidate queue:

```bash
symmband-research train-surrogate --task stability_proxy \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --labels research_data/dft_release/candidate_labels.csv \
  --output-dir research_models

symmband-research train-surrogate --task topology \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --labels research_data/dft_release/candidate_labels.csv \
  --output-dir research_models

symmband-research predict \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --models research_models \
  --output research_data/surrogate_predictions.csv

symmband-research pre-funnel \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --predictions research_data/surrogate_predictions.csv \
  --output research_data/funnel_release/prenovelty_funnel_500.csv \
  --budget 500
```

`pre-funnel` never labels a candidate as novel. It selects Tier A strict-stoichiometry
candidates first and uses Tier B element-set-only candidates only to fill the fixed budget.
Run `novelty` with frozen external database snapshots and then the strict `funnel` command
before reporting a novel-material discovery rate.

The local MP20 analysis uses all 45,229 train/validation/test structures. A candidate whose
formula is absent from MP20 is `composition_novel`; a same-formula candidate with no
StructureMatcher match is `structure_novel`. Both are novel relative to the model training
set. `global_novelty_evaluated` remains false until comparison against a broader current
materials database, so MP20 training-set novelty must not be presented as global novelty.

## Decision Gate for Modifying SymmCD

Only add a topology condition to the generative network if the `symmcd_spacegroup_particle`
or `multiobjective` arm produces a statistically credible fixed-budget improvement and the
post-generation selector is a demonstrated bottleneck. Otherwise, retain the generator and
publish the inverse-design contribution as a symmetry-aware agentic screening policy.

`study_status.json` is the machine-readable checkpoint separating completed generation
metrics from novelty, stability, topology, and surrogate claims that still require data.
