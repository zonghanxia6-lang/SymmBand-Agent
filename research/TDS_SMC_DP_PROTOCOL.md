# Frozen SymmCD DP Guidance Protocol

## Current decision

Use a DP-specific random-forest surrogate as a twisting potential inside a
sequential Monte Carlo sampler. Keep every parameter in `epoch699.ckpt` frozen.
This is the highest-probability first experiment for the current small dataset
because it does not fit a diffusion adapter or a noisy-state neural classifier.

The implementation is described as **TDS-inspired SMC twisting**, not exact TDS.
SymmCD does not expose the normalized trajectory likelihood and look-ahead proposal
required for the exact TDS guarantee.

## Frozen evidence gate

The current model uses 234 eligible records from the nominal 235-label set:

- 97 strict DP positives;
- 137 DP-compatible space-group hard negatives;
- 1 record excluded because its catalog structure is invalid;
- 6 strict-label/encyclopedia compatibility conflicts excluded and audited;
- 22 chemical-system groups in five-fold `StratifiedGroupKFold` validation.

The group-held-out OOF results are:

| Metric | Result |
| --- | ---: |
| DP base rate | 0.4145 |
| PR-AUC | 0.6778 |
| ROC-AUC | 0.7162 |
| Top-20% precision | 0.7234 |
| Top-20% enrichment | 1.7451 |
| Group bootstrap enrichment 95% CI | [1.3553, 2.3860] |
| SG194 top-20% enrichment | 1.9563 |

This passes the pre-registered SMC gate: enrichment at least 1.5 and a group
bootstrap lower confidence bound above 1. The surrogate is a screening model;
it is not a DP confirmation method.

## Sampling policy

Primary guided configuration:

- 64 equal-composition particles per run;
- 500 reverse-diffusion steps;
- atom types locked to the requested formula;
- score every 25 reverse steps over the final 50% of the trajectory;
- gradually temper `P(DP | structure)^alpha` from alpha 0 to alpha 3;
- systematic resampling when ESS is below `0.8 N`;
- preserve coordinates, lattice, atom types, and site-symmetry state together
  when an ancestor is copied;
- rank and audit every final particle, including invalid projections.

Run the exact same seeds with alpha 0 as the paired SG-only baseline. Alpha 1 and
alpha 6 are sensitivity arms and must not replace the primary alpha 3 result after
looking at DFT outcomes.

## Commands

Rebuild the deployable model and its OOF audit files:

```bash
symmband-research train-particle-surrogate \
  --catalog research_data/catalog_release/structure_catalog.jsonl \
  --labels research_data/dft_release/candidate_labels.csv \
  --output-dir research_models/particles \
  --particle DP
```

Run one primary guided batch:

```bash
symmband-research smc-generate \
  --formula BiTe \
  --spacegroup 194 \
  --model research_models/particles/particle_dp_surrogate.joblib \
  --output-dir research_data/smc_runs/dp_sg194/BiTe/alpha3/seed20260813 \
  --particles 64 \
  --diffusion-steps 500 \
  --resample-interval 25 \
  --guidance-start-fraction 0.5 \
  --alpha 3 \
  --ess-threshold 0.8 \
  --seed 20260813 \
  --device cuda
```

For the paired baseline, change only `--alpha 0` and the output directory. Use at
least five pre-registered seeds. Keep the formula panel, particle count, diffusion
steps, MACE budget, and DFT budget identical between arms.

The conversational equivalent is:

```text
symmband > 生成64个可能具有DP的194号空间群BiTe结构
```

The Agent uses the validated DP model by default. A DNL request fails explicitly
until `particle_dnl_surrogate.joblib` has independently passed the same gate.

## Publication endpoint

The primary endpoint is strict SOC band plus IRVSP-confirmed DP discoveries at a
fixed DFT budget. Report validity, target-SG retention, StructureMatcher uniqueness,
MP20 novelty, DFT stability, chemical-system diversity, and DP hit rate as secondary
endpoints. Proxy probability is a selection diagnostic, never the discovery label.

Before DFT, freeze candidate IDs and apply the same deterministic MACE/geometry and
diversity filters to both arms. Use paired-seed bootstrap confidence intervals for
the difference in DP hits and report failures in the denominator.
