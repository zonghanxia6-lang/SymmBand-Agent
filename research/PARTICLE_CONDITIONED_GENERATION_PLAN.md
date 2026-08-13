# Particle-conditioned generation plan

## Objective

The first supported inverse-design requests will be:

- generate candidates likely to contain a DP under a requested compatible space group;
- generate candidates likely to contain a DNL in space group 194 with SOC;
- preserve the existing SymmCD validity, symmetry, novelty, and stability objectives.

WNL and C-1 WP are deferred until more DFT labels are available. A predicted particle
probability is a prioritization score, not a physical confirmation. The publication-level
endpoint is a strict SOC band/IRVSP hit at a fixed DFT budget.

## Current label boundary

The current mapped DFT set supports a pilot particle-specific surrogate but not a strong
claim for direct conditional fine-tuning:

| Target | Positive | Compatible hard negative | Positive chemical systems | Scope |
| --- | ---: | ---: | ---: | --- |
| DP | 97 | 138 | 12 | compatible SOC space groups |
| DNL | 55 | 61 | 9 | space group 194 only |

An additional six DNL-compatible negatives occur in space groups 63 and 129. They should
be retained for a future general DNL model, but excluded from the first SG-194-specific
model. All current usable DNL positives are in SG 194, so the first model must be reported
as `P(DNL | SG=194, SOC)`, not as a space-group-general DNL predictor.

Hard negatives must satisfy all of the following: completed SOC band and IRVSP analysis,
target particle allowed by the encyclopedia for the relaxed space group, and no strict hit
for that particle. Missing calculations and incompatible space groups are unknown/masked,
not negatives.

## Stage 1: particle-specific screening

### Training table

Create one immutable row per relaxed structure with:

- candidate ID, StructureMatcher cluster ID, chemical-system group, and relaxed SG;
- strict binary labels `particle_dp` and `particle_dnl` plus an eligibility mask per label;
- elemental statistics, valence-electron count/parity, heavy-element/SOC descriptors;
- Wyckoff multiplicities, site-symmetry descriptors, density, lattice, and local geometry;
- DFT provenance and a label-quality flag.

Equivalent cells and duplicate structures must remain in the same split. Split first by
chemical system and then audit StructureMatcher clusters across folds. Never use directory
names or requested space groups as final labels.

### Models

Train independent calibrated binary heads first, while exposing their outputs as a
multi-label vector `[P(DP), P(DNL)]`. Independent heads are safer than a jointly trained
neural network at the present sample size because each target has a different eligibility
mask.

1. Publication baseline: class-balanced random forest on auditable descriptors.
2. Transfer baseline: frozen MACE graph embeddings followed by logistic/MLP heads.
3. Later shared model: a frozen crystal encoder with two sigmoid heads and masked weighted
   BCE, only after the label set grows.

Use grouped out-of-fold predictions, probability calibration on held-out groups, and no
threshold selected on the test fold. Report PR-AUC, ROC-AUC, balanced accuracy, Brier score,
precision at the DFT budget, and enrichment over the SG-only base rate. PR-AUC and
fixed-budget enrichment are the primary screening metrics.

### Agent workflow available before adapter training

For `generate structures likely to have DNL in space group 194`, the Agent should:

1. Check that DNL is allowed for SG 194 with SOC in the local encyclopedia.
2. Ask SymmCD for 500-2,000 SG-194 candidates using the unchanged epoch699 model.
3. Re-identify symmetry, reject invalid/non-retained structures, and deduplicate.
4. Score `P(DNL | SG=194)`, stability, novelty, and model uncertainty.
5. Apply a StructureMatcher diversity cap and select a fixed DFT queue.
6. Allocate about 70% of the queue to high-score exploitation and 30% to uncertain/diverse
   exploration.
7. Run MACE relaxation, recheck SG, then SOC DFT, band calculation, and IRVSP.
8. Append confirmed positives and hard negatives to the immutable training table.

This is agentic inverse design by generation plus rejection/ranking. It is useful and fully
testable now, but it must not be described as direct particle-conditioned diffusion.

## Stage 2: SymmCD particle adapter

Do not replace SymmCD with MatterGen. Reuse the adapter principle while preserving
SymmCD's existing SG-conditioned score model:

- freeze the epoch699 backbone, SG embedding, and original decoder weights;
- represent the requested particle by a multi-hot vector `[DP, DNL]` plus a null condition;
- learn a particle embedding and zero-initialized bottleneck residual adapters in each
  decoder message-passing block;
- modulate adapter residuals with the existing SG embedding and particle embedding;
- drop the particle condition in 10-20% of training examples and use classifier-free
  guidance during sampling;
- train only particle embeddings, adapters, and optional normalization parameters with the
  original diffusion denoising losses;
- balance positive and compatible-negative sampling, but do not duplicate augmented cells
  across train and validation splits.

A 116-structure DNL pilot can test that the code trains, but is not sufficient for a strong
scientific claim. Start paper-level adapter experiments after reaching at least 300 DNL
positives and 300 matched hard negatives in SG 194, preferably spanning at least 15-20
chemical systems. Collect labels prospectively with the Stage-1 active-learning loop.

Sweep classifier-free guidance scales including 0, 1, 2, and 3. Stop increasing guidance
when SG retention, validity, diversity, or MACE stability deteriorates.

## Controlled evaluation

Freeze formulas, seeds, raw generation count, MACE budget, and DFT budget before running:

| Arm | Generator condition | Ranking |
| --- | --- | --- |
| A | original SymmCD | random within valid candidates |
| B | SG-only SymmCD | stability/novelty only |
| C | SG-only SymmCD | particle surrogate + stability/novelty |
| D | SG + particle adapter | stability/novelty only |
| E | SG + particle adapter | particle surrogate + stability/novelty |

Generate at least 1,000 raw candidates per arm and seed. Evaluate all arms with the same
prospective DFT budget and report bootstrap confidence intervals. The main result is the
number and rate of strict DNL or DP discoveries per fixed DFT budget, plus enrichment over
Arm B. Secondary metrics are validity, relaxed SG retention, MP20 structural novelty,
MACE/DFT stability, uniqueness, and chemical-system diversity.

Use a chemistry-held-out prospective test set for the final claim. Existing labels may be
used for training and cross-validation, but not counted again as prospective discoveries.

## Decision gates

1. Deploy Stage-1 ranking only if grouped out-of-fold precision at the planned DFT budget
   is above the SG-only base rate and remains positive on held-out chemical systems.
2. Train the adapter pilot only after Stage-1 labels and split audits are reproducible.
3. Scale adapter experiments only after the label-count and diversity threshold is met.
4. Claim direct particle-conditioned generation only if Arm D beats Arm B on strict DFT
   hits; an improvement only in Arm E supports ranking, not generator conditioning.

## MatterGen evidence and limitation

MatterGen demonstrates that residual adapters and classifier-free guidance can steer a
pretrained crystal diffusion model toward chemistry, space-group, and scalar-property
targets. Its published property tasks used approximately 605,000 magnetic-density labels,
42,000 band-gap labels, and 5,000 bulk-modulus labels; the generated property distributions
shifted toward requested values and improved discoveries under fixed DFT budgets. This is
strong evidence for the adapter design pattern.

It is not direct evidence for emergent-particle conditioning. DP/DNL labels are sparse,
discrete, selected by a previous screening policy, and depend on downstream SOC band
analysis. MatterGen therefore supports the architecture choice, while the controlled arms
above must establish whether it works for this task.

Primary references:

- Zeni et al., *A generative model for inorganic materials design*, Nature 639, 624-632
  (2025), https://doi.org/10.1038/s41586-025-08628-5
- Official MatterGen implementation and custom/multi-property adapter instructions,
  https://github.com/microsoft/mattergen
