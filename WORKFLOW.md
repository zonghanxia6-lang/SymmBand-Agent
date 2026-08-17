# SymmBand-Agent complete workflow

This document defines the reproducible workflow and the boundary between files
required by the software and local scientific data that must not be published.

## 1. Conversational entry point

Start the interactive agent with:

```bash
symmband
```

The Pydantic AI layer converts natural-language requests into validated typed tool
arguments. The language model never performs structure generation, energy evaluation,
or band classification itself. Those operations are delegated to deterministic local
programs and their structured results are returned to the conversation.

## 2. Standard crystal generation

Example request:

```text
生成10个194号空间群的BN结构
```

Execution path:

1. `structure_agent.py` extracts formula, space group, and sample count.
2. `workflow_sym.py` loads the frozen `epoch699.ckpt` SymmCD checkpoint.
3. SymmCD performs 500-step symmetry-constrained reverse diffusion.
4. The generated asymmetric unit is projected to compatible Wyckoff positions.
5. MACE performs FIRE pre-relaxation and BFGS refinement.
6. `pymatgen`/`spglib` reanalyzes the relaxed structure instead of trusting its name.
7. Only matching element sets and recovered space groups are accepted as POSCAR files.
8. The agent records each accepted structure, actual space group, and MACE potential
   energy for conversational follow-up.

MACE potential energy is not a DFT formation energy or energy above hull.

## 3. Emergent-particle-conditioned generation

Example request:

```text
生成64个可能具有DP、空间群194的BiTe结构
```

Execution path:

1. The frozen SymmCD checkpoint defines the structure prior conditioned on composition
   and space group.
2. `research_models/particles/particle_dp_surrogate.joblib` scores intermediate
   structures for the DP condition.
3. `inverse_design/smc.py` applies TDS-inspired sequential Monte Carlo twisting during
   the latter half of reverse diffusion.
4. Complete latent states are systematically resampled when effective sample size falls
   below the configured threshold.
5. Final particles are symmetry-projected, validated, ranked, and written with a full
   SMC event audit.

This is a prioritization method, not a topology confirmation. DNL requests intentionally
fail until a DNL surrogate independently passes the same deployment gate.

## 4. Existing-structure MACE energy

Place a CIF or POSCAR in `input_structures/` and ask:

```text
计算 graphene.cif 的能量
```

`mace_energy.py` calculates an unrelaxed MACE single-point total and per-atom potential
energy. It does not report formation energy or `E_hull`.

## 5. Band and IRVSP workflow

Example request:

```text
生成10个194号空间群的NaBi结构，然后计算能带
```

After accepted structures are generated, `band_workflow.py` launches the code in
`band_analysis/`. The cluster stage uses atomate2/jobflow to run VASP relaxation,
SOC self-consistent, SOC line-mode band, and IRVSP jobs. IRVSP, VASP, licensed POTCAR
files, scheduler configuration, and pseudopotential paths must be provided by the user;
they are not distributed in this repository.

## 6. Local encyclopedia retrieval

Example request:

```text
194号空间群考虑SOC时可能存在的演生粒子有哪些
```

`emergent_particles.py` reads only the requested record from
`emergent particles/emergent_particles_index.json`. The index contains the essential
and accidental particles, indexed high-symmetry paths, table provenance, and PDF page
numbers. The copyrighted source article and supplementary PDF are intentionally not
published and are not needed for normal queries.

## 7. Completed-result accidental-degeneracy analysis

Place a downloaded calculation directory under `calculation_results/` and ask:

```text
分析 NaBi_sg186_007 中的偶然简并
```

`band_result_analysis.py` locates the completed SOC band job, reproduces the local-gap
minimum plus adjacent-band IRVSP representation-exchange detector, maps each crossing to
its plotted high-symmetry branch, and compares the branch with the local encyclopedia
index. The report distinguishes:

- `confirmed_by_unique_path`: the indexed path has one compatible accidental particle;
- `path_compatible_ambiguous`: multiple particles share the path and remain candidates;
- `not_indexed_for_this_path`: a representation exchange was detected without a matching
  indexed accidental-particle path.

The agent prints a separate encyclopedia path comparison table containing path, line
label, crossing count, particle abbreviation and name, classification, and source page.
The machine-readable report is stored at
`calculation_results/<result>/agent_analysis/accidental_degeneracy_report.json`.

## 8. Publication-scale inverse-design evaluation

The `inverse_design/` package and `symmband-research` command provide structure catalog
reclassification, MP20 novelty analysis using `StructureMatcher`, DFT label extraction,
random and ablation baselines, surrogate training, fixed-budget funnel construction, and
particle-guided SMC generation. `research/` contains the frozen protocols and current
scientific scope. Large local datasets in `research_data/` remain ignored; only the
deployable DP model and its audit files are versioned.

## 9. Required files in the repository

| Component | Required paths |
|---|---|
| Agent | `structure_agent.py`, `pyproject.toml`, `.env.agent.example` |
| SymmCD inference | `workflow_sym.py`, `symmcd/`, `epoch699.ckpt` |
| MACE | `mace_energy.py`, `macemodel/*.model` |
| Particle guidance | `inverse_design/`, `inverse_design_cli.py`, `research_models/particles/` |
| Band bridge | `band_workflow.py`, `band_analysis/` |
| Encyclopedia | `emergent_particles.py`, `emergent particles/emergent_particles_index.json` |
| Result analysis | `band_result_analysis.py` |
| Pydantic AI runtime | `vendor/pydantic-ai/`, `.local_packages/` |
| Installation | `cluster_environment/`, requirement files |
| Verification | `tests/`, `scripts/release_audit.py` |
| Documentation | `README.md`, `WORKFLOW.md`, `THIRD_PARTY_NOTICES.md` |

`epoch699.ckpt` and the MACE model are Git LFS objects. A clone is incomplete until
`git lfs pull` succeeds.

## 10. Files that must remain local

- `.env.agent` and every API key;
- POTCAR and other licensed pseudopotential files;
- VASP/IRVSP raw outputs and `calculation_results/`;
- generated structures and band output directories;
- `research_data/` and non-deployable intermediate models;
- source papers and supplementary PDFs;
- machine-specific atomate2, jobflow, Slurm, and absolute-path configuration.

Run `python scripts/release_audit.py` before every public push.
