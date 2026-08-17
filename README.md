# SymmBand-Agent

For the research workflow—including structure reclassification, Materials Project
novelty matching, ablation studies, surrogate models, and the fixed-budget DFT
funnel—see [`research/README.md`](research/README.md).

The complete end-to-end execution logic, required files, and data that must not be
published are documented in [`WORKFLOW.md`](WORKFLOW.md). The GitHub release boundary
is defined in [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md).

![SymmBand-Agent integrated workflow](topological-material-discovery-workflow-v3-agent-integrated.svg)

SymmBand-Agent is a conversational workflow for crystal-structure generation and
electronic-band calculations. A user can submit a request such as:

```text
Generate 10 NaBi structures in space group 194, then calculate their band structures.
```

The agent then performs the following stages:

1. Pydantic AI calls DeepSeek to parse the natural-language request into a chemical
   formula, space-group number, and sample count.
2. SymmCD uses `epoch699.ckpt` for conditional diffusion sampling.
3. MACE relaxes the generated structures and retains POSCAR files whose elements and
   recovered space groups satisfy the request.
4. atomate2/jobflow runs VASP relaxation, SOC-SCF, SOC band, and symmetry calculations
   for the accepted POSCAR files.
5. IRVSP analyzes irreducible representations and the workflow produces `band_*.png`.

## Repository layout

```text
SymmBand-Agent/
├── structure_agent.py          Main conversational-agent entry point
├── workflow_sym.py             SymmCD + MACE structure generation
├── mace_energy.py              MACE single-point energy for an existing CIF/POSCAR
├── band_workflow.py            Subprocess bridge for the band workflow
├── band_result_analysis.py     Accidental-degeneracy and path-index comparison
├── emergent_particles.py       Local emergent-particle index lookup
├── input_structures/           User-provided CIF/POSCAR inputs
├── symmcd/                     Source required for checkpoint inference
├── inverse_design/             Novelty, surrogates, funnel, and TDS-inspired SMC
├── research_models/particles/  Deployable DP surrogate and audit artifacts
├── emergent particles/         Token-efficient encyclopedia JSON index
├── band_analysis/              atomate2/VASP/IRVSP band workflow
├── vendor/pydantic-ai/         Vendored Pydantic AI runtime snapshot
├── epoch699.ckpt               SymmCD checkpoint tracked with Git LFS
├── macemodel/                  Local MACE model tracked with Git LFS
├── cluster_environment/        Conda/pip, cluster, and Slurm configuration
├── tests/                      Agent and bridge-layer tests
└── scripts/                    GitHub preparation and audit scripts
```

## Quick start

Recommended setup for a Linux GPU cluster:

```bash
cd cluster_environment
bash install.sh conda gpu
conda activate symmcd-band-agent
cd ..
```

Copy and edit the agent configuration:

```bash
cp .env.agent.example .env.agent
chmod 600 .env.agent
vi .env.agent
```

Validate the environment and configuration:

```bash
python cluster_environment/validate_environment.py --python-only
python structure_agent.py --show-config
python structure_agent.py --check-band
```

Start an interactive session or submit a one-shot request:

```bash
python structure_agent.py
python structure_agent.py --prompt "Generate 10 NaBi structures in space group 194, then calculate their band structures."
```

### Interactive use on Windows

Activate the Conda environment in the project root and perform a one-time editable
installation:

```powershell
conda activate symmcd
python -m pip install --no-deps --no-build-isolation -e .
```

You can then start a continuous conversation from any directory:

```text
symmband
symmband ➤ Generate 10 BN structures in space group 194.
symmband ➤ Report the energy and space group of the third generated structure.
```

Sampling progress is displayed in real time. Every accepted structure is reported with
its acceptance number, total MACE potential energy, energy per atom, and recovered space
group. A MACE potential energy is not a rigorous thermodynamic formation energy. If a
user asks for a formation energy, the agent states this limitation and returns only the
available MACE quantities. Enter `/exit` to leave the interactive session.

Windows uses `MACE_DEVICE=cpu` by default for maximum compatibility. If you switch to
`cuda`, the PyTorch/CUDA environment must contain a compatible
`nvrtc-builtins64_*.dll`. Otherwise, MACE relaxation will fail, although the SymmCD
generation stage remains unaffected.

### Querying the emergent-particle encyclopedia

Supplemental Tables S1/S2 have been preprocessed into a local index. After starting
`symmband`, you can ask:

```text
symmband ➤ Which emergent particles are allowed in space group 194 when SOC is included?
```

The agent reports essential and accidental degeneracies separately and cites the source
table and PDF page. A normal query reads only the record for the requested space group
from `emergent particles/emergent_particles_index.json`; it does not send the 1,228-page
supplement to the language model. The index needs to be rebuilt in an environment with
`pypdf` only when the supplemental material changes.

You can also request the high-symmetry paths associated with accidental degeneracies:

```text
symmband ➤ List the accidental degeneracies and their high-symmetry paths for space group 216 with SOC.
```

The path data come from the space-group-resolved tables in supplemental Sections S7B/S8B
and include the line label, path endpoints, and source PDF page.

### Analyzing completed band results

Place one material-result directory retrieved from the cluster under
`calculation_results/`, for example:

```text
calculation_results/NaBi_sg186_007/
```

The result must contain a completed SOC line-mode VASP job (`vasprun.xml` or
`vasprun.xml.gz`), the IRVSP output `outir`, and `INCAR`. In an interactive session,
ask:

```text
symmband ➤ Analyze the accidental degeneracies in the NaBi result.
```

The agent locates the material directory and band job, reproduces the detector used for
the red circles in the band plot—local small-gap minima plus adjacent-band irreducible-
representation exchange—maps each hit to its high-symmetry path, and compares it with
the supplemental S2/S8B index for the observed space group and SOC setting. The report
is written to:

```text
calculation_results/<result>/agent_analysis/accidental_degeneracy_report.json
```

`confirmed_by_unique_path` is a compatibility label meaning that only one indexed
accidental-particle category matches that path. It is a path-taxonomy result, not proof
of a topological phase. `path_compatible_ambiguous` means that multiple particle types
share the path and must remain candidates; a one-dimensional band path alone cannot
distinguish a point, line, or nodal-line network. This analysis uses the local JSON index
and does not send the full supplemental material to the language model.

Rebuild the encyclopedia index with:

```powershell
python scripts/build_emergent_particle_index.py
```

### Calculating a MACE energy for an existing structure

Place a CIF or POSCAR file in `input_structures/`, for example:

```text
input_structures/graphene.cif
```

Then use either an interactive request or a one-shot command:

```bash
python structure_agent.py --prompt "Calculate the energy of graphene.cif."
```

The agent performs an unrelaxed MACE single-point calculation and reports the total
energy in eV and energy per atom in eV/atom. The JSON report is written to:

```text
calculation_results/mace_energy/<structure>_<timestamp>/mace_energy.json
```

This value is a potential-energy prediction from the configured MACE model. It is not a
DFT energy, formation energy, or energy above hull (`E_hull`). Computing `E_hull`
requires elemental reference states and competing-phase energies for the same chemical
system; it cannot be inferred from the MACE total energy of a single structure.

For the complete VASP, POTCAR, IRVSP, and Slurm configuration, see
[`cluster_environment/README.md`](cluster_environment/README.md).

## Outputs

Generated structures are written by default to:

```text
generated_structures/<formula>_sg<spacegroup>/run_<timestamp>/
```

Band images are written under the corresponding task directory:

```text
band_analysis/bands/band_*.png
```

If `BAND_OUTPUT_ROOT` is configured, images are written to:

```text
<BAND_OUTPUT_ROOT>/<formula>_sg<spacegroup>/run_<timestamp>/bands/
```

“Generate 10” means that the workflow performs 10 diffusion samples. Only structures
that pass the final element and recovered-space-group checks enter VASP, so fewer than
10 band images may be produced.

## Publishing to GitHub

`epoch699.ckpt` is approximately 725 MB and exceeds GitHub's standard 100 MiB per-file
limit. It must be stored with Git LFS. The required model patterns are already present
in `.gitattributes`.

Run the automated release audit before publishing:

```bash
python scripts/release_audit.py
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\prepare_github.ps1
```

On Linux, macOS, or Git Bash:

```bash
bash scripts/prepare_github.sh
```

After reviewing the LFS objects and staged files, commit and publish:

```bash
git status --short
git lfs ls-files
git commit -m "Initial SymmBand-Agent monorepo"
gh repo create SymmBand-Agent --source=. --private --push
```

If GitHub CLI is unavailable, create an empty private repository on GitHub and run:

```bash
git remote add origin https://github.com/<YOUR_ACCOUNT>/SymmBand-Agent.git
git push -u origin main
```

A private repository is recommended initially. Review the rights for SymmCD, the band
workflow, and the models referenced in `THIRD_PARTY_NOTICES.md` before making the
repository public.

## Security and version control

- `.env.agent`, API keys, POTCAR files, VASP outputs, generated structures, and band
  results are excluded by `.gitignore`.
- The repository retains only the two Pydantic AI packages required at runtime and
  preserves their complete MIT license text.
- Install Git LFS before cloning, then run `git lfs pull` to retrieve model files.
- Never upload POTCAR files to GitHub.

## License

The Pydantic AI MIT license is preserved under `vendor/pydantic-ai/`. The remaining code
and models did not include licenses in their original directories. Confirm and add the
appropriate permissions before public redistribution. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.
