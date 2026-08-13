# Release manifest

## Included and versioned

- Frozen SymmCD inference source and `epoch699.ckpt` through Git LFS.
- Local MACE model through Git LFS.
- Conversational Pydantic AI agent and vendored minimal runtime.
- Standard structure generation, MACE energy, band bridge, and result analysis tools.
- Local emergent-particle JSON index with table and page provenance.
- DP-specific surrogate and its validation/audit files.
- TDS-inspired SMC implementation and inverse-design research utilities.
- Conda/venv installation, environment validation, Slurm examples, and tests.
- Reproducible SVG workflow diagrams.

## Explicitly excluded

- API keys and `.env.agent`.
- POTCAR and all licensed VASP pseudopotentials.
- Generated structures, VASP/IRVSP outputs, downloaded calculation results, and logs.
- Large research datasets and intermediate non-deployable models.
- Copyrighted source articles and supplementary PDFs.
- Local manuscript drafts and unpublished figure data that are not runtime dependencies.

## External runtime requirements

- Git LFS for model retrieval.
- A DeepSeek-compatible API key for conversational use.
- VASP license, POTCAR library, IRVSP executable, scheduler, and MongoDB/jobflow setup for
  cluster band calculations.
- Current full Materials Project data/API access only for database-wide novelty analysis.

## Scientific scope

- MACE values are potential energies, not formation energies or `E_hull`.
- The DP surrogate is a screening model; SOC DFT and IRVSP remain required.
- Unique path matching is strong path-level evidence, not a complete topological invariant.
- Fixed-budget acceleration claims require completion of all preregistered comparison arms.
- Redistribution rights for third-party code and model weights must be confirmed before a
  public release; see `THIRD_PARTY_NOTICES.md`.
