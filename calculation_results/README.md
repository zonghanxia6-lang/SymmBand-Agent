# Calculation results

This directory is the local discovery root used by the conversational result-analysis
tool and by MACE energy reports. Its generated contents are ignored by Git.

For accidental-degeneracy analysis, place one downloaded result directory here, for
example:

```text
calculation_results/NaBi_sg186_007/
```

The result must contain one completed SOC line-mode band job with `vasprun.xml` (or
`vasprun.xml.gz`), `outir`, and `INCAR` (or `INCAR.gz`). VASP outputs, POTCAR files,
calculation reports, and generated band images must remain local and must not be committed.
