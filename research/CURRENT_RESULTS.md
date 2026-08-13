# Current Retrospective Results

Data freeze: local `calculation_results` audit on 2026-08-13.

## Generation Catalog

- 17,212 generated candidates; 16,496 pass combined parse, geometry, and symmetry checks.
- 39.33% retain the requested space group at strict tolerance.
- 31.66% retain strict target stoichiometry.
- 4,891 candidates are internally unique by StructureMatcher.

## DFT and Particle Labels

- 340 material result directories contain 1,401 recognized VASP jobs.
- 333 relaxations converged; 328 materials completed SOC band and IRVSP stages.
- 338 relaxed structures could be symmetry-reclassified; 337 agree with the space group
  used for band interpretation at `symprec=0.01` Angstrom.
- 298 candidate-level results are usable for topology labels; 148 have strict unique-path
  particle hits.
- 102 mapped candidates combine a strict particle hit with relative polymorph energy no more
  than 0.1 eV/atom above the lowest downloaded structure of the same formula.

The 102 joint hits are not 102 stable topological materials. No downloaded phonon outputs,
formation-energy references, convex-hull data, or complete topological invariants are present.

`_archive_metadata2` contains 91 archive rows, including one overlap with the first archive,
so it adds 90 unique materials. Of these, 89 completed IRVSP analysis, 18 show candidate
crossings, and 16 have strict particle hits (11 DP and five C-1 WP). Cr-, Fe-, and
Mn-containing results were run with `ISPIN=1` and require magnetic-state validation.

## Proxy Models and Queue

- Stability proxy: 301 labels, chemical-system grouped CV ROC-AUC 0.681.
- Topology screen: 297 labels, chemical-system grouped CV ROC-AUC 0.743.
- All 16,496 valid structures were scored without prediction failures.
- The pre-novelty fixed-budget queue contains 500 candidates: 381 Tier A strict-stoichiometry
  candidates and 119 clearly marked Tier B element-set-only candidates.
- The queue spans 192 reduced-formula/space-group groups and balances assignments across ten
  SOC accidental-particle types allowed by the local encyclopedia index.

## MP20 Novelty

- All 45,229 MP20 train/validation/test structures were imported without parse failures.
- MP20 contains same-formula references for 9,180 of 16,496 valid generated candidates
  (55.65% formula coverage).
- StructureMatcher finds 603 known structural matches.
- Among formula-covered candidates, 8,577 do not match an MP20 structure (93.43%).
- The 7,316 valid candidates with formulas absent from MP20 are composition-novel relative
  to the model training set.
- Overall, 15,893 of 16,496 valid structures are novel relative to MP20 (96.34%).
- After requiring strict stoichiometry, target-space-group retention, internal uniqueness,
  and MP20 training-set novelty, 387 candidates remain. The fixed 500-candidate queue adds
  113 explicitly marked Tier B element-set-only candidates; every queued candidate is novel
  relative to MP20.

Both composition novelty and StructureMatcher novelty are valid training-set novelty. Neither
is automatically equivalent to global novelty against the complete current Materials Project.

## Required Prospective Work

1. Freeze the complete current Materials Project snapshot to resolve MP20 coverage gaps.
2. Execute matched random-space-group and four ablation arms under identical DFT budgets.
3. Compute compatible formation energies/Ehull and phonons for final candidates.
4. Confirm topology hits with irreducible-representation checks, Wilson loops or appropriate
   invariants, and expert inspection.
5. Report fixed-budget acceleration only after all arms share the same evaluation funnel.
