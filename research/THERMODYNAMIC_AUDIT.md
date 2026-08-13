# Thermodynamic Energy Audit

Audit date: 2026-08-13.

## What Is Present

- `calculation_results` contains 1,036 `OUTCAR` files.
- 1,034 contain VASP `TOTEN` and `energy(sigma->0)` values.
- Zero contain an explicit formation-energy field.
- Zero contain an explicit energy-above-hull field.
- The workflow uses PBE PAW potentials through `MPRelaxSet`/`MPStaticSet`, with local
  overrides including `ENCUT=500 eV`, non-spin-polarized relaxations, and SOC static jobs.

VASP reports a total energy for one calculated structure. Formation energy requires
elemental reference chemical potentials:

`Delta E_f = E(compound) - sum_i n_i mu_i`

Energy above hull additionally requires every relevant competing phase in the complete
chemical system. Elemental energies alone are not sufficient for `E_hull`.

## What MP20 Provides

The local MP20 snapshot contains 45,229 structures with Materials Project
`formation_energy_per_atom` and `e_above_hull` labels. These labels belong to those known
MP20 structures. They cannot be assigned to a newly generated structure and cannot be
combined directly with the current SOC `TOTEN` values because they use a corrected
Materials Project thermodynamic energy scale.

## Recommended Open-Data Route

Materials Project can provide corrected competing-phase entries using
`MPRester.get_entries_in_chemsys`. Build the phase diagram with pymatgen `PhaseDiagram`,
but first calculate each new candidate with a compatible non-SOC Materials Project static
workflow and process it under the same compatibility/mixing scheme. Do not mix the current
SOC static energies directly with MP20 formation energies.

Official methodology:

- https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds
- https://pymatgen.org/pymatgen.analysis.compatibility.html

## Recommended Next Calculation

1. Run a dedicated non-SOC thermodynamic static calculation for selected candidates using
   a frozen MP-compatible input set and POTCAR mapping.
2. Download all corrected entries for each candidate chemical system, not only elemental
   entries.
3. Apply the same compatibility scheme to candidate entries and construct `PhaseDiagram`.
4. Report the current same-formula relative energy only as a screening proxy until this is
   complete.

An alternative fully self-consistent route is to calculate all elemental and competing
phases with exactly the same local settings. A universal ML potential can create an
approximate ML hull if applied consistently to candidates and references, but that result
must be labeled `ML E_hull`, not DFT `E_hull`.
