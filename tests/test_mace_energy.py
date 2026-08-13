import json
import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from mace_energy import MaceEnergyConfig, calculate_mace_energy, resolve_structure_file


class ConstantEnergyCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, energy: float):
        super().__init__()
        self.energy = energy

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.energy


class MaceEnergyTests(unittest.TestCase):
    def test_cif_single_point_energy_and_json_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            input_dir.mkdir()
            structure_path = input_dir / "graphene.cif"
            atoms = Atoms(
                "C2",
                scaled_positions=[(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)],
                cell=[(2.46, 0.0, 0.0), (-1.23, 2.13, 0.0), (0.0, 0.0, 15.0)],
                pbc=True,
            )
            write(structure_path, atoms, format="cif")

            result = calculate_mace_energy(
                MaceEnergyConfig(
                    structure_filename="graphene.cif",
                    input_directory=input_dir,
                    output_root=root / "results",
                    mace_model="test.model",
                    mace_device="cpu",
                ),
                calculator=ConstantEnergyCalculator(-18.0),
            )

            self.assertEqual(result.formula, "C2")
            self.assertEqual(result.atom_count, 2)
            self.assertEqual(result.total_energy_ev, -18.0)
            self.assertEqual(result.energy_per_atom_ev, -9.0)
            self.assertIn("unrelaxed", result.calculation_type)
            report = json.loads(Path(result.report_file).read_text(encoding="utf-8"))
            self.assertEqual(report["structure_filename"], "graphene.cif")
            self.assertEqual(report["total_energy_ev"], -18.0)

    def test_poscar_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            input_dir.mkdir()
            structure_path = input_dir / "POSCAR_graphene"
            atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], cell=[3.0, 3.0, 10.0], pbc=True)
            write(structure_path, atoms, format="vasp")

            result = calculate_mace_energy(
                MaceEnergyConfig(
                    structure_filename="POSCAR_graphene",
                    input_directory=input_dir,
                    output_root=root / "results",
                    mace_model="test.model",
                    mace_device="cpu",
                ),
                calculator=ConstantEnergyCalculator(-7.5),
            )

            self.assertEqual(result.atom_count, 1)
            self.assertEqual(result.energy_per_atom_ev, -7.5)

    def test_path_traversal_and_unsupported_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "notes.txt").write_text("not a structure", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain"):
                resolve_structure_file(input_dir, "../graphene.cif")
            with self.assertRaisesRegex(ValueError, "unsupported structure format"):
                resolve_structure_file(input_dir, "notes.txt")


if __name__ == "__main__":
    unittest.main()
