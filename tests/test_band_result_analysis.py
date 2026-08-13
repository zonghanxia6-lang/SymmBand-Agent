import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from band_result_analysis import (
    analyze_band_result,
    find_result_directory,
    match_accidental_particles,
)
from emergent_particles import EmergentParticle, HighSymmetryPath


class FakeAnalyzer:
    valid = True
    bs = SimpleNamespace(
        branches=[
            {"start_index": 0, "end_index": 9, "name": "\\Gamma-A"},
            {"start_index": 10, "end_index": 19, "name": "K-H"},
        ]
    )

    def find_crossings_by_irreps(self, _outir, e_window, gap_tol):
        self.parameters = (e_window, gap_tol)
        return [
            {
                "k_interval": (4, 6),
                "k1_coords": [0.0, 0.0, 0.1],
                "k2_coords": [0.0, 0.0, 0.2],
                "band_indices": (4, 5),
                "irreps_swapped": ("DT7", "DT8"),
                "energy_approx": 0.02,
            },
            {
                "k_interval": (14, 16),
                "k1_coords": [1 / 3, 1 / 3, 0.1],
                "k2_coords": [1 / 3, 1 / 3, 0.2],
                "band_indices": (6, 7),
                "irreps_swapped": ("P4 + P5", "P6"),
                "energy_approx": -0.49,
            },
        ]


class BandResultAnalysisTests(unittest.TestCase):
    def test_result_name_resolves_material_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "NaBi_sg186_007"
            target.mkdir()
            self.assertEqual(find_result_directory("NaBi结果", root), target.resolve())

    def test_path_matching_is_orientation_independent(self):
        particles = [
            EmergentParticle(
                abbreviation="DP",
                name="Dirac point",
                paths=[HighSymmetryPath(line_label="Δ", path="ΓA", source_pdf_page=1058)],
            )
        ]
        matches = match_accidental_particles("A-\\Gamma", particles)
        self.assertEqual([item.abbreviation for item in matches], ["DP"])

    def test_analysis_distinguishes_unique_and_ambiguous_path_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "NaBi_sg186_007"
            job_dir = result_dir / "job_band"
            job_dir.mkdir(parents=True)
            (job_dir / "outir").write_text("Current command : irvsp -sg 186\n", encoding="utf-8")
            (job_dir / "INCAR").write_text("LSORBIT = True\n", encoding="utf-8")
            (job_dir / "vasprun.xml").touch()

            fake_image = result_dir / "band_NaBi.png"
            fake_image.touch()
            with patch("band_result_analysis._load_analyzer", return_value=FakeAnalyzer()):
                report = analyze_band_result("NaBi", results_root=root)

            self.assertTrue(Path(report.report_file).is_file())

        self.assertEqual(report.spacegroup_number, 186)
        self.assertEqual(report.spacegroup_symbol, "P6_3mc")
        self.assertTrue(report.soc)
        self.assertEqual(report.crossing_count, 2)
        self.assertEqual(report.confirmed_particle_types, ["DP"])
        self.assertEqual(
            report.path_compatible_particle_types,
            ["DP", "TP", "WNL", "WNL net"],
        )
        self.assertEqual(report.path_summaries[0].classification, "confirmed_by_unique_path")
        self.assertEqual(report.path_summaries[1].classification, "path_compatible_ambiguous")
        self.assertEqual(len(report.encyclopedia_path_table), 4)
        self.assertEqual(
            [item.abbreviation for item in report.encyclopedia_path_table],
            ["DP", "TP", "WNL", "WNL net"],
        )
        self.assertEqual(report.encyclopedia_path_table[0].crossing_count, 1)
        self.assertEqual(report.encyclopedia_path_table[1].high_symmetry_path, "KH")
        self.assertEqual(report.encyclopedia_path_table[1].line_label, "P")
        self.assertEqual(report.encyclopedia_path_table[1].source_pdf_page, 1058)


if __name__ == "__main__":
    unittest.main()
