import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from inverse_design.catalog import CatalogConfig, _minimum_distance, build_catalog, discover_candidates
from inverse_design.baselines import create_matched_random_targets
from inverse_design.funnel import build_funnel, build_prenovelty_funnel
from inverse_design.dft_results import DFTMaterialRecord, _is_vasp_job_directory, classify_stage
from inverse_design.metrics import create_assignment_template, evaluate_benchmark
from inverse_design.references import evaluate_novelty, import_mp20_snapshot, import_reference_structures
from inverse_design.surrogate import train_surrogate
from inverse_design.smc import (
    _resample_nodes,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
)


class InverseDesignTests(unittest.TestCase):
    def _write_candidate(self, root: Path, folder: str, identifier: str, structure: Structure) -> Path:
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"gen_{identifier}.cif"
        structure.to(filename=path)
        return path

    def test_catalog_uses_unique_collection_id_and_reclassifies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "STRUCTURE"
            structure = Structure(
                Lattice.cubic(5.4),
                ["Na", "Cl"],
                [[0, 0, 0], [0.5, 0.5, 0.5]],
            )
            self._write_candidate(source, "batch", "NaCl_sg221_001", structure)
            self._write_candidate(source, "batch_2", "NaCl_sg221_001", structure)
            records = build_catalog(
                CatalogConfig(
                    source_root=source,
                    output_root=root / "catalog",
                    curated_root=root / "curated",
                )
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(len({record.candidate_id for record in records}), 2)
            self.assertTrue(all(record.target_spacegroup_match for record in records))
            self.assertTrue(all(Path(record.curated_file).is_file() for record in records))
            self.assertEqual(sum(record.is_unique is True for record in records), 1)

    def test_nested_task_candidates_have_unique_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            structure = Structure(Lattice.cubic(3.5), ["C"], [[0, 0, 0]])
            self._write_candidate(source, "a/carbon", "task0_sg221_6", structure)
            self._write_candidate(source, "b/carbon", "task0_sg221_6", structure)
            candidates = discover_candidates(source)
            self.assertEqual(len({item[1] for item in candidates}), 2)

    def test_novelty_requires_structure_match_not_formula_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = Structure(
                Lattice.cubic(5.4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            candidate_file = root / "candidate.cif"
            candidate.to(filename=candidate_file)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                json.dumps(
                    {
                        "candidate_id": "one",
                        "valid_structure": True,
                        "reduced_formula": candidate.composition.reduced_formula,
                        "selected_file": str(candidate_file),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            refs = root / "references.jsonl"
            import_reference_structures([candidate_file], refs, "test")
            rows = evaluate_novelty(catalog, [refs], root / "novelty.csv")
            self.assertFalse(rows[0]["is_novel"])
            self.assertTrue(rows[0]["matched_reference_ids"])

    def test_mp20_import_and_uncovered_formula_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            structure = Structure(Lattice.cubic(3.0), ["C"], [[0, 0, 0]])
            fields = [
                "material_id", "pretty_formula", "formation_energy_per_atom",
                "e_above_hull", "spacegroup.number", "cif",
            ]
            for split in ("train", "val", "test"):
                with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({
                        "material_id": f"mp-{split}", "pretty_formula": "C",
                        "formation_energy_per_atom": "0", "e_above_hull": "0",
                        "spacegroup.number": "221", "cif": structure.to(fmt="cif"),
                    })
            snapshot = root / "mp20.jsonl"
            metadata = import_mp20_snapshot(root, snapshot)
            self.assertEqual(metadata["structure_count"], 3)
            candidate = Structure(Lattice.cubic(4.0), ["Na"], [[0, 0, 0]])
            candidate_file = root / "na.cif"
            candidate.to(filename=candidate_file)
            catalog = root / "catalog.jsonl"
            catalog.write_text(json.dumps({
                "candidate_id": "na", "valid_structure": True,
                "reduced_formula": "Na", "selected_file": str(candidate_file),
            }) + "\n", encoding="utf-8")
            rows = evaluate_novelty(catalog, [snapshot], root / "novelty.csv")
            self.assertEqual(rows[0]["novelty_status"], "composition_novel")
            self.assertFalse(rows[0]["reference_formula_covered"])
            self.assertTrue(rows[0]["is_novel"])
            self.assertTrue(rows[0]["composition_novel"])
            self.assertFalse(rows[0]["global_novelty_evaluated"])

    def test_empty_dft_denominators_are_not_reported_as_zero_rates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                json.dumps(
                    {
                        "candidate_id": "one",
                        "valid_structure": True,
                        "target_spacegroup_match": True,
                        "is_unique": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            assignments = root / "assignments.csv"
            create_assignment_template(catalog, assignments)
            with assignments.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            row["arm"] = "symmcd_spacegroup"
            with assignments.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            report = evaluate_benchmark(catalog, assignments, root / "report.json")
            arm = report["arms"]["symmcd_spacegroup"]
            self.assertIsNone(arm["stability"]["rate"])
            self.assertIsNone(arm["dft_topology_hit"]["rate"])

    def test_surrogate_refuses_insufficient_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "catalog.jsonl").write_text("", encoding="utf-8")
            (root / "labels.csv").write_text("candidate_id,topology_evaluated,topology_hit\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "only 0 labeled"):
                train_surrogate(
                    root / "catalog.jsonl",
                    root / "labels.csv",
                    root / "models",
                    "topology",
                )

    def test_smc_weight_normalization_and_ess(self):
        weights = normalize_log_weights(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(weights, np.full(4, 0.25))
        self.assertAlmostEqual(effective_sample_size(weights), 4.0)

    def test_systematic_resampling_is_seeded_and_in_range(self):
        weights = np.array([0.05, 0.05, 0.1, 0.8])
        first = systematic_resample(weights, np.random.default_rng(7))
        second = systematic_resample(weights, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len(weights))
        self.assertTrue(np.all((first >= 0) & (first < len(weights))))

    def test_smc_node_resampling_preserves_particle_blocks(self):
        import torch

        nodes = torch.tensor([[0], [1], [10], [11], [20], [21]])
        sampled = _resample_nodes(nodes, np.array([2, 2, 0]), atoms_per_particle=2)
        self.assertEqual(sampled.reshape(3, 2).tolist(), [[20, 21], [20, 21], [0, 1]])

    def test_funnel_budget_is_publication_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("catalog.jsonl", "novelty.csv", "predictions.csv"):
                (root / name).write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "between 500 and 2000"):
                build_funnel(
                    root / "catalog.jsonl",
                    root / "novelty.csv",
                    root / "predictions.csv",
                    root / "funnel.csv",
                    10,
                )

    def test_prenovelty_funnel_budget_is_publication_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("catalog.jsonl", "predictions.csv"):
                (root / name).write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "between 500 and 2000"):
                build_prenovelty_funnel(
                    root / "catalog.jsonl",
                    root / "predictions.csv",
                    root / "pre_funnel.csv",
                    20,
                )

    def test_random_targets_skip_unknown_legacy_formula(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                json.dumps(
                    {
                        "candidate_id": "task",
                        "target_formula": None,
                        "target_spacegroup": 194,
                        "stoichiometry_valid": True,
                        "target_spacegroup_match": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            targets = create_matched_random_targets(catalog, root / "targets.json", 20)
            self.assertEqual(targets, [])

    def test_dft_stage_classification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "INCAR").write_text("NSW = 99\nIBRION = 2\n", encoding="ascii")
            self.assertEqual(classify_stage(root), "relax")
            (root / "INCAR").write_text(
                "NSW = 0\nLSORBIT = True\nICHARG = 11\n", encoding="ascii"
            )
            self.assertEqual(classify_stage(root), "band_soc")

    def test_dft_material_schema_tracks_relaxed_symmetry(self):
        names = DFTMaterialRecord.__dataclass_fields__
        self.assertIn("relaxed_spacegroup_strict", names)
        self.assertIn("band_symmetry_consistent", names)
        self.assertIn("target_spacegroup_retained_strict", names)

    def test_completed_irvsp_is_not_rejected_by_independent_spglib_audit(self):
        from inverse_design.labels import merge_dft_topology_labels

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dft = root / "dft.csv"
            topology = root / "topology.csv"
            with dft.open("w", encoding="utf-8-sig", newline="") as handle:
                fields = [
                    "candidate_id", "result_name", "relax_completed",
                    "relative_energy_per_atom_ev", "low_energy_polymorph",
                    "band_symmetry_consistent", "final_structure_file",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "one", "result_name": "result", "relax_completed": True,
                    "relative_energy_per_atom_ev": 0, "low_energy_polymorph": True,
                    "band_symmetry_consistent": False, "final_structure_file": "POSCAR",
                })
            with topology.open("w", encoding="utf-8-sig", newline="") as handle:
                fields = [
                    "result_name", "analysis_completed", "strict_topology_hit",
                    "confirmed_particles", "crossing_count", "compatible_particles", "report_file",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "result_name": "result", "analysis_completed": True,
                    "strict_topology_hit": True, "confirmed_particles": "DP",
                    "crossing_count": 1, "compatible_particles": "DP", "report_file": "report.json",
                })
            rows = merge_dft_topology_labels(dft, topology, root / "labels.csv")
            self.assertTrue(rows[0]["topology_evaluated"])
            self.assertEqual(rows[0]["topology_hit"], "True")

    def test_agent_analysis_directory_is_not_a_vasp_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = Path(temp_dir) / "agent_analysis"
            analysis.mkdir()
            (analysis / "report.json").write_text("{}", encoding="ascii")
            self.assertFalse(_is_vasp_job_directory(analysis))

    def test_single_site_minimum_distance_uses_periodic_image(self):
        structure = Structure(Lattice.cubic(3.0), ["C"], [[0, 0, 0]])
        self.assertAlmostEqual(_minimum_distance(structure), 3.0)


if __name__ == "__main__":
    unittest.main()
