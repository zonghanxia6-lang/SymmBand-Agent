import unittest
from types import SimpleNamespace

from physics_eval import evaluate_response
from physics_evidence import (
    build_band_evidence_assessment,
    crossing_diagnostics,
    validate_analysis_parameters,
)


class PhysicsEvidenceTests(unittest.TestCase):
    def test_crossing_diagnostics_preserve_units_reference_and_threshold_margin(self):
        diagnostics = crossing_diagnostics(
            {
                "energy_approx": -0.49,
                "minimum_gap_ev": 0.006,
                "neighboring_gaps_ev": (0.04, 0.05),
            },
            energy_window_ev=1.0,
            gap_tolerance_ev=0.03,
        )
        self.assertEqual(diagnostics["energy_reference"], "E - E_F (eV)")
        self.assertEqual(diagnostics["gap_to_tolerance_ratio"], 0.2)
        self.assertEqual(diagnostics["threshold_margin"], "well_below_threshold")
        self.assertEqual(diagnostics["fermi_proximity"], "within_0.5_eV_of_fermi")

    def test_evidence_ladder_never_promotes_unique_path_to_topology_confirmation(self):
        crossing = SimpleNamespace(classification="confirmed_by_unique_path")
        assessment = build_band_evidence_assessment(
            [crossing], electronic_converged=True, kpoint_count=80
        )
        self.assertEqual(
            assessment.evidence_level,
            "L3_path_taxonomy_unique_symmetry_candidate",
        )
        self.assertIn("do not state that topology is confirmed", assessment.claim_boundary)
        self.assertTrue(any("three-dimensional" in item for item in assessment.limitations))

    def test_nonconvergence_is_an_explicit_failed_quality_check(self):
        crossing = SimpleNamespace(classification="path_compatible_ambiguous")
        assessment = build_band_evidence_assessment(
            [crossing], electronic_converged=False, kpoint_count=None
        )
        convergence = assessment.quality_checks[0]
        self.assertEqual(convergence.status, "fail")
        self.assertIn("not reliable", convergence.consequence)

    def test_analysis_parameters_reject_dimensionally_bad_thresholds(self):
        with self.assertRaises(ValueError):
            validate_analysis_parameters(1.0, 1.0)
        with self.assertRaises(ValueError):
            validate_analysis_parameters(-1.0, 0.03)

    def test_response_eval_detects_overclaim(self):
        case = {
            "id": "topology",
            "required_concepts": [["候选", "candidate"], ["berry"]],
            "forbidden_claims": ["拓扑已确认"],
        }
        result = evaluate_response(case, "这是候选，但拓扑已确认。")
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_concepts, [["berry"]])
        self.assertEqual(result.forbidden_hits, ["拓扑已确认"])


if __name__ == "__main__":
    unittest.main()
