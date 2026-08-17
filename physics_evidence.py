"""Evidence-calibrated helpers for physics analysis.

This module is deliberately independent of the language model.  It converts numerical
outputs into explicit evidence levels, limitations, and follow-up calculations so that
the agent cannot silently promote a detector hit into a stronger physical claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PhysicsQualityCheck:
    name: str
    status: str
    evidence: str
    consequence: str


@dataclass(frozen=True)
class PhysicsEvidenceAssessment:
    evidence_level: str
    conclusion: str
    claim_boundary: str
    quality_checks: list[PhysicsQualityCheck]
    limitations: list[str]
    recommended_validations: list[str]


def validate_analysis_parameters(energy_window_ev: float, gap_tolerance_ev: float) -> None:
    """Reject physically meaningless detector settings before parsing expensive data."""
    if energy_window_ev <= 0:
        raise ValueError("energy_window_ev must be positive and referenced to the Fermi level")
    if gap_tolerance_ev <= 0:
        raise ValueError("gap_tolerance_ev must be positive")
    if gap_tolerance_ev >= energy_window_ev:
        raise ValueError("gap_tolerance_ev must be smaller than energy_window_ev")


def crossing_diagnostics(
    raw: dict[str, Any],
    *,
    energy_window_ev: float,
    gap_tolerance_ev: float,
) -> dict[str, Any]:
    """Preserve the numerical context that supports one crossing-detector hit."""
    minimum_gap = _optional_float(raw.get("minimum_gap_ev"))
    neighboring = raw.get("neighboring_gaps_ev")
    neighboring_gaps = (
        tuple(float(value) for value in neighboring) if neighboring is not None else None
    )
    energy = float(raw["energy_approx"])
    if abs(energy) <= 0.1:
        proximity = "within_0.1_eV_of_fermi"
    elif abs(energy) <= 0.5:
        proximity = "within_0.5_eV_of_fermi"
    else:
        proximity = "inside_requested_energy_window"

    ratio = minimum_gap / gap_tolerance_ev if minimum_gap is not None else None
    if ratio is None:
        threshold_margin = "not_recorded"
    elif ratio <= 0.25:
        threshold_margin = "well_below_threshold"
    elif ratio <= 0.75:
        threshold_margin = "below_threshold"
    else:
        threshold_margin = "near_threshold"
    return {
        "minimum_gap_ev": minimum_gap,
        "neighboring_gaps_ev": neighboring_gaps,
        "gap_to_tolerance_ratio": round(ratio, 6) if ratio is not None else None,
        "threshold_margin": threshold_margin,
        "fermi_proximity": proximity,
        "energy_reference": "E - E_F (eV)",
        "inside_energy_window": abs(energy) <= energy_window_ev,
    }


def build_band_evidence_assessment(
    crossings: Iterable[Any],
    *,
    electronic_converged: bool | None,
    kpoint_count: int | None,
) -> PhysicsEvidenceAssessment:
    """Build a conservative evidence ladder for a line-mode band analysis."""
    items = list(crossings)
    unique_path_hits = [
        item for item in items if item.classification == "confirmed_by_unique_path"
    ]
    ambiguous_hits = [
        item for item in items if item.classification == "path_compatible_ambiguous"
    ]

    if electronic_converged is True:
        convergence_check = PhysicsQualityCheck(
            name="electronic_convergence",
            status="pass",
            evidence="vasprun reports electronic convergence",
            consequence="No electronic non-convergence warning is attached to the detector hits.",
        )
    elif electronic_converged is False:
        convergence_check = PhysicsQualityCheck(
            name="electronic_convergence",
            status="fail",
            evidence="vasprun reports electronic non-convergence",
            consequence="Band energies and apparent crossings are not reliable until reconverged.",
        )
    else:
        convergence_check = PhysicsQualityCheck(
            name="electronic_convergence",
            status="not_available",
            evidence="the parser did not expose an electronic convergence flag",
            consequence="Convergence must be checked directly from the VASP outputs.",
        )

    checks = [
        convergence_check,
        PhysicsQualityCheck(
            name="line_path_sampling",
            status="reported_not_converged",
            evidence=(
                f"line-mode band structure contains {kpoint_count} k-points"
                if kpoint_count is not None
                else "line-mode k-point count is unavailable"
            ),
            consequence="A single path sampling does not establish k-point convergence.",
        ),
        PhysicsQualityCheck(
            name="symmetry_exchange_test",
            status="pass" if items else "no_hit",
            evidence=(
                f"{len(items)} local-gap minima also exchange adjacent-band IRVSP irreps"
                if items
                else "no local-gap minimum passed the adjacent-band irrep-exchange test"
            ),
            consequence=(
                "Hits are symmetry-supported crossing candidates on sampled lines."
                if items
                else "This excludes hits only within the sampled paths, energy window, and threshold."
            ),
        ),
        PhysicsQualityCheck(
            name="three_dimensional_topology",
            status="not_tested",
            evidence="only a one-dimensional high-symmetry path was analyzed",
            consequence="Point, line, line-net topology and topological charge are not confirmed.",
        ),
    ]

    if not items:
        level = "L0_no_candidate_within_search_scope"
        conclusion = "No symmetry-supported accidental crossing was detected within the configured search scope."
    elif unique_path_hits:
        level = "L3_path_taxonomy_unique_symmetry_candidate"
        conclusion = (
            f"Detected {len(items)} symmetry-supported crossing candidate(s); "
            f"{len(unique_path_hits)} has a unique encyclopedia path taxonomy."
        )
    else:
        level = "L2_symmetry_supported_crossing_candidate"
        conclusion = f"Detected {len(items)} symmetry-supported crossing candidate(s) on sampled lines."

    limitations = [
        "A path-unique encyclopedia match is a taxonomy match, not proof of a topological phase.",
        "The finite k-point mesh gives an upper bound/sample estimate for a crossing gap, not an exact zero-gap proof.",
        "No off-path three-dimensional search, Berry phase, Wilson loop, surface state, or topological charge is included.",
    ]
    if ambiguous_hits:
        limitations.append(
            f"{len(ambiguous_hits)} hit(s) share their path with multiple particle types and cannot be uniquely named."
        )
    if electronic_converged is not True:
        limitations.append("Electronic convergence is not positively established for this report.")

    validations = [
        "Repeat with tighter electronic convergence, higher ENCUT, and denser k sampling; report the crossing-gap change.",
        "Perform a local k-space refinement around every candidate and test whether the minimum gap tends to zero.",
        "Scan a three-dimensional neighborhood or build a symmetry-faithful Wannier model to determine point/line dimensionality.",
        "Compute the relevant Berry phase, Wilson loop, Chern/topological charge, and surface or hinge signatures.",
        "Verify that the magnetic order, SOC setting, relaxed structure, and space group match the intended physical state.",
    ]
    return PhysicsEvidenceAssessment(
        evidence_level=level,
        conclusion=conclusion,
        claim_boundary=(
            "The strongest permitted claim is a symmetry-supported crossing candidate with path taxonomy; "
            "do not state that topology is confirmed."
        ),
        quality_checks=checks,
        limitations=limitations,
        recommended_validations=validations,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
