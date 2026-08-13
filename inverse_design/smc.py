"""Frozen-checkpoint sequential Monte Carlo guidance for SymmCD sampling."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from inverse_design.catalog import _minimum_distance
from inverse_design.surrogate import structure_features_from_structure


@dataclass(frozen=True)
class SMCConfig:
    formula: str
    spacegroup_number: int
    model_path: Path
    output_dir: Path
    checkpoint_path: Path
    particles: int = 32
    diffusion_steps: int = 500
    resample_interval: int = 50
    guidance_start_fraction: float = 0.5
    alpha: float = 3.0
    ess_threshold: float = 0.8
    seed: int = 20260813
    device: str = "auto"
    allow_unvalidated_surrogate: bool = False

    def validate(self) -> None:
        if not self.formula.strip():
            raise ValueError("formula cannot be empty")
        if not 1 <= self.spacegroup_number <= 230:
            raise ValueError("spacegroup_number must be between 1 and 230")
        if self.particles < 2:
            raise ValueError("particles must be at least 2")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if self.resample_interval < 1:
            raise ValueError("resample_interval must be positive")
        if not 0 <= self.guidance_start_fraction < 1:
            raise ValueError("guidance_start_fraction must be in [0, 1)")
        if self.alpha < 0:
            raise ValueError("alpha cannot be negative")
        if not 0 < self.ess_threshold <= 1:
            raise ValueError("ess_threshold must be in (0, 1]")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"particle surrogate not found: {self.model_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SymmCD checkpoint not found: {self.checkpoint_path}")


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / len(values))
    shifted = np.where(finite, values - np.max(values[finite]), -np.inf)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(np.square(values)))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-variance resampling with deterministic behavior for a fixed RNG seed."""
    values = np.asarray(weights, dtype=float)
    count = len(values)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(values)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def build_particle_batch(
    spacegroup_number: int,
    atom_numbers: list[int],
    particles: int,
    device: str,
) -> Any:
    from torch_geometric.data import Batch
    from workflow_sym import build_constrained_batch

    single = build_constrained_batch(spacegroup_number, atom_numbers, "cpu")
    data = single.to_data_list()[0]
    return Batch.from_data_list([data.clone() for _ in range(particles)]).to(device)


def _split_nodes(tensor: torch.Tensor, atoms_per_particle: int) -> torch.Tensor:
    return tensor.reshape(-1, atoms_per_particle, *tensor.shape[1:])


def _resample_nodes(
    tensor: torch.Tensor,
    ancestors: np.ndarray,
    atoms_per_particle: int,
) -> torch.Tensor:
    index = torch.as_tensor(ancestors, device=tensor.device, dtype=torch.long)
    return _split_nodes(tensor, atoms_per_particle).index_select(0, index).reshape_as(tensor)


def _resample_graphs(tensor: torch.Tensor, ancestors: np.ndarray) -> torch.Tensor:
    index = torch.as_tensor(ancestors, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, index)


def _representative_structure(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    atom_numbers: list[int],
) -> Structure:
    return Structure(
        Lattice(lattice.detach().cpu().numpy()),
        atom_numbers,
        frac_coords.detach().cpu().numpy() % 1.0,
        coords_are_cartesian=False,
    )


class ParticleSurrogateScorer:
    def __init__(self, model_path: Path, requested_spacegroup: int):
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.calibrator = self.bundle.get("calibrator")
        self.metrics = self.bundle.get("metrics", {})
        self.particle = self.bundle.get("particle", "unknown")
        scope = self.bundle.get("spacegroup_scope")
        if scope is not None and int(scope) != requested_spacegroup:
            raise ValueError(
                f"surrogate is scoped to SG {scope}, not requested SG {requested_spacegroup}"
            )

    def score(
        self,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        atom_numbers: list[int],
        spacegroup_number: int,
        site_symm: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        from symmcd.pl_modules.mainmodelfun import modify_frac_coords_one

        feature_rows = []
        errors: list[str] = []
        for index in range(len(lattices)):
            try:
                if site_symm is None:
                    structure = _representative_structure(
                        frac_coords[index], lattices[index], atom_numbers
                    )
                else:
                    projected_frac, count, projected_types, _, _, _ = modify_frac_coords_one(
                        frac_coords[index],
                        site_symm[index],
                        torch.as_tensor(atom_numbers, device=frac_coords.device),
                        torch.tensor(spacegroup_number),
                    )
                    if count <= 0:
                        raise ValueError("symmetry projection produced an empty structure")
                    structure = Structure(
                        Lattice(lattices[index].detach().cpu().numpy()),
                        projected_types,
                        projected_frac,
                        coords_are_cartesian=False,
                    )
                features, _ = structure_features_from_structure(
                    structure,
                    spacegroup_number,
                    minimum_distance=_minimum_distance(structure),
                )
                feature_rows.append(features)
                errors.append("")
            except Exception as exc:
                feature_rows.append([0.0] * len(self.bundle["features"]))
                errors.append(f"{type(exc).__name__}: {exc}")
        raw = self.model.predict_proba(np.asarray(feature_rows, dtype=float))[:, 1]
        probabilities = (
            self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            if self.calibrator is not None
            else raw
        )
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities[[bool(error) for error in errors]] = 1e-6
        return np.clip(probabilities, 1e-6, 1 - 1e-6), errors


class SMCTwistingCallback:
    def __init__(
        self,
        scorer: ParticleSurrogateScorer,
        atom_numbers: list[int],
        spacegroup_number: int,
        particles: int,
        interval: int,
        start_step: int,
        end_step: int,
        alpha: float,
        ess_threshold: float,
        seed: int,
    ):
        self.scorer = scorer
        self.atom_numbers = atom_numbers
        self.spacegroup_number = spacegroup_number
        self.particles = particles
        self.interval = interval
        self.start_step = start_step
        self.end_step = end_step
        self.alpha = alpha
        self.ess_threshold = ess_threshold
        self.rng = np.random.default_rng(seed)
        self.log_weights = np.zeros(particles, dtype=float)
        self.previous_log_potential = np.zeros(particles, dtype=float)
        self.events: list[dict[str, Any]] = []

    def __call__(self, state: dict[str, Any]) -> dict[str, Any] | None:
        step = int(state["step"])
        is_final_step = bool(state["is_final_step"])
        if step < self.start_step:
            return None
        if not is_final_step and step % self.interval:
            return None
        atoms_per_particle = len(self.atom_numbers)
        coords = _split_nodes(state["frac_coords"], atoms_per_particle)
        site_symm = _split_nodes(state["site_symm"], atoms_per_particle)
        probabilities, errors = self.scorer.score(
            coords,
            state["lattices"],
            self.atom_numbers,
            self.spacegroup_number,
            site_symm=site_symm,
        )
        tempering_fraction = min(
            1.0,
            max(0.0, (step - self.start_step) / max(1, self.end_step - self.start_step)),
        )
        beta = self.alpha * tempering_fraction
        log_potential = beta * np.log(probabilities)
        self.log_weights += log_potential - self.previous_log_potential
        self.previous_log_potential = log_potential
        weights = normalize_log_weights(self.log_weights)
        ess = effective_sample_size(weights)
        should_resample = not is_final_step and ess < self.ess_threshold * self.particles
        event: dict[str, Any] = {
            "step": step,
            "diffusion_time": int(state["diffusion_time"]),
            "guidance_beta": round(beta, 6),
            "probability_min": round(float(probabilities.min()), 8),
            "probability_mean": round(float(probabilities.mean()), 8),
            "probability_max": round(float(probabilities.max()), 8),
            "effective_sample_size": round(ess, 6),
            "resampled": should_resample,
            "score_error_count": sum(bool(error) for error in errors),
        }
        if not should_resample:
            self.events.append(event)
            return None
        ancestors = systematic_resample(weights, self.rng)
        event["unique_ancestor_count"] = int(len(set(ancestors.tolist())))
        event["ancestors"] = ancestors.tolist()
        self.events.append(event)
        self.log_weights.fill(0.0)
        self.previous_log_potential = log_potential[ancestors]
        return {"ancestors": ancestors}


def _project_final_structure(
    frac_coords: torch.Tensor,
    site_symm: torch.Tensor,
    atom_numbers: list[int],
    spacegroup_number: int,
    lattice: torch.Tensor,
) -> Structure:
    from symmcd.pl_modules.mainmodelfun import modify_frac_coords_one

    projected_frac, count, projected_types, _, _, _ = modify_frac_coords_one(
        frac_coords,
        site_symm,
        torch.as_tensor(atom_numbers, device=frac_coords.device),
        torch.tensor(spacegroup_number),
    )
    if count <= 0:
        raise ValueError("symmetry projection produced an empty structure")
    structure = Structure(
        Lattice(lattice.detach().cpu().numpy()),
        projected_types,
        projected_frac,
        coords_are_cartesian=False,
    )
    analyzer = SpacegroupAnalyzer(structure, symprec=0.2)
    refined = analyzer.get_refined_structure()
    return SpacegroupAnalyzer(refined, symprec=0.1).get_primitive_standard_structure()


def run_smc_generation(config: SMCConfig) -> dict:
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.image")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pyxtal\.molecule")
    with contextlib.redirect_stdout(io.StringIO()):
        from workflow_sym import (
            WorkflowConfig,
            _load_generation_model,
            atomic_numbers_from_formula,
        )

    config.validate()
    device = (
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto"
        else config.device
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    atom_numbers = atomic_numbers_from_formula(config.formula)
    scorer = ParticleSurrogateScorer(config.model_path, config.spacegroup_number)
    validated = bool(scorer.metrics.get("validated_for_smc"))
    if not validated and not config.allow_unvalidated_surrogate:
        raise RuntimeError(
            "Particle surrogate did not pass the fixed-budget enrichment gate; "
            "use --allow-unvalidated-surrogate only for a method smoke test"
        )
    workflow_config = WorkflowConfig(
        formula=config.formula,
        spacegroup_number=config.spacegroup_number,
        checkpoint_path=config.checkpoint_path,
        enable_relax=False,
        diffusion_steps=config.diffusion_steps,
    )
    model = _load_generation_model(workflow_config, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        model.requires_grad_(False)
    model.eval()
    batch = build_particle_batch(
        config.spacegroup_number, atom_numbers, config.particles, device
    )
    callback = SMCTwistingCallback(
        scorer=scorer,
        atom_numbers=atom_numbers,
        spacegroup_number=config.spacegroup_number,
        particles=config.particles,
        interval=config.resample_interval,
        start_step=max(1, math.ceil(config.diffusion_steps * config.guidance_start_fraction)),
        end_step=config.diffusion_steps,
        alpha=config.alpha,
        ess_threshold=config.ess_threshold,
        seed=config.seed,
    )
    with torch.inference_mode():
        output, _ = model.sample(
            batch,
            num_steps=config.diffusion_steps,
            fixed_atom_types=batch.atom_types - 1,
            smc_callback=callback,
            show_progress=False,
        )

    atoms_per_particle = len(atom_numbers)
    representative_coords = _split_nodes(output["frac_coords"], atoms_per_particle)
    representative_symm = _split_nodes(output["site_symm"], atoms_per_particle)
    final_probabilities, score_errors = scorer.score(
        representative_coords,
        output["lattices"],
        atom_numbers,
        config.spacegroup_number,
        site_symm=representative_symm,
    )
    order = np.argsort(final_probabilities)[::-1]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    requested_composition = Composition(config.formula).fractional_composition
    rows = []
    for rank, index in enumerate(order, start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "particle_index": int(index),
            "particle": scorer.particle,
            "surrogate_probability": round(float(final_probabilities[index]), 8),
            "requested_spacegroup": config.spacegroup_number,
            "actual_spacegroup": "",
            "actual_formula": "",
            "composition_retained": False,
            "requested_spacegroup_retained": False,
            "minimum_distance_angstrom": "",
            "valid": False,
            "condition_valid": False,
            "structure_file": "",
            "error": score_errors[index],
        }
        try:
            structure = _project_final_structure(
                representative_coords[index],
                representative_symm[index],
                atom_numbers,
                config.spacegroup_number,
                output["lattices"][index],
            )
            actual_sg = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
            actual_composition = structure.composition.fractional_composition
            composition_retained = actual_composition.almost_equals(
                requested_composition, rtol=0.0, atol=1e-6
            )
            minimum_distance = _minimum_distance(structure)
            requested_sg_retained = actual_sg == config.spacegroup_number
            condition_valid = composition_retained and requested_sg_retained
            filename = (
                f"rank_{rank:03d}_{config.formula}_sg{config.spacegroup_number}_"
                f"p{final_probabilities[index]:.4f}.cif"
            )
            path = config.output_dir / filename
            structure.to(filename=path)
            row.update(
                {
                    "actual_spacegroup": actual_sg,
                    "actual_formula": structure.composition.reduced_formula,
                    "composition_retained": composition_retained,
                    "requested_spacegroup_retained": requested_sg_retained,
                    "minimum_distance_angstrom": round(float(minimum_distance), 8),
                    "valid": True,
                    "condition_valid": condition_valid,
                    "structure_file": str(path.resolve()),
                    "error": "",
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    with (config.output_dir / "smc_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (config.output_dir / "smc_events.json").write_text(
        json.dumps(callback.events, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "method": "TDS-inspired sequential Monte Carlo twisting",
        "checkpoint_frozen": True,
        "particle": scorer.particle,
        "surrogate_validated_for_smc": validated,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "resampling_event_count": sum(event["resampled"] for event in callback.events),
        "scoring_event_count": len(callback.events),
        "valid_output_count": sum(bool(row["valid"]) for row in rows),
        "composition_retained_count": sum(bool(row["composition_retained"]) for row in rows),
        "requested_spacegroup_retained_count": sum(
            bool(row["requested_spacegroup_retained"]) for row in rows
        ),
        "condition_valid_count": sum(bool(row["condition_valid"]) for row in rows),
        "probability_min": round(float(final_probabilities.min()), 8),
        "probability_mean": round(float(final_probabilities.mean()), 8),
        "probability_max": round(float(final_probabilities.max()), 8),
        "warning": "Surrogate-guided candidates require prospective SOC band and IRVSP validation.",
    }
    (config.output_dir / "smc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
