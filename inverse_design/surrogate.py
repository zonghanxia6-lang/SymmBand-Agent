"""Leakage-aware baseline surrogates for stability and topology prioritization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
from pymatgen.core import Element, Structure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .metrics import _bool
from .catalog import _minimum_distance


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def structure_features_from_structure(
    structure: Structure,
    spacegroup_number: int,
    minimum_distance: float | None = None,
) -> tuple[list[float], list[str]]:
    """Build the auditable feature vector used by clean and SMC particle scoring."""
    if minimum_distance is None or not math.isfinite(minimum_distance):
        minimum_distance = _minimum_distance(structure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        density = float(structure.density)
    fractions = structure.composition.fractional_composition
    elements = list(fractions.elements)
    weights = np.array([float(fractions[element]) for element in elements])

    def property_values(name: str) -> list[float]:
        values = []
        for element in elements:
            value = getattr(Element(element.symbol), name, None)
            try:
                values.append(float(value or 0.0))
            except (TypeError, ValueError):
                values.append(0.0)
        return values

    atomic_numbers = property_values("Z")
    masses = property_values("atomic_mass")
    electronegativities = property_values("X")
    rows = property_values("row")
    groups = property_values("group")
    feature_names = [
        "atom_count",
        "element_count",
        "volume_per_atom",
        "density",
        "min_distance",
        "spacegroup_number",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "alpha",
        "beta",
        "gamma",
    ]
    values = [
        len(structure),
        len(elements),
        structure.volume / len(structure),
        density,
        minimum_distance,
        float(spacegroup_number),
        structure.lattice.a,
        structure.lattice.b,
        structure.lattice.c,
        structure.lattice.alpha,
        structure.lattice.beta,
        structure.lattice.gamma,
    ]
    for name, data in (
        ("atomic_number", atomic_numbers),
        ("atomic_mass", masses),
        ("electronegativity", electronegativities),
        ("period", rows),
        ("group", groups),
    ):
        array = np.asarray(data, dtype=float)
        values.extend([float(np.average(array, weights=weights)), _std(data), min(data), max(data)])
        feature_names.extend([f"{name}_weighted_mean", f"{name}_std", f"{name}_min", f"{name}_max"])
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"Non-finite structure features for {row.get('candidate_id', 'unknown')}")
    return values, feature_names


def structure_features(row: dict) -> tuple[list[float], list[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        structure = Structure.from_file(row["selected_file"])
    minimum_distance = float(row.get("min_distance_angstrom") or 0.0)
    return structure_features_from_structure(
        structure,
        int(row.get("actual_spacegroup") or 0),
        minimum_distance=minimum_distance,
    )


def _chemical_group(structure: Structure) -> str:
    return "-".join(sorted(element.symbol for element in structure.composition.elements))


def _label_for_task(label: dict, task: str, stability_threshold: float) -> bool | None:
    if task == "topology":
        return _bool(label.get("topology_hit")) if _bool(label.get("topology_evaluated")) else None
    if task == "stability_proxy":
        return _bool(label.get("low_energy_polymorph"))
    ehull = str(label.get("energy_above_hull_ev_per_atom", "")).strip()
    phonon = _bool(label.get("phonon_stable"))
    if not ehull and phonon is None:
        return None
    return (bool(ehull) and float(ehull) <= stability_threshold) or phonon is True


def train_surrogate(
    catalog_jsonl: Path,
    labels_csv: Path,
    output_dir: Path,
    task: str,
    stability_threshold: float = 0.1,
    minimum_labeled: int = 30,
) -> dict:
    if task not in {"stability", "stability_proxy", "topology"}:
        raise ValueError("task must be 'stability', 'stability_proxy', or 'topology'")
    with labels_csv.open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["candidate_id"]: row for row in csv.DictReader(handle) if row.get("candidate_id")}
    catalog = [json.loads(line) for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()]
    train_rows = []
    y = []
    groups = []
    x = []
    feature_names: list[str] = []
    for row in catalog:
        label = _label_for_task(labels.get(row["candidate_id"], {}), task, stability_threshold)
        if label is None or not row.get("valid_structure"):
            continue
        features, feature_names = structure_features(row)
        structure = Structure.from_file(row["selected_file"])
        train_rows.append(row)
        x.append(features)
        y.append(int(label))
        groups.append(_chemical_group(structure))
    if len(y) < minimum_labeled:
        raise RuntimeError(
            f"Refusing to train {task} surrogate: only {len(y)} labeled structures; "
            f"at least {minimum_labeled} are required"
        )
    class_counts = np.bincount(np.asarray(y), minlength=2)
    if min(class_counts) < 5:
        raise RuntimeError(f"Refusing to train: class counts {class_counts.tolist()} are too imbalanced")
    unique_groups = len(set(groups))
    positive_groups = len({group for group, label in zip(groups, y) if label == 1})
    negative_groups = len({group for group, label in zip(groups, y) if label == 0})
    folds = min(5, positive_groups, negative_groups)
    if folds < 2:
        raise RuntimeError(
            "Both classes must span at least two independent chemical-system groups"
        )
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=20260812,
        n_jobs=-1,
    )
    probabilities = cross_val_predict(
        model,
        np.asarray(x),
        np.asarray(y),
        groups=np.asarray(groups),
        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=20260812),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    predictions = probabilities >= 0.5
    metrics = {
        "task": task,
        "labeled_count": len(y),
        "positive_count": int(sum(y)),
        "chemical_system_group_count": unique_groups,
        "cv": f"{folds}-fold StratifiedGroupKFold by chemical system",
        "balanced_accuracy": round(balanced_accuracy_score(y, predictions), 6),
        "average_precision": round(average_precision_score(y, probabilities), 6),
        "roc_auc": round(roc_auc_score(y, probabilities), 6),
        "warning": "Baseline prioritization model; prospective fixed-budget validation is required.",
    }
    model.fit(np.asarray(x), np.asarray(y))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names, "task": task}, output_dir / f"{task}_surrogate.joblib")
    (output_dir / f"{task}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _particle_tokens(value: object) -> set[str]:
    return {token.strip() for token in str(value or "").split(";") if token.strip()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_particle_surrogate(
    catalog_jsonl: Path,
    labels_csv: Path,
    output_dir: Path,
    particle: str = "DP",
    spacegroup_number: int | None = None,
    minimum_labeled: int = 30,
) -> dict:
    """Train a particle-specific classifier using only symmetry-compatible hard negatives."""
    from emergent_particles import lookup_emergent_particles

    particle = particle.strip().upper()
    if particle not in {"DP", "DNL"}:
        raise ValueError("particle must be DP or DNL")
    with labels_csv.open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["candidate_id"]: row for row in csv.DictReader(handle) if row.get("candidate_id")}
    catalog = [json.loads(line) for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()]

    x: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    actual_spacegroups: list[int] = []
    audit_rows: list[dict] = []
    conflict_rows: list[dict] = []
    exclusion_rows: list[dict] = []
    feature_names: list[str] = []
    compatibility_cache: dict[int, bool] = {}
    for row in catalog:
        label = labels.get(row.get("candidate_id"), {})
        if not _bool(label.get("topology_evaluated")):
            continue
        actual_sg = int(row.get("actual_spacegroup") or 0)
        if spacegroup_number is not None and actual_sg != spacegroup_number:
            continue
        if actual_sg not in compatibility_cache:
            accidental = lookup_emergent_particles(actual_sg, soc=True).accidental
            compatibility_cache[actual_sg] = particle in {item.abbreviation.upper() for item in accidental}
        hit = particle in {token.upper() for token in _particle_tokens(label.get("topology_particles"))}
        if not compatibility_cache[actual_sg]:
            if hit:
                conflict_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "actual_spacegroup": actual_sg,
                        "reported_particles": label.get("topology_particles", ""),
                        "reason": "strict hit conflicts with encyclopedia compatibility",
                    }
                )
            continue
        if not row.get("valid_structure"):
            exclusion_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "actual_spacegroup": actual_sg,
                    "reported_particles": label.get("topology_particles", ""),
                    "reason": "catalog valid_structure is false",
                }
            )
            continue
        features, feature_names = structure_features(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = Structure.from_file(row["selected_file"])
        group = _chemical_group(structure)
        x.append(features)
        y.append(int(hit))
        groups.append(group)
        actual_spacegroups.append(actual_sg)
        audit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "actual_spacegroup": actual_sg,
                "chemical_system": group,
                "label": int(hit),
                "label_kind": "strict_positive" if hit else "compatible_hard_negative",
            }
        )

    if len(y) < minimum_labeled:
        raise RuntimeError(
            f"Refusing to train {particle} surrogate: only {len(y)} compatible labeled structures; "
            f"at least {minimum_labeled} are required"
        )
    class_counts = np.bincount(np.asarray(y), minlength=2)
    if min(class_counts) < 5:
        raise RuntimeError(f"Refusing to train: class counts {class_counts.tolist()} are too imbalanced")
    positive_groups = len({group for group, target in zip(groups, y) if target == 1})
    negative_groups = len({group for group, target in zip(groups, y) if target == 0})
    folds = min(5, positive_groups, negative_groups)
    if folds < 2:
        raise RuntimeError("Both classes must span at least two independent chemical-system groups")

    model = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=20260813,
        n_jobs=-1,
    )
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=20260813)
    probabilities = cross_val_predict(
        model,
        np.asarray(x),
        np.asarray(y),
        groups=np.asarray(groups),
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    calibrator = LogisticRegression(random_state=20260813)
    calibrator.fit(probabilities.reshape(-1, 1), np.asarray(y))
    # Report strictly held-out scores. The calibrator is fitted on all OOF pairs only
    # for deployment and is not reused to claim cross-validation performance.
    predictions = probabilities >= 0.5
    budget = max(1, math.ceil(0.2 * len(y)))
    top_indices = np.argsort(probabilities)[-budget:]
    precision_at_20 = float(np.mean(np.asarray(y)[top_indices]))
    base_rate = float(np.mean(y))
    enrichment = precision_at_20 / base_rate if base_rate else None
    group_array = np.asarray(groups)
    unique_group_values = np.unique(group_array)
    bootstrap_rng = np.random.default_rng(20260813)
    enrichment_bootstrap = []
    for _ in range(2000):
        sampled_groups = bootstrap_rng.choice(
            unique_group_values, size=len(unique_group_values), replace=True
        )
        sampled_indices = np.concatenate(
            [np.flatnonzero(group_array == group) for group in sampled_groups]
        )
        sampled_y = np.asarray(y)[sampled_indices]
        sampled_p = probabilities[sampled_indices]
        sampled_base_rate = float(np.mean(sampled_y))
        if sampled_base_rate <= 0:
            continue
        sampled_budget = max(1, math.ceil(0.2 * len(sampled_y)))
        sampled_top = np.argsort(sampled_p)[-sampled_budget:]
        enrichment_bootstrap.append(
            float(np.mean(sampled_y[sampled_top])) / sampled_base_rate
        )
    enrichment_ci = np.percentile(enrichment_bootstrap, [2.5, 97.5]).tolist()

    sg194_mask = np.asarray(actual_spacegroups) == 194
    sg194_metrics = None
    if sg194_mask.sum() >= 10 and len(np.unique(np.asarray(y)[sg194_mask])) == 2:
        sg194_y = np.asarray(y)[sg194_mask]
        sg194_p = probabilities[sg194_mask]
        sg194_budget = max(1, math.ceil(0.2 * len(sg194_y)))
        sg194_top = np.argsort(sg194_p)[-sg194_budget:]
        sg194_base = float(np.mean(sg194_y))
        sg194_precision = float(np.mean(sg194_y[sg194_top]))
        sg194_metrics = {
            "labeled_count": int(sg194_mask.sum()),
            "positive_count": int(sg194_y.sum()),
            "positive_base_rate": round(sg194_base, 6),
            "average_precision": round(average_precision_score(sg194_y, sg194_p), 6),
            "roc_auc": round(roc_auc_score(sg194_y, sg194_p), 6),
            "precision_at_top_20_percent": round(sg194_precision, 6),
            "enrichment_at_top_20_percent": round(sg194_precision / sg194_base, 6),
        }
    task_name = f"particle_{particle.lower()}"
    if spacegroup_number is not None:
        task_name += f"_sg{spacegroup_number}"
    metrics = {
        "task": task_name,
        "particle": particle,
        "spacegroup_scope": spacegroup_number or "all compatible SOC space groups",
        "labeled_count": len(y),
        "positive_count": int(class_counts[1]),
        "compatible_hard_negative_count": int(class_counts[0]),
        "excluded_compatibility_conflict_count": len(conflict_rows),
        "excluded_invalid_structure_count": len(exclusion_rows),
        "chemical_system_group_count": len(set(groups)),
        "positive_chemical_system_group_count": positive_groups,
        "negative_chemical_system_group_count": negative_groups,
        "cv": f"{folds}-fold StratifiedGroupKFold by chemical system",
        "positive_base_rate": round(base_rate, 6),
        "balanced_accuracy_at_0_5": round(balanced_accuracy_score(y, predictions), 6),
        "average_precision": round(average_precision_score(y, probabilities), 6),
        "roc_auc": round(roc_auc_score(y, probabilities), 6),
        "brier_score": round(brier_score_loss(y, probabilities), 6),
        "top_20_percent_count": budget,
        "precision_at_top_20_percent": round(precision_at_20, 6),
        "enrichment_at_top_20_percent": round(enrichment, 6) if enrichment is not None else None,
        "enrichment_at_top_20_percent_group_bootstrap_95_ci": [
            round(value, 6) for value in enrichment_ci
        ],
        "sg194_oof_metrics": sg194_metrics,
        "validated_for_smc": bool(
            enrichment is not None
            and enrichment >= 1.5
            and enrichment_ci[0] > 1.0
        ),
        "validation_probability": "raw group-held-out random-forest probability",
        "deployment_calibration": "Platt scaling fitted to all OOF prediction-label pairs",
        "catalog_sha256": _sha256(catalog_jsonl),
        "labels_sha256": _sha256(labels_csv),
        "random_seed": 20260813,
        "warning": "Screening surrogate only; strict SOC band/IRVSP validation remains required.",
    }
    for row, probability in zip(audit_rows, probabilities):
        row["oof_probability"] = round(float(probability), 8)

    model.fit(np.asarray(x), np.asarray(y))
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f"{task_name}_surrogate.joblib"
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "features": feature_names,
            "task": task_name,
            "particle": particle,
            "spacegroup_scope": spacegroup_number,
            "metrics": metrics,
        },
        bundle_path,
    )
    (output_dir / f"{task_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    for filename, rows in (
        (f"{task_name}_oof_predictions.csv", audit_rows),
        (f"{task_name}_compatibility_conflicts.csv", conflict_rows),
        (f"{task_name}_training_exclusions.csv", exclusion_rows),
    ):
        path = output_dir / filename
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    return metrics


def predict_surrogates(catalog_jsonl: Path, model_dir: Path, output_csv: Path) -> Path:
    catalog = [json.loads(line) for line in catalog_jsonl.read_text(encoding="utf-8").splitlines()]
    bundles = {}
    for task in ("stability", "topology"):
        path = model_dir / f"{task}_surrogate.joblib"
        if task == "stability" and not path.is_file():
            path = model_dir / "stability_proxy_surrogate.joblib"
        if path.is_file():
            bundles[task] = joblib.load(path)
    if not bundles:
        raise FileNotFoundError(f"No surrogate models found in {model_dir}")
    rows = []
    feature_rows: list[list[float]] = []
    feature_result_indices: list[int] = []
    for row in catalog:
        if not row.get("valid_structure"):
            continue
        result = {"candidate_id": row["candidate_id"], "prediction_error": ""}
        try:
            features, _ = structure_features(row)
            feature_rows.append(features)
            feature_result_indices.append(len(rows))
        except Exception as exc:
            result["prediction_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(result)
    matrix = np.asarray(feature_rows, dtype=float)
    for task, bundle in bundles.items():
        model = bundle["model"]
        probabilities = model.predict_proba(matrix)[:, 1]
        tree_probabilities = np.asarray(
            [tree.predict_proba(matrix)[:, 1] for tree in model.estimators_], dtype=float
        )
        uncertainties = np.std(tree_probabilities, axis=0)
        for index, probability, uncertainty in zip(
            feature_result_indices, probabilities, uncertainties
        ):
            rows[index][f"{task}_probability"] = round(float(probability), 8)
            rows[index][f"{task}_uncertainty"] = round(float(uncertainty), 8)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: key != "candidate_id")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
