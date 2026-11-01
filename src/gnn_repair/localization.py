"""Leakage-safe class profiling and class-specific neuron-localization baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    LoadedRun,
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.data import GraphData
from gnn_repair.metrics import classification_metrics
from gnn_repair.protocol import (
    canonical_payload_sha256,
    evaluation_splits,
    split_masks,
    study_record,
    test_status,
)
from gnn_repair.reproducibility import set_deterministic


ATTRIBUTION_METHODS = (
    "gradient_x_activation",
    "absolute_gradient_x_activation",
    "centered_gradient_x_activation",
    "absolute_centered_gradient_x_activation",
    "gradient_magnitude",
    "activation_magnitude",
    "activation_distance",
    "mean_ablation",
)

LOCALIZATION_PROTOCOL_CODE_FILES = (
    "localization.py",
    "model.py",
    "data.py",
    "metrics.py",
    "checkpoints.py",
    "protocol.py",
    "reproducibility.py",
)


def prediction_profile(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Return overall and error-count metrics for every class on one split."""

    profile = classification_metrics(logits, y, mask)
    truth = y[mask]
    predictions = logits[mask].argmax(dim=-1)
    for class_id, class_metrics in profile["per_class"].items():
        label = int(class_id)
        true_class = truth == label
        predicted_class = predictions == label
        true_positives = int((true_class & predicted_class).sum().item())
        false_positives = int((~true_class & predicted_class).sum().item())
        false_negatives = int((true_class & ~predicted_class).sum().item())
        support = int(true_class.sum().item())
        class_metrics.update(
            {
                "predicted": int(predicted_class.sum().item()),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "error_rate": false_negatives / support if support else 0.0,
            }
        )
    return profile


def _target_loss(
    logits: torch.Tensor,
    data: GraphData,
    target_class: int,
) -> torch.Tensor:
    target_mask = data.val_mask & (data.y == target_class)
    if not bool(target_mask.any()):
        raise ValueError(f"Validation split has no examples of class {target_class}")
    return F.cross_entropy(logits[target_mask], data.y[target_mask])


def gradient_attribution_scores(
    loaded: LoadedRun,
    target_class: int,
) -> list[dict[str, list[float]]]:
    """Compute class-loss attribution scores on validation data only.

    Gradient x activation follows the thesis's first-order Taylor form: take the
    absolute value after summing gradient-activation products over graph nodes.
    """

    model, data = loaded.model, loaded.data
    model.zero_grad(set_to_none=True)
    logits, activations = model(
        data.x,
        data.adjacency,
        return_activations=True,
    )
    for activation in activations:
        activation.retain_grad()
    loss = _target_loss(logits, data, target_class)
    loss.backward()

    target_mask = data.val_mask & (data.y == target_class)
    scores: list[dict[str, list[float]]] = []
    for activation in activations:
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("Activation gradients were not retained")
        training_mean = activation[data.train_mask].mean(dim=0, keepdim=True)
        centered_activation = activation - training_mean
        scores.append(
            {
                "gradient_x_activation": (
                    (gradient * activation).sum(dim=0).abs().detach().cpu().tolist()
                ),
                "absolute_gradient_x_activation": (
                    (gradient * activation).abs().sum(dim=0).detach().cpu().tolist()
                ),
                "centered_gradient_x_activation": (
                    (gradient * centered_activation)
                    .sum(dim=0)
                    .abs()
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "absolute_centered_gradient_x_activation": (
                    (gradient * centered_activation)
                    .abs()
                    .sum(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "gradient_magnitude": (
                    gradient.abs().sum(dim=0).detach().cpu().tolist()
                ),
                "activation_magnitude": (
                    activation[target_mask].abs().mean(dim=0).detach().cpu().tolist()
                ),
                "activation_distance": (
                    centered_activation[target_mask]
                    .abs()
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                ),
            }
        )
    model.zero_grad(set_to_none=True)
    return scores


@torch.no_grad()
def mean_ablation_scores(
    loaded: LoadedRun,
    target_classes: list[int],
) -> list[dict[int, list[float]]]:
    """Measure target validation-loss change after replacing one neuron by its mean."""

    model, data = loaded.model, loaded.data
    base_logits, activations = model(
        data.x,
        data.adjacency,
        return_activations=True,
    )
    training_means = [
        activation[data.train_mask].mean(dim=0) for activation in activations
    ]
    base_losses = {
        target_class: float(_target_loss(base_logits, data, target_class).item())
        for target_class in target_classes
    }
    scores: list[dict[int, list[float]]] = []
    for layer_index, layer_means in enumerate(training_means):
        layer_scores = {target_class: [] for target_class in target_classes}
        for neuron_index in range(layer_means.numel()):

            def replace_neuron(
                active_layer: int,
                activation: torch.Tensor,
                *,
                expected_layer: int = layer_index,
                expected_neuron: int = neuron_index,
            ) -> torch.Tensor:
                if active_layer != expected_layer:
                    return activation
                modified = activation.clone()
                modified[:, expected_neuron] = layer_means[expected_neuron]
                return modified

            ablated_logits = model(
                data.x,
                data.adjacency,
                activation_intervention=replace_neuron,
            )
            for target_class in target_classes:
                ablated_loss = float(
                    _target_loss(ablated_logits, data, target_class).item()
                )
                layer_scores[target_class].append(
                    ablated_loss - base_losses[target_class]
                )
        scores.append(layer_scores)
    return scores


def _ranking(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _agreement(
    candidate_scores: list[float],
    ablation_scores: list[float],
    *,
    top_k: int,
) -> dict[str, float | None]:
    count = min(top_k, len(candidate_scores))
    candidate_ranking = _ranking(candidate_scores)
    ablation_ranking = _ranking(ablation_scores)
    candidate_top = set(candidate_ranking[:count])
    ablation_top = set(ablation_ranking[:count])
    candidate_array = np.asarray(candidate_scores, dtype=np.float64)
    ablation_array = np.asarray(ablation_scores, dtype=np.float64)
    if np.ptp(candidate_array) == 0 or np.ptp(ablation_array) == 0:
        correlation: float | None = None
    else:
        correlation = float(spearmanr(candidate_array, ablation_array).statistic)
    return {
        "spearman_rho_vs_mean_ablation": correlation,
        "top_k": count,
        "top_k_overlap_fraction": len(candidate_top & ablation_top) / count,
        "selected_mean_ablation_loss_change": float(
            ablation_array[candidate_ranking[:count]].mean()
        ),
        "optimal_mean_ablation_loss_change": float(
            ablation_array[ablation_ranking[:count]].mean()
        ),
    }


def localize_loaded_run(
    loaded: LoadedRun,
    target_classes: list[int],
    *,
    top_k: int,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    """Profile one checkpoint and rank its hidden neurons for selected classes."""

    set_deterministic(
        int(loaded.result["seed"]),
        num_threads=int(
            loaded.result["config"]["training"].get("num_threads", 1)
        ),
    )

    with torch.no_grad():
        logits = loaded.model(loaded.data.x, loaded.data.adjacency)
        profiles = {
            name: prediction_profile(logits, loaded.data.y, mask)
            for name, mask in split_masks(loaded.data, selected_splits).items()
        }
    gradient_scores = {
        target_class: gradient_attribution_scores(loaded, target_class)
        for target_class in target_classes
    }
    ablation_scores = mean_ablation_scores(loaded, target_classes)

    classes: dict[str, Any] = {}
    for target_class in target_classes:
        layers = []
        for layer_index, method_scores in enumerate(gradient_scores[target_class]):
            combined_scores = {
                **method_scores,
                "mean_ablation": ablation_scores[layer_index][target_class],
            }
            layers.append(
                {
                    "layer": layer_index,
                    "width": len(combined_scores["mean_ablation"]),
                    "methods": {
                        method: {
                            "scores": scores,
                            "ranking": _ranking(scores),
                            "agreement": _agreement(
                                scores,
                                combined_scores["mean_ablation"],
                                top_k=top_k,
                            ),
                        }
                        for method, scores in combined_scores.items()
                    },
                }
            )
        classes[str(target_class)] = {"layers": layers}

    return {
        "experiment": loaded.result["experiment"],
        "seed": int(loaded.result["seed"]),
        "checkpoint": loaded.result["checkpoint"],
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": loaded.data.metadata["sha256"],
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "localization_code_sha256": source_code_sha256(
            LOCALIZATION_PROTOCOL_CODE_FILES
        ),
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "test_status": test_status(selected_splits),
        "profiles": profiles,
        "classes": classes,
    }


def _run_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No run records found under {root / 'runs'}")

    def seed(path: Path) -> int:
        with path.open("r", encoding="utf-8") as stream:
            return int(json.load(stream)["seed"])

    return sorted(paths, key=seed)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "population_std": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _aggregate_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    split_names = tuple(
        split for split in ("validation", "test") if split in profiles[0]
    )
    if any(
        tuple(split for split in ("validation", "test") if split in profile)
        != split_names
        for profile in profiles
    ):
        raise ValueError("Profile evaluation splits are inconsistent")
    for split in split_names:
        split_profiles = [profile[split] for profile in profiles]
        aggregate[split] = {
            metric: _summary([float(profile[metric]) for profile in split_profiles])
            for metric in ("accuracy", "macro_f1", "micro_f1")
        }
        class_ids = split_profiles[0]["per_class"].keys()
        aggregate[split]["per_class"] = {
            class_id: {
                metric: _summary(
                    [
                        float(profile["per_class"][class_id][metric])
                        for profile in split_profiles
                    ]
                )
                for metric in (
                    "precision",
                    "recall",
                    "f1",
                    "support",
                    "predicted",
                    "false_positives",
                    "false_negatives",
                    "error_rate",
                )
            }
            for class_id in class_ids
        }
    return aggregate


def _select_targets(
    profiles_by_backbone: dict[str, list[dict[str, Any]]],
    target_count: int,
) -> tuple[list[int], dict[str, Any]]:
    all_profiles = [
        profile
        for backbone_profiles in profiles_by_backbone.values()
        for profile in backbone_profiles
    ]
    class_ids = all_profiles[0]["validation"]["per_class"].keys()
    scores = {
        int(class_id): _summary(
            [
                float(profile["validation"]["per_class"][class_id]["f1"])
                for profile in all_profiles
            ]
        )
        for class_id in class_ids
    }
    selected = sorted(scores, key=lambda class_id: (scores[class_id]["mean"], class_id))[
        :target_count
    ]
    return selected, {str(class_id): score for class_id, score in scores.items()}


def _aggregate_agreement(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for run in runs:
        for class_id, class_record in run["classes"].items():
            for layer in class_record["layers"]:
                for method, method_record in layer["methods"].items():
                    for metric, value in method_record["agreement"].items():
                        if metric == "top_k" or value is None:
                            continue
                        grouped[(class_id, int(layer["layer"]), method, metric)].append(
                            float(value)
                        )
    summary: dict[str, Any] = {}
    for (class_id, layer, method, metric), values in grouped.items():
        summary.setdefault(class_id, {}).setdefault(str(layer), {}).setdefault(
            method, {}
        )[metric] = _summary(values)
    return summary


def _localization_cache_identity(
    loaded: LoadedRun,
    target_classes: list[int],
    top_k: int,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    return {
        "version": 4,
        "experiment": loaded.result["experiment"],
        "seed": int(loaded.result["seed"]),
        "checkpoint": loaded.result["checkpoint"],
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": loaded.data.metadata["sha256"],
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "localization_code_sha256": source_code_sha256(
            LOCALIZATION_PROTOCOL_CODE_FILES
        ),
        "target_classes": target_classes,
        "agreement_top_k": top_k,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "study": study_record(loaded.result["config"]),
        "methods": list(ATTRIBUTION_METHODS),
    }


def _localize_with_cache(
    loaded: LoadedRun,
    target_classes: list[int],
    *,
    top_k: int,
    cache_path: Path,
    resume: bool,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    identity = _localization_cache_identity(
        loaded, target_classes, top_k, selected_splits
    )

    def with_provenance(run: dict[str, Any]) -> dict[str, Any]:
        return {
            **run,
            "checkpoint_model_sha256": identity["checkpoint_model_sha256"],
            "dataset_sha256": identity["dataset_sha256"],
            "split_indices_sha256": identity["split_indices_sha256"],
            "localization_code_sha256": identity[
                "localization_code_sha256"
            ],
        }

    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(
                f"Localization cache {cache_path} differs from the active protocol"
            )
        return with_provenance(cached["run"])

    localized = localize_loaded_run(
        loaded,
        target_classes,
        top_k=top_k,
        selected_splits=selected_splits,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {"identity": identity, "run": localized},
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(cache_path)
    return with_provenance(localized)


def build_localization_report(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Build profiles first, select target classes, then localize without test leakage."""

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Localization config requires a non-empty datasets mapping")
    target_count = int(config.get("target_classes_per_dataset", 3))
    top_k = int(config.get("agreement_top_k", 5))
    include_shared_targets = bool(config.get("include_shared_targets", True))
    cache_root = Path(config.get("cache_root", "artifacts/localization_runs"))
    study = study_record(config)
    selected_splits = tuple(study["evaluation_splits"])
    if target_count < 1 or top_k < 1:
        raise ValueError("Target-class count and agreement top-k must be positive")

    report: dict[str, Any] = {
        "protocol": {
            "study": study,
            "config_sha256": canonical_payload_sha256(config),
            "target_selection": (
                "Lowest mean validation class F1 across configured backbones and runs"
            ),
            "localization_split": "validation",
            "evaluation_splits": list(selected_splits),
            "test_status": test_status(selected_splits),
            "test_label_use": (
                "sealed and absent from every development profile"
                if "test" not in selected_splits
                else "descriptive profiles only; never used in neuron ranking"
            ),
            "ablation": "replace a hidden neuron with its training-node activation mean",
            "gradient_x_activation_aggregation": (
                "absolute sum of class-loss gradient times activation over graph nodes"
            ),
            "absolute_gradient_x_activation_aggregation": (
                "sum of absolute class-loss gradient times activation over graph nodes"
            ),
            "centered_reference": "training-node activation mean for each hidden channel",
            "include_shared_targets": include_shared_targets,
            "agreement_top_k": top_k,
            "localization_code_sha256": source_code_sha256(
                LOCALIZATION_PROTOCOL_CODE_FILES
            ),
        },
        "datasets": {},
    }
    for dataset_name, dataset_config in datasets.items():
        backbone_roots = {
            name: Path(root)
            for name, root in dataset_config.get("backbones", {}).items()
        }
        if not backbone_roots:
            raise ValueError(f"Dataset {dataset_name!r} has no configured backbones")
        profiles_by_backbone: dict[str, list[dict[str, Any]]] = {}
        for backbone_name, root in backbone_roots.items():
            profiles_by_backbone[backbone_name] = []
            for run_number, path in enumerate(_run_paths(root), start=1):
                if progress:
                    print(
                        f"[{dataset_name}/{backbone_name}] profiling run {run_number}",
                        flush=True,
                    )
                loaded = load_trained_run(path, allow_download=allow_download)
                if study_record(loaded.result["config"]) != study:
                    raise RuntimeError(
                        f"Study protocol mismatch for {dataset_name}/{backbone_name}"
                    )
                set_deterministic(
                    int(loaded.result["seed"]),
                    num_threads=int(
                        loaded.result["config"]["training"].get(
                            "num_threads", 1
                        )
                    ),
                )
                with torch.no_grad():
                    logits = loaded.model(loaded.data.x, loaded.data.adjacency)
                profiles_by_backbone[backbone_name].append(
                    {
                        "seed": int(loaded.result["seed"]),
                        **{
                            name: prediction_profile(logits, loaded.data.y, mask)
                            for name, mask in split_masks(
                                loaded.data, selected_splits
                            ).items()
                        },
                    }
                )

        targets, target_scores = _select_targets(profiles_by_backbone, target_count)
        backbone_specific_targets = {}
        analysis_targets = {}
        for backbone_name, backbone_profiles in profiles_by_backbone.items():
            primary = _select_targets(
                {backbone_name: backbone_profiles}, target_count
            )[0]
            backbone_specific_targets[backbone_name] = primary
            analysis_targets[backbone_name] = [
                *primary,
                *(
                    [target for target in targets if target not in primary]
                    if include_shared_targets
                    else []
                ),
            ]
        dataset_report: dict[str, Any] = {
            "target_classes": targets,
            "backbone_specific_target_classes": backbone_specific_targets,
            "analysis_target_classes": analysis_targets,
            "target_selection_validation_f1": target_scores,
            "backbones": {},
        }
        for backbone_name, root in backbone_roots.items():
            localized_runs = []
            active_targets = analysis_targets[backbone_name]
            paths = _run_paths(root)
            for run_number, path in enumerate(paths, start=1):
                if progress:
                    print(
                        f"[{dataset_name}/{backbone_name}] localizing "
                        f"run {run_number}/{len(paths)} for classes {active_targets}",
                        flush=True,
                    )
                loaded = load_trained_run(path, allow_download=allow_download)
                if study_record(loaded.result["config"]) != study:
                    raise RuntimeError(
                        f"Study protocol mismatch for {dataset_name}/{backbone_name}"
                    )
                localized_runs.append(
                    _localize_with_cache(
                        loaded,
                        active_targets,
                        top_k=top_k,
                        cache_path=(
                            cache_root
                            / dataset_name
                            / backbone_name
                            / f"{path.stem}.json"
                        ),
                        resume=resume,
                        selected_splits=selected_splits,
                    )
                )
            profiles = profiles_by_backbone[backbone_name]
            dataset_report["backbones"][backbone_name] = {
                "runs": localized_runs,
                "profile_summary": _aggregate_profiles(profiles),
                "agreement_summary": _aggregate_agreement(localized_runs),
            }
        report["datasets"][dataset_name] = dataset_report
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/localization_profiles.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Download a missing dataset instead of requiring the audited local archive",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-run localization only when its checkpoint and protocol match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_localization_report(
        config,
        allow_download=args.allow_download,
        progress=True,
        resume=args.resume,
    )
    output = args.output or Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote localization report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
