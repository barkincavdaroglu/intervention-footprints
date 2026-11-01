"""Grouped neuron-intervention faithfulness experiments and matched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    LoadedRun,
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.data import GraphData
from gnn_repair.localization import (
    ATTRIBUTION_METHODS,
    LOCALIZATION_PROTOCOL_CODE_FILES,
    prediction_profile,
)
from gnn_repair.protocol import (
    canonical_payload_sha256,
    evaluation_splits,
    split_masks,
    study_record,
    test_status,
)
from gnn_repair.reproducibility import set_deterministic


GROUPED_METHODS = (*ATTRIBUTION_METHODS, "mean_ablation_bottom")

GROUPED_PROTOCOL_CODE_FILES = (
    "interventions.py",
    "localization.py",
    "model.py",
    "data.py",
    "metrics.py",
    "checkpoints.py",
    "protocol.py",
    "reproducibility.py",
)


EFFECT_METRICS = (
    "target_precision",
    "target_recall",
    "target_f1",
    "non_target_macro_f1",
    "accuracy",
    "macro_f1",
    "target_loss",
)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "population_std": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def compact_class_metrics(
    profile: dict[str, Any], target_class: int
) -> dict[str, float]:
    target = profile["per_class"][str(target_class)]
    non_target_f1 = [
        float(metrics["f1"])
        for class_id, metrics in profile["per_class"].items()
        if int(class_id) != target_class
    ]
    return {
        "target_precision": float(target["precision"]),
        "target_recall": float(target["recall"]),
        "target_f1": float(target["f1"]),
        "non_target_macro_f1": mean(non_target_f1),
        "accuracy": float(profile["accuracy"]),
        "macro_f1": float(profile["macro_f1"]),
    }


def _target_loss(
    logits: torch.Tensor,
    data: GraphData,
    mask: torch.Tensor,
    target_class: int,
) -> float:
    target_mask = mask & (data.y == target_class)
    if not bool(target_mask.any()):
        raise ValueError(f"Split has no examples of class {target_class}")
    return float(F.cross_entropy(logits[target_mask], data.y[target_mask]).item())


def intervention_effects(
    base_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    data: GraphData,
    target_classes: list[int],
    *,
    base_profiles: dict[str, dict[str, Any]] | None = None,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[int, dict[str, Any]]:
    """Measure class benefit/damage on explicitly authorized held-out splits."""

    effects = {target_class: {} for target_class in target_classes}
    base_predictions = base_logits.argmax(dim=-1)
    modified_predictions = modified_logits.argmax(dim=-1)
    for split_name, mask in split_masks(data, selected_splits).items():
        base_profile = (
            base_profiles[split_name]
            if base_profiles is not None
            else prediction_profile(base_logits, data.y, mask)
        )
        modified_profile = prediction_profile(modified_logits, data.y, mask)
        truth = data.y[mask]
        base_split_predictions = base_predictions[mask]
        modified_split_predictions = modified_predictions[mask]
        base_correct = base_split_predictions == truth
        modified_correct = modified_split_predictions == truth
        all_repaired = int((~base_correct & modified_correct).sum().item())
        all_broken = int((base_correct & ~modified_correct).sum().item())

        for target_class in target_classes:
            base_metrics = compact_class_metrics(base_profile, target_class)
            modified_metrics = compact_class_metrics(modified_profile, target_class)
            base_metrics["target_loss"] = _target_loss(
                base_logits, data, mask, target_class
            )
            modified_metrics["target_loss"] = _target_loss(
                modified_logits, data, mask, target_class
            )
            delta = {
                metric: modified_metrics[metric] - base_metrics[metric]
                for metric in EFFECT_METRICS
            }
            target_truth = truth == target_class
            target_repaired = int(
                (target_truth & ~base_correct & modified_correct).sum().item()
            )
            target_broken = int(
                (target_truth & base_correct & ~modified_correct).sum().item()
            )
            base_per_class = base_profile["per_class"]
            modified_per_class = modified_profile["per_class"]
            harmed_non_target_classes = sum(
                float(modified_per_class[class_id]["f1"])
                < float(metrics["f1"]) - 1e-12
                for class_id, metrics in base_per_class.items()
                if int(class_id) != target_class
            )
            non_target_truth = ~target_truth
            introduced_target_false_positives = int(
                (
                    non_target_truth
                    & (base_split_predictions != target_class)
                    & (modified_split_predictions == target_class)
                ).sum().item()
            )
            removed_target_false_positives = int(
                (
                    non_target_truth
                    & (base_split_predictions == target_class)
                    & (modified_split_predictions != target_class)
                ).sum().item()
            )
            effects[target_class][split_name] = {
                "base": base_metrics,
                "modified": modified_metrics,
                "delta": delta,
                "transitions": {
                    "repaired_errors": all_repaired,
                    "newly_broken_predictions": all_broken,
                    "net_corrected_predictions": all_repaired - all_broken,
                    "target_repaired_errors": target_repaired,
                    "target_newly_broken_predictions": target_broken,
                    "target_net_corrected_predictions": target_repaired
                    - target_broken,
                    "introduced_target_false_positives": (
                        introduced_target_false_positives
                    ),
                    "removed_target_false_positives": removed_target_false_positives,
                    "non_target_classes_harmed": harmed_non_target_classes,
                },
            }
    return effects


def _random_ranking(width: int, experiment: str, seed: int, layer: int, repeat: int) -> list[int]:
    material = f"{experiment}:{seed}:{layer}:{repeat}".encode("utf-8")
    random_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(random_seed).permutation(width).tolist()


@torch.no_grad()
def evaluate_grouped_run(
    loaded: LoadedRun,
    localization_run: dict[str, Any],
    *,
    target_classes: list[int],
    budget_fractions: list[float],
    random_repeats: int,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    """Evaluate top-k and matched random mean-replacement interventions."""

    set_deterministic(
        int(loaded.result["seed"]),
        num_threads=int(
            loaded.result["config"]["training"].get("num_threads", 1)
        ),
    )

    if int(localization_run["seed"]) != int(loaded.result["seed"]):
        raise RuntimeError("Localization run seed does not match loaded checkpoint")
    if localization_run["experiment"] != loaded.result["experiment"]:
        raise RuntimeError("Localization experiment does not match loaded checkpoint")
    if localization_run["checkpoint"] != loaded.result["checkpoint"]:
        raise RuntimeError("Localization run checkpoint does not match loaded checkpoint")
    if localization_run.get("checkpoint_model_sha256") != model_state_sha256(
        loaded.checkpoint
    ):
        raise RuntimeError("Localization model hash does not match loaded checkpoint")
    if localization_run.get("dataset_sha256") != loaded.data.metadata["sha256"]:
        raise RuntimeError("Localization dataset hash does not match loaded checkpoint")
    if localization_run.get("split_indices_sha256") != split_indices_sha256(
        loaded.data
    ):
        raise RuntimeError("Localization split hash does not match loaded checkpoint")
    if localization_run.get("localization_code_sha256") != source_code_sha256(
        LOCALIZATION_PROTOCOL_CODE_FILES
    ):
        raise RuntimeError("Localization implementation hash is stale")
    model, data = loaded.model, loaded.data
    base_logits, activations = model(
        data.x,
        data.adjacency,
        return_activations=True,
    )
    training_means = [
        activation[data.train_mask].mean(dim=0) for activation in activations
    ]
    base_profiles = {
        split_name: prediction_profile(base_logits, data.y, mask)
        for split_name, mask in split_masks(data, selected_splits).items()
    }
    logits_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    def modified_logits(layer: int, neurons: list[int]) -> torch.Tensor:
        key = (layer, tuple(sorted(neurons)))
        if key in logits_cache:
            return logits_cache[key]
        neuron_index = torch.tensor(neurons, dtype=torch.long, device=data.x.device)

        def replace_group(
            active_layer: int,
            activation: torch.Tensor,
        ) -> torch.Tensor:
            if active_layer != layer:
                return activation
            modified = activation.clone()
            modified[:, neuron_index] = training_means[layer][neuron_index]
            return modified

        output = model(
            data.x,
            data.adjacency,
            activation_intervention=replace_group,
        )
        logits_cache[key] = output
        return output

    records: list[dict[str, Any]] = []
    for target_class in target_classes:
        class_record = localization_run["classes"][str(target_class)]
        for layer_record in class_record["layers"]:
            layer = int(layer_record["layer"])
            width = int(layer_record["width"])
            for method in GROUPED_METHODS:
                source_method = (
                    "mean_ablation" if method == "mean_ablation_bottom" else method
                )
                source_record = layer_record["methods"][source_method]
                if method == "mean_ablation_bottom":
                    scores = [float(value) for value in source_record["scores"]]
                    ranking = sorted(
                        range(width), key=lambda index: (scores[index], index)
                    )
                else:
                    ranking = [int(index) for index in source_record["ranking"]]
                for budget_fraction in budget_fractions:
                    budget = min(max(1, round(width * budget_fraction)), width)
                    neurons = ranking[:budget]
                    effects = intervention_effects(
                        base_logits,
                        modified_logits(layer, neurons),
                        data,
                        [target_class],
                        base_profiles=base_profiles,
                        selected_splits=selected_splits,
                    )[target_class]
                    records.append(
                        {
                            "target_class": target_class,
                            "layer": layer,
                            "width": width,
                            "method": method,
                            "budget": budget,
                            "budget_fraction": budget_fraction,
                            "repeat": None,
                            "neurons": neurons,
                            "effects": effects,
                        }
                    )

    for layer, layer_means in enumerate(training_means):
        width = int(layer_means.numel())
        for repeat in range(random_repeats):
            ranking = _random_ranking(
                width,
                str(loaded.result["experiment"]),
                int(loaded.result["seed"]),
                layer,
                repeat,
            )
            for budget_fraction in budget_fractions:
                budget = min(max(1, round(width * budget_fraction)), width)
                neurons = ranking[:budget]
                all_effects = intervention_effects(
                    base_logits,
                    modified_logits(layer, neurons),
                    data,
                    target_classes,
                    base_profiles=base_profiles,
                    selected_splits=selected_splits,
                )
                for target_class in target_classes:
                    records.append(
                        {
                            "target_class": target_class,
                            "layer": layer,
                            "width": width,
                            "method": "random",
                            "budget": budget,
                            "budget_fraction": budget_fraction,
                            "repeat": repeat,
                            "neurons": neurons,
                            "effects": all_effects[target_class],
                        }
                    )

    return {
        "experiment": loaded.result["experiment"],
        "seed": int(loaded.result["seed"]),
        "checkpoint": loaded.result["checkpoint"],
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": loaded.data.metadata["sha256"],
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "grouped_code_sha256": source_code_sha256(GROUPED_PROTOCOL_CODE_FILES),
        "localization_classes_sha256": hashlib.sha256(
            json.dumps(
                localization_run["classes"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "target_classes": target_classes,
        "budget_fractions": budget_fractions,
        "random_repeats": random_repeats,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "test_status": test_status(selected_splits),
        "unique_interventions": len(logits_cache),
        "records": records,
    }


def _cache_identity(
    loaded: LoadedRun,
    localization_run: dict[str, Any],
    target_classes: list[int],
    budget_fractions: list[float],
    random_repeats: int,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    localization_payload = json.dumps(
        localization_run["classes"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "version": 3,
        "experiment": loaded.result["experiment"],
        "seed": int(loaded.result["seed"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": loaded.data.metadata["sha256"],
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "grouped_code_sha256": source_code_sha256(GROUPED_PROTOCOL_CODE_FILES),
        "localization_classes_sha256": hashlib.sha256(localization_payload).hexdigest(),
        "target_classes": target_classes,
        "budget_fractions": budget_fractions,
        "random_repeats": random_repeats,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "study": study_record(loaded.result["config"]),
        "methods": [*GROUPED_METHODS, "random"],
    }


def _evaluate_with_cache(
    loaded: LoadedRun,
    localization_run: dict[str, Any],
    *,
    target_classes: list[int],
    budget_fractions: list[float],
    random_repeats: int,
    cache_path: Path,
    resume: bool,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    identity = _cache_identity(
        loaded,
        localization_run,
        target_classes,
        budget_fractions,
        random_repeats,
        selected_splits,
    )

    def with_provenance(run: dict[str, Any]) -> dict[str, Any]:
        return {
            **run,
            "checkpoint_model_sha256": identity["checkpoint_model_sha256"],
            "dataset_sha256": identity["dataset_sha256"],
            "split_indices_sha256": identity["split_indices_sha256"],
            "grouped_code_sha256": identity["grouped_code_sha256"],
            "localization_classes_sha256": identity[
                "localization_classes_sha256"
            ],
        }

    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(
                f"Grouped-intervention cache {cache_path} differs from the active protocol"
            )
        return with_provenance(cached["run"])

    run = evaluate_grouped_run(
        loaded,
        localization_run,
        target_classes=target_classes,
        budget_fractions=budget_fractions,
        random_repeats=random_repeats,
        selected_splits=selected_splits,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {"identity": identity, "run": run},
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(cache_path)
    return with_provenance(run)


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_run: dict[
        tuple[int, int, str, int, str, str],
        dict[int, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        seed = int(run["seed"])
        for record in run["records"]:
            base_key = (
                int(record["target_class"]),
                int(record["layer"]),
                str(record["method"]),
                int(record["budget"]),
            )
            for split_name, split_record in record["effects"].items():
                for metric, value in split_record["delta"].items():
                    per_run[(*base_key, split_name, f"delta_{metric}")][seed].append(
                        float(value)
                    )
                for metric, value in split_record["transitions"].items():
                    per_run[(*base_key, split_name, metric)][seed].append(float(value))

    summary: dict[str, Any] = {}
    for (target, layer, method, budget, split, metric), values_by_seed in per_run.items():
        seed_values = [
            mean(repeat_values)
            for _, repeat_values in sorted(values_by_seed.items())
        ]
        summary.setdefault(str(target), {}).setdefault(str(layer), {}).setdefault(
            method, {}
        ).setdefault(str(budget), {}).setdefault(split, {})[metric] = _summary(
            seed_values
        )
    return summary


def _run_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No run records found under {root / 'runs'}")

    def seed(path: Path) -> int:
        with path.open("r", encoding="utf-8") as stream:
            return int(json.load(stream)["seed"])

    return sorted(paths, key=seed)


def _index_unique_runs(
    runs: list[dict[str, Any]], *, source: str
) -> dict[int, dict[str, Any]]:
    indexed = {}
    for run in runs:
        seed = int(run["seed"])
        if seed in indexed:
            raise RuntimeError(f"Duplicate seed {seed} in {source}")
        indexed[seed] = run
    return indexed


def build_grouped_report(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    localization_path = Path(config["localization_report"])
    with localization_path.open("r", encoding="utf-8") as stream:
        localization = json.load(stream)
    budget_fractions = [
        float(value)
        for value in config.get("budget_fractions", [0.03125, 0.0625, 0.125])
    ]
    random_repeats = int(config.get("random_repeats", 3))
    target_mode = str(config.get("target_mode", "backbone_specific"))
    study = study_record(config)
    selected_splits = tuple(study["evaluation_splits"])
    localization_study = localization.get("protocol", {}).get("study")
    if study["phase"] == "confirmatory_blind" and localization_study != study:
        raise RuntimeError("Localization report does not match the confirmatory study")
    if not budget_fractions or any(not 0 < value <= 1 for value in budget_fractions):
        raise ValueError("Intervention budget fractions must be in (0, 1]")
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    cache_root = Path(config.get("cache_root", "artifacts/grouped_intervention_runs"))

    report: dict[str, Any] = {
        "protocol": {
            "study": study,
            "config_sha256": canonical_payload_sha256(config),
            "selection_split": "validation rankings from the localization report",
            "evaluation_splits": list(selected_splits),
            "test_status": test_status(selected_splits),
            "intervention": "replace selected neurons with training-node activation means",
            "budget_fractions": budget_fractions,
            "random_repeats": random_repeats,
            "target_mode": target_mode,
            "random_control": (
                "same deterministic random ranking is reused across target classes "
                "within a checkpoint, layer, and repeat"
            ),
            "delta_sign": "modified minus base; negative F1 is damage, positive loss is damage",
            "grouped_code_sha256": source_code_sha256(
                GROUPED_PROTOCOL_CODE_FILES
            ),
        },
        "datasets": {},
    }
    for dataset_name, dataset_config in config["datasets"].items():
        localized_dataset = localization["datasets"][dataset_name]
        dataset_report = {
            "shared_target_classes": localized_dataset["target_classes"],
            "backbones": {},
        }
        for backbone_name, root_value in dataset_config["backbones"].items():
            if target_mode == "backbone_specific":
                target_classes = [
                    int(value)
                    for value in localized_dataset["backbone_specific_target_classes"][
                        backbone_name
                    ]
                ]
            elif target_mode == "shared":
                target_classes = [
                    int(value) for value in localized_dataset["target_classes"]
                ]
            else:
                raise ValueError("target_mode must be 'backbone_specific' or 'shared'")
            root = Path(root_value)
            localized_runs = _index_unique_runs(
                localized_dataset["backbones"][backbone_name]["runs"],
                source=f"{dataset_name}/{backbone_name} localization report",
            )
            runs = []
            paths = _run_paths(root)
            path_seeds = []
            for path in paths:
                with path.open("r", encoding="utf-8") as stream:
                    path_seeds.append(int(json.load(stream)["seed"]))
            if len(path_seeds) != len(set(path_seeds)):
                raise RuntimeError(f"Duplicate checkpoint seeds under {root}")
            if set(path_seeds) != set(localized_runs):
                raise RuntimeError(
                    f"Checkpoint and localization seed sets differ for "
                    f"{dataset_name}/{backbone_name}"
                )
            for run_number, run_path in enumerate(paths, start=1):
                loaded = load_trained_run(run_path, allow_download=allow_download)
                if study_record(loaded.result["config"]) != study:
                    raise RuntimeError(
                        f"Study protocol mismatch for {dataset_name}/{backbone_name}"
                    )
                seed = int(loaded.result["seed"])
                if progress:
                    print(
                        f"[{dataset_name}/{backbone_name}] grouped run "
                        f"{run_number}/{len(paths)} seed {seed}",
                        flush=True,
                    )
                runs.append(
                    _evaluate_with_cache(
                        loaded,
                        localized_runs[seed],
                        target_classes=target_classes,
                        budget_fractions=budget_fractions,
                        random_repeats=random_repeats,
                        cache_path=(
                            cache_root
                            / dataset_name
                            / backbone_name
                            / f"{run_path.stem}.json"
                        ),
                        resume=resume,
                        selected_splits=selected_splits,
                    )
                )
            dataset_report["backbones"][backbone_name] = {
                "target_classes": target_classes,
                "runs": runs,
                "summary": _aggregate_runs(runs),
            }
        report["datasets"][dataset_name] = dataset_report
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grouped_interventions.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_grouped_report(
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
    print(f"Wrote grouped intervention report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
