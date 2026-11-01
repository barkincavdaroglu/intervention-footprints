"""Command-line training entry point for the deterministic reference baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from gnn_repair.data import GraphData, prepare_dataset
from gnn_repair.metrics import classification_metrics
from gnn_repair.model import build_model
from gnn_repair.protocol import (
    blind_test_labels,
    configured_num_classes,
    split_masks,
    study_record,
    test_status,
)
from gnn_repair.reproducibility import environment_metadata, set_deterministic


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("Device must be 'cpu' or 'cuda'; sparse MPS is not used by this baseline")


@torch.no_grad()
def _loss_on_mask(
    model: torch.nn.Module, data: GraphData, mask: torch.Tensor
) -> float:
    model.eval()
    logits = model(data.x, data.adjacency)
    return float(F.cross_entropy(logits[mask], data.y[mask]).item())


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    data: GraphData,
    selected_splits: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    logits = model(data.x, data.adjacency)
    metrics = {
        "train": classification_metrics(logits, data.y, data.train_mask),
    }
    metrics.update(
        {
            name: classification_metrics(logits, data.y, mask)
            for name, mask in split_masks(data, selected_splits).items()
        }
    )
    return metrics


def train_one_seed(
    config: dict[str, Any],
    seed: int,
    *,
    allow_download: bool = True,
) -> dict[str, Any]:
    training = config["training"]
    study = study_record(config)
    selected_splits = tuple(study["evaluation_splits"])
    set_deterministic(seed, num_threads=int(training.get("num_threads", 1)))
    device = _resolve_device(str(training.get("device", "cpu")))

    data_config = config["data"]
    data = prepare_dataset(
        data_config,
        seed=seed,
        allow_download=allow_download,
    )
    data = blind_test_labels(data, selected_splits).to(device)

    model_config = config["model"]
    num_classes = configured_num_classes(data_config, data)
    model = build_model(
        model_config,
        in_channels=data.x.shape[1],
        out_channels=num_classes,
    ).to(device)
    model_record = {
        "in_channels": int(data.x.shape[1]),
        "out_channels": num_classes,
        **copy.deepcopy(model_config),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    optimizer_name = str(training.get("optimizer", "adam")).lower()
    optimizer_types = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    if optimizer_name not in optimizer_types:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'")
    optimizer = optimizer_types[optimizer_name](
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    min_delta = float(training.get("min_delta", 0.0))
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.adjacency)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        val_loss = _loss_on_mask(model, data, data.val_mask)
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss.item()), "val_loss": val_loss}
        )
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = _evaluate(model, data, selected_splits)

    output_root = Path(config["output"]["root"])
    checkpoint_dir = output_root / "checkpoints"
    run_dir = output_root / "runs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config["experiment"]["name"])
    checkpoint_path = checkpoint_dir / f"{experiment_name}_seed_{seed}.pt"
    run_path = run_dir / f"{experiment_name}_seed_{seed}.json"

    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "config": copy.deepcopy(config),
        "model": model_record,
        "seed": seed,
        "best_epoch": best_epoch,
        "split_indices": {
            "train": data.train_mask.nonzero(as_tuple=False).flatten().cpu(),
            "validation": data.val_mask.nonzero(as_tuple=False).flatten().cpu(),
            "test": data.test_mask.nonzero(as_tuple=False).flatten().cpu(),
        },
        "data_metadata": data.metadata,
    }
    torch.save(checkpoint, checkpoint_path)

    result = {
        "experiment": experiment_name,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_loss": best_val_loss,
        "metrics": metrics,
        "evaluation_splits": list(selected_splits),
        "study": study,
        "test_status": test_status(selected_splits),
        "data": data.metadata,
        "config": copy.deepcopy(config),
        "environment": environment_metadata(),
        "model": model_record,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    with run_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def summarize_runs(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_splits = tuple(results[0].get("evaluation_splits", ("validation", "test")))
    if any(
        tuple(result.get("evaluation_splits", ("validation", "test")))
        != selected_splits
        for result in results
    ):
        raise ValueError("Cannot summarize runs with different evaluation_splits")
    summary: dict[str, Any] = {
        "experiment": results[0]["experiment"],
        "seeds": [result["seed"] for result in results],
        "runs": len(results),
        "evaluation_splits": list(selected_splits),
        "test_status": test_status(selected_splits),
    }
    for split_name in selected_splits:
        split_metrics = [result["metrics"][split_name] for result in results]
        summary[split_name] = {}
        for metric in ("accuracy", "macro_f1", "micro_f1"):
            values = [float(item[metric]) for item in split_metrics]
            summary[split_name][metric] = {
                "mean": mean(values),
                "population_std": pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def load_completed_run(config: dict[str, Any], seed: int) -> dict[str, Any] | None:
    """Load a completed run only when its identity, config, and checkpoint match."""

    experiment_name = str(config["experiment"]["name"])
    run_path = (
        Path(config["output"]["root"])
        / "runs"
        / f"{experiment_name}_seed_{seed}.json"
    )
    if not run_path.exists():
        return None
    with run_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("experiment") != experiment_name or result.get("seed") != seed:
        raise RuntimeError(f"Run identity does not match {run_path}")
    if result.get("config") != config:
        raise RuntimeError(
            f"Run config at {run_path} differs from the active config; rerun without --resume"
        )
    checkpoint_path = Path(str(result.get("checkpoint", "")))
    if not checkpoint_path.exists():
        raise RuntimeError(f"Completed run is missing its checkpoint: {checkpoint_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/citeseer_gcn.yaml"),
        help="YAML experiment configuration",
    )
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--no-download", action="store_true", help="Fail if the dataset is not already present"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed runs only when their saved config and checkpoint match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.output_root is not None:
        config["output"]["root"] = str(args.output_root)
    if args.device is not None:
        config["training"]["device"] = args.device

    seeds = args.seeds or [int(seed) for seed in config["experiment"]["seeds"]]
    results = []
    selected_splits = tuple(study_record(config)["evaluation_splits"])
    display_split = "test" if "test" in selected_splits else "validation"
    for run_number, seed in enumerate(seeds, start=1):
        completed = load_completed_run(config, seed) if args.resume else None
        if completed is not None:
            results.append(completed)
            displayed = completed["metrics"][display_split]
            print(
                f"[{run_number}/{len(seeds)}] seed {seed} resumed: "
                f"{display_split}_accuracy={displayed['accuracy']:.6f}, "
                f"{display_split}_macro_f1={displayed['macro_f1']:.6f}",
                flush=True,
            )
            continue
        print(f"[{run_number}/{len(seeds)}] training seed {seed}", flush=True)
        result = train_one_seed(config, seed, allow_download=not args.no_download)
        results.append(result)
        displayed = result["metrics"][display_split]
        print(
            f"[{run_number}/{len(seeds)}] seed {seed} complete: "
            f"{display_split}_accuracy={displayed['accuracy']:.6f}, "
            f"{display_split}_macro_f1={displayed['macro_f1']:.6f}, "
            f"best_epoch={result['best_epoch']}",
            flush=True,
        )
    summary = summarize_runs(results)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
