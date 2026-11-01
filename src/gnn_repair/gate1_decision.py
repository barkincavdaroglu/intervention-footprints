"""Frozen, runner-independent decision logic for topology Gate 1.

The runner is responsible for producing validation-only checkpoint alignments and
normalizing them into :class:`CheckpointEvidence` records.  This module performs
no model fitting and reads no artifacts.  It only applies the predeclared Gate 1
rules, returning a deterministic, JSON-serializable audit record.

The API deliberately separates a *candidate* (for example, constant-label WL at
depth two) from its views (q90/q95/q99, or individual diffusion scales).  This
prevents support quantiles and diffusion scales from being averaged before their
required sensitivity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable, Mapping


CONTROL_BLOCK_NAMES = (
    "magnitude",
    "raw_effect_field",
    "mask_jaccard",
    "uniform_node_role",
)
SUBSET_NAMES = ("active_channels", "all_channels")
NORMALIZED_WL = "normalized_wl"
DIFFUSION = "diffusion"
CANDIDATE_FAMILIES = (NORMALIZED_WL, DIFFUSION)
TIE_TOLERANCE = 0.01
DEFAULT_EXPECTED_CELLS = (
    "amazon_computers/thesis_gcn",
    "amazon_computers/residual_gcn",
    "roman_empire/thesis_gcn",
    "roman_empire/residual_gcn",
)


def _payload_float(value: Any) -> Any:
    """Preserve a JSON primitive so validation can reject invalid numeric types."""

    return value


def _payload_int(value: Any) -> Any:
    """Preserve integer evidence; floats and booleans must not be truncated."""

    return value


@dataclass(frozen=True)
class CandidateVariant:
    """One support-quantile or diffusion-scale view of a candidate."""

    name: str
    quantile: float | None = None
    diffusion_scale: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateVariant":
        return cls(
            name=payload.get("name"),
            quantile=(
                None
                if payload.get("quantile") is None
                else _payload_float(payload["quantile"])
            ),
            diffusion_scale=(
                None
                if payload.get("diffusion_scale") is None
                else _payload_float(payload["diffusion_scale"])
            ),
        )


@dataclass(frozen=True)
class CandidateSpec:
    """Frozen definition of one representation candidate.

    ``primary_variant`` defines the view used for the primary checkpoint median,
    cell rule, rewiring rule, and candidate-selection mean.  Quantile or scale
    variants remain separate required sensitivities.
    """

    candidate_id: str
    family: str
    primary_variant: str
    variants: tuple[CandidateVariant, ...]
    depth: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateSpec":
        raw_variants = payload.get("variants", ())
        if not isinstance(raw_variants, (list, tuple)):
            raw_variants = ()
        return cls(
            candidate_id=payload.get("candidate_id"),
            family=payload.get("family"),
            primary_variant=payload.get("primary_variant"),
            variants=tuple(
                value
                if isinstance(value, CandidateVariant)
                else CandidateVariant.from_mapping(value)
                for value in raw_variants
            ),
            depth=(
                None
                if payload.get("depth") is None
                else _payload_int(payload["depth"])
            ),
        )


@dataclass(frozen=True)
class RewireAlignment:
    """Alignment obtained from one degree-preserving graph rewire."""

    alignment: float | None
    requested_swaps: int
    completed_swaps: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewireAlignment":
        raw_alignment = payload.get("alignment", payload.get("rewired_alignment"))
        return cls(
            alignment=(
                None if raw_alignment is None else _payload_float(raw_alignment)
            ),
            requested_swaps=_payload_int(payload.get("requested_swaps")),
            completed_swaps=_payload_int(
                payload.get("completed_swaps", payload.get("successful_swaps"))
            ),
        )


@dataclass(frozen=True)
class CheckpointEvidence:
    """Normalized evidence for one candidate view in one checkpoint."""

    checkpoint_id: str
    cell: str
    candidate_id: str
    variant: str
    subset: str
    candidate_alignment: float | None
    control_alignments: Mapping[str, float | None]
    rewires: tuple[RewireAlignment, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckpointEvidence":
        raw_alignment = payload.get(
            "candidate_alignment",
            payload.get("alignment"),
        )
        controls = payload.get("control_alignments", payload.get("controls", {}))
        if not isinstance(controls, Mapping):
            controls = {}
        raw_rewires = payload.get("rewires", ())
        if not isinstance(raw_rewires, (list, tuple)):
            raw_rewires = ()
        return cls(
            checkpoint_id=payload.get("checkpoint_id"),
            cell=payload.get("cell"),
            candidate_id=payload.get("candidate_id"),
            variant=payload.get("variant"),
            subset=payload.get("subset"),
            candidate_alignment=(
                None if raw_alignment is None else _payload_float(raw_alignment)
            ),
            control_alignments=dict(controls),
            rewires=tuple(
                value
                if isinstance(value, RewireAlignment)
                else RewireAlignment.from_mapping(value)
                for value in raw_rewires
            ),
        )


@dataclass(frozen=True)
class LocoFlags:
    """Caller-supplied held-checkpoint prediction conclusions."""

    non_worse_than_best_control: bool | None
    improves_predeclared_metric: bool | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LocoFlags":
        return cls(
            non_worse_than_best_control=payload.get(
                "non_worse_than_best_control",
                payload.get("non_worse"),
            ),
            improves_predeclared_metric=payload.get(
                "improves_predeclared_metric",
                payload.get("improves_metric"),
            ),
        )


def _reason(code: str, message: str, **context: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if context:
        result["context"] = context
    return result


def _append_reason(
    reasons: list[dict[str, Any]],
    code: str,
    message: str,
    **context: Any,
) -> None:
    candidate = _reason(code, message, **context)
    if candidate not in reasons:
        reasons.append(candidate)


def _coerce_spec(value: CandidateSpec | Mapping[str, Any]) -> CandidateSpec:
    return value if isinstance(value, CandidateSpec) else CandidateSpec.from_mapping(value)


def _coerce_evidence(
    value: CheckpointEvidence | Mapping[str, Any],
) -> CheckpointEvidence:
    return (
        value
        if isinstance(value, CheckpointEvidence)
        else CheckpointEvidence.from_mapping(value)
    )


def _coerce_loco(value: LocoFlags | Mapping[str, Any]) -> LocoFlags:
    return value if isinstance(value, LocoFlags) else LocoFlags.from_mapping(value)


def _finite_alignment(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and -1.0 <= float(value) <= 1.0
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_spec(spec: CandidateSpec) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if not isinstance(spec.candidate_id, str) or not spec.candidate_id.strip():
        _append_reason(
            reasons,
            "invalid_candidate_id",
            "candidate_id must be a non-empty string",
        )
    if spec.family not in CANDIDATE_FAMILIES:
        _append_reason(
            reasons,
            "invalid_candidate_family",
            f"family must be one of {CANDIDATE_FAMILIES}",
            family=spec.family,
        )
    names = [variant.name for variant in spec.variants]
    valid_names = all(
        isinstance(name, str) and bool(name.strip()) for name in names
    )
    if not names or not valid_names or len(set(names)) != len(names):
        _append_reason(
            reasons,
            "invalid_candidate_variants",
            "variant names must be non-empty and unique",
        )
    if spec.primary_variant not in names:
        _append_reason(
            reasons,
            "missing_primary_variant_spec",
            "primary_variant is not present in variants",
            primary_variant=spec.primary_variant,
        )

    if spec.family == NORMALIZED_WL:
        if (
            isinstance(spec.depth, bool)
            or not isinstance(spec.depth, int)
            or not 0 <= spec.depth <= 4
        ):
            _append_reason(
                reasons,
                "invalid_wl_depth",
                "normalized WL depth must be an integer from zero through four",
                depth=spec.depth,
            )
        quantiles = []
        for variant in spec.variants:
            if (
                variant.quantile is None
                or not _finite_number(variant.quantile)
                or not 0 < variant.quantile < 1
                or variant.diffusion_scale is not None
            ):
                _append_reason(
                    reasons,
                    "invalid_wl_variant",
                    "every normalized WL variant must define exactly one quantile",
                    variant=variant.name,
                )
                continue
            quantiles.append(variant.quantile)
        if len(set(quantiles)) != len(quantiles):
            _append_reason(
                reasons,
                "duplicate_wl_quantile",
                "normalized WL quantiles must be unique",
            )
        for required in (0.90, 0.95, 0.99):
            if not any(math.isclose(value, required, abs_tol=1e-12) for value in quantiles):
                _append_reason(
                    reasons,
                    f"missing_q{int(required * 100)}_variant",
                    f"normalized WL requires a q{int(required * 100)} view",
                )

    if spec.family == DIFFUSION:
        if spec.depth is not None:
            _append_reason(
                reasons,
                "diffusion_depth_not_allowed",
                "diffusion candidates do not have a WL depth",
                depth=spec.depth,
            )
        scales = []
        for variant in spec.variants:
            if (
                variant.diffusion_scale is None
                or not _finite_number(variant.diffusion_scale)
                or variant.diffusion_scale <= 0
                or variant.quantile is not None
            ):
                _append_reason(
                    reasons,
                    "invalid_diffusion_variant",
                    "every diffusion variant must define exactly one positive scale",
                    variant=variant.name,
                )
                continue
            scales.append(variant.diffusion_scale)
        if len(scales) < 2:
            _append_reason(
                reasons,
                "insufficient_diffusion_scales",
                "diffusion requires at least two scales for the adjacency rule",
            )
        if len(set(scales)) != len(scales):
            _append_reason(
                reasons,
                "duplicate_diffusion_scale",
                "diffusion scales must be unique",
            )
    return reasons


def _view_key(record: CheckpointEvidence) -> tuple[str, str]:
    return record.variant, record.subset


def _checkpoint_key(record: CheckpointEvidence) -> tuple[str, str]:
    return record.cell, record.checkpoint_id


def _group_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(detail["incremental_alignment"]) for detail in details]
    by_cell: dict[str, list[float]] = {}
    for detail, value in zip(details, values, strict=True):
        by_cell.setdefault(str(detail["cell"]), []).append(value)
    cell_summaries = {
        cell: {
            "count": len(cell_values),
            "mean": float(mean(cell_values)),
            "median": float(median(cell_values)),
        }
        for cell, cell_values in sorted(by_cell.items())
    }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "positive_count": int(sum(value > 0 for value in values)),
        "positive_fraction": float(sum(value > 0 for value in values) / len(values)),
        "cells": cell_summaries,
        "checkpoints": details,
    }


def _rewiring_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(detail["real_minus_rewired"]) for detail in details]
    by_cell: dict[str, list[float]] = {}
    for detail, value in zip(details, values, strict=True):
        by_cell.setdefault(str(detail["cell"]), []).append(value)
    cells = {
        cell: {
            "count": len(cell_values),
            "mean": float(mean(cell_values)),
            "median": float(median(cell_values)),
        }
        for cell, cell_values in sorted(by_cell.items())
    }
    negative_cells = [
        cell for cell, summary in cells.items() if float(summary["median"]) < 0
    ]
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "cells": cells,
        "negative_cells": negative_cells,
        "checkpoints": details,
    }


def _evaluate_group(
    records: list[CheckpointEvidence],
    *,
    require_rewiring: bool,
    required_rewire_repetitions: int,
    reasons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    rewiring_details: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda value: (value.cell, value.checkpoint_id)):
        context = {
            "candidate_id": record.candidate_id,
            "variant": record.variant,
            "subset": record.subset,
            "cell": record.cell,
            "checkpoint_id": record.checkpoint_id,
        }
        if not _finite_alignment(record.candidate_alignment):
            _append_reason(
                reasons,
                "invalid_candidate_alignment",
                "candidate alignment must be finite and in [-1, 1]",
                **context,
            )
            continue
        missing_controls = [
            name
            for name in CONTROL_BLOCK_NAMES
            if not _finite_alignment(record.control_alignments.get(name))
        ]
        if missing_controls:
            _append_reason(
                reasons,
                "missing_or_invalid_control_alignment",
                "all four frozen control alignments must be finite and in [-1, 1]",
                controls=missing_controls,
                **context,
            )
            continue
        strongest_control = max(
            CONTROL_BLOCK_NAMES,
            key=lambda name: (
                float(record.control_alignments[name]),
                -CONTROL_BLOCK_NAMES.index(name),
            ),
        )
        strongest_value = float(record.control_alignments[strongest_control])
        observed = float(record.candidate_alignment)
        detail = {
            "cell": record.cell,
            "checkpoint_id": record.checkpoint_id,
            "candidate_alignment": observed,
            "strongest_control": strongest_control,
            "strongest_control_alignment": strongest_value,
            "incremental_alignment": float(observed - strongest_value),
        }
        details.append(detail)

        if not require_rewiring:
            continue
        if len(record.rewires) != required_rewire_repetitions:
            _append_reason(
                reasons,
                "rewire_count_mismatch",
                "active evidence requires the predeclared number of rewires",
                expected_rewires=required_rewire_repetitions,
                observed_rewires=len(record.rewires),
                **context,
            )
            count_is_valid = False
        else:
            count_is_valid = True
        rewired_values = []
        valid_rewires = count_is_valid
        for repetition, rewire in enumerate(record.rewires):
            if (
                isinstance(rewire.requested_swaps, bool)
                or isinstance(rewire.completed_swaps, bool)
                or not isinstance(rewire.requested_swaps, int)
                or not isinstance(rewire.completed_swaps, int)
                or rewire.requested_swaps <= 0
                or rewire.completed_swaps != rewire.requested_swaps
            ):
                _append_reason(
                    reasons,
                    "incomplete_rewire",
                    "failed, zero-swap, or incomplete rewires cannot enter the null",
                    repetition=repetition,
                    requested_swaps=rewire.requested_swaps,
                    completed_swaps=rewire.completed_swaps,
                    **context,
                )
                valid_rewires = False
            if not _finite_alignment(rewire.alignment):
                _append_reason(
                    reasons,
                    "invalid_rewired_alignment",
                    "rewired alignment must be finite and in [-1, 1]",
                    repetition=repetition,
                    **context,
                )
                valid_rewires = False
            else:
                rewired_values.append(float(rewire.alignment))
        if valid_rewires:
            rewired_mean = float(mean(rewired_values))
            rewire_detail = {
                **detail,
                "rewire_count": len(rewired_values),
                "rewired_alignment_mean": rewired_mean,
                "real_minus_rewired": float(observed - rewired_mean),
            }
            rewiring_details.append(rewire_detail)
    return details, rewiring_details


def _required_views(spec: CandidateSpec) -> tuple[tuple[str, str], ...]:
    views: list[tuple[str, str]] = [
        (spec.primary_variant, "active_channels"),
        (spec.primary_variant, "all_channels"),
    ]
    if spec.family == NORMALIZED_WL:
        sensitivity_names = [
            variant.name
            for variant in spec.variants
            if variant.quantile is not None
            and any(
                math.isclose(variant.quantile, required, abs_tol=1e-12)
                for required in (0.90, 0.95, 0.99)
            )
        ]
    else:
        sensitivity_names = [variant.name for variant in spec.variants]
    for name in sensitivity_names:
        views.extend(((name, "active_channels"), (name, "all_channels")))
    return tuple(dict.fromkeys(views))


def _evaluate_candidate(
    spec: CandidateSpec,
    records: list[CheckpointEvidence],
    loco: LocoFlags | None,
    expected_cells: tuple[str, ...],
    required_rewire_repetitions: int,
) -> tuple[dict[str, Any], frozenset[tuple[str, str]] | None]:
    reasons = _validate_spec(spec)
    known_variants = {variant.name for variant in spec.variants}
    for record in records:
        invalid_identity_fields = [
            name
            for name, value in (
                ("checkpoint_id", record.checkpoint_id),
                ("cell", record.cell),
                ("candidate_id", record.candidate_id),
                ("variant", record.variant),
            )
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_identity_fields:
            _append_reason(
                reasons,
                "invalid_record_identity",
                "checkpoint record identity fields must be non-empty strings",
                fields=invalid_identity_fields,
            )
        if record.subset not in SUBSET_NAMES:
            _append_reason(
                reasons,
                "invalid_subset",
                f"subset must be one of {SUBSET_NAMES}",
                subset=record.subset,
            )
        if record.variant not in known_variants:
            _append_reason(
                reasons,
                "undeclared_variant",
                "record references a variant absent from the candidate specification",
                variant=record.variant,
            )
    if not records:
        _append_reason(
            reasons,
            "missing_candidate_evidence",
            "candidate has no checkpoint evidence",
            candidate_id=spec.candidate_id,
        )

    grouped: dict[tuple[str, str], list[CheckpointEvidence]] = {}
    for record in records:
        grouped.setdefault(_view_key(record), []).append(record)
    primary_key = (spec.primary_variant, "active_channels")
    primary_records = grouped.get(primary_key, [])
    universe = frozenset(_checkpoint_key(record) for record in primary_records)
    primary_cells = {cell for cell, _ in universe}
    if primary_cells != set(expected_cells):
        _append_reason(
            reasons,
            "candidate_cell_set_mismatch",
            "primary active evidence must contain all and only the four frozen cells",
            missing_cells=sorted(set(expected_cells) - primary_cells, key=str),
            extra_cells=sorted(primary_cells - set(expected_cells), key=str),
        )

    required_views = _required_views(spec) if not _validate_spec(spec) else ()
    for variant, subset in required_views:
        view_records = grouped.get((variant, subset), [])
        view_universe = frozenset(_checkpoint_key(record) for record in view_records)
        if not view_records:
            _append_reason(
                reasons,
                "missing_required_view",
                "candidate is missing a required variant/subset view",
                variant=variant,
                subset=subset,
            )
        elif view_universe != universe:
            _append_reason(
                reasons,
                "checkpoint_set_mismatch",
                "every required view must use the identical checkpoint universe",
                variant=variant,
                subset=subset,
                missing=sorted(universe - view_universe, key=str),
                extra=sorted(view_universe - universe, key=str),
            )

    if loco is None:
        _append_reason(
            reasons,
            "missing_loco_flags",
            "candidate is missing caller-supplied LOCO flags",
        )
    elif (
        not isinstance(loco.non_worse_than_best_control, bool)
        or not isinstance(loco.improves_predeclared_metric, bool)
    ):
        _append_reason(
            reasons,
            "incomplete_loco_flags",
            "LOCO non-worsening and improvement flags must both be boolean",
        )

    variant_results: dict[str, dict[str, Any]] = {}
    rewiring_results: dict[str, dict[str, Any]] = {}
    if not reasons:
        for variant, subset in required_views:
            group_reasons: list[dict[str, Any]] = []
            details, rewiring_details = _evaluate_group(
                grouped[(variant, subset)],
                require_rewiring=subset == "active_channels",
                required_rewire_repetitions=required_rewire_repetitions,
                reasons=group_reasons,
            )
            reasons.extend(group_reasons)
            if details:
                variant_results.setdefault(variant, {})[subset] = _group_summary(details)
            if rewiring_details:
                rewiring_results[variant] = _rewiring_summary(rewiring_details)

    if reasons:
        return (
            {
                "candidate_id": spec.candidate_id,
                "family": spec.family,
                "depth": spec.depth,
                "primary_variant": spec.primary_variant,
                "status": "not_evaluable",
                "selection_score": None,
                "not_evaluable_reasons": reasons,
                "failed_criteria": [],
                "criteria": {},
                "variants": variant_results,
                "rewiring_by_variant": rewiring_results,
            },
            universe or None,
        )

    primary_active = variant_results[spec.primary_variant]["active_channels"]
    primary_all = variant_results[spec.primary_variant]["all_channels"]
    rewiring = rewiring_results[spec.primary_variant]
    positive_cells = [
        cell
        for cell, summary in primary_active["cells"].items()
        if float(summary["median"]) > 0
    ]

    if spec.family == NORMALIZED_WL:
        q_summaries: dict[str, Any] = {}
        q_passes = []
        for required in (0.95, 0.99):
            variant = next(
                value
                for value in spec.variants
                if value.quantile is not None
                and math.isclose(value.quantile, required, abs_tol=1e-12)
            )
            active = variant_results[variant.name]["active_channels"]
            all_channels = variant_results[variant.name]["all_channels"]
            quantile_rewiring = rewiring_results[variant.name]
            rewiring_pass = (
                quantile_rewiring["median"] > 0
                and len(quantile_rewiring["negative_cells"]) <= 1
            )
            passed = (
                active["median"] > 0
                and all_channels["median"] > 0
                and rewiring_pass
            )
            q_summaries[f"q{int(required * 100)}"] = {
                "variant": variant.name,
                "active_median": active["median"],
                "all_median": all_channels["median"],
                "real_minus_rewired_median": quantile_rewiring["median"],
                "rewiring_pass": rewiring_pass,
                "pass": passed,
            }
            q_passes.append(passed)
        sensitivity_pass = all(q_passes)
        sensitivity_details: dict[str, Any] = {
            "rule": "positive active and all-channel medians at q95 and q99",
            "quantiles": q_summaries,
        }
    else:
        scale_results = []
        for variant in sorted(
            spec.variants,
            key=lambda value: float(value.diffusion_scale),
        ):
            active = variant_results[variant.name]["active_channels"]
            all_channels = variant_results[variant.name]["all_channels"]
            scale_rewiring = rewiring_results[variant.name]
            scale_rewiring_pass = (
                scale_rewiring["median"] > 0
                and len(scale_rewiring["negative_cells"]) <= 1
            )
            scale_results.append(
                {
                    "variant": variant.name,
                    "scale": float(variant.diffusion_scale),
                    "active_median": active["median"],
                    "all_median": all_channels["median"],
                    "real_minus_rewired_median": scale_rewiring["median"],
                    "rewiring_pass": scale_rewiring_pass,
                    "positive": (
                        active["median"] > 0
                        and all_channels["median"] > 0
                        and scale_rewiring_pass
                    ),
                }
            )
        adjacent_pairs = [
            [first["variant"], second["variant"]]
            for first, second in zip(scale_results, scale_results[1:])
            if first["positive"] and second["positive"]
        ]
        sensitivity_pass = bool(adjacent_pairs)
        sensitivity_details = {
            "rule": "positive active and all-channel medians at two adjacent scales",
            "scales": scale_results,
            "passing_adjacent_pairs": adjacent_pairs,
        }

    assert loco is not None
    criteria = {
        "median_incremental_positive": {
            "pass": primary_active["median"] > 0,
            "value": primary_active["median"],
            "rule": "primary active-channel median incremental alignment > 0",
        },
        "positive_in_at_least_three_cells": {
            "pass": len(positive_cells) >= 3,
            "positive_cells": positive_cells,
            "count": len(positive_cells),
            "rule": "primary active-channel cell median > 0 in at least 3 of 4 cells",
        },
        "active_and_all_positive": {
            "pass": primary_active["median"] > 0 and primary_all["median"] > 0,
            "active_median": primary_active["median"],
            "all_median": primary_all["median"],
            "rule": "primary variant is positive for active and all channels",
        },
        "candidate_sensitivity": {
            "pass": sensitivity_pass,
            **sensitivity_details,
        },
        "real_minus_rewired": {
            "pass": rewiring["median"] > 0 and len(rewiring["negative_cells"]) <= 1,
            "median": rewiring["median"],
            "negative_cells": rewiring["negative_cells"],
            "rule": (
                "primary active real-minus-rewired median > 0 with negative "
                "cell medians in at most one cell"
            ),
        },
        "loco_prediction": {
            "pass": bool(
                loco.non_worse_than_best_control
                and loco.improves_predeclared_metric
            ),
            "non_worse_than_best_control": loco.non_worse_than_best_control,
            "improves_predeclared_metric": loco.improves_predeclared_metric,
            "rule": (
                "LOCO is not worse than the strongest confound model and improves "
                "at least one predeclared metric"
            ),
        },
    }
    failed = [name for name, result in criteria.items() if not result["pass"]]
    return (
        {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "depth": spec.depth,
            "primary_variant": spec.primary_variant,
            "status": "pass" if not failed else "fail",
            "selection_score": float(primary_active["mean"]),
            "not_evaluable_reasons": [],
            "failed_criteria": failed,
            "criteria": criteria,
            "variants": variant_results,
            "rewiring": rewiring,
            "rewiring_by_variant": rewiring_results,
            "primary_active_checkpoint_contrasts": primary_active["checkpoints"],
        },
        universe,
    )


def _select_candidate(
    passing: list[dict[str, Any]],
    *,
    tolerance: float,
) -> tuple[str, dict[str, Any]]:
    highest = max(float(candidate["selection_score"]) for candidate in passing)
    tied = [
        candidate
        for candidate in passing
        if highest - float(candidate["selection_score"]) <= tolerance + 1e-12
    ]

    def preference(candidate: dict[str, Any]) -> tuple[Any, ...]:
        if candidate["family"] == NORMALIZED_WL:
            return (
                0,
                int(candidate["depth"]),
                -float(candidate["selection_score"]),
                str(candidate["candidate_id"]),
            )
        return (
            1,
            math.inf,
            -float(candidate["selection_score"]),
            str(candidate["candidate_id"]),
        )

    selected = min(tied, key=preference)
    return str(selected["candidate_id"]), {
        "largest_mean_incremental": highest,
        "tolerance": tolerance,
        "within_tolerance": [
            str(candidate["candidate_id"])
            for candidate in sorted(tied, key=lambda value: str(value["candidate_id"]))
        ],
        "rule": (
            "within 0.01 of the largest mean: normalized WL before diffusion, "
            "then shallower WL depth; remaining ties use score then candidate_id"
        ),
    }


def evaluate_gate1_decision(
    records: Iterable[CheckpointEvidence | Mapping[str, Any]],
    candidate_specs: Iterable[CandidateSpec | Mapping[str, Any]],
    loco_flags: Mapping[str, LocoFlags | Mapping[str, Any]],
    *,
    expected_cells: Iterable[str] = DEFAULT_EXPECTED_CELLS,
    tie_tolerance: float = 0.01,
    required_rewire_repetitions: int = 20,
) -> dict[str, Any]:
    """Apply the frozen Gate 1 rules to normalized checkpoint evidence.

    The function fails closed: any undeclared candidate, duplicate record,
    incomplete four-cell universe, missing required view, incomplete rewire, or
    missing LOCO conclusion makes the overall decision ``not_evaluable``.
    """

    normalized_records = tuple(_coerce_evidence(value) for value in records)
    specs = tuple(_coerce_spec(value) for value in candidate_specs)
    normalized_loco = {
        candidate_id: _coerce_loco(value)
        for candidate_id, value in loco_flags.items()
    }
    expected = tuple(expected_cells)
    global_reasons: list[dict[str, Any]] = []
    if (
        len(expected) != 4
        or len(set(expected)) != 4
        or any(not isinstance(value, str) or not value.strip() for value in expected)
    ):
        _append_reason(
            global_reasons,
            "invalid_expected_cells",
            "Gate 1 requires exactly four unique expected cells",
            expected_cells=list(expected),
        )
    if (
        not math.isfinite(tie_tolerance)
        or not math.isclose(tie_tolerance, TIE_TOLERANCE, abs_tol=1e-12)
    ):
        _append_reason(
            global_reasons,
            "non_frozen_tie_tolerance",
            "the frozen Gate 1 tie tolerance is exactly 0.01",
            tie_tolerance=tie_tolerance,
        )
    if (
        isinstance(required_rewire_repetitions, bool)
        or not isinstance(required_rewire_repetitions, int)
        or required_rewire_repetitions <= 0
    ):
        _append_reason(
            global_reasons,
            "invalid_required_rewire_repetitions",
            "required_rewire_repetitions must be a positive integer",
            required_rewire_repetitions=required_rewire_repetitions,
        )
    if not specs:
        _append_reason(
            global_reasons,
            "missing_candidate_specs",
            "at least one frozen candidate specification is required",
        )
    spec_ids = [spec.candidate_id for spec in specs]
    if len(set(spec_ids)) != len(spec_ids):
        _append_reason(
            global_reasons,
            "duplicate_candidate_spec",
            "candidate specifications must have unique candidate_id values",
        )

    declared = set(spec_ids)
    undeclared = sorted(
        {record.candidate_id for record in normalized_records} - declared,
        key=str,
    )
    if undeclared:
        _append_reason(
            global_reasons,
            "undeclared_candidate_evidence",
            "evidence contains candidates absent from the frozen specification",
            candidate_ids=undeclared,
        )
    observed_cells = {record.cell for record in normalized_records}
    if observed_cells != set(expected):
        _append_reason(
            global_reasons,
            "exact_four_cells_required",
            "evidence must contain all and only the four frozen cells",
            missing_cells=sorted(set(expected) - observed_cells, key=str),
            extra_cells=sorted(observed_cells - set(expected), key=str),
        )

    keys: set[tuple[str, str, str, str, str]] = set()
    checkpoint_cells: dict[str, set[str]] = {}
    for record in normalized_records:
        checkpoint_cells.setdefault(record.checkpoint_id, set()).add(record.cell)
        key = (
            record.candidate_id,
            record.variant,
            record.subset,
            record.cell,
            record.checkpoint_id,
        )
        if key in keys:
            _append_reason(
                global_reasons,
                "duplicate_checkpoint_record",
                "candidate/view/subset/checkpoint evidence must be unique",
                key=list(key),
            )
        keys.add(key)
    reused_checkpoint_ids = {
        checkpoint_id: sorted(cells, key=str)
        for checkpoint_id, cells in sorted(
            checkpoint_cells.items(), key=lambda item: str(item[0])
        )
        if len(cells) > 1
    }
    if reused_checkpoint_ids:
        _append_reason(
            global_reasons,
            "checkpoint_id_reused_across_cells",
            "a normalized checkpoint identity may belong to only one cell",
            checkpoint_cells=reused_checkpoint_ids,
        )

    decisions: dict[str, dict[str, Any]] = {}
    universes: dict[str, frozenset[tuple[str, str]]] = {}
    for spec in specs:
        decision, universe = _evaluate_candidate(
            spec,
            [
                record
                for record in normalized_records
                if record.candidate_id == spec.candidate_id
            ],
            normalized_loco.get(spec.candidate_id),
            expected,
            max(1, required_rewire_repetitions),
        )
        decisions[spec.candidate_id] = decision
        if universe is not None:
            universes[spec.candidate_id] = universe

    if universes:
        reference_id = next(iter(universes))
        reference = universes[reference_id]
        mismatched = sorted(
            [
                candidate_id
                for candidate_id, universe in universes.items()
                if universe != reference
            ],
            key=str,
        )
        if mismatched:
            _append_reason(
                global_reasons,
                "candidate_checkpoint_universe_mismatch",
                "all candidates must use the same frozen checkpoints",
                reference_candidate=reference_id,
                mismatched_candidates=mismatched,
            )

    not_evaluable_candidates = sorted(
        [
            candidate_id
            for candidate_id, decision in decisions.items()
            if decision["status"] == "not_evaluable"
        ],
        key=str,
    )
    if not_evaluable_candidates:
        _append_reason(
            global_reasons,
            "candidate_not_evaluable",
            "every frozen candidate must be evaluable before selection",
            candidate_ids=not_evaluable_candidates,
        )

    if global_reasons:
        status = "not_evaluable"
        selected_candidate = None
        tie_break = None
    else:
        passing = [
            decision for decision in decisions.values() if decision["status"] == "pass"
        ]
        if not passing:
            status = "fail"
            selected_candidate = None
            tie_break = None
        else:
            status = "pass"
            selected_candidate, tie_break = _select_candidate(
                passing,
                tolerance=TIE_TOLERANCE,
            )

    return {
        "status": status,
        "evaluable": status != "not_evaluable",
        "selected_candidate": selected_candidate,
        "expected_cells": list(expected),
        "observed_cells": sorted(observed_cells, key=str),
        "control_blocks": list(CONTROL_BLOCK_NAMES),
        "required_rewire_repetitions": required_rewire_repetitions,
        "not_evaluable_reasons": global_reasons,
        "candidates": decisions,
        "tie_break": tie_break,
    }


__all__ = [
    "CANDIDATE_FAMILIES",
    "CONTROL_BLOCK_NAMES",
    "DEFAULT_EXPECTED_CELLS",
    "DIFFUSION",
    "NORMALIZED_WL",
    "SUBSET_NAMES",
    "TIE_TOLERANCE",
    "CandidateSpec",
    "CandidateVariant",
    "CheckpointEvidence",
    "LocoFlags",
    "RewireAlignment",
    "evaluate_gate1_decision",
]
