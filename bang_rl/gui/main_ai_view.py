"""Pure (Qt-free) helpers that turn main-AI training metrics into display data.

The training monitor and the saved-model dialog share these so the text and
series logic can be tested without a display. ``metrics`` is the public
``progress.metrics`` mapping published by ``train_main_ai``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .i18n import tr

ROLE_NAMES = ("sheriff", "deputy", "outlaw", "traitor")
BASELINE_NAMES = ("dummy", "random")
TRUNCATION_WARNING_RATE = 0.01


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed:.{digits}f}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def series(metrics: Mapping[str, Any] | None, key: str) -> list[float]:
    """A published point list as plain floats (non-numbers are skipped)."""

    values = _mapping(metrics).get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [float(item) for item in values if number(item) is not None]


def sampled_axis(count: int, interval: int | float | None) -> list[int]:
    """Game/update numbers of sampled graph points (1, 1+N, 1+2N, ...)."""

    step = max(1, int(interval or 1))
    return [1 + index * step for index in range(max(0, int(count)))]


def baseline_win_rates(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    """Overall dummy/random win rate on the evaluation deals (None if unknown)."""

    baselines = _mapping(_mapping(metrics).get("main_baselines"))
    return {name: number(_mapping(baselines.get(name)).get("win_rate")) for name in BASELINE_NAMES}


def truncation_is_high(rate: Any) -> bool:
    parsed = number(rate)
    return parsed is not None and parsed > TRUNCATION_WARNING_RATE


def role_label(name: str) -> str:
    return tr(f"mainai.role.{name}")


def baseline_role_rate(baselines: Mapping[str, Any], baseline: str, role: str) -> float | None:
    """Win rate of the ``dummy``/``random`` baseline for one role."""

    by_role = _mapping(_mapping(baselines.get(baseline)).get("by_role"))
    return number(_mapping(by_role.get(role)).get("win_rate"))


def by_role_line(
    main_by_role: Mapping[str, Any], baselines: Mapping[str, Any], *, indent: str = "  "
) -> str:
    """``role main / dummy / random`` win rates on one line."""

    parts = [
        f"{role_label(name)} {fmt(main_by_role.get(name), 2)} / "
        f"{fmt(baseline_role_rate(baselines, 'dummy', name), 2)} / "
        f"{fmt(baseline_role_rate(baselines, 'random', name), 2)}"
        for name in ROLE_NAMES
    ]
    return indent + tr("mainai.by_role") + " " + " · ".join(parts)


def format_eval_row(
    title_key: str, row: Mapping[str, Any] | None, baselines: Mapping[str, Any] | None
) -> list[str]:
    """Readable lines for one evaluation row (``main_eval_latest`` / ``main_best_eval``)."""

    if not row:
        return [tr(title_key) + " " + tr("mainai.no_eval")]
    lines = [
        tr(
            title_key + ".line",
            update=row.get("ppo_update", "—"),
            games=row.get("training_games", "—"),
            win=fmt(row.get("win_rate")),
            score=fmt(row.get("normalized_score")),
            accuracy=fmt(row.get("role_accuracy")),
            mse=fmt(row.get("emotion_mse"), 4),
        )
    ]
    lines.append(by_role_line(_mapping(row.get("by_role")), _mapping(baselines)))
    return lines


def format_main_ai_summary(
    metrics: Mapping[str, Any] | None,
    *,
    ppo_updates: int = 0,
    artifact_path: str | None = None,
) -> dict[str, str]:
    """Text for the monitor's main-AI labels, keyed by label."""

    data = _mapping(metrics)
    baselines = _mapping(data.get("main_baselines"))
    rates = baseline_win_rates(data)
    training_rate = number(data.get("main_training_win_rate"))
    progress = tr(
        "mainai.progress",
        games=int(number(data.get("main_games_completed")) or 0),
        updates=int(ppo_updates or 0),
        win=fmt(training_rate),
    )
    if any(value is not None for value in rates.values()):
        base_text = tr("mainai.baselines", dummy=fmt(rates["dummy"]), random=fmt(rates["random"]))
    else:
        base_text = tr("mainai.baselines_pending")
    return {
        "progress": progress,
        "baselines": base_text,
        "latest": "\n".join(format_eval_row("mainai.latest", _mapping(data.get("main_eval_latest")), baselines)),
        "best": "\n".join(format_eval_row("mainai.best", _mapping(data.get("main_best_eval")), baselines)),
        "saved": tr("mainai.saved", path=artifact_path) if artifact_path else tr("mainai.not_saved"),
    }


# --- inference refinement -------------------------------------------------------

REFINE_SERIES = ("hidden_role_accuracy", "role_accuracy", "emotion_mse", "win_rate")


def refine_series(metrics: Mapping[str, Any] | None) -> dict[str, list[float | None]]:
    """Per-round values (index = round, 0 = starting modules); None where unmeasured."""

    data = _mapping(metrics)
    result: dict[str, list[float | None]] = {}
    for key in REFINE_SERIES:
        values = data.get(f"refine_{key}_points")
        result[key] = (
            [number(item) for item in values]
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
            else []
        )
    return result


def _refine_values(row: Mapping[str, Any]) -> dict[str, str]:
    values = _mapping(row.get("metrics"))
    return {
        "hidden": fmt(values.get("hidden_role_accuracy")),
        "role": fmt(values.get("role_accuracy")),
        "mse": fmt(values.get("emotion_mse"), 4),
        "win": fmt(values.get("win_rate")),
    }


def refine_round_line(row: Mapping[str, Any]) -> str:
    """One readable line: round, accepted/rejected, the reason and the metrics."""

    verdict = tr("refine.verdict.accepted" if row.get("accepted") else "refine.verdict.rejected")
    return tr(
        "refine.round_line",
        round=row.get("round", "—"),
        verdict=verdict,
        reason=tr(f"refine.reason.{row.get('reason')}"),
        **_refine_values(row),
    )


def format_refinement_summary(
    metrics: Mapping[str, Any] | None, *, artifact_path: str | None = None
) -> dict[str, str]:
    """Text for the refinement labels: progress, base, best, per-round lines, rule, saved."""

    data = _mapping(metrics)
    history = [item for item in (data.get("refine_history") or ()) if isinstance(item, Mapping)]
    stage = str(data.get("refine_stage") or "none")
    best_round = int(number(data.get("refine_best_round")) or 0)
    progress = tr(
        "refine.progress",
        round=int(number(data.get("refine_round")) or 0),
        total=int(number(data.get("refine_total_rounds")) or 0),
        stage=tr(f"refine.stage.{stage}"),
        best=best_round,
        games=int(number(data.get("refine_main_games_total")) or number(data.get("episodes_completed")) or 0),
    )
    base = next((item for item in history if item.get("round") == 0), None)
    best = next((item for item in history if item.get("round") == best_round), None)
    rounds = [refine_round_line(item) for item in history if item.get("round") != 0]
    stop = data.get("stop_reason")
    if stop:
        rounds.append(tr(f"refine.stop.{stop}"))
    return {
        "progress": progress,
        "base": tr("refine.base", **_refine_values(base)) if base else "",
        "best": tr("refine.best", round=best_round, **_refine_values(best)) if best else "",
        "rounds": "\n".join(rounds),
        "rule": tr("refine.rule"),
        "saved": tr("refine.saved", path=artifact_path) if artifact_path else tr("refine.not_saved"),
    }


def inference_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Table cells of an inference evaluation: name, hidden, role, MSE, win, by target role."""

    table: list[list[str]] = []
    for row in rows:
        by_role = _mapping(row.get("role_accuracy_by_target_role"))
        table.append(
            [
                tr("infer.base_label") if row.get("label") == "base" else str(row.get("label", "")),
                fmt(row.get("hidden_role_accuracy")),
                fmt(row.get("role_accuracy")),
                fmt(row.get("emotion_mse"), 4),
                fmt(row.get("win_rate")),
                " · ".join(f"{role_label(name)} {fmt(by_role.get(name), 2)}" for name in ROLE_NAMES),
            ]
        )
    return table


__all__ = [
    "BASELINE_NAMES",
    "REFINE_SERIES",
    "ROLE_NAMES",
    "TRUNCATION_WARNING_RATE",
    "format_refinement_summary",
    "inference_rows",
    "refine_round_line",
    "refine_series",
    "baseline_role_rate",
    "baseline_win_rates",
    "by_role_line",
    "fmt",
    "format_eval_row",
    "format_main_ai_summary",
    "number",
    "role_label",
    "sampled_axis",
    "series",
    "truncation_is_high",
]
