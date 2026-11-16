"""Qt-free logic behind the data library screen ("데이터 보관함").

* which action buttons a selection allows (and why one is disabled),
* the :class:`~bang_rl.gui.training.TrainingPreset` an "apply" action stands for,
* readable table cells and the details text (HTML) of an entry.

Everything here is plain Python so it is unit tested without a display. The
dialog in :mod:`bang_rl.gui.dialogs.data_library` only renders it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape
from typing import Any

from bang_rl.data_library import (
    KIND_DATASET,
    KIND_MAIN_AI,
    KIND_MODULE_PAIR,
    KIND_REPLAY,
    KIND_REPORT,
    LIBRARY_KINDS,
    Fact,
    LibraryEntry,
)

from .i18n import tr
from .training import (
    TRAINING_MODE_DATA_COLLECTION,
    TRAINING_MODE_INFERENCE_REFINEMENT,
    TRAINING_MODE_MAIN_AI,
    TRAINING_MODE_MODULE_TRAINING,
    TrainingPreset,
)

# --- actions ---------------------------------------------------------------------

ACTION_MODULE_TRAINING_DATA = "module_training_data"
ACTION_REFINEMENT_DATA = "refinement_data"
ACTION_RESUME_MODULES = "resume_modules"
ACTION_MAIN_AI_TRAIN = "main_ai_train"
ACTION_EVALUATE = "evaluate"
ACTION_PLAY = "play"
ACTION_CONTINUE_PPO = "continue_ppo"
ACTION_REFINE_BASE = "refine_base"
ACTION_COLLECT = "collect"
ACTION_VIEW_REPLAY = "view_replay"
ACTION_OPEN_REPORT = "open_report"
ACTION_EDIT_NOTES = "edit_notes"
ACTION_SAVE_COPY = "save_copy"
ACTION_OPEN_FOLDER = "open_folder"
ACTION_DELETE = "delete"

# Actions that start another screen or job setup and therefore close the library.
APPLY_ACTIONS = frozenset(
    {
        ACTION_MODULE_TRAINING_DATA,
        ACTION_REFINEMENT_DATA,
        ACTION_RESUME_MODULES,
        ACTION_MAIN_AI_TRAIN,
        ACTION_CONTINUE_PPO,
        ACTION_REFINE_BASE,
        ACTION_COLLECT,
    }
)
LEAVING_ACTIONS = APPLY_ACTIONS | {ACTION_PLAY, ACTION_VIEW_REPLAY}

# Button order: what the entry kind is for first, then the general actions.
ACTIONS_BY_KIND: dict[str, tuple[str, ...]] = {
    KIND_DATASET: (ACTION_MODULE_TRAINING_DATA, ACTION_REFINEMENT_DATA),
    KIND_MODULE_PAIR: (ACTION_MAIN_AI_TRAIN, ACTION_RESUME_MODULES, ACTION_EVALUATE),
    KIND_MAIN_AI: (ACTION_PLAY, ACTION_CONTINUE_PPO, ACTION_REFINE_BASE, ACTION_COLLECT, ACTION_EVALUATE),
    KIND_REPLAY: (ACTION_VIEW_REPLAY,),
    KIND_REPORT: (ACTION_OPEN_REPORT,),
}
GENERAL_ACTIONS: tuple[str, ...] = (ACTION_EDIT_NOTES, ACTION_SAVE_COPY, ACTION_OPEN_FOLDER, ACTION_DELETE)
ALL_ACTIONS: tuple[str, ...] = (
    ACTION_MODULE_TRAINING_DATA,
    ACTION_REFINEMENT_DATA,
    ACTION_MAIN_AI_TRAIN,
    ACTION_RESUME_MODULES,
    ACTION_PLAY,
    ACTION_CONTINUE_PPO,
    ACTION_REFINE_BASE,
    ACTION_COLLECT,
    ACTION_EVALUATE,
    ACTION_VIEW_REPLAY,
    ACTION_OPEN_REPORT,
    *GENERAL_ACTIONS,
)
# Double click: the most common action per kind (the first usable one of the list).
DEFAULT_ACTIONS: dict[str, tuple[str, ...]] = {
    KIND_DATASET: (ACTION_MODULE_TRAINING_DATA,),
    KIND_MODULE_PAIR: (ACTION_MAIN_AI_TRAIN, ACTION_RESUME_MODULES),
    KIND_MAIN_AI: (ACTION_PLAY,),
    KIND_REPLAY: (ACTION_VIEW_REPLAY,),
    KIND_REPORT: (ACTION_OPEN_REPORT,),
}
# Kinds a "save a copy" plan exists for (see bang_rl.saved_artifacts).
COPYABLE_KINDS = frozenset({KIND_DATASET, KIND_MODULE_PAIR, KIND_MAIN_AI})

# Kind filter entries: ``None`` is "all"; the tuple groups several kinds.
KIND_FILTER_ALL = "all"
KIND_FILTER_MODELS = "models"
KIND_FILTERS: tuple[str, ...] = (KIND_FILTER_ALL, *LIBRARY_KINDS, KIND_FILTER_MODELS)


def kinds_for_filter(value: str | None) -> tuple[str, ...] | None:
    """Kinds a kind-filter value stands for (``None`` = every kind)."""

    if value in (None, "", KIND_FILTER_ALL):
        return None
    if value == KIND_FILTER_MODELS:
        return (KIND_MODULE_PAIR, KIND_MAIN_AI)
    return (str(value),)


@dataclass(frozen=True, slots=True)
class ActionState:
    """One button: usable or not, and the reason shown in its tooltip when not."""

    action: str
    enabled: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class LibraryChoice:
    """What the user picked in the library, handed to the screen that opened it."""

    action: str
    entries: tuple[LibraryEntry, ...]
    params: dict[str, Any] = field(default_factory=dict)


def _compat_reason(entry: LibraryEntry) -> str:
    ok, code = entry.compatibility()
    if ok:
        return ""
    prefix = "training.resume_module_reason" if entry.kind == KIND_MODULE_PAIR else "training.base_model_reason"
    return tr(f"{prefix}.{code}") if code else tr("library.reason.incompatible")


def action_states(entries: Sequence[LibraryEntry]) -> dict[str, ActionState]:
    """State of every action for the current selection (disabled ones carry a reason)."""

    states = {action: ActionState(action, False, tr("library.reason.select_first")) for action in ALL_ACTIONS}
    if not entries:
        return states
    kinds = {entry.kind for entry in entries}
    if len(entries) > 1:
        for action in ALL_ACTIONS:
            states[action] = ActionState(action, False, tr("library.reason.select_one"))
        if kinds == {KIND_DATASET}:
            for action in ACTIONS_BY_KIND[KIND_DATASET]:
                states[action] = ActionState(action, True)
        states[ACTION_OPEN_FOLDER] = ActionState(ACTION_OPEN_FOLDER, False, tr("library.reason.select_one"))
        states[ACTION_DELETE] = ActionState(ACTION_DELETE, True)
        return states
    entry = entries[0]
    wrong_kind = tr("library.reason.other_kind")
    for action in ALL_ACTIONS:
        if action in GENERAL_ACTIONS:
            continue
        states[action] = ActionState(action, False, wrong_kind)
    for action in ACTIONS_BY_KIND.get(entry.kind, ()):
        enabled, reason = True, ""
        if entry.kind in (KIND_MODULE_PAIR, KIND_MAIN_AI) and action != ACTION_RESUME_MODULES:
            ok, _code = entry.compatibility()
            if not ok:
                enabled, reason = False, _compat_reason(entry)
        if action == ACTION_RESUME_MODULES and not entry.id:
            enabled, reason = False, tr("library.reason.no_id")
        states[action] = ActionState(action, enabled, reason)
    if entry.info is None:
        states[ACTION_EDIT_NOTES] = ActionState(ACTION_EDIT_NOTES, False, tr("library.reason.no_info"))
    else:
        states[ACTION_EDIT_NOTES] = ActionState(ACTION_EDIT_NOTES, True)
    states[ACTION_OPEN_FOLDER] = ActionState(ACTION_OPEN_FOLDER, True)
    states[ACTION_DELETE] = ActionState(ACTION_DELETE, True)
    if entry.kind not in COPYABLE_KINDS:
        states[ACTION_SAVE_COPY] = ActionState(ACTION_SAVE_COPY, False, tr("library.reason.no_copy_plan"))
    elif entry.saved:
        states[ACTION_SAVE_COPY] = ActionState(ACTION_SAVE_COPY, False, tr("library.reason.already_saved"))
    else:
        states[ACTION_SAVE_COPY] = ActionState(ACTION_SAVE_COPY, True)
    return states


def default_action(entries: Sequence[LibraryEntry]) -> str | None:
    """Action for a double click: the usual one for the kind, if it is usable."""

    if not entries:
        return None
    states = action_states(entries)
    kinds = {entry.kind for entry in entries}
    if len(kinds) != 1:
        return None
    for action in DEFAULT_ACTIONS.get(next(iter(kinds)), ()):
        if states[action].enabled:
            return action
    return None


def build_preset(action: str, entries: Sequence[LibraryEntry]) -> TrainingPreset | None:
    """Training form preset for an apply action, or ``None`` for other actions."""

    if action == ACTION_MODULE_TRAINING_DATA:
        return TrainingPreset(TRAINING_MODE_MODULE_TRAINING, dataset_paths=tuple(str(entry.path) for entry in entries))
    if action == ACTION_REFINEMENT_DATA:
        return TrainingPreset(
            TRAINING_MODE_INFERENCE_REFINEMENT, dataset_paths=tuple(str(entry.path) for entry in entries)
        )
    if not entries:
        return None
    entry = entries[0]
    if action == ACTION_RESUME_MODULES:
        return TrainingPreset(TRAINING_MODE_MODULE_TRAINING, module_run_id=entry.id)
    if action == ACTION_MAIN_AI_TRAIN:
        return TrainingPreset(TRAINING_MODE_MAIN_AI, module_run_id=entry.id)
    if action == ACTION_CONTINUE_PPO:
        return TrainingPreset(TRAINING_MODE_MAIN_AI, bangmodel_path=str(entry.path))
    if action == ACTION_REFINE_BASE:
        return TrainingPreset(TRAINING_MODE_INFERENCE_REFINEMENT, bangmodel_path=str(entry.path))
    if action == ACTION_COLLECT:
        return TrainingPreset(TRAINING_MODE_DATA_COLLECTION, players="main1-dummy4", bangmodel_path=str(entry.path))
    return None


# --- text ------------------------------------------------------------------------


def kind_text(kind: str) -> str:
    return tr(f"library.kind.{kind}")


def location_text(location: str) -> str:
    return tr(f"library.location.{location}")


def _translated_or(key: str, fallback: str) -> str:
    text = tr(key)
    return fallback if text == key else text


def _fact_value(key: str, value: str) -> str:
    if key == "players":
        return _translated_or("library.players." + value.replace("-", "_"), value)
    if key == "action_mode":
        return _translated_or("training.action_mode." + value, value)
    if key == "workflow":
        return _translated_or("training.mode." + value, value)
    if key == "parts":
        return tr(f"perf.parts_{value}") if value in ("intent", "emotion") else value
    if key == "human_won":
        return tr("library.yes") if value == "yes" else tr("library.no")
    return value


def fact_text(fact: Fact) -> str:
    """One fact as a short readable phrase (``("games", "100")`` -> "100판")."""

    key, value = fact
    if key == "main_ai_ready":
        return tr("library.fact.main_ai_ready.ok") if value == "ok" else tr("library.fact.main_ai_ready.no")
    template = tr(f"library.fact.{key}", value=_fact_value(key, value))
    return template if template != f"library.fact.{key}" else f"{key} {value}"


def facts_text(facts: Sequence[Fact]) -> str:
    return " · ".join(fact_text(fact) for fact in facts)


@dataclass(frozen=True, slots=True)
class TableRow:
    key: str
    cells: tuple[str, ...]
    tooltip: str


TABLE_COLUMNS: tuple[str, ...] = (
    "library.col.kind",
    "library.col.name",
    "library.col.created",
    "library.col.conditions",
    "library.col.summary",
    "library.col.location",
    "library.col.notes",
)


def table_row(entry: LibraryEntry) -> TableRow:
    cells = (
        kind_text(entry.kind),
        entry.name,
        entry.created_text,
        facts_text(entry.condition_facts),
        facts_text(entry.summary_facts),
        location_text(entry.location),
        entry.first_note_line,
    )
    tooltip = "\n".join(part for part in (entry.name, entry.created_text, entry.notes.strip()) if part)
    return TableRow(entry.key, cells, tooltip)


def human_size(size: int) -> str:
    from bang_rl.saved_artifacts import human_size as _human_size

    return _human_size(int(size))


def _info_lines(values: Any) -> list[tuple[str, str]]:
    import json

    rows: list[tuple[str, str]] = []
    if not isinstance(values, dict):
        return rows
    for key, value in values.items():
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False)
            text = text if len(text) <= 300 else text[:297] + "..."
        else:
            text = str(value)
        rows.append((str(key), text))
    return rows


def _link(entry: LibraryEntry) -> str:
    where = f" ({location_text(entry.location)})" if entry.saved else ""
    return f'<a href="entry:{escape(entry.key, quote=True)}">{escape(entry.name)}</a> ' f"[{escape(kind_text(entry.kind))}]{escape(where)}"


def entry_details_html(entry: LibraryEntry, *, performance: str | None = None, with_size: bool = True) -> str:
    """Details pane content: identity, conditions, results, sources, used-by links, notes."""

    def heading(key: str) -> str:
        return f"<h4>{escape(tr(key))}</h4>"

    def table(rows: Sequence[tuple[str, str]]) -> str:
        body = "".join(
            f"<tr><td style='padding-right:12px'><b>{escape(label)}</b></td><td>{escape(value)}</td></tr>"
            for label, value in rows
        )
        return f"<table>{body}</table>"

    info = entry.info
    basics = [
        (tr("library.detail.kind"), f"{kind_text(entry.kind)}" + (f" / {entry.kind_label}" if entry.kind_label else "")),
        (tr("library.detail.name"), entry.name),
        (tr("library.detail.created"), entry.created_text or "-"),
        (tr("library.detail.created_by"), entry.created_by or "-"),
        (tr("library.detail.location"), location_text(entry.location)),
        (tr("library.detail.id"), entry.id or "-"),
        (tr("library.detail.path"), "\n".join(str(path) for path in (entry.paths or (entry.path,)))),
    ]
    if with_size:
        basics.append((tr("library.detail.size"), human_size(entry.size_bytes())))
    parts = [f"<h3>{escape(entry.name)}</h3>", table(basics)]

    conditions = _info_lines(info.conditions) if info is not None else []
    summary = _info_lines(info.summary) if info is not None else []
    parts.append(heading("library.detail.conditions"))
    parts.append(table(conditions) if conditions else f"<p>{escape(facts_text(entry.condition_facts) or '-')}</p>")
    parts.append(heading("library.detail.summary"))
    parts.append(table(summary) if summary else f"<p>{escape(facts_text(entry.summary_facts) or '-')}</p>")

    parts.append(heading("library.detail.sources"))
    if entry.source_links:
        items = []
        for ref, target in entry.source_links:
            if target is not None:
                items.append(f"<li>{_link(target)}</li>")
            else:
                label = ref.get("name") or ref.get("id") or "-"
                items.append(f"<li>{escape(str(label))} [{escape(str(ref.get('kind_label') or ref.get('kind') or ''))}] "
                             f"<i>{escape(tr('library.detail.not_stored'))}</i></li>")
        parts.append(f"<ul>{''.join(items)}</ul>")
    else:
        parts.append(f"<p>{escape(tr('library.detail.none'))}</p>")

    parts.append(heading("library.detail.used_by"))
    if entry.used_by:
        parts.append("<ul>" + "".join(f"<li>{_link(item)}</li>" for item in entry.used_by) + "</ul>")
    else:
        parts.append(f"<p>{escape(tr('library.detail.none'))}</p>")
    if entry.related:
        parts.append(heading("library.detail.related"))
        parts.append("<ul>" + "".join(f"<li>{_link(item)}</li>" for item in entry.related) + "</ul>")

    parts.append(heading("library.detail.notes"))
    notes = entry.notes.strip()
    parts.append(f"<pre>{escape(notes)}</pre>" if notes else f"<p>{escape(tr('library.detail.none'))}</p>")
    if performance:
        parts.append(heading("library.detail.performance"))
        parts.append(f"<pre>{escape(performance)}</pre>")
    return "".join(parts)


def performance_text(entry: LibraryEntry) -> str:
    """Saved-model performance text of a module pair or main AI (reuses the performance dialog's formatters)."""

    from .dialogs.module_performance import format_main_ai_model_summary, format_module_run_summary

    if entry.kind == KIND_MODULE_PAIR:
        from bang_rl.training.module_artifacts import summarize_module_run

        run = entry.extra.get("run")
        return format_module_run_summary(summarize_module_run(run)) if run is not None else ""
    if entry.kind == KIND_MAIN_AI:
        from bang_rl.training.main_ai_artifacts import summarize_main_ai_artifact

        return format_main_ai_model_summary(summarize_main_ai_artifact(entry.path))
    return ""


def save_plan_for(entry: LibraryEntry, *, saved_root: Any = None) -> Any:
    """The existing "save a copy" plan for an entry (``None`` when its kind has none)."""

    from bang_rl.saved_artifacts import plan_dataset, plan_main_ai, plan_modules

    if entry.kind == KIND_DATASET:
        return plan_dataset(entry.path, saved_root=saved_root)
    if entry.kind == KIND_MODULE_PAIR:
        return plan_modules(entry.intent_path, entry.emotion_path, saved_root=saved_root)
    if entry.kind == KIND_MAIN_AI:
        return plan_main_ai(entry.path, saved_root=saved_root)
    return None


def report_text(path: Any, *, limit: int = 4000) -> str:
    """Read-only text of a report: each top-level key with a shortened value."""

    import json
    from pathlib import Path

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return str(exc)
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)[: limit * 4]
    lines: list[str] = []
    for key, value in data.items():
        text = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
        if len(text) > limit:
            text = text[:limit] + f"\n... ({len(text) - limit})"
        lines.append(f"[{key}]\n{text}\n")
    return "\n".join(lines)


def entry_delete_paths(entry: LibraryEntry) -> tuple[Any, ...]:
    """Files/folders that make up ``entry`` (both modules of a pair)."""

    return tuple(entry.paths or (entry.path,))


def delete_confirmation_text(entries: Sequence[LibraryEntry]) -> str:
    """Confirmation text: what is deleted, total size, and entries that depend on it."""

    selected = {id(entry) for entry in entries}
    lines = [tr("library.delete_prompt", count=len(entries))]
    lines += [f"  - {entry.name} [{kind_text(entry.kind)}]" for entry in entries]
    total = sum(entry.size_bytes() for entry in entries)
    lines.append(tr("library.delete_size", size=human_size(total)))
    dependents = [
        item for entry in entries for item in (*entry.used_by, *entry.related) if id(item) not in selected
    ]
    if dependents:
        lines.append("")
        lines.append(tr("library.delete_dependents"))
        seen: set[int] = set()
        for item in dependents:
            if id(item) not in seen:
                seen.add(id(item))
                lines.append(f"  - {item.name} [{kind_text(item.kind)}]")
    if any(entry.saved for entry in entries):
        lines.append("")
        lines.append(tr("library.delete_saved_warning"))
    lines.append("")
    lines.append(tr("library.delete_trash_note"))
    return "\n".join(lines)


def multi_details_html(entries: Sequence[LibraryEntry]) -> str:
    items = "".join(f"<li>{escape(entry.name)} [{escape(kind_text(entry.kind))}]</li>" for entry in entries)
    return f"<h3>{escape(tr('library.detail.multi', count=len(entries)))}</h3><ul>{items}</ul>"


__all__ = [
    "ACTIONS_BY_KIND",
    "ACTION_COLLECT",
    "ACTION_CONTINUE_PPO",
    "ACTION_DELETE",
    "ACTION_EDIT_NOTES",
    "ACTION_EVALUATE",
    "ACTION_MAIN_AI_TRAIN",
    "ACTION_MODULE_TRAINING_DATA",
    "ACTION_OPEN_FOLDER",
    "ACTION_OPEN_REPORT",
    "ACTION_PLAY",
    "ACTION_REFINE_BASE",
    "ACTION_REFINEMENT_DATA",
    "ACTION_RESUME_MODULES",
    "ACTION_SAVE_COPY",
    "ACTION_VIEW_REPLAY",
    "ALL_ACTIONS",
    "APPLY_ACTIONS",
    "ActionState",
    "KIND_FILTERS",
    "KIND_FILTER_ALL",
    "KIND_FILTER_MODELS",
    "LEAVING_ACTIONS",
    "LibraryChoice",
    "TABLE_COLUMNS",
    "TableRow",
    "action_states",
    "build_preset",
    "default_action",
    "delete_confirmation_text",
    "entry_delete_paths",
    "entry_details_html",
    "fact_text",
    "facts_text",
    "kind_text",
    "kinds_for_filter",
    "location_text",
    "multi_details_html",
    "performance_text",
    "report_text",
    "save_plan_for",
    "table_row",
]
