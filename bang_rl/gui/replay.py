"""Pure, headless-testable replay model for the graphical board.

This module turns a recorded episode into an ordered list of per-turn public
snapshots that the SAME :class:`~bang_rl.gui.widgets.game_widgets.GraphicalBoard`
and :class:`~bang_rl.gui.widgets.game_widgets.GameLog` can render unchanged.

It imports **no Qt** so the reconstruction, index stepping (next/prev/seek),
and the log-row -> game-moment lookup can all be exercised in CI where PyQt6 is
unavailable.

A replay snapshot duck-types :class:`bang_rl.app.services.PublicGameState` (and
the ``SimpleNamespace`` used in the GUI tests): it exposes ``observation`` (a
dict carrying the keys the board reads plus a *cumulative* ``event_sequence``
up to that turn), ``action_mask``, ``current_player``, ``environment_step``,
``terminated``, ``truncated`` and ``termination_status``.  Feeding consecutive
snapshots to the board animates bang shots via the board's own event diff.

Accepted episode sources:

* a :class:`bang_rl.data.collector.CanonicalEpisode` (anything exposing
  ``to_dict``),
* its ``to_dict()`` mapping (with ordered ``steps`` + a full ``events`` list),
* or a plain list of observation dicts (one per turn).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


# Observation keys the GraphicalBoard / GameLog read from a snapshot. Kept here
# so reconstruction always carries the full set (missing keys default safely).
#
# Privacy: these are exactly the keys of the environment's PUBLIC projection
# (``bang_rl.env.state.PublicEnvironmentState.to_public``): ``revealed_roles``,
# ``role_visibility`` and ``self_role`` already only expose the sheriff, any
# eliminated player, and the viewer's own role -- a living non-sheriff, non-self
# role is 255 / HIDDEN. Recorded episodes are stored public too: the collector's
# ``_observation`` strips the private role keys and ``_event_mapping`` rejects
# non-public events. The private ``role_labels`` list (top-level, only under
# ``include_private=True``) is deliberately NOT in this set, so ReplayModel never
# copies it into a snapshot and hidden roles cannot leak into the board/log.
_OBSERVATION_KEYS = (
    "healths",
    "alive",
    "role_visibility",
    "revealed_roles",
    "self_role",
    "current_player_idx",
    "environment_step",
    "round_index",
)


@dataclass(frozen=True)
class ReplaySnapshot:
    """A single replay turn shaped like ``app.services.PublicGameState``.

    Duck-typed so the graphical board/log render it unchanged.  ``observation``
    carries a *cumulative* ``event_sequence`` (all events at or before this
    turn's ``environment_step``) so the log grows across the replay and the
    board animates newly-appended bang events between consecutive snapshots.
    """

    observation: Mapping[str, Any]
    action_mask: tuple[int, ...]
    environment_step: int
    current_player: int | None
    terminated: bool = False
    truncated: bool = False
    termination_status: str = "running"


def _as_mapping(episode: Any) -> Mapping[str, Any] | None:
    """Return a ``to_dict``-style mapping for a CanonicalEpisode-like input."""

    if isinstance(episode, Mapping):
        return episode
    to_dict = getattr(episode, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    # Recorded observations frequently carry scalar-like fields (e.g.
    # ``environment_step``) as a 1-element list/tuple such as ``[3]`` because
    # the public projection stores them as single-element arrays. Unwrap a
    # single-element sequence before ``int()`` (mirroring ``_current_player``)
    # so those values are read as scalars instead of collapsing to ``None`` and
    # dragging every snapshot back to step 0 (item 9 no-log/no-playback bug).
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
        if value is None:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_step(event: Mapping[str, Any]) -> int | None:
    return _int_or_none(event.get("environment_step"))


def _events_up_to(events: Sequence[Mapping[str, Any]], step: int | None) -> tuple[Mapping[str, Any], ...]:
    """Return every event recorded at or before ``step`` (in recorded order).

    ``step`` of ``None`` means "no bound" and returns every event, so an
    observation with no ``environment_step`` still gets a full log.
    """

    result: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if step is None:
            result.append(dict(event))
            continue
        event_step = _event_step(event)
        if event_step is None or event_step <= step:
            result.append(dict(event))
    return tuple(result)


def _observation_step(observation: Mapping[str, Any], fallback: int) -> int:
    step = _int_or_none(observation.get("environment_step"))
    if step is not None:
        return step
    current = observation.get("current_player_idx")
    if isinstance(current, (list, tuple)) and current:
        # Some observations only carry current_player_idx as a 1-tuple; the
        # environment_step still governs event ordering, so fall back to the
        # per-turn index when the observation omits an explicit step.
        return fallback
    return fallback


def _base_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the board-relevant observation keys (never the private/heavy ones)."""

    base: dict[str, Any] = {}
    for key in _OBSERVATION_KEYS:
        if key in observation:
            base[key] = observation[key]
    return base


class ReplayModel:
    """An ordered, index-addressable reconstruction of a recorded episode.

    Provides ``__len__``, a mutable ``index`` cursor, ``next``/``prev``
    (clamped at the ends), ``seek``, ``snapshot_at`` and a
    ``snapshot_for_moment`` / ``index_for_moment`` lookup that maps an
    ``environment_step`` (or a log-row index / an event mapping) to the
    snapshot representing the game at that moment.  A log-row lookup resolves
    against the COMPLETE cumulative log (:meth:`full_events`), independent of
    the cursor, so it matches what the live GameLog renders.  All pure and
    Qt-free.
    """

    def __init__(self, episode: Any) -> None:
        self._snapshots: tuple[ReplaySnapshot, ...] = tuple(self._reconstruct(episode))
        self._index = 0

    # -- construction ------------------------------------------------
    @classmethod
    def _reconstruct(cls, episode: Any) -> list[ReplaySnapshot]:
        mapping = _as_mapping(episode)
        if mapping is not None:
            return cls._from_mapping(mapping)
        if isinstance(episode, Sequence) and not isinstance(episode, (str, bytes)):
            return cls._from_observations(list(episode))
        raise TypeError(
            "ReplayModel accepts a CanonicalEpisode, its to_dict mapping, "
            "or a sequence of observation dicts"
        )

    @classmethod
    def _from_mapping(cls, mapping: Mapping[str, Any]) -> list[ReplaySnapshot]:
        events = tuple(
            event for event in (mapping.get("events") or ()) if isinstance(event, Mapping)
        )
        steps = mapping.get("steps")
        snapshots: list[ReplaySnapshot] = []
        if steps:
            for turn, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    continue
                snapshots.append(cls._snapshot_from_step(step, turn, events))
            # Append the final settled state so the replay ends on the result.
            final = mapping.get("final_state")
            if isinstance(final, Mapping) and snapshots:
                last_step = snapshots[-1].environment_step
                snapshots.append(
                    cls._snapshot_from_observation(
                        final,
                        turn=len(snapshots),
                        events=events,
                        step_override=last_step,
                        terminal=cls._terminal_status(mapping),
                    )
                )
            return snapshots
        # No 'steps': treat 'initial_state' + observation list style mappings as
        # a plain observation stream keyed under 'observations'.
        observations = mapping.get("observations")
        if isinstance(observations, Sequence):
            return cls._from_observations(list(observations), events=events)
        initial = mapping.get("initial_state")
        if isinstance(initial, Mapping):
            return cls._from_observations([initial], events=events)
        return []

    @classmethod
    def _snapshot_from_step(
        cls,
        step: Mapping[str, Any],
        turn: int,
        events: Sequence[Mapping[str, Any]],
    ) -> ReplaySnapshot:
        pre = step.get("pre_observation")
        observation = pre if isinstance(pre, Mapping) else {}
        env_step = _int_or_none(step.get("environment_step"))
        if env_step is None:
            env_step = _observation_step(observation, turn)
        base = _base_observation(observation)
        base.setdefault("environment_step", env_step)
        current = _current_player(observation, step.get("actor_idx"))
        base.setdefault("current_player_idx", (current,) if current is not None else ())
        base["event_sequence"] = _events_up_to(events, env_step)
        mask = _mask(step.get("action_mask") or observation.get("action_mask"))
        terminated = bool(step.get("terminated", False))
        truncated = bool(step.get("truncated", False))
        status = "normal" if terminated else "truncated" if truncated else "running"
        return ReplaySnapshot(
            observation=base,
            action_mask=mask,
            environment_step=int(env_step),
            current_player=current,
            terminated=terminated,
            truncated=truncated,
            termination_status=status,
        )

    @classmethod
    def _from_observations(
        cls,
        observations: Sequence[Any],
        *,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> list[ReplaySnapshot]:
        # When observations already embed their own event_sequence and the
        # caller passes no explicit events, use each observation's own events
        # as the cumulative source (deduplicated by recorded order).
        aggregated: list[Mapping[str, Any]] = list(events)
        if not aggregated:
            # Observations carry their own cumulative event_sequence; the last
            # (longest) one is the complete log. Using it avoids duplicating the
            # shared prefix that every earlier observation repeats.
            longest: tuple[Mapping[str, Any], ...] = ()
            for observation in observations:
                if not isinstance(observation, Mapping):
                    continue
                sequence = tuple(
                    dict(event)
                    for event in (observation.get("event_sequence") or ())
                    if isinstance(event, Mapping)
                )
                if len(sequence) >= len(longest):
                    longest = sequence
            aggregated = list(longest)
        snapshots: list[ReplaySnapshot] = []
        # Carry the last-known board keys forward (item 4): a hovered/clicked
        # moment must render a COMPLETE board even when an individual recorded
        # observation omitted, say, healths/alive. ``carried`` accumulates the
        # most recent value seen for each board key so an early snapshot missing
        # a key inherits the previous turn's value instead of rendering empty.
        # Per-turn identity keys must NOT be inherited: they describe the
        # specific turn, not the board contents, so a later turn that omits them
        # falls back to its own turn index rather than an earlier value.
        carry_forward = tuple(
            key
            for key in _OBSERVATION_KEYS
            if key not in ("environment_step", "current_player_idx")
        )
        carried: dict[str, Any] = {}
        for turn, observation in enumerate(observations):
            if not isinstance(observation, Mapping):
                continue
            merged = dict(carried)
            # Preserve non-board keys the snapshot builder reads (action_mask,
            # terminated/truncated) straight from the raw observation.
            for key, value in observation.items():
                merged[key] = value
            snapshots.append(
                cls._snapshot_from_observation(merged, turn=turn, events=aggregated)
            )
            for key in carry_forward:
                if key in observation:
                    carried[key] = observation[key]
        return snapshots

    @classmethod
    def _snapshot_from_observation(
        cls,
        observation: Mapping[str, Any],
        *,
        turn: int,
        events: Sequence[Mapping[str, Any]],
        step_override: int | None = None,
        terminal: str | None = None,
    ) -> ReplaySnapshot:
        env_step = step_override if step_override is not None else _observation_step(observation, turn)
        base = _base_observation(observation)
        base.setdefault("environment_step", env_step)
        current = _current_player(observation, None)
        if "current_player_idx" not in base:
            base["current_player_idx"] = (current,) if current is not None else ()
        base["event_sequence"] = _events_up_to(events, env_step)
        mask = _mask(observation.get("action_mask"))
        terminated = bool(observation.get("terminated", False)) or terminal == "normal"
        truncated = bool(observation.get("truncated", False)) or terminal == "truncated"
        if terminal is not None:
            status = terminal
        else:
            status = "normal" if terminated else "truncated" if truncated else "running"
        return ReplaySnapshot(
            observation=base,
            action_mask=mask,
            environment_step=int(env_step),
            current_player=current,
            terminated=terminated,
            truncated=truncated,
            termination_status=status,
        )

    @staticmethod
    def _terminal_status(mapping: Mapping[str, Any]) -> str:
        status = mapping.get("termination_status")
        if isinstance(status, str) and status:
            return status
        return "normal"

    # -- sequence protocol -------------------------------------------
    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def index(self) -> int:
        return self._index

    @property
    def snapshots(self) -> tuple[ReplaySnapshot, ...]:
        return self._snapshots

    def is_empty(self) -> bool:
        return not self._snapshots

    def is_at_end(self) -> bool:
        return not self._snapshots or self._index >= len(self._snapshots) - 1

    def is_at_start(self) -> bool:
        return self._index <= 0

    # -- stepping ----------------------------------------------------
    def _clamp(self, index: int) -> int:
        if not self._snapshots:
            return 0
        return max(0, min(int(index), len(self._snapshots) - 1))

    def seek(self, index: int) -> ReplaySnapshot | None:
        """Move the cursor to ``index`` (clamped) and return that snapshot."""

        self._index = self._clamp(index)
        return self.current()

    def next(self) -> ReplaySnapshot | None:
        """Advance one turn (clamped at the last turn)."""

        self._index = self._clamp(self._index + 1)
        return self.current()

    def prev(self) -> ReplaySnapshot | None:
        """Step back one turn (clamped at the first turn)."""

        self._index = self._clamp(self._index - 1)
        return self.current()

    def current(self) -> ReplaySnapshot | None:
        return self.snapshot_at(self._index)

    def snapshot_at(self, index: int) -> ReplaySnapshot | None:
        if not self._snapshots:
            return None
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None

    # -- moment lookup (log-row hover) -------------------------------
    def index_for_step(self, environment_step: int) -> int | None:
        """Return the snapshot index whose state represents ``environment_step``.

        Picks the last snapshot whose ``environment_step`` is <= the requested
        step, so hovering a log row shows the game *as of* that moment.  Returns
        ``None`` only when the model is empty.
        """

        if not self._snapshots:
            return None
        target = _int_or_none(environment_step)
        if target is None:
            return None
        # Prefer the FIRST snapshot whose recorded step equals the target so a
        # hovered event maps to the turn it happened on (not a later, duplicated
        # settled state carrying the same step).
        exact: int | None = None
        below: int | None = None
        for position, snapshot in enumerate(self._snapshots):
            if snapshot.environment_step == target and exact is None:
                exact = position
            elif snapshot.environment_step < target:
                below = position
        if exact is not None:
            return exact
        return below if below is not None else 0

    def index_for_event(self, event: Mapping[str, Any]) -> int | None:
        """Map a log event mapping to the snapshot index at that moment."""

        if not isinstance(event, Mapping):
            return None
        step = _event_step(event)
        if step is None:
            return None
        return self.index_for_step(step)

    def index_for_log_row(self, row: int) -> int | None:
        """Map a hovered GameLog row to the snapshot index at that moment.

        The live GameLog renders the FULL cumulative event log (that of the
        latest/longest snapshot), so a row is resolved against
        :meth:`full_events` rather than the cursor's (possibly short) current
        log.  This makes hover-preview / click-to-rewind work for every row the
        user actually sees, not just the opening turn's few events.
        """

        events = self.full_events()
        if row is None or not (0 <= int(row) < len(events)):
            return None
        return self.index_for_event(events[int(row)])

    def current_events(self) -> tuple[Mapping[str, Any], ...]:
        """Return the cumulative event log for the current snapshot."""

        snapshot = self.current()
        if snapshot is None:
            return ()
        return tuple(snapshot.observation.get("event_sequence") or ())

    def full_events(self) -> tuple[Mapping[str, Any], ...]:
        """Return the COMPLETE cumulative event log for the whole episode.

        Each snapshot carries the cumulative ``event_sequence`` up to its own
        ``environment_step``, so the longest one (the latest turn) is the full
        log the live GameLog renders.  Resolving a clicked/hovered row against
        this log -- independent of the cursor position -- is what makes rows
        past the opening turn map to the correct moment.
        """

        longest: tuple[Mapping[str, Any], ...] = ()
        for snapshot in self._snapshots:
            events = tuple(snapshot.observation.get("event_sequence") or ())
            if len(events) > len(longest):
                longest = events
        return longest

    def snapshot_for_moment(self, moment: Any) -> ReplaySnapshot | None:
        """Return the snapshot at ``moment`` (an event mapping or a step int)."""

        if isinstance(moment, Mapping):
            index = self.index_for_event(moment)
        else:
            step = _int_or_none(moment)
            index = None if step is None else self.index_for_step(step)
        return None if index is None else self.snapshot_at(index)


def _current_player(observation: Mapping[str, Any], actor: Any) -> int | None:
    actor_int = _int_or_none(actor)
    if actor_int is not None:
        return actor_int
    current = observation.get("current_player_idx")
    if isinstance(current, (list, tuple)):
        return _int_or_none(current[0]) if current else None
    return _int_or_none(current)


def _mask(value: Any) -> tuple[int, ...]:
    if value is None:
        return (0, 0, 0, 0, 0, 0)
    try:
        values = tuple(int(item) for item in value)
    except TypeError:
        return (0, 0, 0, 0, 0, 0)
    if not values:
        return (0, 0, 0, 0, 0, 0)
    return values


__all__ = ["ReplayModel", "ReplaySnapshot"]
