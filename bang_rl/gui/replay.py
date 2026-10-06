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
from dataclasses import dataclass, field, replace
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
    # Item 1: the finished game's terminal role of every seat. Defaults to EMPTY
    # so every running / mid-replay snapshot and every legacy construction
    # carries NO roles -> no mid-replay leak. ReplayModel sets it ONLY on the
    # terminal snapshot (from the record's top-level final_roles) so a replay
    # reveals all roles at its END. GraphicalBoard reads it via
    # getattr(state, "final_roles", ()) exactly like the live terminal
    # PublicGameState, so a terminal replay snapshot reveals every seat.
    final_roles: tuple[int, ...] = ()
    # FEAT-002: the per-turn AI internals captured for this moment, shaped like
    # GameService.export_agent_internals ({"health": [...], "seats": {seat_str:
    # descriptor}}). Defaults to None so legacy snapshots, every observation
    # stream construction, and all existing tests carry NO internals (the board
    # then renders no emotion bars / "no inference info" on hover). ReplayModel
    # attaches the record's top-level ``internals`` list index-parallel; the
    # spectator / replay board reads it to draw emotion bars and the hover
    # role-inference overlay. Qt-free and numpy-free: it is a plain mapping.
    internals: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LogSnapshot:
    """A log-only snapshot carrying the COMPLETE cumulative event log.

    Item 2: during replay playback the LEFT game log must show the ENTIRE
    episode log from the very first frame while the graphical BOARD plays from
    turn 1. The board renders the stepped :class:`ReplaySnapshot` (its
    cumulative ``event_sequence`` grows turn-by-turn), but the log is rendered
    from this snapshot instead so every turn is visible at once.

    It duck-types exactly what
    :meth:`bang_rl.gui.widgets.game_widgets.GameLog.update_state` reads:
    ``getattr(state, "observation", None)`` then ``observation.get(
    "event_sequence")``. It is Qt-free and numpy-free, carrying only the full
    log mapping and no board-state keys (the log never reads them).
    """

    observation: Mapping[str, Any]


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
    # ``environment_step``) as a 1-element sequence such as ``[3]`` because the
    # public projection (``make_observation_space``) stores them as ``Box``
    # arrays of shape ``(1,)``. The LIVE GameScreen records ``state.observation``
    # verbatim, so ``environment_step`` arrives as a ``numpy.ndarray`` (not a
    # ``list``/``tuple``); a saved-and-reloaded public replay may instead carry
    # a plain 1-element list. Unwrap BOTH forms (numpy arrays via ``.tolist()``,
    # plain sequences directly) before ``int()`` so the value is read as a
    # scalar instead of collapsing to ``None`` and dragging every snapshot back
    # to step 0 -- the confirmed root cause of hover/click always showing turn 0
    # (items 3 and 8). Numpy is NOT imported here (this module stays Qt-free and
    # numpy-free); duck-typing ``.tolist()`` keeps it importable in the sandbox.
    if value is None:
        return None
    # numpy scalars/arrays and other array-likes expose ``.tolist()`` which
    # returns a builtin int (scalar) or a builtin list (array) with no numpy
    # dependency here.
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (list, tuple, dict)):
        try:
            value = tolist()
        except Exception:
            return None
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
        # Item 1: the record's TERMINAL-ONLY reveal. Read once here; it is
        # applied to the terminal snapshot only (never a running one) so a
        # replay reveals all roles at its END without any mid-replay leak. A
        # legacy record without the field yields an empty tuple -> no reveal.
        final_roles = cls._record_final_roles(mapping)
        # FEAT-002: the record's top-level per-turn internals list, index-
        # parallel to steps / observations (may contain None placeholders). A
        # legacy record without the key yields an empty tuple, so every snapshot
        # keeps internals=None (graceful: no bars / "no inference info").
        internals = cls._record_internals(mapping)
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
                        final_roles=final_roles,
                    )
                )
            else:
                # No appended settled state: the reveal belongs on the last
                # snapshot only when it is itself terminal (terminated/truncated).
                cls._apply_terminal_reveal(snapshots, final_roles)
            cls._attach_internals(snapshots, internals)
            return snapshots
        # No 'steps': treat 'initial_state' + observation list style mappings as
        # a plain observation stream keyed under 'observations'.
        observations = mapping.get("observations")
        if isinstance(observations, Sequence):
            snapshots = cls._from_observations(list(observations), events=events)
            cls._apply_terminal_reveal(snapshots, final_roles)
            cls._attach_internals(snapshots, internals)
            return snapshots
        initial = mapping.get("initial_state")
        if isinstance(initial, Mapping):
            snapshots = cls._from_observations([initial], events=events)
            cls._apply_terminal_reveal(snapshots, final_roles)
            cls._attach_internals(snapshots, internals)
            return snapshots
        return []

    @staticmethod
    def _record_internals(mapping: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the record's top-level ``internals`` list as a tuple.

        FEAT-002: the full replay record carries an ``internals`` list that is
        index-parallel to ``steps`` / ``observations`` and may hold ``None``
        placeholders where a per-turn export was missed. Returns an EMPTY tuple
        when the field is absent / not a list (a legacy or public record), so
        every snapshot then keeps ``internals=None``.
        """

        raw = mapping.get("internals")
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(raw)

    @staticmethod
    def _attach_internals(
        snapshots: list[ReplaySnapshot], internals: tuple[Any, ...]
    ) -> None:
        """Attach each record internals entry to its snapshot by index.

        FEAT-002: snapshot ``i`` receives ``internals[i]`` when it exists and is
        a mapping; otherwise it keeps ``internals=None``. Length mismatches are
        guarded (a trailing appended ``final_state`` snapshot, or a shorter
        internals list, simply leaves the extra snapshots with ``None``), so a
        legacy record without internals and a record with ``None`` placeholders
        both degrade gracefully. Replaces the frozen snapshot in place.
        """

        if not internals:
            return
        for index in range(min(len(snapshots), len(internals))):
            entry = internals[index]
            if isinstance(entry, Mapping):
                snapshots[index] = replace(snapshots[index], internals=entry)

    @staticmethod
    def _record_final_roles(mapping: Mapping[str, Any]) -> tuple[int, ...]:
        """Parse the record's top-level ``final_roles`` into a tuple of ints.

        Returns an EMPTY tuple when the field is absent/empty/unparsable, so a
        legacy record (or any record saved before this feature) carries no
        reveal and every snapshot's ``final_roles`` stays empty.
        """

        raw = mapping.get("final_roles")
        if not isinstance(raw, (list, tuple)) or not raw:
            return ()
        roles: list[int] = []
        for value in raw:
            parsed = _int_or_none(value)
            if parsed is None:
                return ()
            roles.append(parsed)
        return tuple(roles)

    @staticmethod
    def _apply_terminal_reveal(
        snapshots: list[ReplaySnapshot], final_roles: tuple[int, ...]
    ) -> None:
        """Attach ``final_roles`` to the last snapshot IFF it is terminal.

        A running last snapshot keeps ``final_roles == ()`` so a non-terminal
        snapshot never reports a reveal. Replaces the frozen snapshot in place.
        """

        if not final_roles or not snapshots:
            return
        last = snapshots[-1]
        if not (last.terminated or last.truncated):
            return
        snapshots[-1] = replace(last, final_roles=final_roles)

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
        final_roles: tuple[int, ...] = (),
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
        # Item 1: the TERMINAL-ONLY reveal is honoured only when this snapshot
        # is actually terminal, so an appended settled state carrying final_roles
        # that somehow is not terminal never leaks a role.
        reveal = tuple(final_roles) if (final_roles and (terminated or truncated)) else ()
        return ReplaySnapshot(
            observation=base,
            action_mask=mask,
            environment_step=int(env_step),
            current_player=current,
            terminated=terminated,
            truncated=truncated,
            termination_status=status,
            final_roles=reveal,
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

        ROBUSTNESS (items 3 & 8): the row -> snapshot mapping is computed for the
        WHOLE log at once (:meth:`_row_to_index_map`) with a monotonic guarantee:
        a later row never maps to an earlier snapshot, and a row past the first
        never collapses back to index 0.  When per-event ``environment_step``
        data lines up with the recorded snapshots this reproduces the exact
        step-based mapping; when the step data is missing, constant, or ambiguous
        (so a naive per-event lookup would drag every row to turn 0) it falls
        back to a positional distribution of rows across the recorded snapshots.
        Either way a hovered/clicked mid or late row lands on its own moment
        instead of resetting the board to the opening turn.
        """

        events = self.full_events()
        if row is None:
            return None
        try:
            row_int = int(row)
        except (TypeError, ValueError):
            return None
        if not (0 <= row_int < len(events)):
            return None
        return self._row_to_index_map()[row_int]

    def log_row_for_index(self, index: int) -> int | None:
        """Return the first full-log row that maps to snapshot ``index``.

        FEAT-004 (item 5): the inverse of :meth:`index_for_log_row`, used to
        highlight the currently playing / seeked step in the log. It mirrors
        :meth:`index_for_log_row` exactly by scanning the SAME monotonic
        ``row -> snapshot index`` mapping (:meth:`_row_to_index_map`) and
        returning the first row whose snapshot index equals ``index`` (clamped
        into range). Because the mapping is non-decreasing, the first matching
        row is the earliest log row representing that snapshot's moment, so the
        highlight lands on the row the hover-preview / click would resolve back
        to the same snapshot (round-trip stable). Returns ``None`` for an empty
        model. Pure, Qt-free and numpy-free.
        """

        if not self._snapshots:
            return None
        clamped = self._clamp(index)
        mapping = self._row_to_index_map()
        if not mapping:
            return None
        for row, snapshot_index in enumerate(mapping):
            if snapshot_index == clamped:
                return row
        # No row maps exactly to this snapshot (e.g. an appended settled state
        # with no distinct log row): fall back to the last row whose snapshot
        # index is <= the target so the highlight tracks the nearest earlier
        # moment instead of vanishing.
        fallback: int | None = None
        for row, snapshot_index in enumerate(mapping):
            if snapshot_index <= clamped:
                fallback = row
        return fallback

    def _row_to_index_map(self) -> tuple[int, ...]:
        """Return a monotonic ``row -> snapshot index`` mapping for the log.

        Each entry is the index of the snapshot representing the game *as of*
        that log row.  The mapping is non-decreasing (a later row maps to a
        later-or-equal snapshot) and only the row(s) belonging to the opening
        turn map to index 0, so a mid/late row can never collapse to turn 0.

        Two signals drive the mapping, preferring the authoritative one:

        * When events carry a usable ``environment_step`` AND the recorded
          snapshots carry distinct steps, each row resolves to the first
          snapshot whose step is >= the event's step (clamped monotonically),
          which matches :meth:`index_for_step`'s intent exactly.
        * Otherwise (steps missing/constant/ambiguous) rows are spread across
          the snapshots by position so the mapping stays monotonic and the last
          row reaches the final snapshot.
        """

        events = self.full_events()
        count = len(events)
        snapshot_count = len(self._snapshots)
        if count == 0 or snapshot_count == 0:
            return ()
        if snapshot_count == 1:
            return (0,) * count

        snapshot_steps = [snapshot.environment_step for snapshot in self._snapshots]
        distinct_steps = len(set(snapshot_steps)) > 1
        event_steps = [_event_step(event) for event in events]
        usable_event_steps = distinct_steps and all(step is not None for step in event_steps)

        mapping: list[int] = []
        if usable_event_steps:
            # Step-driven mapping: the first snapshot at or beyond the event's
            # step, kept monotonic so duplicated/settled steps never regress.
            cursor = 0
            for step in event_steps:
                target = int(step) if step is not None else snapshot_steps[cursor]
                while (
                    cursor + 1 < snapshot_count
                    and snapshot_steps[cursor] < target
                ):
                    cursor += 1
                mapping.append(cursor)
            if any(value > 0 for value in mapping):
                return tuple(mapping)
            # Every row mapped to 0 despite "usable" steps (e.g. the snapshots'
            # steps are all >= the events' steps): fall through to positional.
            mapping = []

        # Positional fallback: distribute the rows across the snapshots so a
        # later row maps to a later-or-equal snapshot and the final row reaches
        # the last snapshot. Guarantees no mid/late row collapses to turn 0.
        if count == 1:
            return (0,)
        last_snapshot = snapshot_count - 1
        last_row = count - 1
        for row in range(count):
            index = (row * last_snapshot) // last_row
            if index > last_snapshot:
                index = last_snapshot
            mapping.append(index)
        return tuple(mapping)

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

    def full_log_snapshot(self) -> LogSnapshot:
        """Return a log-only snapshot carrying the COMPLETE episode log.

        Item 2: the returned snapshot's ``observation['event_sequence']`` is
        :meth:`full_events` (the longest cumulative log), independent of the
        cursor ``index``, so the live GameLog can render the ENTIRE log from the
        very first replay frame while the board keeps stepping from turn 1.
        Because :meth:`GameLog.update_state` never shrinks on a shorter incoming
        sequence (the ``merge_rendered_log`` contract), feeding this once keeps
        the full log stable as the board advances. For an empty model the
        ``event_sequence`` is ``()``. Pure, Qt-free and numpy-free.
        """

        return LogSnapshot(observation={"event_sequence": self.full_events()})

    def snapshot_for_moment(self, moment: Any) -> ReplaySnapshot | None:
        """Return the snapshot at ``moment`` (an event mapping or a step int)."""

        if isinstance(moment, Mapping):
            index = self.index_for_event(moment)
        else:
            step = _int_or_none(moment)
            index = None if step is None else self.index_for_step(step)
        return None if index is None else self.snapshot_at(index)


def merge_rendered_log(
    rendered: Sequence[Any], incoming: Sequence[Any]
) -> tuple[tuple[Any, ...], bool]:
    """Pure form of the GameLog "never shrink on hover" render contract.

    Given the rows currently rendered by the log and the ``incoming``
    cumulative ``event_sequence`` from a new snapshot, return
    ``(rows_to_render, grew)`` where:

    * ``rows_to_render`` is the FULL set of rows the log should show. When the
      incoming sequence is the same length or longer (live play growing the
      log) it IS the incoming sequence. When the incoming sequence is SHORTER
      (a hover preview carrying a truncated event_sequence) the previously
      rendered rows are KEPT so later rows never disappear (items 2 & 3).
    * ``grew`` is True only when rows were actually appended, so the caller
      auto-scrolls to the bottom only for live play and never on a hover
      preview (keeping the hovered row under the cursor, item 3).

    This mirrors :meth:`bang_rl.gui.widgets.game_widgets.GameLog.update_state`
    so the behavioural invariant is testable headlessly (no Qt).
    """

    rendered_rows = tuple(rendered)
    incoming_rows = tuple(incoming)
    if len(incoming_rows) < len(rendered_rows):
        # Hover preview: keep the full rendered log, do not shrink or scroll.
        return rendered_rows, False
    grew = len(incoming_rows) > len(rendered_rows)
    return incoming_rows, grew


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


__all__ = ["LogSnapshot", "ReplayModel", "ReplaySnapshot", "merge_rendered_log"]
