"""Convert trusted full GUI replays into canonical training episodes.

A ``full`` replay is an archival format: it contains authoritative states and
DummyAgent internals.  This module is the explicit trust boundary for using
those files as supervised-training data.  Authoritative roles and emotion
values become targets only; model inputs are regenerated from the public
projection of each saved state and from public action/event deltas.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bang_rl.data.collector import CanonicalEpisode, CanonicalStep
from bang_rl.env import EnvironmentState
from bang_rl.exceptions import DataError

PLAYER_COUNT = 5


def _canonical_seed(value: Any) -> int:
    """Store uint64 replay seeds in the canonical signed int64 column."""

    seed = int(value)
    if seed >= 2**63:
        seed -= 2**64
    if seed < -(2**63) or seed >= 2**63:
        raise DataError("full replay seed does not fit the canonical int64 seed field")
    return seed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataError(f"full replay {name} must be a mapping")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataError(f"full replay {name} must be a list")
    return value


def _event_dicts(state: EnvironmentState) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(event.to_public_dict() if hasattr(event, "to_public_dict") else event.to_dict())
        for event in state.event_log
        if bool(getattr(event, "public", True))
    )


def _public_observation(state: EnvironmentState) -> dict[str, Any]:
    viewer = int(state.agent_selection)
    public_state = state.to_public(viewer)
    public = (
        public_state.to_observation_dict()
        if hasattr(public_state, "to_observation_dict")
        else public_state.to_dict()
    )
    # Full-state serialization is never allowed to become model input.
    for key in ("roles", "true_roles", "private_state", "privileged_state"):
        public.pop(key, None)
    return public


def _action_event(delta: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    actions = [
        event for event in delta
        if str(event.get("event_type", "")).lower() in {"pass", "bang"}
    ]
    if len(actions) > 1:
        raise DataError("full replay contains more than one action in an environment-step delta")
    return actions[0] if actions else None


def _emotion_labels(internals: Any, *, environment_step: int, observation: Mapping[str, Any], event_ids: tuple[int, ...]) -> dict[str, Any]:
    if not isinstance(internals, Mapping):
        return {}
    seats = internals.get("seats")
    if not isinstance(seats, Mapping):
        return {}
    labels: dict[str, Any] = {}
    for player in range(PLAYER_COUNT):
        descriptor = seats.get(str(player), seats.get(player))
        if not isinstance(descriptor, Mapping) or str(descriptor.get("kind", "")).lower() != "dummy":
            continue
        state = descriptor.get("state")
        emotion = state.get("emotion_module") if isinstance(state, Mapping) else None
        if not isinstance(emotion, Mapping) or not isinstance(emotion.get("values"), list):
            continue
        values = [float(item) for item in emotion["values"]]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise DataError("full replay emotion labels must be normalized to [0, 1]")
        labels[str(player)] = {
            "episode_id": observation.get("episode_id"),
            "environment_step": int(environment_step),
            "player_idx": player,
            "values": values,
            "emotion_values": values,
            "emotion_names": list(emotion.get("emotion_names", ())),
            "emotion_schema_version": str(emotion.get("schema_version", "full-replay")),
            "input_event_ids": list(event_ids),
            "public_observation": dict(observation),
        }
    return labels


def full_replay_to_canonical(
    replay: Mapping[str, Any] | str | Path,
    *,
    strict: bool = True,
) -> CanonicalEpisode:
    """Convert one full replay into one canonical, training-safe episode.

    The replay must contain index-parallel ``states`` and ``observations``.
    Each canonical step is built from a public state projection before an
    action and the public action/event delta leading to the next state.
    ``internals`` are optional targets; missing exports produce missing emotion
    labels rather than fabricated zeros.
    """

    if isinstance(replay, (str, Path)):
        try:
            with Path(replay).open("r", encoding="utf-8") as handle:
                replay = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot read full replay {replay}") from exc
    record = _mapping(replay, "record")
    if record.get("replay_format") != "full":
        raise DataError("training conversion requires replay_format='full'")
    states_raw = _list(record.get("states"), "states")
    observations_raw = _list(record.get("observations"), "observations")
    if len(states_raw) != len(observations_raw) or len(states_raw) < 2:
        raise DataError("full replay states and observations must be index-parallel and contain an action")
    internals_raw = record.get("internals")
    if internals_raw is not None:
        internals = _list(internals_raw, "internals")
        if len(internals) != len(states_raw):
            raise DataError("full replay internals must align with states")
    else:
        internals = [None] * len(states_raw)

    states = [EnvironmentState.from_dict(_mapping(item, "state")) for item in states_raw]
    if strict:
        for previous, current in zip(states, states[1:]):
            if int(current.environment_step) < int(previous.environment_step):
                raise DataError("full replay environment steps are not monotonic")
    observations = [_public_observation(state) for state in states]
    events_by_state = [_event_dicts(state) for state in states]
    steps: list[CanonicalStep] = []
    for index, (state, next_state) in enumerate(zip(states, states[1:])):
        before = events_by_state[index]
        after = events_by_state[index + 1]
        before_ids = {int(item.get("event_id")) for item in before if item.get("event_id") is not None}
        delta = tuple(item for item in after if item.get("event_id") is not None and int(item["event_id"]) not in before_ids)
        action_event = _action_event(delta)
        if action_event is None:
            if strict:
                raise DataError(f"full replay turn {index} has no public action event")
            continue
        action_type = str(action_event.get("event_type", "")).lower()
        actor = int(action_event.get("actor_idx", state.agent_selection))
        target = action_event.get("target_idx")
        target = None if target is None else int(target)
        action = 0 if action_type == "pass" else target + 1 if target is not None else None
        if action is None or not 0 <= action < 6:
            raise DataError(f"full replay turn {index} has an invalid action")
        event_ids = tuple(int(item["event_id"]) for item in delta if item.get("event_id") is not None)
        pre_event_ids = tuple(int(item["event_id"]) for item in before if item.get("event_id") is not None)
        hp_before = action_event.get("hp_before")
        hp_after = action_event.get("hp_after")
        eliminated = next((item.get("eliminated_idx") for item in delta if item.get("eliminated_idx") is not None), None)
        labels = _emotion_labels(
            internals[index],
            environment_step=int(state.environment_step),
            observation=observations[index],
            event_ids=pre_event_ids,
        )
        steps.append(CanonicalStep(
            environment_step=int(state.environment_step),
            actor_idx=actor,
            action=int(action),
            action_type=action_type,
            target_idx=target,
            action_mask=tuple(int(item) for item in observations[index].get("action_mask", (1, 1, 1, 1, 1, 1))),
            pre_observation=observations[index],
            post_observation=observations[index + 1],
            pre_event_ids=pre_event_ids,
            event_ids=event_ids,
            hp_before=None if hp_before is None else int(hp_before),
            hp_after=None if hp_after is None else int(hp_after),
            eliminated_idx=None if eliminated is None else int(eliminated),
            reward=0.0,
            emotion_labels=labels,
            terminated=bool(next_state.terminated),
            truncated=bool(next_state.truncated),
        ))
    if not steps:
        raise DataError("full replay produced no canonical action steps")
    final = states[-1]
    terminal = None if final.terminal_result is None else final.terminal_result.to_dict()
    return CanonicalEpisode(
        episode_id=str(record.get("episode_id") or f"full-replay-{record.get('seed') or states[0].seed}"),
        seed=_canonical_seed(record.get("seed") or states[0].seed),
        agent_order=tuple(range(PLAYER_COUNT)),
        metadata={
            "source": "full_replay",
            "replay_format": "full",
            "mode": record.get("mode", record.get("game_type")),
            "emotion_schema_version": "full-replay",
            "observation_schema_version": "1.0",
            "role_label_schema": "environment_role_index",
        },
        initial_state=observations[0],
        steps=tuple(steps),
        events=events_by_state[-1],
        termination_status="truncated" if final.truncated else "normal" if final.terminated else "running",
        termination_reason="max_episode_steps_reached" if final.truncated else None,
        terminal_result=terminal,
        final_state=observations[-1],
        role_labels=tuple(int(role) for role in final.roles),
    )


def load_full_replay_episodes(paths: Iterable[str | Path]) -> tuple[CanonicalEpisode, ...]:
    return tuple(full_replay_to_canonical(path) for path in paths)


__all__ = ["full_replay_to_canonical", "load_full_replay_episodes"]
