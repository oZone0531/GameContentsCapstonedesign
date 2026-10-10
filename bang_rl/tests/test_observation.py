"""Focused tests for TASK-1-5 public observations and role disclosure."""

from __future__ import annotations

import numpy as np
import pytest

from bang_rl.config import EnvConfig
from bang_rl.env import (
    BangEnvironment,
    EnvironmentState,
    EventType,
    GameEvent,
    HIDDEN_ROLE_SENTINEL,
    PublicEnvironmentState,
    PublicPlayerState,
    Role,
    RoleVisibility,
)


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def make_state(**overrides) -> EnvironmentState:
    values = {
        "episode_id": "observation-episode",
        "seed": 7,
        "roles": ROLES,
        "health": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "agent_selection": 0,
    }
    values.update(overrides)
    return EnvironmentState(**values)


def test_observation_has_fixed_numpy_shapes_dtypes_mask_and_schema() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)
    observation = environment.observe(agent=1)
    values = observation.to_observation_dict()

    assert observation.observation_schema_version == "1.0"
    assert values["current_player_idx"].shape == (1,)
    assert values["current_player_idx"].dtype == np.uint8
    assert values["environment_step"].shape == (1,)
    assert values["environment_step"].dtype == np.uint32
    assert values["healths"].shape == (5,)
    assert values["healths"].dtype == np.uint8
    assert values["alive"].shape == (5,)
    assert values["alive"].dtype == np.bool_
    assert values["turn_order"].shape == (5,)
    assert values["turn_order"].dtype == np.uint8
    assert values["role_visibility"].shape == (5,)
    assert values["revealed_roles"].shape == (5,)
    assert values["confirmed_role_distribution"].shape == (5, 5)
    assert values["confirmed_role_distribution"].dtype == np.float32
    assert values["action_mask"].shape == (6,)
    assert values["action_mask"].dtype == np.uint8
    np.testing.assert_array_equal(values["action_mask"], [1, 0, 1, 1, 1, 1])


def test_each_viewer_gets_self_and_public_sheriff_but_not_hidden_roles() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)

    for viewer_idx in range(5):
        observation = environment.observe(agent=viewer_idx)
        values = observation.to_observation_dict()
        assert values["self_role"].tolist() == [int(ROLES[viewer_idx])]
        assert values["sheriff_idx"].tolist() == [0]
        assert values["revealed_roles"][0] == Role.SHERIFF
        assert values["revealed_roles"][viewer_idx] == ROLES[viewer_idx]
        assert values["role_visibility"][viewer_idx] == RoleVisibility.SELF
        for target_idx in range(1, 5):
            if target_idx != viewer_idx:
                assert values["revealed_roles"][target_idx] == HIDDEN_ROLE_SENTINEL
                assert values["role_visibility"][target_idx] == RoleVisibility.HIDDEN
                assert not np.any(values["confirmed_role_distribution"][target_idx])

        serialized = observation.to_dict()
        assert "roles" not in serialized
        assert all(player["role"] is None for player in serialized["players"][1:] if player["player_idx"] != viewer_idx)


def test_elimination_creates_public_role_reveal_for_every_viewer() -> None:
    state = make_state(health=(5, 1, 4, 4, 4), alive=(True, True, True, True, True))
    environment = BangEnvironment(state=state, max_episode_steps=None)
    result = environment.step(2)

    assert [event.event_type for event in result.events] == [
        EventType.BANG,
        EventType.PLAYER_ELIMINATED,
        EventType.ROLE_REVEALED,
    ]
    reveal = result.events[-1]
    assert reveal.public is True
    assert reveal.eliminated_idx == 1
    assert reveal.revealed_role == Role.DEPUTY

    for viewer_idx in range(5):
        observation = environment.observe(agent=viewer_idx)
        values = observation.to_observation_dict()
        assert values["revealed_roles"][1] == Role.DEPUTY
        assert values["confirmed_role_distribution"][1].tolist() == [0, 1, 0, 0, 0]
        assert values["role_visibility"][1] == (
            RoleVisibility.SELF if viewer_idx == 1 else RoleVisibility.REVEALED
        )
        assert values["event_sequence"][-1]["revealed_role"] == Role.DEPUTY


def test_living_non_sheriff_role_is_never_revealed_by_deduction() -> None:
    # Game rule: only the Sheriff, the viewer's own seat and eliminated players
    # are public.  Even when the last living seat's role could be deduced by
    # elimination (here: the Traitor), the environment does NOT reveal it; any
    # deduction is left to the agent's own role-belief estimator.
    state = make_state(
        health=(5, 0, 0, 0, 4),
        alive=(True, False, False, False, True),
        agent_selection=0,
    )
    environment = BangEnvironment(state=state, max_episode_steps=None)
    values = environment.observe(agent=0).to_observation_dict()

    assert values["revealed_roles"].tolist() == [0, 1, 2, 3, HIDDEN_ROLE_SENTINEL]
    assert values["role_visibility"].tolist() == [1, 2, 2, 2, 0]
    assert values["confirmed_role_distribution"][4].tolist() == [0, 0, 0, 0, 0]


def test_event_history_is_bounded_only_in_observation_and_reset_step_integrate_it() -> None:
    config = EnvConfig(
        max_episode_steps=100,
        warn_turn_threshold=10,
        critical_turn_threshold=20,
        event_history={"max_events": 2},
    )
    environment = BangEnvironment(config=config, episode_id="configured", seed=9)
    reset_observation = environment.reset(episode_id="reset", seed=10)
    assert reset_observation["environment_step"].tolist() == [0]
    assert reset_observation["event_sequence"] == ()

    for _ in range(3):
        environment.step(0)
    assert len(environment.state.event_log) == 3
    assert len(environment.observe()["event_sequence"]) == 2
    assert [event["environment_step"] for event in environment.observe()["event_sequence"]] == [2, 3]
    assert environment.step(0).observation["action_mask"].dtype == np.uint8


def test_public_event_projection_is_shared_but_observation_containers_are_detached() -> None:
    events = tuple(
        GameEvent(
            event_id=index + 1,
            environment_step=index + 1,
            event_type=EventType.PASS,
            actor_idx=index % 5,
        )
        for index in range(2_000)
    )
    state = make_state(event_log=events)
    first = state.to_public(viewer_idx=0)
    second = state.to_public(viewer_idx=1)

    assert first.event_log is second.event_log
    assert len(first.event_sequence) == 2_000
    assert first.event_sequence[0].event_id == 1
    assert first.event_sequence[-1].event_id == 2_000

    first_values = first.to_observation_dict()
    first_values["event_sequence"][0]["event_id"] = 999_999
    assert second.to_observation_dict()["event_sequence"][0]["event_id"] == 1



    environment = BangEnvironment(state=make_state(), max_episode_steps=None)
    space = environment.observation_space
    values = environment.observe().to_observation_dict()
    fixed_values = {key: value for key, value in values.items() if key not in {"event_sequence", "event_history", "observation_schema_version"}}

    assert space.contains(fixed_values)
    assert space["action_mask"].shape == (6,)
    assert space["action_mask"].dtype == np.uint8
    assert space["confirmed_role_distribution"].shape == (5, 5)




def test_public_observation_serializes_long_history_once_and_detaches_mappings(monkeypatch) -> None:
    events = tuple(
        GameEvent(
            event_id=index + 1,
            environment_step=index + 1,
            event_type=EventType.PASS,
            actor_idx=index % 5,
        )
        for index in range(2_000)
    )
    state = make_state(event_log=events)
    calls = 0
    original = GameEvent.to_public_dict

    def counted(event):
        nonlocal calls
        calls += 1
        return original(event)

    monkeypatch.setattr(GameEvent, "to_public_dict", counted)
    # EnvironmentState owns the serialized projection.  Constructing a viewer
    # DTO must consume that trusted payload rather than reserializing events.
    observation = state.to_public(viewer_idx=0)
    assert calls == 0

    first = observation.to_observation_dict()
    second = observation.to_observation_dict()
    _ = observation["event_history"]
    _ = tuple(observation.keys())
    assert calls == 0

    first["event_sequence"][0]["event_id"] = 999_999
    assert second["event_sequence"][0]["event_id"] == 1
    assert observation["event_sequence"][0]["event_id"] == 1


def test_unlimited_event_history_preserves_all_public_events() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)

    for _ in range(4):
        environment.step(0)

    observation = environment.observe()
    assert len(environment.state.event_log) == 4
    assert len(observation.event_sequence) == 4
    assert [event["event_id"] for event in observation["event_sequence"]] == [1, 2, 3, 4]


def test_public_state_rejects_private_events_and_accepts_integer_role_inputs() -> None:
    # Plain integer role inputs are normalized before the public projection so
    # role identity comparisons cannot accidentally expose or omit a role.
    integer_state = make_state(roles=tuple(int(role) for role in ROLES))
    observation = BangEnvironment(state=integer_state, max_episode_steps=None).observe(agent=1)
    assert observation["self_role"].tolist() == [int(Role.DEPUTY)]
    assert observation["confirmed_role_distribution"][0].tolist() == [1, 0, 0, 0, 0]

    private_event = GameEvent(
        event_id=1,
        environment_step=1,
        event_type=EventType.ROLE_REVEALED,
        eliminated_idx=1,
        revealed_role=Role.DEPUTY,
        public=False,
    )
    with pytest.raises(ValueError, match="private events"):
        PublicEnvironmentState(
            episode_id="public-boundary",
            current_player_idx=0,
            fixed_turn_order=(0, 1, 2, 3, 4),
            environment_step=1,
            players=tuple(
                PublicPlayerState(
                    player_idx=index,
                    health=5 if index == 0 else 4,
                    alive=True,
                )
                for index in range(5)
            ),
            sheriff_idx=0,
            role_visibility=(RoleVisibility.SELF, RoleVisibility.HIDDEN, RoleVisibility.HIDDEN, RoleVisibility.HIDDEN, RoleVisibility.HIDDEN),
            revealed_roles=(0, HIDDEN_ROLE_SENTINEL, HIDDEN_ROLE_SENTINEL, HIDDEN_ROLE_SENTINEL, HIDDEN_ROLE_SENTINEL),
            confirmed_role_distribution=((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)),
            event_log=(private_event,),
            viewer_idx=0,
        )


def test_public_state_to_dict_reuses_cached_event_payload(monkeypatch) -> None:
    events = tuple(
        GameEvent(
            event_id=index + 1,
            environment_step=index + 1,
            event_type=EventType.PASS,
            actor_idx=index % 5,
        )
        for index in range(2_000)
    )
    observation = make_state(event_log=events).to_public(viewer_idx=0)
    calls = 0
    original = GameEvent.to_public_dict

    def counted(event):
        nonlocal calls
        calls += 1
        return original(event)

    monkeypatch.setattr(GameEvent, "to_public_dict", counted)
    payload = observation.to_dict()
    assert calls == 0
    assert payload["event_log"] == payload["event_sequence"]
    payload["event_log"][0]["event_id"] = 999_999
    assert observation.event_log[0].event_id == 1


def test_external_public_event_cache_must_match_event_log() -> None:
    event = GameEvent(
        event_id=1,
        environment_step=1,
        event_type=EventType.PASS,
        actor_idx=0,
    )
    with pytest.raises(ValueError, match="public event projection"):
        make_state(event_log=(event,), _public_event_log=())

    private_event = GameEvent(
        event_id=2,
        environment_step=1,
        event_type=EventType.ROLE_REVEALED,
        actor_idx=0,
        eliminated_idx=1,
        revealed_role=Role.DEPUTY,
        public=False,
    )
    with pytest.raises(ValueError, match="only public"):
        make_state(event_log=(event, private_event), _public_event_log=(private_event,))
