"""Focused tests for TASK-1-1 environment domain contracts."""

import json

import numpy as np
import pytest

from bang_rl.env import (
    Action,
    ActionType,
    EnvironmentState,
    EventType,
    GameEvent,
    Role,
    RoleVisibility,
    Team,
    TerminationStatus,
    TerminalResult,
    assign_roles,
    canonical_agent_order,
    canonical_player_order,
    initial_health,
)


def test_role_team_and_action_enum_values_are_stable() -> None:
    assert [role.value for role in Role] == [0, 1, 2, 3, 4]
    assert Role.SHERIFF.team is Team.SHERIFF_TEAM
    assert Role.DEPUTY.team is Team.SHERIFF_TEAM
    assert Role.OUTLAW_A.team is Team.OUTLAW_TEAM
    assert Role.OUTLAW_B.team is Team.OUTLAW_TEAM
    assert Role.TRAITOR.team is Team.TRAITOR_TEAM
    assert ActionType.PASS.value == 0
    assert ActionType.BANG.value == 1
    assert EventType.PASS.value == "pass"
    assert EventType.BANG.value == "bang"
    assert TerminationStatus.TRUNCATED.value == "truncated"


def test_canonical_order_and_seeded_role_permutation() -> None:
    assert canonical_player_order() == (0, 1, 2, 3, 4)
    assert canonical_agent_order() == (
        "player_0",
        "player_1",
        "player_2",
        "player_3",
        "player_4",
    )

    seed = 314159
    expected = tuple(
        Role(int(value))
        for value in np.random.Generator(np.random.PCG64(seed)).permutation(
            np.asarray(tuple(role.value for role in Role), dtype=np.uint8)
        )
    )
    assert assign_roles(seed) == expected
    assert assign_roles(seed) == assign_roles(seed)
    assert sorted(assign_roles(seed)) == sorted(Role)


def test_initial_state_has_fixed_health_and_validated_slots() -> None:
    state = EnvironmentState.initial("episode-1", 7)

    assert len(state.players) == 5
    assert sum(role is Role.SHERIFF for role in state.roles) == 1
    assert sum(role is Role.DEPUTY for role in state.roles) == 1
    assert sum(role in (Role.OUTLAW_A, Role.OUTLAW_B) for role in state.roles) == 2
    assert sum(role is Role.TRAITOR for role in state.roles) == 1
    assert state.health == initial_health(state.roles)
    assert state.health[state.sheriff_idx] == 5
    assert all(hp == 4 for index, hp in enumerate(state.health) if index != state.sheriff_idx)
    assert state.fixed_turn_order == canonical_player_order()
    assert state.termination_status is TerminationStatus.RUNNING

    with pytest.raises(ValueError):
        EnvironmentState(
            episode_id="bad",
            seed=1,
            roles=(Role.SHERIFF,) * 5,
            health=(5, 5, 5, 5, 5),
            alive=(True,) * 5,
        )


def test_action_and_event_serialization_preserve_typed_values() -> None:
    action = Action.bang(actor_idx=1, target_idx=4)
    assert Action.from_dict(action.to_dict()) == action
    assert Action.pass_action(0).to_dict()["target_idx"] is None

    event = GameEvent(
        event_id=12,
        environment_step=3,
        event_type=EventType.BANG,
        actor_idx=1,
        target_idx=4,
        hp_before=4,
        hp_after=3,
    )
    restored = GameEvent.from_dict(event.to_dict())
    assert restored == event
    assert json.loads(json.dumps(event.to_dict()))["event_id"] == 12


def test_terminal_result_and_termination_contracts() -> None:
    result = TerminalResult(
        terminated=True,
        winners=(0, 1),
        losers=(2, 3, 4),
        winning_team=Team.SHERIFF_TEAM,
        reason="all enemies eliminated",
    )
    assert TerminalResult.from_dict(result.to_dict()) == result
    assert TerminationStatus.NORMAL.value == "normal"

    with pytest.raises(ValueError):
        TerminalResult(
            terminated=False,
            winners=(),
            losers=(),
            winning_team=None,
            reason="not a normal result",
        )


def test_public_state_never_exposes_hidden_roles() -> None:
    roles = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)
    state = EnvironmentState(
        episode_id="privacy",
        seed=11,
        roles=roles,
        health=(5, 4, 4, 4, 4),
        alive=(True, True, True, True, True),
        event_log=(
            GameEvent(
                event_id=1,
                environment_step=1,
                event_type=EventType.ROLE_REVEALED,
                eliminated_idx=2,
                revealed_role=Role.OUTLAW_A,
                public=True,
            ),
            GameEvent(
                event_id=2,
                environment_step=1,
                event_type=EventType.ROLE_REVEALED,
                eliminated_idx=3,
                revealed_role=Role.OUTLAW_B,
                public=False,
            ),
        ),
    )

    public = state.to_public(viewer_idx=1)
    payload = public.to_dict()
    assert "roles" not in payload
    assert payload["players"][0]["role"] == Role.SHERIFF.value
    assert payload["players"][1]["role"] == Role.DEPUTY.value
    assert payload["players"][2]["role"] is None
    assert payload["players"][3]["role"] is None
    assert payload["players"][4]["role"] is None
    assert payload["revealed_roles"] == [0, 1, 255, 255, 255]
    assert payload["role_visibility"] == [RoleVisibility.REVEALED.value, RoleVisibility.SELF.value, 0, 0, 0]
    assert len(payload["event_log"]) == 1
    assert payload["event_log"][0]["revealed_role"] == Role.OUTLAW_A.value
    assert Role.OUTLAW_B.value not in {
        player["role"] for player in payload["players"] if player["role"] is not None
    }

    # A privileged serialization retains the true roles and is distinct from
    # the public representation used for agent observations.
    privileged = state.to_privileged_dict()
    assert privileged["roles"] == [role.value for role in roles]
    assert privileged["event_log"][1]["revealed_role"] == Role.OUTLAW_B.value
    assert json.loads(public.to_json())["players"][4]["role"] is None


def test_deputy_viewer_with_all_alive_only_sees_self_and_sheriff() -> None:
    # Game rule: only the Sheriff is public. A living Deputy must never learn any
    # other living player's role (no elimination-of-alternatives deduction).
    roles = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)
    deputy_idx = 1
    state = EnvironmentState(
        episode_id="deputy-privacy",
        seed=7,
        roles=roles,
        health=(5, 4, 4, 4, 4),
        alive=(True, True, True, True, True),
    )

    public = state.to_public(viewer_idx=deputy_idx)

    assert public.role_visibility[deputy_idx] == RoleVisibility.SELF
    assert public.role_visibility[state.sheriff_idx] == RoleVisibility.REVEALED
    for other in (2, 3, 4):
        assert public.role_visibility[other] == RoleVisibility.HIDDEN
        assert public.revealed_roles[other] == 255
        assert public.players[other].role is None
    # No living non-self, non-sheriff role is ever CONFIRMED/REVEALED.
    assert RoleVisibility.CONFIRMED not in public.role_visibility


def test_public_and_privileged_state_round_trip() -> None:
    state = EnvironmentState.initial("round-trip", 99)
    assert EnvironmentState.from_dict(state.to_dict()) == state
    public = state.to_public(viewer_idx=state.agent_selection)
    assert type(public).from_dict(public.to_dict()) == public


def test_initial_state_always_starts_on_the_sheriff_seat() -> None:
    # Item 2: the game must always start on the Sheriff's seat (then proceed
    # clockwise via the environment's ascending-seat advance).
    for seed in (0, 1, 7, 42, 99, 314159, 271828):
        state = EnvironmentState.initial("turn-order", seed)
        sheriff_seat = state.roles.index(Role.SHERIFF)
        assert state.agent_selection == sheriff_seat
        assert state.agent_selection == state.sheriff_idx
        # The canonical turn order is preserved (ascending == clockwise).
        assert state.fixed_turn_order == (0, 1, 2, 3, 4)
