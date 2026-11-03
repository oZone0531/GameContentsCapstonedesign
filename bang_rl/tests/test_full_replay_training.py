from __future__ import annotations

from bang_rl.agents.main_agent.role_taxonomy import semantic_role_labels
from bang_rl.data.full_replay import full_replay_to_canonical
from bang_rl.env import (
    EnvironmentState,
    EventType,
    GameEvent,
    Role,
    Team,
    TerminalResult,
    TerminationStatus,
)


def _state(step: int, event_log=(), *, terminated: bool = False) -> dict:
    terminal = None
    if terminated:
        terminal = TerminalResult(
            terminated=True,
            winners=(0,),
            losers=(1,),
            winning_team=Team.SHERIFF_TEAM,
            reason="test",
        )
    state = EnvironmentState(
        episode_id="full-test",
        seed=7,
        roles=(Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR),
        health=(5, 3, 4, 4, 4) if step else (5, 4, 4, 4, 4),
        alive=(True, True, True, True, True),
        agent_selection=1,
        environment_step=step,
        terminated=terminated,
        termination_status=TerminationStatus.NORMAL if terminated else TerminationStatus.RUNNING,
        terminal_result=terminal,
        event_log=tuple(event_log),
    )
    return state.to_dict()


def test_full_replay_converts_behavior_and_private_targets_without_leaking_roles() -> None:
    event = GameEvent(
        event_id=1,
        environment_step=1,
        event_type=EventType.BANG,
        actor_idx=0,
        target_idx=1,
        hp_before=4,
        hp_after=3,
    )
    internals = {
        "seats": {
            str(index): {
                "kind": "dummy",
                "state": {
                    "emotion_module": {
                        "values": [0.1, 0.2],
                        "emotion_names": ["fear", "anger"],
                        "schema_version": "test-emotion",
                    }
                },
            }
            for index in range(5)
        }
    }
    episode = full_replay_to_canonical(
        {
            "replay_format": "full",
            "mode": "dummy_five",
            "seed": 7,
            "states": [_state(0), _state(1, (event,), terminated=True)],
            "observations": [{}, {}],
            "internals": [internals, internals],
        }
    )
    assert len(episode.steps) == 1
    assert episode.steps[0].action == 2  # BANG target player 1
    assert episode.role_labels == (0, 1, 2, 3, 4)
    assert set(episode.steps[0].emotion_labels) == {"0", "1", "2", "3", "4"}
    assert "roles" not in episode.steps[0].pre_observation
    assert semantic_role_labels(episode.role_labels) == (0, 1, 2, 2, 3)
