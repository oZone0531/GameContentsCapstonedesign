"""Headless regression tests for item 1 (reveal all roles at a replay's END).

These import ONLY the Qt-free modules (no PyQt6, numpy, or pydantic) so they
collect and run in the sandbox as well as in CI. ``import bang_rl.gui.replay``
would execute ``bang_rl/gui/__init__`` which pulls in pydantic/PyQt6-backed
modules absent in the sandbox, so each module is loaded straight from its path
(the same pattern as ``test_replay_log_row.py`` / ``test_board_role_reveal.py``).

They pin the TERMINAL-ONLY reveal contract: a saved replay record carrying a
top-level ``final_roles`` must surface those roles on the FINAL snapshot only,
so a replay reveals every player's role at its END while every mid-replay
snapshot stays public (``final_roles == ()``). A legacy record without the field
must still load with no reveal (backward compatible). The second half pins that
``board_layout.role_for_display`` reveals every seat from a terminal snapshot's
``final_roles`` at ``game_over`` and HIDES a non-sheriff living seat during play.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(relative: str, name: str):
    path = Path(__file__).resolve().parents[1] / Path(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A frozen dataclass looks up its module via sys.modules[cls.__module__]
    # during processing, so register the module before executing it.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_replay = _load_module("gui/replay.py", "_replay_terminal_isolated")
_board_layout = _load_module(
    "gui/widgets/board_layout.py", "_board_layout_terminal_isolated"
)
ReplayModel = _replay.ReplayModel
role_for_display = _board_layout.role_for_display
HIDDEN_ROLE_SENTINEL = _board_layout.HIDDEN_ROLE_SENTINEL


# A terminal reveal: seat 0 is Sheriff, the rest are hidden mid-game.
_FINAL_ROLES = [0, 2, 1, 3, 4]
# Mid-game public observations reveal only the sheriff (seat 0); every living
# non-sheriff seat is HIDDEN (255).
_SHERIFF_ONLY_REVEALED = [0, 255, 255, 255, 255]


def _step(step: int, *, terminated: bool = False, truncated: bool = False) -> dict:
    return {
        "environment_step": step,
        "actor_idx": step % 5,
        "terminated": terminated,
        "truncated": truncated,
        "pre_observation": {
            "healths": [5, 5, 5, 5, 5],
            "alive": [True, True, True, True, True],
            "role_visibility": [3, 0, 0, 0, 0],
            "revealed_roles": list(_SHERIFF_ONLY_REVEALED),
            "current_player_idx": [step % 5],
            "environment_step": [step],
        },
    }


def _record(*, with_final_roles: bool) -> dict:
    """Build a to_dict-style replay record with steps + events + final_state."""

    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
        {
            "environment_step": 3,
            "event_type": "player_eliminated",
            "eliminated_idx": 3,
        },
    ]
    record: dict = {
        "schema_version": "1.0",
        "replay_format": "public",
        "mode": "human_random",
        "termination_status": "normal",
        "events": events,
        "steps": [_step(step) for step in range(4)],
        # The final settled state terminates the episode.
        "final_state": {
            "healths": [5, 5, 5, 0, 5],
            "alive": [True, True, True, False, True],
            "role_visibility": [3, 0, 0, 3, 0],
            "revealed_roles": [0, 255, 255, 3, 255],
            # The human at seat 0 knows their own role (Sheriff) via self_role,
            # mirroring the live terminal PublicGameState the board reads.
            "self_role": [0],
            "current_player_idx": [0],
            "environment_step": [3],
            "terminated": True,
        },
    }
    if with_final_roles:
        record["final_roles"] = list(_FINAL_ROLES)
    return record


def _record_terminal_status_only() -> dict:
    """Build a record that mirrors the PRODUCTION writer's appended-final_state.

    The real writer (game.py / replay_store.py) saves a ``final_state``
    observation that carries NO ``terminated`` / ``truncated`` flags of its own;
    the appended settled snapshot is flagged terminal SOLELY because
    ``ReplayModel._terminal_status`` reads the record's top-level
    ``termination_status`` (defaulting to ``"normal"``). The other record builder
    sets ``terminated: True`` directly on the observation, which bypasses that
    ``terminal == "normal"`` fallback. This builder exercises the real path so
    the privacy-critical reveal gate is revert-sensitive on it too.
    """

    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
        {
            "environment_step": 3,
            "event_type": "player_eliminated",
            "eliminated_idx": 3,
        },
    ]
    return {
        "schema_version": "1.0",
        "replay_format": "public",
        "mode": "human_random",
        # The ONLY terminal signal for the appended final_state below.
        "termination_status": "normal",
        "events": events,
        "steps": [_step(step) for step in range(4)],
        "final_state": {
            "healths": [5, 5, 5, 0, 5],
            "alive": [True, True, True, False, True],
            "role_visibility": [3, 0, 0, 3, 0],
            "revealed_roles": [0, 255, 255, 3, 255],
            "self_role": [0],
            "current_player_idx": [0],
            "environment_step": [3],
            # NOTE: no "terminated" / "truncated" here, matching the production
            # writer; the appended snapshot is terminal only via termination_status.
        },
        "final_roles": list(_FINAL_ROLES),
    }


def test_terminal_status_only_final_state_reveals_all_roles() -> None:
    # Production writer path: the appended final_state omits terminated/truncated
    # and relies SOLELY on the top-level termination_status ("normal"). The
    # appended snapshot must still report terminal AND carry final_roles.
    model = ReplayModel(_record_terminal_status_only())
    assert len(model) >= 2
    terminal = model.snapshot_at(len(model) - 1)
    assert terminal is not None
    # The reveal gate fires via the terminal == "normal" fallback, not an
    # explicit flag on the observation.
    assert terminal.terminated or terminal.truncated
    assert terminal.final_roles == (0, 2, 1, 3, 4)


def test_terminal_status_only_has_no_mid_replay_leak() -> None:
    # Every non-terminal snapshot still carries NO final_roles on this path.
    model = ReplayModel(_record_terminal_status_only())
    for snapshot in model.snapshots[:-1]:
        assert snapshot.final_roles == ()
        assert not snapshot.terminated
        assert not snapshot.truncated


def test_terminal_snapshot_reveals_all_roles() -> None:
    model = ReplayModel(_record(with_final_roles=True))
    assert len(model) >= 2
    terminal = model.snapshot_at(len(model) - 1)
    assert terminal is not None
    # (b) the TERMINAL snapshot carries final_roles and reports terminal.
    assert terminal.final_roles == (0, 2, 1, 3, 4)
    assert terminal.terminated or terminal.truncated


def test_no_mid_replay_leak() -> None:
    model = ReplayModel(_record(with_final_roles=True))
    # (a) every NON-terminal snapshot carries NO final_roles.
    for snapshot in model.snapshots[:-1]:
        assert snapshot.final_roles == ()
        assert not snapshot.terminated
        assert not snapshot.truncated


def test_legacy_record_without_final_roles_reveals_nothing() -> None:
    # (c) a legacy record WITHOUT a top-level final_roles yields snapshots whose
    # final_roles are all empty (backward compatible, no reveal).
    model = ReplayModel(_record(with_final_roles=False))
    assert len(model) >= 2
    for snapshot in model.snapshots:
        assert snapshot.final_roles == ()
    # The final snapshot still terminates the episode, it simply reveals nothing.
    assert model.snapshots[-1].terminated or model.snapshots[-1].truncated


def test_board_reveals_every_seat_from_terminal_final_roles() -> None:
    # role_for_display reveals every seat from the terminal snapshot's
    # final_roles when game_over is True.
    model = ReplayModel(_record(with_final_roles=True))
    terminal = model.snapshot_at(len(model) - 1)
    assert terminal is not None
    observation = terminal.observation
    for seat, expected in enumerate(_FINAL_ROLES):
        assert (
            role_for_display(
                seat,
                observation,
                terminal.final_roles,
                True,
                human_player=0,
            )
            == expected
        )


def test_board_hides_living_nonsheriff_seat_during_play() -> None:
    # With game_over False and empty final_roles a living non-sheriff seat stays
    # hidden (confirming no mid-replay leak at the reveal boundary).
    model = ReplayModel(_record(with_final_roles=True))
    mid = model.snapshot_at(0)
    assert mid is not None
    assert mid.final_roles == ()
    assert (
        role_for_display(1, mid.observation, mid.final_roles, False, human_player=0)
        == HIDDEN_ROLE_SENTINEL
    )
