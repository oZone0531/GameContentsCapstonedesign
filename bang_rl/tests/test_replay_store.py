"""Headless tests for the Qt-free replay-store persistence (FEAT-003).

These import ONLY the Qt-free, numpy-free modules bang_rl.gui.replay_store and
bang_rl.gui.replay, so they run in CI where PyQt6/numpy are available for the
package __init__ but no display is present.
"""
from __future__ import annotations

import time

from bang_rl.gui import replay_store
from bang_rl.gui.replay import ReplayModel


def _observations() -> list[dict]:
    """Three per-turn public observations with a growing cumulative log."""

    return [
        {
            "environment_step": 0,
            "current_player_idx": (0,),
            "healths": [4, 4, 4, 4, 4],
            "alive": [1, 1, 1, 1, 1],
            "event_sequence": [{"environment_step": 0, "type": "start"}],
        },
        {
            "environment_step": 1,
            "current_player_idx": (1,),
            "healths": [4, 3, 4, 4, 4],
            "alive": [1, 1, 1, 1, 1],
            "event_sequence": [
                {"environment_step": 0, "type": "start"},
                {"environment_step": 1, "type": "bang"},
            ],
        },
        {
            "environment_step": 2,
            "current_player_idx": (0,),
            "healths": [4, 2, 4, 4, 4],
            "alive": [1, 1, 1, 1, 1],
            "event_sequence": [
                {"environment_step": 0, "type": "start"},
                {"environment_step": 1, "type": "bang"},
                {"environment_step": 2, "type": "bang"},
            ],
        },
    ]


def test_build_replay_record_has_all_metadata_and_feeds_replay_model() -> None:
    record = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={"1": "random", "2": "random", "3": "random", "4": "random"},
        winning_team="sheriff_team",
        winners=[0, 2],
        human_player=0,
        seed=7,
        observations=_observations(),
        terminated=True,
        truncated=False,
    )
    assert record["schema_version"] == "1.0"
    assert record["mode"] == "human_random"
    assert record["game_type"] == "human_random"
    assert record["ai_versions"] == {
        "1": "random",
        "2": "random",
        "3": "random",
        "4": "random",
    }
    assert record["winning_team"] == "sheriff_team"
    assert record["winners"] == [0, 2]
    assert record["human_won"] is True
    assert record["outcome"] == "normal"
    assert len(record["observations"]) == 3

    model = ReplayModel(record)
    assert len(model) == 3
    # The last snapshot carries the full cumulative event log.
    model.seek(len(model) - 1)
    assert len(model.current_events()) == 3


def test_human_won_true_when_human_seat_in_winners_else_false() -> None:
    won = replay_store.build_replay_record(
        mode="human_dummy",
        mode_label="Human 1 vs Dummy AI 4",
        ai_versions={},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=None,
        observations=_observations(),
        terminated=True,
    )
    lost = replay_store.build_replay_record(
        mode="human_dummy",
        mode_label="Human 1 vs Dummy AI 4",
        ai_versions={},
        winning_team="outlaw_team",
        winners=[1, 3],
        human_player=0,
        seed=None,
        observations=_observations(),
        terminated=True,
    )
    assert won["human_won"] is True
    assert lost["human_won"] is False


def test_build_handles_truncated_game() -> None:
    record = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={},
        winning_team=None,
        winners=[],
        human_player=0,
        seed=1,
        observations=_observations(),
        terminated=False,
        truncated=True,
    )
    assert record["outcome"] == "truncated"
    assert record["winning_team"] is None
    assert record["human_won"] is False


def test_save_and_load_round_trip(tmp_path) -> None:
    record = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={"1": "random"},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=3,
        observations=_observations(),
        terminated=True,
    )
    path = replay_store.save_replay(record, tmp_path)
    assert path.exists()
    assert path.parent == tmp_path
    loaded = replay_store.load_replay(path)
    assert loaded == record


def test_list_replays_newest_first(tmp_path) -> None:
    record = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={},
        winning_team=None,
        winners=[],
        human_player=0,
        seed=None,
        observations=_observations(),
        terminated=True,
    )
    first = replay_store.save_replay(record, tmp_path, filename="replay-a.json")
    time.sleep(0.01)
    second = replay_store.save_replay(record, tmp_path, filename="replay-b.json")
    listing = replay_store.list_replays(tmp_path)
    assert set(listing) == {first, second}
    # Newest-first: the file saved second sorts before the first.
    assert listing[0] == second


def test_list_replays_missing_directory_returns_empty(tmp_path) -> None:
    assert replay_store.list_replays(tmp_path / "does-not-exist") == []


def test_replay_store_i18n_keys_exist_in_both_languages() -> None:
    from bang_rl.gui.i18n import _STRINGS

    for key in ("replay.repath", "replay.folder_title", "replay.browser"):
        assert key in _STRINGS
        assert "en" in _STRINGS[key]
        assert "ko" in _STRINGS[key]
        for text in _STRINGS[key].values():
            assert "\u2014" not in text
