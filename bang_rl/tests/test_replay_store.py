"""Headless tests for the Qt-free replay-store persistence (FEAT-003).

These import ONLY the Qt-free, numpy-free modules bang_rl.gui.replay_store and
bang_rl.gui.replay, so they run in CI where PyQt6/numpy are available for the
package __init__ but no display is present.
"""
from __future__ import annotations

import time

import pytest

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


def test_public_record_is_tagged_public_with_no_states() -> None:
    record = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=7,
        observations=_observations(),
        terminated=True,
    )
    assert record["replay_format"] == "public"
    assert "states" not in record


def _states() -> list[dict]:
    """Three authoritative EnvironmentState-style dicts (one per turn)."""

    return [
        {
            "episode_id": "e",
            "seed": 7,
            "roles": [0, 1, 2, 3, 4],
            "health": [5, 4, 4, 4, 4],
            "alive": [True, True, True, True, True],
            "agent_selection": 0,
            "environment_step": step,
        }
        for step in range(3)
    ]


def test_build_full_replay_record_carries_states_and_public_observations() -> None:
    record = replay_store.build_full_replay_record(
        mode="dummy_five",
        mode_label="Dummy AI x5",
        ai_versions={str(seat): "dummy" for seat in range(5)},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=7,
        states=_states(),
        observations=_observations(),
        terminated=True,
        truncated=False,
    )
    assert record["replay_format"] == "full"
    assert record["seed"] == 7
    assert len(record["states"]) == 3
    # A full record ALSO carries the public observations so it renders like a
    # public replay on the board/log.
    assert len(record["observations"]) == 3
    model = ReplayModel(record)
    assert len(model) == 3


def test_full_record_round_trips_and_summary_reports_format(tmp_path) -> None:
    record = replay_store.build_full_replay_record(
        mode="main_one_dummy_four",
        mode_label="Main AI 1 vs Dummy AI 4",
        ai_versions={"0": {"kind": "main"}},
        winning_team="outlaw_team",
        winners=[1],
        human_player=0,
        seed=3,
        states=_states(),
        observations=_observations(),
        terminated=True,
    )
    path = replay_store.save_replay(record, tmp_path)
    loaded = replay_store.load_replay(path)
    assert loaded == record
    summary = replay_store.replay_summary(loaded)
    assert summary["replay_format"] == "full"
    assert summary["has_full_state"] is True
    # A public record's summary reports 'public'.
    public = replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human 1 vs Random AI 4",
        ai_versions={},
        winning_team=None,
        winners=[],
        human_player=0,
        seed=1,
        observations=_observations(),
        terminated=True,
    )
    assert replay_store.replay_summary(public)["replay_format"] == "public"
    assert replay_store.replay_summary(public)["has_full_state"] is False


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


def test_rename_replay_preserves_suffix_and_sanitizes(tmp_path) -> None:
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
    path = replay_store.save_replay(record, tmp_path, filename="replay-orig.json")
    # A name without a suffix keeps the source's .json suffix; spaces become _.
    renamed = replay_store.rename_replay(path, "My Cool Game")
    assert renamed.name == "My_Cool_Game.json"
    assert renamed.parent == tmp_path
    assert renamed.exists()
    assert not path.exists()
    # A supplied .jsonl suffix is preserved as-is.
    renamed2 = replay_store.rename_replay(renamed, "next.jsonl")
    assert renamed2.name == "next.jsonl"


def test_rename_replay_defends_against_path_traversal(tmp_path) -> None:
    path = replay_store.save_replay({}, tmp_path, filename="replay-x.json")
    result = replay_store.rename_replay(path, "../../escape")
    # The directory component is stripped so the file stays inside tmp_path.
    assert result.parent == tmp_path
    assert result.name == "escape.json"


def test_rename_replay_rejects_existing_target(tmp_path) -> None:
    a = replay_store.save_replay({}, tmp_path, filename="a.json")
    replay_store.save_replay({}, tmp_path, filename="b.json")
    with pytest.raises(FileExistsError):
        replay_store.rename_replay(a, "b.json")


def test_rename_replay_missing_source_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        replay_store.rename_replay(tmp_path / "missing.json", "new.json")


def test_delete_replay_removes_file_and_tolerates_missing(tmp_path) -> None:
    path = replay_store.save_replay({}, tmp_path, filename="doomed.json")
    assert path.exists()
    replay_store.delete_replay(path)
    assert not path.exists()
    # A second delete (missing file) is a tolerated no-op.
    replay_store.delete_replay(path)
    # Deleting a directory is refused (only regular files are removed).
    replay_store.delete_replay(tmp_path)
    assert tmp_path.exists()


def test_build_replay_filename_encodes_mode(tmp_path) -> None:
    name = replay_store.build_replay_filename("human_random")
    assert name.startswith("replay-human_random-")
    assert name.endswith(".json")
    assert "human_random" in name
    # A timestamp component is present between the mode and the suffix.
    assert len(name) > len("replay-human_random-.json")


def test_build_replay_filename_falls_back_when_mode_missing() -> None:
    for bad in (None, "", "   "):
        name = replay_store.build_replay_filename(bad)
        assert name.startswith("replay-")
        assert name.endswith(".json")
        # No mode token: the mode-less shape 'replay-<stamp>.json'.
        assert name.count("-") >= 1


def test_build_replay_filename_sanitizes_mode_no_path_leak() -> None:
    name = replay_store.build_replay_filename("../../evil/mode.json")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name.endswith(".json")


def test_save_replay_uses_mode_from_record_for_auto_name(tmp_path) -> None:
    record = replay_store.build_replay_record(
        mode="random_five",
        mode_label="Random AI x5",
        ai_versions={str(seat): "random" for seat in range(5)},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=1,
        observations=_observations(),
        terminated=True,
    )
    path = replay_store.save_replay(record, tmp_path)
    assert "random_five" in path.name
    assert path.exists()


def test_describe_ai_versions_random_five() -> None:
    entries = replay_store.describe_ai_versions(
        {str(seat): "random" for seat in range(5)}
    )
    assert len(entries) == 5
    assert all(kind == "random" and version is None for _tag, _seat, kind, version in entries)
    # Sorted by numeric seat.
    assert [seat for _tag, seat, _kind, _v in entries] == ["0", "1", "2", "3", "4"]


def test_describe_ai_versions_main_carries_version_and_dummies() -> None:
    entries = replay_store.describe_ai_versions(
        {
            "0": "dummy",
            "1": {"kind": "main", "version": "1.0", "metadata": {"x": 1}},
            "2": "dummy",
            "3": "dummy",
            "4": "dummy",
        }
    )
    kinds = {seat: (kind, version) for _tag, seat, kind, version in entries}
    assert kinds["1"] == ("main", "1.0")
    assert kinds["0"] == ("dummy", None)
    assert kinds["4"] == ("dummy", None)


def test_describe_ai_versions_legacy_and_empty_safe() -> None:
    assert replay_store.describe_ai_versions({}) == []
    assert replay_store.describe_ai_versions(None) == []
    # An unknown descriptor is reported as 'unknown', never crashes.
    entries = replay_store.describe_ai_versions({"0": 123, "1": {"kind": "wat"}})
    kinds = {seat: kind for _tag, seat, kind, _v in entries}
    assert kinds["0"] == "unknown"
    assert kinds["1"] == "unknown"


def test_replay_format_label_tag_full_public_and_legacy() -> None:
    # FEAT-001: single Qt-free seam both the browser row and the info panel
    # resolve the format label through.
    assert replay_store.replay_format_label_tag({"replay_format": "full"}) == "replay.format_full"
    assert replay_store.replay_format_label_tag({"replay_format": "public"}) == "replay.format_public"
    # A legacy record with no replay_format defaults to public.
    assert replay_store.replay_format_label_tag({}) == "replay.format_public"
    assert replay_store.replay_format_label_tag(None) == "replay.format_public"
    # Accepts a replay_summary result too.
    summary = replay_store.replay_summary({"replay_format": "full", "states": [{}]})
    assert replay_store.replay_format_label_tag(summary) == "replay.format_full"


def test_replay_store_i18n_keys_exist_in_both_languages() -> None:
    from bang_rl.gui.i18n import _STRINGS

    for key in (
        "replay.repath",
        "replay.folder_title",
        "replay.browser",
        "replay.info_title",
        "replay.info_mode",
        "replay.info_format",
        "replay.info_outcome",
        "replay.info_seed",
        "replay.info_seat",
        "replay.ai_random",
        "replay.ai_dummy",
        "replay.ai_main",
        "replay.ai_unknown",
    ):
        assert key in _STRINGS
        assert "en" in _STRINGS[key]
        assert "ko" in _STRINGS[key]
        for text in _STRINGS[key].values():
            assert "\u2014" not in text


# --- item 4 / item 6: per-step internals round-trip (needs numpy DummyAgent) --
def test_full_replay_internals_round_trip_back_to_dummy_agent(tmp_path) -> None:
    """A captured DummyAgent state survives the full-replay JSON round-trip and
    can be handed back to ``DummyAgent.from_state_dict`` to resume its internal
    emotion / role-belief state."""

    from bang_rl.agents import DummyAgent

    dummy = DummyAgent("player_1", seed=13, player_idx=1)
    per_seat_state = dummy.state_dict()
    internals = [
        {"health": [5, 5, 5, 5, 5], "seats": {"1": {"kind": "dummy", "state": per_seat_state}}},
        None,  # a per-turn export-miss placeholder must survive as JSON null
        {"health": [5, 4, 5, 5, 5], "seats": {"1": {"kind": "dummy", "state": per_seat_state}}},
    ]
    record = replay_store.build_full_replay_record(
        mode="human_dummy",
        mode_label="Human vs Dummy",
        ai_versions={"1": "dummy", "2": "dummy", "3": "dummy", "4": "dummy"},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=7,
        states=[{"episode_id": "e", "environment_step": step} for step in range(3)],
        internals=internals,
        observations=_observations(),
        terminated=True,
    )
    path = replay_store.save_replay(record, tmp_path)
    loaded = replay_store.load_replay(path)
    assert loaded["internals"][1] is None
    restored_state = loaded["internals"][0]["seats"]["1"]["state"]
    # The state round-trips cleanly back into a live DummyAgent.
    restored_agent = DummyAgent.from_state_dict(restored_state)
    assert restored_agent.role_estimator.beliefs.shape == dummy.role_estimator.beliefs.shape
    assert replay_store.replay_summary(loaded)["has_internals"] is True


def test_public_replay_never_stores_internals() -> None:
    public = replay_store.build_replay_record(
        mode="human_dummy",
        mode_label="Human vs Dummy",
        ai_versions={"1": "dummy"},
        winning_team=None,
        winners=[],
        human_player=0,
        seed=1,
        observations=_observations(),
    )
    assert "internals" not in public
    assert replay_store.replay_summary(public)["has_internals"] is False
