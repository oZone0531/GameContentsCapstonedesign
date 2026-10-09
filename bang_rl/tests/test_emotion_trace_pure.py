"""Emotion decision trace: built from full replays and saved next to them (Qt-free)."""

from __future__ import annotations

import json
from pathlib import Path

import bang_rl.paths as paths
from bang_rl.app import GameService
from bang_rl.config import SystemConfig
from bang_rl.gui.emotion_trace import build_trace_lines, main
from bang_rl.gui.game import GameSessionController

ROOT = Path(__file__).resolve().parents[2]


def _play_spectator_game(tmp_path, monkeypatch, emotion=None):
    monkeypatch.setattr(paths, "REPLAYS_DIR", tmp_path)
    cfg = SystemConfig.model_validate({"preset": {"emotion": emotion}} if emotion else {})
    controller = GameSessionController(GameService(), config=cfg, seed=5)
    controller.start("dummy_five")
    guard = 0
    while controller.active and controller.is_ai_turn() and guard < 500:
        controller.step_single_ai_turn()
        guard += 1
    return sorted(tmp_path.glob("*.json")), sorted(tmp_path.glob("*.txt"))


def test_trace_is_saved_next_to_full_replay_with_formula(tmp_path, monkeypatch):
    emotion = json.loads((ROOT / "emotion_formula.example.json").read_text(encoding="utf-8"))
    replays, traces = _play_spectator_game(tmp_path, monkeypatch, emotion)
    assert len(replays) == 1 and len(traces) == 1
    assert traces[0].stem == replays[0].stem
    text = traces[0].read_text(encoding="utf-8")
    assert "emotion_formula" in text
    record = json.loads(replays[0].read_text(encoding="utf-8"))
    decisions = sum(1 for line in text.splitlines() if line.startswith("step "))
    assert decisions == len(record["states"]) - 1  # one decision per recorded turn
    assert "intent alone" in text and "Game over" in text


def test_trace_also_works_for_the_default_module(tmp_path, monkeypatch):
    _, traces = _play_spectator_game(tmp_path, monkeypatch, None)
    assert len(traces) == 1
    assert "rule_table" in traces[0].read_text(encoding="utf-8")


def test_public_record_has_no_trace_and_cli_regenerates(tmp_path, monkeypatch):
    assert build_trace_lines({"observations": []}) == []
    replays, traces = _play_spectator_game(tmp_path, monkeypatch, None)
    traces[0].unlink()
    assert main([str(replays[0])]) == 0
    assert replays[0].with_suffix(".txt").exists()
