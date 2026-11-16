"""Smoke test for train_main_ai.py: a tiny PPO run produces a loadable .bangmodel."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule
from bang_rl.data.game_collection import formula_label_emotion_config, load_collection_config
from bang_rl.evaluation import summarize_protocol_report
from bang_rl.evaluation.ablation import ablated_protocol
from bang_rl.config import EvaluationConfig
from bang_rl.training.inference_artifact import save_phase2_inference_artifact

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("train_main_ai_cli", ROOT / "train_main_ai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_training_run_saves_an_evaluable_model(tmp_path):
    phase2 = save_phase2_inference_artifact(
        tmp_path / "p2.phase2model",
        IntentInferenceModule(seed=1),
        EmotionInferenceModule(config=formula_label_emotion_config(), seed=2),
    )
    out = tmp_path / "main.bangmodel"
    assert _script().main([
        "--phase2", str(phase2), "--updates", "2", "--games-per-update", "3",
        "--eval-every", "2", "--eval-games", "3", "--out", str(out),
    ]) == 0
    log = json.loads((tmp_path / "main_work" / "train_log.json").read_text(encoding="utf-8"))
    assert [row["update"] for row in log] == [1, 2]
    assert "check" in log[-1]
    report = ablated_protocol(out, "full", config=load_collection_config()).run(
        EvaluationConfig(protocol_id="custom_v1", games=3, games_per_seat=1, base_seed=0)
    )
    assert summarize_protocol_report(report)["games"] == 3
