"""Four-class role scoring, formula emotion scoring and baseline rows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule, PPOModule
from bang_rl.config import EvaluationConfig
from bang_rl.data.game_collection import formula_label_emotion_config, load_collection_config
from bang_rl.evaluation import (
    CustomEvaluationProtocol,
    baseline_factory,
    run_seat_evaluation,
    summarize_protocol_report,
)
from bang_rl.evaluation.protocol import (
    _version_expectations,
    observe_boundaries,
    opponent_emotion_target,
    semantic_role_prediction,
)
from bang_rl.training.final_inference_artifact import save_final_inference_artifact
from bang_rl.training.inference_artifact import (
    load_phase2_inference_artifact,
    save_phase2_inference_artifact,
)

SEMANTIC = {"SHERIFF": "sheriff", "DEPUTY": "deputy", "OUTLAW_A": "outlaw", "OUTLAW_B": "outlaw", "TRAITOR": "traitor"}


@pytest.fixture(scope="module")
def config():
    return load_collection_config()


@pytest.fixture(scope="module")
def untrained_model(config, tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    intent = IntentInferenceModule(seed=1)
    emotion = EmotionInferenceModule(config=formula_label_emotion_config(), seed=2)
    phase2 = load_phase2_inference_artifact(save_phase2_inference_artifact(root / "p2.phase2model", intent, emotion))
    versions = dict(_version_expectations(config, None))
    return save_final_inference_artifact(
        root / "untrained.bangmodel",
        PPOModule(seed=3, emotion_dim=4 * 9),
        phase2_artifact=phase2,
        job_snapshot={"job_id": "untrained", "seed": 0, "split_seed": 0, "config_snapshot": {}, "versions": versions},
        config_snapshot={},
    )


def test_semantic_prediction_merges_outlaw_columns():
    # Mass split across outlaw A/B beats a larger single traitor column.
    assert semantic_role_prediction([0.0, 0.1, 0.3, 0.3, 0.4]) == 2
    assert semantic_role_prediction([0.1, 0.6, 0.1, 0.1, 0.1]) == 1
    assert semantic_role_prediction([0.1, 0.1, 0.1, 0.7]) == 3  # already four-way
    with pytest.raises(ValueError):
        semantic_role_prediction([0.5, 0.5])


def test_formula_opponents_are_scored_on_their_label(config):
    agent = baseline_factory("dummy", config)(2, 0, (0, 1, 2, 3, 4))
    values, names = opponent_emotion_target(agent)
    assert values.shape == (9,) and len(names) == 9 and names[4] == "relation_0"


def test_boundaries_reach_only_living_non_acting_formula_dummies():
    calls = []

    class Formula:
        uses_emotion_formula = True

        def __init__(self, index):
            self.index = index

        def observe_public_boundary(self, observation):
            calls.append(self.index)

    env = SimpleNamespace(state=SimpleNamespace(alive=[True, True, False, True, True]), observe=lambda i: {"i": i})
    agents = {0: Formula(0), 1: Formula(1), 2: Formula(2), 3: SimpleNamespace(uses_emotion_formula=False)}
    observe_boundaries(agents, env, actor=1)
    assert calls == [0]  # 1 acts, 2 is dead, 3 is not a formula dummy


def test_baselines_are_reproducible_and_ordered(config):
    first = run_seat_evaluation(baseline_factory("dummy", config), config, games=30, label="dummy")
    second = run_seat_evaluation(baseline_factory("dummy", config), config, games=30, label="dummy")
    assert first["game_results"] == second["game_results"]
    random = run_seat_evaluation(baseline_factory("random", config), config, games=100, label="random")
    dummy = run_seat_evaluation(baseline_factory("dummy", config), config, games=100, label="dummy")
    assert dummy["win_rate"] > random["win_rate"]
    assert set(dummy["by_role"]) == {"sheriff", "deputy", "outlaw", "traitor"}
    with pytest.raises(ValueError):
        baseline_factory("nonsense", config)


def test_main_agent_and_baselines_play_the_same_deals(config, untrained_model):
    games = 10
    report = CustomEvaluationProtocol(untrained_model, config=config).run(
        EvaluationConfig(protocol_id="custom_v1", games=games, games_per_seat=2, base_seed=5)
    )
    baseline = run_seat_evaluation(baseline_factory("random", config), config, games=games, base_seed=5)
    main_rows = report.to_dict()["game_results"]
    assert [(row["seed"], row["main_agent_seat"], SEMANTIC[row["main_agent_role"]]) for row in main_rows] == [
        (row["seed"], row["seat"], row["role"]) for row in baseline["game_results"]
    ]
    summary = summarize_protocol_report(report)
    assert sum(item["games"] for item in summary["by_role"].values()) == summary["normal_games"]
    # Formula emotions are now scored (9 entries per living opponent, constants skipped).
    assert report.metric_samples["estimated_emotion"] > 0
    assert isinstance(report.estimated_emotion_mse, float)


def test_cli_writes_comparison(tmp_path, untrained_model, capsys):
    script = Path(__file__).resolve().parents[2] / "evaluate_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_baselines_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "eval.json"
    emotion_file = Path(__file__).resolve().parents[2] / "emotion_formula.example.json"
    assert module.main([
        "--games", "5", "--baselines", "random,dummy", "--artifact", str(untrained_model),
        "--emotion-file", str(emotion_file), "--out", str(out),
    ]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [row["agent"] for row in rows] == ["random", "dummy", "main"]
    assert "role acc" in capsys.readouterr().out
