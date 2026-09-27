from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import ObservationPreprocessor, RoleFusion
from bang_rl.config import EvaluationConfig, PresetConfig, SystemConfig
from bang_rl.env.types import Role, TerminationStatus
from bang_rl.exceptions import ArtifactCompatibilityError
from bang_rl.evaluation import (
    ConfidenceInterval,
    CustomEvaluationProtocol,
    PrimaryEvaluationProtocol,
    run_custom_evaluation,
    run_primary_evaluation,
)


@dataclass
class _Action:
    action: torch.Tensor


class _InferenceModule:
    def __init__(self, output):
        self.output = output

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}

    def predict_proba(self, _processed):
        return self.output


class _Policy:
    def state_dict(self):
        return {"weight": torch.tensor([2.0])}

    def act(self, _inputs, _mask, *, deterministic=False):
        assert deterministic is True
        return _Action(torch.tensor(0))


def _artifact(environment_version: str = "1.0"):
    metadata = {
        "artifact_type": "inference",
        "artifact_format_version": "1.0",
        "inference_model_artifact_format_version": "1.0",
        "versions": {
            "environment_version": environment_version,
            "observation_schema_version": "1.0",
            "action_space_version": "1.0",
            "reward_config_version": "1.0",
            "emotion_schema_version": "1.0",
            "inference_model_artifact_format_version": "1.0",
        },
        "modules": {
            "intent": {"identifier": "IntentInferenceModule"},
            "emotion": {"identifier": "EmotionInferenceModule"},
            "ppo": {"identifier": "PPOModule"},
        },
    }
    return SimpleNamespace(
        intent=_InferenceModule(np.zeros((4, 5), dtype=np.float32)),
        emotion=_InferenceModule(np.zeros((4, 1), dtype=np.float32)),
        ppo=_Policy(),
        role_fusion=RoleFusion(),
        preprocessor=ObservationPreprocessor(),
        metadata=metadata,
        path=None,
    )


class _TinyEnvironment:
    """One-step normal fixture used to test protocol accounting quickly."""

    def __init__(self, episode_id: str, seed: int, _config):
        self.episode_id = episode_id
        self.seed = seed
        self.agent_selection = "player_0"
        self.environment_step = 0
        self.close_count = 0
        self.state = SimpleNamespace(
            roles=(Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR),
            terminated=False,
            truncated=False,
            termination_status=TerminationStatus.RUNNING,
            terminal_result=None,
        )

    def reset(self, *, seed: int, episode_id: str):
        self.seed = seed
        self.episode_id = episode_id
        return self.observe(0)

    def observe(self, selected: int):
        visibility = np.zeros(5, dtype=np.uint8)
        visibility[selected] = 1
        confirmed = np.zeros((5, 5), dtype=np.float32)
        confirmed[selected, int(self.state.roles[selected])] = 1.0
        return {
            "current_player_idx": np.asarray([selected], dtype=np.uint8),
            "environment_step": np.asarray([self.environment_step], dtype=np.uint32),
            "round_index": np.asarray([0], dtype=np.uint32),
            "healths": np.full(5, 4, dtype=np.uint8),
            "alive": np.ones(5, dtype=np.bool_),
            "turn_order": np.arange(5, dtype=np.uint8),
            "sheriff_idx": np.asarray([0], dtype=np.uint8),
            "self_role": np.asarray([int(self.state.roles[selected])], dtype=np.uint8),
            "role_visibility": visibility,
            "revealed_roles": np.full(5, 255, dtype=np.uint8),
            "confirmed_role_distribution": confirmed,
            "event_sequence": (),
            "action_mask": np.ones(6, dtype=np.uint8),
            "observation_schema_version": "1.0",
        }

    def close(self):
        self.close_count += 1

    def step(self, _action):
        self.environment_step += 1
        self.state.terminated = True
        self.state.termination_status = TerminationStatus.NORMAL
        self.state.terminal_result = SimpleNamespace(
            winners=(0,),
            winning_team=SimpleNamespace(name="SHERIFF_TEAM"),
        )


def _factory(episode_id, seed, config):
    return _TinyEnvironment(episode_id, seed, config)


def test_primary_protocol_has_exact_balanced_counts_and_seed_formula() -> None:
    report = run_primary_evaluation(
        _artifact(), config=SystemConfig(), environment_factory=_factory
    )

    assert report.total_games == 500
    assert report.normal_games == 500
    assert report.truncated_games == 0
    assert report.seat_counts == {str(index): 100 for index in range(5)}
    assert report.seed_set == tuple(range(500))
    assert [game["main_agent_seat"] for game in report.game_results] == [index % 5 for index in range(500)]
    assert [game["seed"] for game in report.game_results] == list(range(500))


def test_primary_resolves_requested_opponent_preset() -> None:
    alternate = PresetConfig(preset_id="alternate", version="2.0")
    config = SystemConfig(
        evaluation=EvaluationConfig(
            opponent_preset="alternate",
            opponent_presets={"alternate": alternate},
        )
    )
    report = run_primary_evaluation(_artifact(), config=config, environment_factory=_factory)
    assert report.opponent["preset_id"] == "alternate"
    assert report.opponent["preset_version"] == "2.0"


def test_each_evaluation_game_closes_environment_on_normal_truncated_and_exception() -> None:
    created = []

    class TruncatedEnvironment(_TinyEnvironment):
        def step(self, action):
            super().step(action)
            self.state.terminated = False
            self.state.truncated = True
            self.state.termination_status = TerminationStatus.TRUNCATED
            self.state.terminal_result = None

    class FailingEnvironment(_TinyEnvironment):
        def step(self, _action):
            raise RuntimeError("game failure")

    def factory_for(environment_type):
        def factory(episode_id, seed, config):
            environment = environment_type(episode_id, seed, config)
            created.append(environment)
            return environment
        return factory

    config = SystemConfig()
    normal_runner = PrimaryEvaluationProtocol(
        _artifact(), config=config, environment_factory=factory_for(_TinyEnvironment)
    )
    normal_runner.evaluation_run_id = "lifecycle-normal"
    normal_runner._run_game(0, 0, 0, config.preset)

    truncated_runner = PrimaryEvaluationProtocol(
        _artifact(), config=config, environment_factory=factory_for(TruncatedEnvironment)
    )
    truncated_runner.evaluation_run_id = "lifecycle-truncated"
    truncated_runner._run_game(0, 0, 0, config.preset)

    failing_runner = PrimaryEvaluationProtocol(
        _artifact(), config=config, environment_factory=factory_for(FailingEnvironment)
    )
    failing_runner.evaluation_run_id = "lifecycle-exception"
    with pytest.raises(RuntimeError, match="game failure"):
        failing_runner._run_game(0, 0, 0, config.preset)

    assert [environment.close_count for environment in created] == [1, 1, 1]


    artifact = _artifact()
    first = PrimaryEvaluationProtocol(artifact, config=SystemConfig(), environment_factory=_factory).run()
    second = PrimaryEvaluationProtocol(artifact, config=SystemConfig(), environment_factory=_factory).run()

    assert first.to_dict() == second.to_dict()
    assert first.output_metadata["training_mutation"] is False
    assert first.output_metadata["main_agent_inputs"] == "public_observation_only"



def test_protocol_logger_ownership_closes_internal_loggers_on_all_run_paths(monkeypatch) -> None:
    class CountingLogger:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.close_count = 0
            self.__class__.instances.append(self)

        def info(self, *_args, **_kwargs):
            return None

        def event(self, *_args, **_kwargs):
            return None

        def close(self):
            self.close_count += 1

    monkeypatch.setattr("bang_rl.evaluation.protocol.StructuredLogger", CountingLogger)

    normal_runner = PrimaryEvaluationProtocol(
        _artifact(), config=SystemConfig(), environment_factory=_factory
    )
    normal_runner.run()
    assert CountingLogger.instances[-1].close_count == 1

    class TruncatedEnvironment(_TinyEnvironment):
        def step(self, action):
            super().step(action)
            self.state.terminated = False
            self.state.truncated = True
            self.state.termination_status = TerminationStatus.TRUNCATED
            self.state.terminal_result = None

    truncated_runner = PrimaryEvaluationProtocol(
        _artifact(), config=SystemConfig(),
        environment_factory=lambda episode_id, seed, config: TruncatedEnvironment(
            episode_id, seed, config
        ),
    )
    truncated_runner.run()
    assert CountingLogger.instances[-1].close_count == 1

    class FailingEnvironment(_TinyEnvironment):
        def step(self, _action):
            raise RuntimeError("evaluation failure")

    failing_runner = PrimaryEvaluationProtocol(
        _artifact(), config=SystemConfig(),
        environment_factory=lambda episode_id, seed, config: FailingEnvironment(
            episode_id, seed, config
        ),
    )
    with pytest.raises(RuntimeError, match="evaluation failure"):
        failing_runner.run()
    assert CountingLogger.instances[-1].close_count == 1


def test_caller_owned_evaluation_logger_is_never_closed() -> None:
    class CountingLogger:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    logger = CountingLogger()
    runner = PrimaryEvaluationProtocol(
        _artifact(), config=SystemConfig(), logger=logger, environment_factory=_factory
    )
    runner.close()
    assert logger.close_count == 0


def test_custom_protocol_closes_replaced_superclass_logger(monkeypatch) -> None:
    class CountingLogger:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.close_count = 0
            self.__class__.instances.append(self)

        def info(self, *_args, **_kwargs):
            return None

        def event(self, *_args, **_kwargs):
            return None

        def close(self):
            self.close_count += 1

    monkeypatch.setattr("bang_rl.evaluation.protocol.StructuredLogger", CountingLogger)
    config = SystemConfig(
        evaluation=EvaluationConfig(
            protocol_id="custom_v1", games=1, seats=[0], base_seed=17
        )
    )
    runner = CustomEvaluationProtocol(
        _artifact(), config=config, environment_factory=_factory
    )
    superclass_logger, custom_logger = CountingLogger.instances[-2:]
    assert superclass_logger.close_count == 1

    runner.run(config.evaluation)
    assert custom_logger.close_count == 1


def test_primary_metrics_and_wilson_ci_are_reported_with_exact_denominators() -> None:
    report = run_primary_evaluation(_artifact(), config=SystemConfig(), environment_factory=_factory)

    assert report.normal_wins == 100
    assert report.normal_losses == 400
    assert report.win_rate == pytest.approx(0.2)
    assert isinstance(report.win_rate_ci, ConfidenceInterval)
    assert 0.0 <= report.win_rate_ci.lower < report.win_rate < report.win_rate_ci.upper <= 1.0
    assert report.truncated_rate == 0.0
    assert report.role_specific_results["SHERIFF"]["normal_games"] == 100
    assert report.metric_samples["role_classification"] == 400
    assert report.metric_samples["hidden_role_classification"] == 400
    assert report.estimated_emotion_mse == pytest.approx(0.0)


def test_artifact_config_compatibility_mismatch_is_rejected() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="environment_version"):
        PrimaryEvaluationProtocol(
            _artifact(environment_version="incompatible"),
            config=SystemConfig(),
            environment_factory=_factory,
        )


def test_report_json_output_contains_protocol_and_compatibility_metadata(tmp_path) -> None:
    path = tmp_path / "evaluation.json"
    report = run_primary_evaluation(
        _artifact(), config=SystemConfig(), environment_factory=_factory, report_path=path
    )
    payload = path.read_text(encoding="utf-8")
    assert payload == report.to_json()
    assert '"id": "deterministic_v1"' in payload
    assert '"seed_formula": "base_seed + evaluation_index"' in payload
    assert report.compatibility["validated"] is True
    assert report.output_metadata["report_path"] == str(path)

class _StochasticPolicy(_Policy):
    def act(self, _inputs, _mask, *, deterministic=False, generator=None):
        assert deterministic is False
        if generator is None:
            action = torch.tensor(0)
        else:
            action = torch.randint(0, 2, (), generator=generator)
        return _Action(action)


def test_custom_evaluation_supports_counts_seats_presets_and_normal_denominators() -> None:
    config = SystemConfig(
        evaluation=EvaluationConfig(
            protocol_id="custom_v1",
            games=4,
            games_per_seat=2,
            seats=[0, 1],
            opponent_preset="alternate",
            opponent_presets={"alternate": PresetConfig(preset_id="alternate", version="2.0")},
            base_seed=41,
        )
    )
    report = run_custom_evaluation(
        _artifact(), config=config, evaluation=config.evaluation, environment_factory=_factory
    )

    assert report.protocol["id"] == "custom_v1"
    assert report.protocol["primary_comparable"] is False
    assert report.total_games == 4
    assert report.seed_set == (41, 42, 43, 44)
    assert report.seat_counts == {"0": 2, "1": 2, "2": 0, "3": 0, "4": 0}
    assert report.opponent["preset_id"] == "alternate"
    assert report.normal_games == 4
    assert report.win_rate == pytest.approx(0.5)
    assert report.win_rate_ci is not None


def test_stochastic_evaluation_is_seed_reproducible_and_does_not_mutate_artifact() -> None:
    artifact = _artifact()
    artifact.ppo = _StochasticPolicy()
    config = SystemConfig(
        evaluation=EvaluationConfig(
            protocol_id="stochastic_v1",
            games=2,
            action_mode="stochastic",
            deterministic=False,
            seat_policy="fixed",
            seats=[0],
            base_seed=9,
        )
    )

    first = CustomEvaluationProtocol(
        artifact, config=config, environment_factory=_factory
    ).run(config.evaluation)
    second = CustomEvaluationProtocol(
        artifact, config=config, environment_factory=_factory
    ).run(config.evaluation)

    assert first.to_dict() == second.to_dict()
    assert first.protocol["kind"] == "stochastic"
    assert first.output_metadata["rng_isolation"] == "per_game_environment_and_agent_streams"
    assert first.output_metadata["training_mutation"] is False


def test_custom_report_contains_primary_comparison_without_merging_protocols() -> None:
    config = SystemConfig()
    primary = run_primary_evaluation(_artifact(), config=config, environment_factory=_factory)
    custom_config = EvaluationConfig(protocol_id="custom_v1", games=2, base_seed=100)
    custom = run_custom_evaluation(
        _artifact(),
        config=config,
        evaluation=custom_config,
        primary_report=primary,
        environment_factory=_factory,
    )

    assert custom.comparison["available"] is True
    assert custom.comparison["primary_evaluation_run_id"] == primary.evaluation_run_id
    assert custom.comparison["same_conditions"] is False
    assert custom.comparison["display_as_continuous_performance"] is False
    assert custom.to_dict()["comparison"]["primary_protocol_id"] == "deterministic_v1"


def test_custom_evaluation_rejects_seed_and_seat_mismatches() -> None:
    with pytest.raises(Exception, match="seeds"):
        EvaluationConfig(protocol_id="custom_v1", games=2, seeds=[1])

    config = SystemConfig()
    evaluation = EvaluationConfig(
        protocol_id="custom_v1",
        games=4,
        games_per_seat=3,
        seats=[0, 1],
    )
    runner = CustomEvaluationProtocol(_artifact(), config=config, environment_factory=_factory)
    with pytest.raises(ValueError, match="games_per_seat"):
        runner.run(evaluation)


def test_explicit_environment_owned_logger_cleanup_runs_once_on_all_paths() -> None:
    class CountingLogger:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class OwnedEnvironment(_TinyEnvironment):
        def __init__(self, episode_id, seed, config):
            super().__init__(episode_id, seed, config)
            self.owned_logger = CountingLogger()
            self.owned_loggers = (self.owned_logger,)
            self.close_owned_calls = 0

        def close_owned_loggers(self):
            self.close_owned_calls += 1
            for logger in self.owned_loggers:
                logger.close()

    class TruncatedEnvironment(OwnedEnvironment):
        def step(self, action):
            super().step(action)
            self.state.terminated = False
            self.state.truncated = True
            self.state.termination_status = TerminationStatus.TRUNCATED
            self.state.terminal_result = None

    class FailingEnvironment(OwnedEnvironment):
        def step(self, _action):
            raise RuntimeError("owned logger failure")

    created = []

    def run(environment_type):
        def factory(episode_id, seed, config):
            environment = environment_type(episode_id, seed, config)
            created.append(environment)
            return environment

        runner = PrimaryEvaluationProtocol(
            _artifact(), config=SystemConfig(), environment_factory=factory
        )
        runner.evaluation_run_id = "owned-logger"
        return runner

    run(OwnedEnvironment)._run_game(0, 0, 0, SystemConfig().preset)
    run(TruncatedEnvironment)._run_game(0, 0, 0, SystemConfig().preset)
    with pytest.raises(RuntimeError, match="owned logger failure"):
        run(FailingEnvironment)._run_game(0, 0, 0, SystemConfig().preset)

    assert [environment.close_owned_calls for environment in created] == [1, 1, 1]
    assert [environment.owned_logger.close_count for environment in created] == [1, 1, 1]


def test_explicit_factory_owned_logger_hook_runs_once() -> None:
    class CountingLogger:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class Factory:
        def __init__(self):
            self.logger = CountingLogger()
            self.close_owned_calls = 0

        def __call__(self, episode_id, seed, config):
            return _TinyEnvironment(episode_id, seed, config)

        def close_owned_loggers(self):
            self.close_owned_calls += 1
            self.logger.close()

    factory = Factory()
    runner = PrimaryEvaluationProtocol(
        _artifact(), config=SystemConfig(), environment_factory=factory
    )
    runner.evaluation_run_id = "factory-owned-logger"
    runner._run_game(0, 0, 0, SystemConfig().preset)

    assert factory.close_owned_calls == 1
    assert factory.logger.close_count == 1


def test_factory_owned_logger_closes_without_environment_cleanup_hook() -> None:
    class CountingLogger:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class Factory:
        def __init__(self, environment_type):
            self.environment_type = environment_type
            self.logger = CountingLogger()
            self.owned_loggers = (self.logger,)

        def __call__(self, episode_id, seed, config):
            return self.environment_type(episode_id, seed, config)

    class TruncatedEnvironment(_TinyEnvironment):
        def step(self, action):
            super().step(action)
            self.state.terminated = False
            self.state.truncated = True
            self.state.termination_status = TerminationStatus.TRUNCATED
            self.state.terminal_result = None

    class FailingEnvironment(_TinyEnvironment):
        def step(self, _action):
            raise RuntimeError("factory-owned evaluation failure")

    for environment_type, error in (
        (_TinyEnvironment, None),
        (TruncatedEnvironment, None),
        (FailingEnvironment, "factory-owned evaluation failure"),
    ):
        factory = Factory(environment_type)
        runner = PrimaryEvaluationProtocol(
            _artifact(), config=SystemConfig(), environment_factory=factory
        )
        runner.evaluation_run_id = "factory-owned-no-environment-hook"
        if error is None:
            runner._run_game(0, 0, 0, SystemConfig().preset)
        else:
            with pytest.raises(RuntimeError, match=error):
                runner._run_game(0, 0, 0, SystemConfig().preset)
        assert factory.logger.close_count == 1
