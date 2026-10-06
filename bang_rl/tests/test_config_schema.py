"""Focused tests for TASK-0-3 configuration schemas."""

import pytest
from pydantic import ValidationError

from bang_rl.config import (
    ArtifactConfig,
    DatasetConfig,
    EmotionConfig,
    EmotionDefinition,
    EnvConfig,
    EvaluationConfig,
    HardwareConfig,
    IntentPolicyConfig,
    PauseConfig,
    PPOConfig,
    PresetConfig,
    RLTrainingConfig,
    RewardConfig,
    SLTrainingConfig,
    SystemConfig,
)


def test_system_defaults_match_design_contract() -> None:
    config = SystemConfig()

    assert config.env.max_episode_steps == 2_000
    assert config.env.event_history.max_events is None
    assert config.dataset.canonical_shard_max_episodes == 10_000
    assert config.dataset.canonical_shard_max_bytes == 256_000_000
    assert config.rl_training.ppo == PPOConfig()
    assert config.rl_training.emotion_inference.initial_baseline == 0.0


def test_environment_thresholds_must_be_strictly_ordered() -> None:
    with pytest.raises(ValidationError):
        EnvConfig(warn_turn_threshold=100, critical_turn_threshold=100, max_episode_steps=200)
    with pytest.raises(ValidationError):
        EnvConfig(warn_turn_threshold=100, critical_turn_threshold=201, max_episode_steps=200)
    with pytest.raises(ValidationError):
        EnvConfig(warn_turn_threshold=0, critical_turn_threshold=100, max_episode_steps=200)


def test_event_history_is_unlimited_or_positive() -> None:
    assert EnvConfig(event_history={"max_events": None}).event_history.max_events is None
    assert EnvConfig(event_history={"max_events": 10}).event_history.max_events == 10

    for invalid in (0, -1, True, "10"):
        with pytest.raises(ValidationError):
            EnvConfig(event_history={"max_events": invalid})


def test_dataset_limits_can_be_overridden() -> None:
    config = DatasetConfig(
        canonical_shard_max_episodes=25,
        canonical_shard_max_bytes=4096,
    )
    assert config.canonical_shard_max_episodes == 25
    assert config.canonical_shard_max_bytes == 4096

    with pytest.raises(ValidationError):
        DatasetConfig(canonical_shard_max_bytes=0)


def test_ppo_and_emotion_defaults_are_config_values() -> None:
    ppo = PPOConfig()
    assert ppo.rollout_transitions_per_worker == 256
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.ppo_epochs == 4
    assert ppo.minibatch_size == 64
    assert ppo.clip_epsilon == 0.2
    assert ppo.optimizer.learning_rate == 3e-4

    assert EmotionConfig().initial_baseline == 0.0
    for invalid in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            EmotionConfig(initial_baseline=invalid)


def test_hardware_settings_are_positive_and_user_controlled() -> None:
    hardware = HardwareConfig(
        cpu_cores=8,
        ram_gb=32,
        use_gpu=True,
        workers=4,
        batch_size=128,
        buffer_size=4096,
    )
    assert hardware.model_dump() == {
        "cpu_cores": 8,
        "ram_gb": 32.0,
        "use_gpu": True,
        "workers": 4,
        "batch_size": 128,
        "buffer_size": 4096,
    }

    for field, value in (("cpu_cores", 0), ("ram_gb", 0), ("workers", 0), ("batch_size", 0), ("buffer_size", 0)):
        with pytest.raises(ValidationError):
            HardwareConfig(**{field: value})

    with pytest.raises(ValidationError):
        HardwareConfig(cpu_cores=True)


def test_reward_dsl_keeps_terminal_and_shaping_namespaces_separate() -> None:
    config = RewardConfig(
        version="reward-v1",
        terminal={
            "components": [
                {
                    "id": "terminal_win",
                    "enabled": True,
                    "condition": {"type": "terminal_result", "result": "win"},
                    "value": 1.25,
                }
            ]
        },
        shaping={
            "components": [
                {
                    "id": "bang_hit",
                    "enabled": True,
                    "condition": {"type": "event", "event": "bang_hit"},
                    "value": 0.5,
                },
                {
                    "id": "pass_penalty",
                    "enabled": False,
                    "condition": {"type": "pass"},
                    "value": -0.25,
                },
            ]
        },
    )

    assert len(config.terminal.components) == 1
    assert len(config.shaping.components) == 2
    assert config.terminal.components[0].value == 1.25
    assert config.shaping.components[1].condition.type == "pass"


def test_reward_conditions_and_component_ids_are_validated() -> None:
    with pytest.raises(ValidationError):
        RewardConfig(
            version="v1",
            terminal={
                "components": [
                    {
                        "id": "terminal_win",
                        "enabled": True,
                        "condition": {"type": "terminal_result", "result": "loss"},
                        "value": 1,
                    }
                ]
            },
        )

    with pytest.raises(ValidationError):
        RewardConfig(
            version="v1",
            shaping={
                "components": [
                    {
                        "id": "bang_hit",
                        "enabled": True,
                        "condition": {"type": "event", "event": "not_an_event"},
                        "value": 1,
                    }
                ]
            },
        )

    with pytest.raises(ValidationError):
        RewardConfig(
            version="v1",
            shaping={
                "components": [
                    {
                        "id": "pass_penalty",
                        "enabled": True,
                        "condition": {"type": "pass", "event": "pass"},
                        "value": -1,
                    }
                ]
            },
        )


def test_fixed_model_and_optimizer_values_are_snapshot_verifiable() -> None:
    config = SystemConfig()
    snapshot = config.job_snapshot()
    metadata = config.artifact_metadata()

    assert snapshot["model_architecture"]["intent_hidden_size"] == 128
    assert snapshot["model_architecture"]["emotion_head"].endswith("Sigmoid")
    assert snapshot["model_architecture"]["ppo_policy_head"] == "Linear(256,6)"
    assert snapshot["optimizers"]["intent"]["learning_rate"] == 1e-3
    assert snapshot["optimizers"]["emotion"]["learning_rate"] == 1e-3
    assert snapshot["optimizers"]["ppo"]["learning_rate"] == 3e-4
    assert metadata["training"]["ppo"]["clip_epsilon"] == 0.2
    assert metadata["reward_config_version"] == config.reward_config.version
    assert metadata["reward_config_id"] == config.reward_config.config_identifier

    with pytest.raises(ValidationError):
        IntentPolicyConfig(optimizer={"learning_rate": 3e-4})


def test_all_task_level_schema_types_are_constructible() -> None:
    assert isinstance(PresetConfig(), PresetConfig)
    assert isinstance(SLTrainingConfig(), SLTrainingConfig)
    assert isinstance(RLTrainingConfig(), RLTrainingConfig)
    assert isinstance(EvaluationConfig(), EvaluationConfig)
    assert isinstance(PauseConfig(), PauseConfig)
    assert isinstance(ArtifactConfig(), ArtifactConfig)


def test_environment_diagnostic_threshold_defaults_match_requirements() -> None:
    env = EnvConfig()

    assert env.warn_turn_threshold == 200
    assert env.critical_turn_threshold == 500
    assert env.max_episode_steps == 2_000


def test_primary_evaluation_protocol_defaults_are_recordable() -> None:
    evaluation = EvaluationConfig()

    assert evaluation.protocol_id == "deterministic_v1"
    assert evaluation.games == 500
    assert evaluation.games_per_seat == 100
    assert evaluation.seat_policy == "balanced"
    assert evaluation.action_mode == "deterministic"
    assert evaluation.deterministic is True


def test_artifact_format_version_uses_design_name_and_legacy_alias() -> None:
    assert ArtifactConfig().inference_model_artifact_format_version == "1.0"
    assert ArtifactConfig(artifact_format_version="1.0").artifact_format_version == "1.0"
    assert SystemConfig().artifact_metadata()["inference_model_artifact_format_version"] == "1.0"


def test_reward_attack_component_conditions_match_component_ids() -> None:
    valid = RewardConfig(
        version="v1",
        shaping={
            "components": [
                {
                    "id": "attack_sheriff_outlaw",
                    "enabled": True,
                    "condition": {
                        "type": "role_attack",
                        "actor_role": "outlaw_a",
                        "target_role": "sheriff",
                    },
                    "value": 1,
                },
                {
                    "id": "role_or_faction_attack_terms",
                    "enabled": True,
                    "condition": {
                        "type": "faction_attack",
                        "actor_faction": "outlaw_team",
                        "target_faction": "sheriff_team",
                    },
                    "value": 1,
                },
            ]
        },
    )
    assert len(valid.shaping.components) == 2

    with pytest.raises(ValidationError):
        RewardConfig(
            version="v1",
            shaping={
                "components": [
                    {
                        "id": "attack_sheriff_deputy",
                        "enabled": True,
                        "condition": {
                            "type": "role_attack",
                            "actor_role": "outlaw_a",
                            "target_role": "sheriff",
                        },
                        "value": -1,
                    }
                ]
            },
        )


def test_custom_and_stochastic_evaluation_settings_validate_and_record_seeds() -> None:
    custom = EvaluationConfig(
        protocol_id="custom_v1",
        games=4,
        games_per_seat=2,
        seats=[0, 1],
        opponent_preset="alternate",
        opponent_presets={"alternate": PresetConfig(preset_id="alternate", version="2.0")},
    )
    assert custom.seats == [0, 1]
    assert custom.opponent_presets["alternate"].version == "2.0"

    stochastic = EvaluationConfig(
        protocol_id="stochastic_v1",
        action_mode="stochastic",
        deterministic=False,
        games=2,
        seed_formula="explicit",
        seeds=[7, 8],
    )
    assert stochastic.action_mode == "stochastic"
    assert stochastic.seeds == [7, 8]

    with pytest.raises(ValidationError):
        EvaluationConfig(protocol_id="stochastic_v1", action_mode="stochastic", games=2, seeds=[7])


def test_emotion_description_defaults_to_empty_and_is_backward_compatible() -> None:
    # An emotion with no description still validates (optional, default ""), so
    # every pre-existing config keeps loading under extra="forbid".
    emotion = EmotionDefinition(name="분노")
    assert emotion.description == ""
    # A full config built without any description also loads unchanged.
    config = EmotionConfig(emotions=[{"name": "분노"}, {"name": "공포"}])
    assert all(item.description == "" for item in config.emotions)


def test_emotion_description_round_trips_through_model_dump() -> None:
    emotion = EmotionDefinition(name="분노", description="상대의 공격에 대한 분노")
    assert emotion.description == "상대의 공격에 대한 분노"
    dumped = emotion.model_dump(mode="json")
    assert dumped["description"] == "상대의 공격에 대한 분노"
    # It survives a full round-trip through EmotionConfig.model_dump too.
    config = EmotionConfig(emotions=[{"name": "분노", "description": "분노 설명"}])
    reloaded = EmotionConfig(**config.model_dump(mode="json"))
    assert reloaded.emotions[0].description == "분노 설명"


def test_emotion_desc_alias_maps_to_description() -> None:
    emotion = EmotionDefinition(name="공포", desc="위협을 느낄 때의 공포")
    assert emotion.description == "위협을 느낄 때의 공포"
    # The alias also works through the emotions list normalization.
    config = EmotionConfig(emotions=[{"name": "공포", "desc": "공포 설명"}])
    assert config.emotions[0].description == "공포 설명"


def test_emotion_description_coerces_none_and_non_string() -> None:
    # A null description degrades to empty; a non-string is coerced via str().
    assert EmotionDefinition(name="x", description=None).description == ""
    assert EmotionDefinition(name="y", description=123).description == "123"
