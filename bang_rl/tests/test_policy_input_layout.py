"""Seat-aligned PPO input layout and role-balanced training seats."""

from __future__ import annotations

import numpy as np
import torch

from bang_rl.agents.main_agent import (
    EmotionInferenceModule,
    IntentInferenceModule,
    ObservationPreprocessor,
    PPOModule,
    RoleFusion,
)
from bang_rl.agents.main_agent.policy_inputs import (
    RELATIVE_LAYOUT,
    SEAT_ALIGNED_LAYOUT,
    build_policy_inputs,
    policy_input_layout,
    ppo_dims_for_layout,
)
from bang_rl.config import PPOConfig
from bang_rl.env.types import Role
from bang_rl.tests.test_main_ai_training import _first_observation
from bang_rl.training.main_ai_training import role_balanced_seat


def _modules():
    intent = IntentInferenceModule(role_count=4, seed=2).eval()
    emotion = EmotionInferenceModule(seed=2).eval()
    return intent, emotion


def _inputs(layout: str):
    intent, emotion = _modules()
    _, seat, observation = _first_observation()
    inputs = build_policy_inputs(
        observation,
        preprocessor=ObservationPreprocessor(),
        intent=intent,
        emotion=emotion,
        role_fusion=RoleFusion(),
        viewer_idx=seat,
        layout=layout,
    )
    return seat, observation, emotion, inputs


def test_seat_aligned_rows_follow_absolute_seats() -> None:
    seat, observation, _, relative = _inputs(RELATIVE_LAYOUT)
    _, _, _, aligned = _inputs(SEAT_ALIGNED_LAYOUT)
    others = [slot for slot in range(5) if slot != seat]

    intent = aligned.features["fused_intent"].reshape(5, 5)
    assert np.array_equal(intent[seat], np.zeros(5, dtype=np.float32))
    assert np.array_equal(intent[others], relative.features["fused_intent"])

    emotion = aligned.features["emotion_features"].reshape(5, -1)
    assert not emotion[seat].any()
    assert np.allclose(emotion[others], relative.estimated_emotion)

    self_state = aligned.features["self_state"]
    assert self_state.shape == (19,)
    assert np.array_equal(self_state[:9], relative.features["self_state"])
    own_role = int(np.asarray(observation["self_role"]).reshape(-1)[0])
    assert self_state[9 + own_role] == 1.0 and self_state[9:14].sum() == 1.0
    assert self_state[14 + seat] == 1.0 and self_state[14:19].sum() == 1.0


def test_seat_aligned_ppo_round_trips_and_reports_layout(tmp_path) -> None:
    _, _, emotion, aligned = _inputs(SEAT_ALIGNED_LAYOUT)
    ppo = PPOModule(**ppo_dims_for_layout(emotion, SEAT_ALIGNED_LAYOUT), seed=4)
    assert policy_input_layout(ppo) == SEAT_ALIGNED_LAYOUT
    chosen = ppo.act(aligned.ppo_inputs(), aligned.action_mask, deterministic=True)
    assert aligned.action_mask[int(chosen.action.reshape(-1)[0])] == 1

    path = tmp_path / "ppo.pt"
    ppo.save(path)
    restored = PPOModule.load(path)
    assert policy_input_layout(restored) == SEAT_ALIGNED_LAYOUT
    again = restored.act(aligned.ppo_inputs(), aligned.action_mask, deterministic=True)
    assert torch.equal(again.action, chosen.action)

    relative = PPOModule(**ppo_dims_for_layout(emotion, RELATIVE_LAYOUT), seed=4)
    assert policy_input_layout(relative) == RELATIVE_LAYOUT


def test_role_balanced_seat_cycles_through_roles() -> None:
    roles = (Role.OUTLAW_A, Role.SHERIFF, Role.TRAITOR, Role.OUTLAW_B, Role.DEPUTY)
    picked = [roles[role_balanced_seat(index, roles)] for index in range(8)]
    assert picked[:4] == [Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.TRAITOR]
    assert picked[6] == Role.OUTLAW_B


def test_ppo_config_defaults_keep_the_original_behaviour() -> None:
    config = PPOConfig()
    assert config.input_layout == RELATIVE_LAYOUT
    assert config.seat_sampling == "rotate"


def _role_heads_ppo(emotion):
    config = PPOConfig(input_layout=SEAT_ALIGNED_LAYOUT, role_heads=True)
    return PPOModule(**ppo_dims_for_layout(emotion, SEAT_ALIGNED_LAYOUT), ppo_config=config, seed=4)


def test_role_heads_need_the_seat_aligned_layout() -> None:
    _, emotion = _modules()
    config = PPOConfig(role_heads=True)
    try:
        PPOModule(**ppo_dims_for_layout(emotion, RELATIVE_LAYOUT), ppo_config=config, seed=4)
    except ValueError:
        return
    raise AssertionError("role_heads with the relative layout must be rejected")


def test_role_heads_pick_the_head_of_the_own_role(tmp_path) -> None:
    _, _, emotion, aligned = _inputs(SEAT_ALIGNED_LAYOUT)
    ppo = _role_heads_ppo(emotion)
    inputs = aligned.ppo_inputs()
    chosen = ppo.act(inputs, aligned.action_mask, deterministic=True)
    assert aligned.action_mask[int(chosen.action.reshape(-1)[0])] == 1

    # Changing only the own-role one-hot must switch to another head.
    swapped = {key: np.array(value, copy=True) for key, value in inputs.items()}
    own_role = int(np.argmax(swapped["self_state"][9:14]))
    other = (own_role + 1) % 5
    if other in (2, 3) and own_role in (2, 3):  # outlaw A/B share one head
        other = 4
    swapped["self_state"][9:14] = 0.0
    swapped["self_state"][9 + other] = 1.0
    with torch.no_grad():
        first = ppo.action_distribution(inputs, aligned.action_mask).logits
        second = ppo.action_distribution(swapped, aligned.action_mask).logits
    assert not torch.allclose(first, second)

    path = tmp_path / "ppo_role_heads.pt"
    ppo.save(path)
    restored = PPOModule.load(path)
    again = restored.act(inputs, aligned.action_mask, deterministic=True)
    assert torch.equal(again.action, chosen.action)


def test_role_heads_default_off() -> None:
    assert PPOConfig().role_heads is False
