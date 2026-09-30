"""Focused tests for TASK-2-2 public role-belief estimation."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from bang_rl.agents import DummyAgent, RoleBeliefEstimator


def public_observation(
    *,
    player_idx: int = 2,
    self_role: int = 4,
    revealed_roles: list[int] | None = None,
    confirmed: np.ndarray | None = None,
    events: tuple[dict, ...] = (),
) -> dict:
    if revealed_roles is None:
        revealed_roles = [0, 255, 255, 255, 255]
    if confirmed is None:
        confirmed = np.zeros((5, 5), dtype=np.float32)
        confirmed[0, 0] = 1.0
        confirmed[player_idx, self_role] = 1.0
    return {
        "current_player_idx": np.asarray([player_idx], dtype=np.uint8),
        "self_role": np.asarray([self_role], dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "revealed_roles": np.asarray(revealed_roles, dtype=np.uint8),
        "confirmed_role_distribution": confirmed,
        "event_sequence": events,
    }


def test_beliefs_are_five_role_normalized_and_use_remaining_role_counts() -> None:
    estimator = RoleBeliefEstimator(player_idx=2)

    beliefs = estimator.initialize(
        public_observation(
            revealed_roles=[0, 255, 255, 255, 1],
            confirmed=np.asarray(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
        )
    )

    assert beliefs.shape == (5, 5)
    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0)
    np.testing.assert_array_equal(beliefs[[0, 2, 4]], np.eye(5, dtype=np.float32)[[0, 4, 1]])
    np.testing.assert_allclose(beliefs[1], [0, 0, 0.5, 0.5, 0])
    np.testing.assert_allclose(beliefs[3], [0, 0, 0.5, 0.5, 0])


def test_self_sheriff_revealed_and_confirmed_roles_are_hard_constraints() -> None:
    estimator = RoleBeliefEstimator(player_idx=3)
    beliefs = estimator.initialize(
        public_observation(
            player_idx=3,
            self_role=2,
            revealed_roles=[0, 1, 255, 2, 255],
        )
    )

    for player_idx, role in ((0, 0), (1, 1), (3, 2)):
        np.testing.assert_array_equal(beliefs[player_idx], np.eye(5, dtype=np.float32)[role])
    # The two unconfirmed slots contain exactly the two remaining roles.
    np.testing.assert_allclose(beliefs[2], [0, 0, 0, 0.5, 0.5])
    np.testing.assert_allclose(beliefs[4], [0, 0, 0, 0.5, 0.5])


def test_public_role_reveal_updates_deterministically_and_duplicate_events_are_idempotent() -> None:
    reveal = {
        "event_id": 7,
        "event_type": "role_revealed",
        "eliminated_idx": 1,
        "revealed_role": 3,
        "public": True,
    }
    estimator = RoleBeliefEstimator(player_idx=2)
    first = estimator.initialize(public_observation(events=(reveal,)))
    second = estimator.update_from_events([reveal, reveal])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(second[1], np.eye(5, dtype=np.float32)[3])
    assert estimator.last_event_id == 7
    assert estimator.state_dict()["processed_event_ids"] == [7]


def test_only_public_inputs_are_accepted_and_privileged_role_fields_are_rejected() -> None:
    estimator = RoleBeliefEstimator(player_idx=0)
    with pytest.raises(ValueError, match="privileged"):
        estimator.initialize({"roles": [0, 1, 2, 3, 4], **public_observation()})

    source = inspect.getsource(RoleBeliefEstimator)
    assert "true_roles" not in source
    assert "BangEnvironment" not in source


def test_state_serialization_reset_and_reproducibility() -> None:
    observation = public_observation(
        revealed_roles=[0, 255, 255, 255, 1],
        self_role=4,
        player_idx=2,
    )
    first = RoleBeliefEstimator(player_idx=2)
    expected = first.initialize(observation)
    restored = RoleBeliefEstimator.from_json(first.to_json())

    np.testing.assert_array_equal(restored.beliefs, expected)
    assert restored.state_dict() == first.state_dict()
    assert "true_roles" not in first.to_json()
    assert "privileged" not in first.to_json()

    reset = restored.reset(observation)
    np.testing.assert_array_equal(reset, expected)
    np.testing.assert_array_equal(RoleBeliefEstimator(player_idx=2).initialize(observation), expected)


def test_dummy_agent_exposes_estimator_without_main_agent_dependency() -> None:
    agent = DummyAgent(agent_id="player_2", seed=12)
    observation = public_observation()
    mask = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.uint8)

    assert agent.act({**observation, "action_mask": mask}, mask) == 0
    np.testing.assert_array_equal(agent.role_beliefs, agent.role_estimator.beliefs)
    restored = DummyAgent.from_state_dict(agent.state_dict())
    np.testing.assert_array_equal(restored.role_beliefs, agent.role_beliefs)


from bang_rl.agents import (
    ActionSelector,
    FactionStrengthEstimator,
    IntentCategory,
    IntentModule,
)


def complete_observation(
    *,
    player_idx: int = 4,
    self_role: int = 4,
    alive: tuple[bool, ...] = (True, True, True, True, True),
    healths: tuple[int, ...] = (5, 4, 4, 4, 4),
    confirmed_roles: dict[int, int] | None = None,
    sheriff_idx: int = 0,
) -> dict:
    confirmed = np.zeros((5, 5), dtype=np.float32)
    known = {sheriff_idx: 0, player_idx: self_role}
    if confirmed_roles:
        known.update(confirmed_roles)
    revealed = [255] * 5
    for index, role in known.items():
        confirmed[index, role] = 1.0
        if index != player_idx:
            revealed[index] = role
    return {
        **public_observation(
            player_idx=player_idx,
            self_role=self_role,
            revealed_roles=revealed,
            confirmed=confirmed,
        ),
        "healths": np.asarray(healths, dtype=np.uint8),
        "alive": np.asarray(alive, dtype=np.bool_),
        "sheriff_idx": np.asarray([sheriff_idx], dtype=np.uint8),
        "action_mask": np.asarray([1, 1, 1, 1, 1, 1], dtype=np.uint8),
    }


def one_hot_beliefs(roles: tuple[int, ...]) -> np.ndarray:
    values = np.zeros((5, 5), dtype=np.float32)
    values[np.arange(5), np.asarray(roles)] = 1.0
    return values


def test_faction_strength_uses_normalized_role_beliefs_and_public_health() -> None:
    observation = complete_observation()
    beliefs = one_hot_beliefs((0, 1, 2, 3, 4))
    report = FactionStrengthEstimator().estimate(observation, beliefs)

    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0)
    assert report["sheriff"].survivor_count == 2.0
    assert report["outlaw"].survivor_count == 2.0
    assert report["sheriff"].expected_health == 9.0
    assert report["outlaw"].expected_health == 8.0
    assert report.strong_faction == "sheriff"


def test_intent_categories_are_role_and_faction_aware() -> None:
    observation = complete_observation(
        player_idx=4,
        self_role=4,
        healths=(5, 4, 5, 4, 4),
    )
    beliefs = one_hot_beliefs((0, 1, 2, 3, 4))
    module = IntentModule(player_idx=4, config={"tie_policy": "outlaw"})
    decision = module.decision(observation, beliefs)

    assert decision.category is IntentCategory.ATTACK_STRONG_FACTION
    assert decision.strong_faction == "outlaw"
    # The public tie policy selects the outlaw faction, not a hidden role.
    assert decision.target_scores[2] > decision.target_scores[0]

    duel = complete_observation(
        player_idx=4,
        self_role=4,
        alive=(True, False, False, False, True),
        healths=(5, 0, 0, 0, 4),
        confirmed_roles={4: 4},
    )
    duel_decision = module.decision(duel, beliefs)
    assert duel_decision.category is IntentCategory.ATTACK_SHERIFF_DUEL
    assert duel_decision.target_scores[0] == max(duel_decision.target_scores)


def test_action_selector_and_dummy_agent_never_choose_masked_actions() -> None:
    scores = np.asarray([0.0, 9.0, 3.0, 3.0, 3.0, 3.0])
    mask = np.asarray([1, 0, 1, 1, 0, 0], dtype=np.uint8)
    assert ActionSelector.select(scores, mask) == 2

    agent = DummyAgent(agent_id="player_4", seed=5)
    observation = complete_observation(player_idx=4, self_role=4)
    action_mask = np.asarray([1, 0, 0, 1, 0, 1], dtype=np.uint8)
    action = agent.act({**observation, "action_mask": action_mask}, action_mask)
    assert action in np.flatnonzero(action_mask)
    assert action != 1


def test_intent_policy_state_is_stable_and_versioned_without_private_roles() -> None:
    observation = complete_observation(player_idx=4, self_role=4)
    agent = DummyAgent(
        agent_id="player_4",
        seed=5,
        policy_config={"tie_policy": "outlaw", "role_target_weights": {"outlaw_a": 2.0}},
    )
    agent.act(observation, observation["action_mask"])
    encoded = agent.to_json()
    restored = DummyAgent.from_json(encoded)

    assert encoded == restored.to_json()
    assert restored.state_dict()["schema_version"] == "2.0"
    assert restored.intent_module.state_dict()["schema_version"] == "1.0"
    assert restored.policy_config.tie_policy == "outlaw"
    assert "true_roles" not in encoded
    assert "actual_roles" not in encoded
    assert "privileged" not in encoded


# TASK-2-5 integration coverage

def test_action_selector_combines_emotion_effects_and_renormalizes_mask() -> None:
    selection = ActionSelector.select_with_info(
        np.asarray([1.0, 1.0, 50.0, 1.0, 1.0, 1.0]),
        np.asarray([1, 1, 0, 1, 0, 0], dtype=np.uint8),
        emotion_contribution=np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        mode="deterministic",
    )

    assert selection.action == 1
    assert selection.masked_probabilities[2] == 0.0
    np.testing.assert_allclose(sum(selection.masked_probabilities), 1.0)
    assert selection.combined_scores[1] == 3.0
    assert selection.to_dict()["selected_probability"] == selection.masked_probabilities[1]


def test_action_selector_deterministic_ties_and_stochastic_mode_are_mask_safe() -> None:
    mask = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.uint8)
    assert ActionSelector.select([4.0] * 6, mask) == 0

    rng = np.random.default_rng(13)
    samples = [
        ActionSelector.select([0.0] * 6, mask, mode="stochastic", rng=rng)
        for _ in range(80)
    ]
    assert set(samples).issubset({0, 1, 3})
    assert all(mask[action] == 1 for action in samples)

    with pytest.raises(ValueError, match="action mode"):
        ActionSelector.select([0.0] * 6, mask, mode="unsupported")


def test_dummy_agent_emotion_influences_integrated_action_and_info() -> None:
    from bang_rl.config import EmotionConfig

    config = EmotionConfig(
        schema_version="selector-emotion-v1",
        emotions=[
            {
                "name": "urgency",
                "initial_value": 1.0,
                "effects": {"pass": 20.0},
            }
        ],
    )
    observation = complete_observation(player_idx=0, self_role=0)
    agent = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=config,
    )
    action = agent.act(observation, observation["action_mask"])

    assert action == 0
    assert agent.last_decision_info is not None
    assert agent.last_decision_info.emotion_contribution[0] == 20.0
    assert agent.decision_info is not None
    assert agent.decision_info["action"] == 0


def test_dummy_agent_stochastic_replay_and_versioned_state_round_trip() -> None:
    observation = complete_observation(player_idx=4, self_role=4)
    first = DummyAgent(agent_id="player_4", seed=41, action_mode="stochastic")
    second = DummyAgent(agent_id="player_4", seed=41, action_mode="stochastic")

    first_actions = [first.act(observation, observation["action_mask"]) for _ in range(30)]
    second_actions = [second.act(observation, observation["action_mask"]) for _ in range(30)]
    assert first_actions == second_actions
    assert all(observation["action_mask"][action] == 1 for action in first_actions)

    encoded = first.to_json()
    restored = DummyAgent.from_json(encoded)
    assert encoded == restored.to_json()
    assert restored.action_mode == "stochastic"
    assert restored.last_decision_info is not None


def test_dummy_agent_invalid_and_private_paths_do_not_update_emotion_or_selection() -> None:
    observation = complete_observation(player_idx=4, self_role=4)
    mask = observation["action_mask"]
    agent = DummyAgent(agent_id="player_4", seed=7)
    before = agent.state_dict()

    with pytest.raises(ValueError, match="does not match"):
        agent.act({**observation, "action_mask": np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)}, mask)
    assert agent.state_dict() == before

    with pytest.raises(ValueError, match="privileged"):
        agent.act({**observation, "true_roles": [0, 1, 2, 3, 4]}, mask)
    assert agent.state_dict() == before

    private_event_observation = {
        **observation,
        "event_sequence": ({"event_id": 1, "event_type": "bang", "true_roles": [0]},),
    }
    with pytest.raises(ValueError, match="privileged"):
        agent.act(private_event_observation, mask)
    assert agent.state_dict() == before
    assert "true_roles" not in json.dumps(agent.decision_info or {})


def test_dummy_agent_lifecycle_reset_and_environment_integration() -> None:
    from bang_rl.env import BangEnvironment

    agent = DummyAgent(agent_id="player_0", seed=17)
    observation = complete_observation(player_idx=0, self_role=0)
    agent.on_episode_start({"episode_id": "selector-episode", "seed": 91})
    agent.act(observation, observation["action_mask"])
    agent.on_episode_end({"episode_id": "selector-episode", "terminated": True})
    with pytest.raises(RuntimeError, match="ended"):
        agent.act(observation, observation["action_mask"])
    agent.on_episode_start({"episode_id": "selector-episode-2", "seed": 92})
    assert agent.last_decision_info is None

    environment = BangEnvironment(episode_id="dummy-integration", seed=23, max_episode_steps=24, warn_turn_threshold=8, critical_turn_threshold=16)
    environment.reset()
    agents = {
        name: DummyAgent(agent_id=name, seed=100 + index)
        for index, name in enumerate(environment.possible_agents)
    }
    for name in environment.possible_agents:
        agents[name].on_episode_start({"episode_id": "dummy-integration", "seed": 23})

    for _ in range(40):
        if environment.state.termination_status.value != "running":
            break
        selected = environment.agent_selection
        assert selected is not None
        public_observation = environment.observe(selected)
        action = agents[selected].act(public_observation, public_observation["action_mask"])
        assert public_observation["action_mask"][action] == 1
        environment.step(action)

    assert environment.environment_step > 0
    assert environment.environment_step <= 24


# FEAT-004 per-target relation weighting integration coverage


def _grudge_emotion_config():
    """EmotionConfig whose ``shot_by`` relation update builds a grudge."""

    from bang_rl.config import EmotionConfig

    return EmotionConfig(
        schema_version="relation-target-v1",
        relation_updates={"shot_by": -0.6},
    )


def _shot_by_observation(*, shooter_idx: int, player_idx: int = 0):
    """Observation carrying a public event where ``shooter_idx`` banged us."""

    observation = complete_observation(player_idx=player_idx, self_role=0)
    event = {
        "event_id": 1,
        "event_type": "bang",
        "actor_idx": shooter_idx,
        "target_idx": player_idx,
        "hp_before": 4,
        "hp_after": 3,
        "public": True,
    }
    return {**observation, "event_sequence": (event,), "environment_step": np.asarray([1], dtype=np.int64)}


def test_relation_target_weight_defaults_to_zero_and_is_a_no_op() -> None:
    # With the default weight (0.0) the combined scores and selected action are
    # identical to a baseline agent that never sees the feature.
    observation = _shot_by_observation(shooter_idx=3, player_idx=0)
    mask = observation["action_mask"]

    baseline = DummyAgent(agent_id="player_0", seed=8, emotion_config=_grudge_emotion_config())
    feature = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=_grudge_emotion_config(),
        policy_config={"relation_target_weight": 0.0},
    )
    baseline_action = baseline.act(observation, mask)
    feature_action = feature.act(observation, mask)

    assert baseline_action == feature_action
    assert (
        baseline.last_decision_info.combined_scores
        == feature.last_decision_info.combined_scores
    )
    # Every emotion contribution entry is identical because the added term is 0.
    assert (
        baseline.last_decision_info.emotion_contribution
        == feature.last_decision_info.emotion_contribution
    )


def test_positive_weight_and_grudge_raises_bang_score_for_resented_seat() -> None:
    # Seat 3 shot us -> relation[3] < 0.  With a positive relation_target_weight
    # the bang-at-seat-3 action (index 4) gains score relative to banging a
    # non-resented seat (for example seat 1, action index 2).
    observation = _shot_by_observation(shooter_idx=3, player_idx=0)
    mask = observation["action_mask"]

    baseline = DummyAgent(agent_id="player_0", seed=8, emotion_config=_grudge_emotion_config())
    feature = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=_grudge_emotion_config(),
        policy_config={"relation_target_weight": 5.0},
    )
    baseline.act(observation, mask)
    feature.act(observation, mask)

    # The resented seat's bang action (index 4) is boosted versus baseline.
    assert (
        feature.last_decision_info.emotion_contribution[4]
        > baseline.last_decision_info.emotion_contribution[4]
    )
    # A non-resented seat (index 2 == bang seat 1) is not boosted.
    assert (
        feature.last_decision_info.emotion_contribution[2]
        == baseline.last_decision_info.emotion_contribution[2]
    )
    # Relative to banging the non-resented seat, the resented seat scores higher.
    contribution = feature.last_decision_info.emotion_contribution
    assert contribution[4] > contribution[2]


def test_relation_state_is_not_mutated_by_action_selection() -> None:
    observation = _shot_by_observation(shooter_idx=3, player_idx=0)
    mask = observation["action_mask"]
    agent = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=_grudge_emotion_config(),
        policy_config={"relation_target_weight": 5.0},
    )
    agent.act(observation, mask)
    relation_after_first = agent.emotion_module.relation.copy()
    # A second act on the SAME event (deduped by event_id) must not change the
    # relation vector, confirming scoring reads but never mutates relation.
    agent.act(observation, mask)
    np.testing.assert_array_equal(agent.emotion_module.relation, relation_after_first)


def test_relation_target_weight_preserves_determinism() -> None:
    observation = _shot_by_observation(shooter_idx=3, player_idx=0)
    mask = observation["action_mask"]
    first = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=_grudge_emotion_config(),
        policy_config={"relation_target_weight": 5.0},
    )
    second = DummyAgent(
        agent_id="player_0",
        seed=8,
        emotion_config=_grudge_emotion_config(),
        policy_config={"relation_target_weight": 5.0},
    )
    assert first.act(observation, mask) == second.act(observation, mask)


def test_relation_target_weight_round_trips_through_policy_state() -> None:
    agent = DummyAgent(
        agent_id="player_0",
        seed=8,
        policy_config={"relation_target_weight": 2.5},
    )
    assert agent.policy_config.relation_target_weight == 2.5
    restored = DummyAgent.from_json(agent.to_json())
    assert restored.policy_config.relation_target_weight == 2.5
