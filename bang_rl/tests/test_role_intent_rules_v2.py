from __future__ import annotations

import numpy as np

from bang_rl.agents import IntentCategory, IntentModule, RoleBeliefEstimator


def obs(
    *,
    player_idx: int,
    self_role: int,
    events=(),
    healths=(5, 4, 4, 4, 4),
    alive=(1, 1, 1, 1, 1),
    revealed_roles=None,
):
    confirmed = np.zeros((5, 5), dtype=np.float32)
    confirmed[0, 0] = 1.0
    confirmed[player_idx, self_role] = 1.0
    revealed = [0, 255, 255, 255, 255] if revealed_roles is None else list(revealed_roles)
    return {
        "current_player_idx": np.asarray([player_idx], dtype=np.uint8),
        "viewer_idx": np.asarray([player_idx], dtype=np.uint8),
        "self_role": np.asarray([self_role], dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "revealed_roles": np.asarray(revealed, dtype=np.uint8),
        "confirmed_role_distribution": confirmed,
        "event_sequence": tuple(events),
        "healths": np.asarray(healths, dtype=np.uint8),
        "alive": np.asarray(alive, dtype=np.bool_),
        "action_mask": np.asarray([1, 1, 1, 1, 1, 1], dtype=np.uint8),
    }


def bang(event_id: int, actor: int, target: int):
    return {
        "event_id": event_id,
        "environment_step": event_id,
        "event_type": "bang",
        "actor_idx": actor,
        "target_idx": target,
        "public": True,
    }


def test_sheriff_belief_shooter_of_sheriff_becomes_more_outlaw_like():
    estimator = RoleBeliefEstimator(player_idx=0)
    before = estimator.initialize(obs(player_idx=0, self_role=0))
    baseline = float(before[1, 2] + before[1, 3])
    after = estimator.update(obs(player_idx=0, self_role=0, events=(bang(1, 1, 0),)))
    assert float(after[1, 2] + after[1, 3]) > baseline


def test_sheriff_belief_twice_targeted_non_sheriff_does_not_change_deputy_probability():
    estimator = RoleBeliefEstimator(player_idx=0)
    before = estimator.initialize(obs(player_idx=0, self_role=0))
    baseline = float(before[2, 1])
    after = estimator.update(
        obs(player_idx=0, self_role=0, events=(bang(1, 1, 2), bang(2, 3, 2)))
    )
    assert np.isclose(float(after[2, 1]), baseline)


def test_deputy_crossfire_actor_traitor_up_and_target_outlaw_up():
    estimator = RoleBeliefEstimator(player_idx=1)
    before = estimator.initialize(obs(player_idx=1, self_role=1))
    actor_traitor_before = float(before[2, 4])
    target_outlaw_before = float(before[3, 2] + before[3, 3])
    after = estimator.update(obs(player_idx=1, self_role=1, events=(bang(1, 2, 3),)))
    assert float(after[2, 4]) > actor_traitor_before
    assert float(after[3, 2] + after[3, 3]) > target_outlaw_before


def test_outlaw_attacker_of_self_becomes_more_deputy_or_traitor_like():
    estimator = RoleBeliefEstimator(player_idx=1)
    before = estimator.initialize(obs(player_idx=1, self_role=2))
    baseline = float(before[2, 1] + before[2, 4])
    after = estimator.update(obs(player_idx=1, self_role=2, events=(bang(1, 2, 1),)))
    assert float(after[2, 1] + after[2, 4]) > baseline


def test_sheriff_intent_prefers_higher_outlaw_probability():
    observation = obs(player_idx=0, self_role=0)
    beliefs = RoleBeliefEstimator(player_idx=0).initialize(
        obs(player_idx=0, self_role=0, events=(bang(1, 1, 0),))
    )
    decision = IntentModule(player_idx=0).decision(observation, beliefs)
    assert decision.category in {IntentCategory.ATTACK_OUTLAW_SUSPECT, IntentCategory.PASS}
    assert decision.target_scores[1] >= max(decision.target_scores[2:])


def test_outlaw_intent_attacks_sheriff_when_pressure_is_enough():
    observation = obs(player_idx=1, self_role=2, healths=(3, 4, 4, 4, 4))
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    assert decision.category is IntentCategory.ATTACK_SHERIFF_TEAM
    assert decision.target_scores[0] == 5.0
    assert decision.goal_stage == "finish_sheriff"


def test_outlaw_opening_identity_reveal_cost_reduces_sheriff_attack():
    observation = obs(player_idx=1, self_role=2, healths=(5, 4, 4, 4, 4))
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    # Setup Sheriff score 1.0 - opening IdentityRevealCost 1.0 = 0.0.
    assert decision.target_scores[0] == 0.0
    assert decision.action_scores[0] == 0.0
    assert decision.goal_stage == "weaken_sheriff_team_before_finish"
    assert max(decision.target_scores[2:]) > decision.target_scores[0]


def test_outlaw_crossfire_does_not_reduce_identity_reveal_cost():
    observation = obs(
        player_idx=1,
        self_role=2,
        events=(bang(1, 2, 3),),
        healths=(5, 4, 4, 4, 4),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    # Ordinary crossfire is not a signal to reveal Outlaw alignment.
    # Setup 1.0 - opening reveal cost 1.0 = 0.0.
    assert np.isclose(decision.target_scores[0], 0.0)
    assert decision.goal_stage == "weaken_sheriff_team_before_finish"


def test_outlaw_reveal_cost_falls_after_someone_attacks_sheriff():
    observation = obs(
        player_idx=1,
        self_role=2,
        events=(bang(1, 2, 0),),
        healths=(5, 4, 4, 4, 4),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    # Setup 1.0 - post-Sheriff-attack cost 0.3 = 0.7.
    assert np.isclose(decision.target_scores[0], 0.7)
    assert decision.goal_stage == "weaken_sheriff_team_before_finish"


def test_outlaw_identity_reveal_cost_is_zero_after_self_shoots_sheriff():
    observation = obs(
        player_idx=1,
        self_role=2,
        events=(bang(1, 1, 0),),
        healths=(5, 4, 4, 4, 4),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    assert decision.target_scores[0] == 1.0
    assert decision.goal_stage == "weaken_sheriff_team_before_finish"


def test_traitor_duel_attacks_sheriff():
    observation = obs(
        player_idx=4,
        self_role=4,
        healths=(2, 0, 0, 0, 4),
        alive=(1, 0, 0, 0, 1),
    )
    beliefs = RoleBeliefEstimator(player_idx=4).initialize(observation)
    decision = IntentModule(player_idx=4).decision(observation, beliefs)
    assert decision.category is IntentCategory.ATTACK_SHERIFF_DUEL
    assert decision.target_scores[0] == 5.0
    assert decision.action_scores[0] == -5.0
    assert decision.goal_stage == "final_sheriff_duel"


def test_traitor_preserves_sheriff_while_outlaws_are_believed_alive():
    observation = obs(player_idx=4, self_role=4, healths=(1, 4, 4, 4, 4))
    beliefs = RoleBeliefEstimator(player_idx=4).initialize(observation)
    decision = IntentModule(player_idx=4).decision(observation, beliefs)
    assert decision.goal_stage in {"balance_against_outlaw_strength", "balance_against_sheriff_strength"}
    assert decision.target_scores[0] == -5.0
    assert max(decision.target_scores[1:4]) > decision.target_scores[0]


def test_two_player_outlaw_vs_sheriff_prefers_sheriff_attack():
    observation = obs(
        player_idx=1,
        self_role=2,
        healths=(5, 4, 0, 0, 0),
        alive=(1, 1, 0, 0, 0),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    assert decision.category is IntentCategory.ATTACK_SHERIFF_TEAM
    assert decision.target_scores[0] > decision.action_scores[0]


def test_deputy_sheriff_attack_is_rare_score_not_hard_banned():
    observation = obs(player_idx=1, self_role=1)
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    assert decision.target_scores[0] == -5.0
    assert np.isfinite(decision.target_scores[0])



def test_outlaw_pressure_uses_public_alive_outlaw_count_two_at_start():
    observation = obs(player_idx=1, self_role=2, healths=(4, 4, 4, 4, 4))
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    # AliveOutlawCount starts at 2, so pressure limit is 2 * 2 = 4.
    assert decision.goal_stage == "finish_sheriff"
    assert decision.target_scores[0] == 5.0


def test_outlaw_pressure_decrements_after_public_outlaw_death():
    observation = obs(
        player_idx=1,
        self_role=2,
        healths=(4, 4, 0, 4, 4),
        alive=(1, 1, 0, 1, 1),
        revealed_roles=(0, 255, 3, 255, 255),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    # One dead Outlaw is public, so AliveOutlawCount is 1 and pressure limit is 2.
    assert decision.goal_stage == "weaken_sheriff_team_before_finish"
    # No prior Bang: setup 1.0 - opening reveal cost 1.0 = 0.0.
    assert decision.target_scores[0] == 0.0


def test_traitor_uses_public_alive_outlaw_count_until_both_outlaws_revealed_dead():
    one_outlaw_dead = obs(
        player_idx=4,
        self_role=4,
        healths=(5, 4, 0, 4, 4),
        alive=(1, 1, 0, 1, 1),
        revealed_roles=(0, 255, 2, 255, 255),
    )
    beliefs = RoleBeliefEstimator(player_idx=4).initialize(one_outlaw_dead)
    decision = IntentModule(player_idx=4).decision(one_outlaw_dead, beliefs)
    assert decision.goal_stage in {"balance_against_outlaw_strength", "balance_against_sheriff_strength"}
    assert decision.target_scores[0] == -5.0

    both_outlaws_dead = obs(
        player_idx=4,
        self_role=4,
        healths=(5, 4, 0, 0, 4),
        alive=(1, 1, 0, 0, 1),
        revealed_roles=(0, 255, 2, 3, 255),
    )
    beliefs = RoleBeliefEstimator(player_idx=4).initialize(both_outlaws_dead)
    decision = IntentModule(player_idx=4).decision(both_outlaws_dead, beliefs)
    assert decision.goal_stage == "remove_deputy_preserve_sheriff"
    assert decision.target_scores[0] == -5.0


def test_outlaw_numerical_advantage_uses_alive_team_count():
    # One non-Outlaw (Traitor) is publicly dead while both Outlaws are alive.
    # AliveOutlawCount = 2, AliveTeamCount = Sheriff + Deputy = 2, so the
    # Outlaws switch to the +5 Sheriff-finishing stage even at full Sheriff HP.
    observation = obs(
        player_idx=1,
        self_role=2,
        healths=(5, 4, 4, 4, 0),
        alive=(1, 1, 1, 1, 0),
        revealed_roles=(0, 255, 255, 255, 4),
    )
    beliefs = RoleBeliefEstimator(player_idx=1).initialize(observation)
    decision = IntentModule(player_idx=1).decision(observation, beliefs)
    assert decision.goal_stage == "finish_sheriff"
    assert decision.target_scores[0] == 5.0
    assert decision.action_scores[0] == 0.0


def test_traitor_targets_deputy_when_sheriff_team_is_stronger():
    observation = obs(player_idx=4, self_role=4)
    beliefs = np.zeros((5, 5), dtype=np.float64)
    beliefs[0, 0] = 1.0
    beliefs[4, 4] = 1.0
    beliefs[1, 1] = 0.90
    beliefs[1, 2] = 0.05
    beliefs[1, 3] = 0.05
    beliefs[2, 1] = 0.05
    beliefs[2, 2] = 0.475
    beliefs[2, 3] = 0.475
    beliefs[3, 1] = 0.05
    beliefs[3, 2] = 0.475
    beliefs[3, 3] = 0.475
    decision = IntentModule(player_idx=4).decision(observation, beliefs)
    assert decision.strong_faction == "sheriff"
    assert decision.goal_stage == "balance_against_sheriff_strength"
    assert decision.target_scores[1] > decision.target_scores[2]
    assert decision.target_scores[1] > decision.target_scores[3]
    assert decision.target_scores[0] == -5.0


def test_traitor_targets_outlaw_when_outlaw_team_is_stronger():
    observation = obs(player_idx=4, self_role=4)
    beliefs = np.zeros((5, 5), dtype=np.float64)
    beliefs[0, 0] = 1.0
    beliefs[4, 4] = 1.0
    for idx in (1, 2, 3):
        beliefs[idx, 1] = 0.10
        beliefs[idx, 2] = 0.45
        beliefs[idx, 3] = 0.45
    decision = IntentModule(player_idx=4).decision(observation, beliefs)
    assert decision.strong_faction == "outlaw"
    assert decision.goal_stage == "balance_against_outlaw_strength"
    assert max(decision.target_scores[1:4]) > 4.0
    assert decision.target_scores[0] == -5.0
