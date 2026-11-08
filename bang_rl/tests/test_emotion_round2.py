"""Emotion formula round 2: relations follow beliefs, grudge-directed aggression."""

from __future__ import annotations

import numpy as np

from bang_rl.agents.dummy_agent.emotion_formula import EmotionParams, EmotionState

ALIVE = [True] * 5
HEALTHS = [5, 4, 4, 4, 4]


def _run(params: EmotionParams, beliefs: np.ndarray, steps: int = 30, self_role: int = 1) -> EmotionState:
    state = EmotionState(1, params, rng=np.random.default_rng(0))
    for step in range(steps):
        state.step(step, [], beliefs=beliefs, self_role=self_role, sheriff_idx=0, healths=HEALTHS, alive=ALIVE)
    return state


def _deputy_beliefs() -> np.ndarray:
    # Seat 1 is the deputy: seat 0 is the public sheriff (ally), seat 2 looks
    # like an outlaw, seats 3 and 4 are uncertain.
    beliefs = np.full((5, 5), 0.2)
    beliefs[0] = [1, 0, 0, 0, 0]
    beliefs[1] = [0, 1, 0, 0, 0]
    beliefs[2] = [0, 0, 0.45, 0.45, 0.1]
    return beliefs


def test_relations_stay_neutral_without_events_when_term_is_off():
    state = _run(EmotionParams(sigma=0.0, sigma_R=0.0), _deputy_beliefs())
    assert np.allclose(state.R, 0.0)


def test_relations_follow_beliefs_when_term_is_on():
    state = _run(EmotionParams(sigma=0.0, sigma_R=0.0, rho_B=0.1, g_B=2.0), _deputy_beliefs())
    assert state.R[0] > 0.3  # believed ally -> liking
    assert state.R[2] < -0.1  # believed enemy -> grudge
    assert state.R[2] < state.R[3] < state.R[0]
    assert state.R[1] == 0.0  # own seat untouched
    assert np.all(np.abs(state.R) <= 1.0)


def test_relation_to_dead_seat_is_cleared_with_term_on():
    params = EmotionParams(sigma=0.0, sigma_R=0.0, rho_B=0.2)
    state = EmotionState(1, params, rng=np.random.default_rng(0))
    beliefs = _deputy_beliefs()
    for step in range(10):
        state.step(step, [], beliefs=beliefs, self_role=1, sheriff_idx=0, healths=HEALTHS, alive=ALIVE)
    dead = [True, True, False, True, True]
    state.step(10, [], beliefs=beliefs, self_role=1, sheriff_idx=0, healths=HEALTHS, alive=dead)
    assert state.R[2] == 0.0 and not state.R_active[2]
    assert state.relation_01()[2] == 0.5


def test_grudge_directed_aggression_only_targets_grudges():
    state = EmotionState(1, EmotionParams(kappa_AR=0.0), rng=np.random.default_rng(0))
    state.A, state.S = 0.5, 0.0
    state.R = np.array([0.0, 0.0, -0.8, 0.0, 0.0])
    base = state.action_contribution()
    state.p = EmotionParams(kappa_AR=2.0)
    aimed = state.action_contribution()
    assert aimed[1 + 2] - base[1 + 2] == np.float64(2.0 * 0.5 * 0.8)
    others = [0, 1 + 0, 1 + 3, 1 + 4]
    assert np.allclose(aimed[others], base[others])
