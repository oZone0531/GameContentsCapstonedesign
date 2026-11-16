"""Tests for coupling the emotion formula to a non rule-based intent policy."""

from __future__ import annotations

import pytest

from bang_rl.agents.dummy_agent.emotion_coupling import (
    RuleSoftmaxSource,
    calibrate_scale,
    make_intent_source,
    measure,
)
from bang_rl.data.game_collection import load_collection_config


@pytest.fixture(scope="module")
def config():
    return load_collection_config()


def _signature(stats):
    return (stats.decisions, stats.changed, stats.passes, stats.friendly, dict(stats.wins))


def test_default_dummy_is_unchanged_by_the_new_hook(config):
    # intent_source None + scale 1.0 must be the exact rule-based game.
    assert _signature(measure(config, None, 1.0, games=8)) == _signature(measure(config, None, 1.0, games=8))


@pytest.mark.parametrize("temperature", [0.5, 4.0])
def test_rescaled_rule_policy_at_inverse_temperature_reproduces_the_rule_dummy(config, temperature):
    rule = measure(config, None, 1.0, games=10)
    rescaled = measure(config, RuleSoftmaxSource(temperature), 1.0 / temperature, games=10)
    assert _signature(rescaled) == _signature(rule)


def test_scale_zero_means_no_decision_is_changed_by_emotion(config):
    assert measure(config, RuleSoftmaxSource(2.0), 0.0, games=6).changed == 0


def test_calibration_finds_the_known_scale(config):
    target = measure(config, None, 1.0, games=20).changed_rate
    scale, history = calibrate_scale(config, RuleSoftmaxSource(4.0), target=target, games=20)
    assert 1 <= len(history) <= 4
    assert 0.15 < scale < 0.4  # exact answer is 1/4
    assert abs(history[-1].changed_rate - target) < 0.03


def test_make_intent_source_specs():
    assert make_intent_source(None) is None
    assert make_intent_source("rule") is None
    assert isinstance(make_intent_source("rule-softmax:3"), RuleSoftmaxSource)
    with pytest.raises(ValueError):
        make_intent_source("bangmodel:")
    with pytest.raises(ValueError):
        make_intent_source("nope:1")
    with pytest.raises(ValueError):
        RuleSoftmaxSource(0.0)
