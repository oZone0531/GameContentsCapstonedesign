"""Focused tests for TASK-2-4 Dummy EmotionModule."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from bang_rl.agents import DummyAgent, EmotionModule
from bang_rl.config import EmotionConfig


def emotion_config() -> EmotionConfig:
    return EmotionConfig(
        schema_version="emotion-test-v1",
        initial_baseline=0.25,
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.8,
                "change_rate": 0.3,
                "decay_rate": 0.1,
                "event_updates": {"bang": 0.4, "player_eliminated": -0.2},
                "action_updates": {"pass": 0.05, "bang": -0.05},
                "reward_updates": {"positive": 0.1, "negative": -0.15},
                "terminal_updates": {"terminated": -0.4, "truncated": 0.2},
                "effects": {"pass": 0.2, "bang": -0.3},
            },
            {
                "name": "trust",
                "initial_value": 0.25,
                "decay_rate": 0.2,
                "event_updates": {"bang": -0.3},
            },
        ],
    )


def test_config_defines_dynamic_baseline_and_state_shape() -> None:
    module = EmotionModule(emotion_config(), player_idx=2)

    assert module.emotion_names == ("fear", "trust")
    assert module.emotion_count == 2
    np.testing.assert_allclose(module.values, [0.8, 0.25])
    np.testing.assert_allclose(module.baseline, [0.8, 0.25])
    assert module.effects_for_action("pass") == {"fear": 0.2}


def test_event_updates_are_idempotent_and_values_are_clamped() -> None:
    module = EmotionModule(emotion_config())
    event = {"event_id": 4, "event_type": "bang", "public": True}

    np.testing.assert_allclose(module.apply_event(event), [1.0, 0.0])
    first = module.values
    np.testing.assert_array_equal(module.apply_event(event), first)
    assert module.processed_event_ids == (4,)

    module.apply_event({"event_id": 5, "event_type": "player_eliminated"})
    np.testing.assert_allclose(module.values, [0.8, 0.0])


def test_decay_is_applied_at_most_once_per_environment_step() -> None:
    module = EmotionModule(emotion_config())
    module.apply_event({"event_id": 1, "event_type": "bang"})

    # Initial step zero has no previous step to decay.
    np.testing.assert_allclose(module.advance_step(0), [1.0, 0.0])
    np.testing.assert_allclose(module.advance_step(1), [0.9, 0.2])
    np.testing.assert_allclose(module.advance_step(1), [0.9, 0.2])

    with pytest.raises(ValueError):
        module.advance_step(0)


def test_action_reward_terminal_rules_and_invalid_mask_paths() -> None:
    module = EmotionModule(emotion_config())
    initial = module.values

    np.testing.assert_array_equal(module.apply_action(1, valid=False), initial)
    np.testing.assert_array_equal(
        module.apply_action(1, action_mask=np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)),
        initial,
    )
    np.testing.assert_allclose(
        module.apply_action(0, action_mask=np.ones(6, dtype=np.uint8)), [0.85, 0.25]
    )
    np.testing.assert_allclose(module.apply_reward(1.0), [0.95, 0.25])
    np.testing.assert_allclose(module.apply_terminal({"terminated": True}), [0.55, 0.25], atol=1e-6)

    terminal_state = module.values
    np.testing.assert_array_equal(module.apply_reward(-1.0, valid=False), terminal_state)


def test_terminal_lifecycle_reset_and_agent_integration() -> None:
    module = EmotionModule(emotion_config(), player_idx=1)
    module.on_episode_start({"episode_id": "episode-a", "seed": 22})
    module.apply_event({"event_id": 1, "event_type": "bang"})
    module.on_episode_end({"terminated": False, "truncated": True})
    assert module.episode_id == "episode-a"
    assert module.truncated is True
    np.testing.assert_allclose(module.values, [1.0, 0.0])

    agent = DummyAgent(
        agent_id="player_1",
        player_idx=1,
        emotion_config=emotion_config(),
    )
    agent.on_episode_start({"episode_id": "episode-b", "seed": 23})
    np.testing.assert_allclose(agent.emotion_state, [0.8, 0.25])
    agent.on_episode_end({"terminated": True})
    assert agent.emotion_module.terminated is True
    agent.emotion_module.reset()
    np.testing.assert_allclose(agent.emotion_state, [0.8, 0.25])


def test_deterministic_replay_and_json_persistence() -> None:
    events = [
        {"event_id": 1, "event_type": "bang"},
        {"event_id": 2, "event_type": "player_eliminated"},
    ]
    first = EmotionModule(emotion_config(), player_idx=3)
    first.prepare_action(0, events[:1])
    first.prepare_action(1, events[1:])
    first.apply_action("pass")
    first.apply_reward(-1.0)
    expected = first.values

    restored = EmotionModule.from_json(first.to_json())
    np.testing.assert_array_equal(restored.values, expected)
    assert restored.to_json() == first.to_json()
    np.testing.assert_array_equal(restored.apply_event(events[1]), expected)

    replay = EmotionModule(emotion_config(), player_idx=3)
    replay.prepare_action(0, events[:1])
    replay.prepare_action(1, events[1:])
    replay.apply_action("pass")
    replay.apply_reward(-1.0)
    np.testing.assert_array_equal(replay.values, expected)


def test_public_privacy_and_schema_versioning() -> None:
    module = EmotionModule(emotion_config())
    with pytest.raises(ValueError, match="privileged"):
        module.apply_event({"event_id": 1, "event_type": "bang", "true_roles": [0, 1, 2, 3, 4]})

    encoded = module.state_dict()
    assert encoded["schema_version"] == "1.0"
    assert encoded["emotion_schema_version"] == "emotion-test-v1"
    assert "true_roles" not in module.to_json()
    assert "reward_breakdown" not in module.to_json()
    assert "BangEnvironment" not in inspect.getsource(EmotionModule)


def test_emotion_config_can_add_and_remove_items_without_code_changes() -> None:
    one = EmotionConfig(emotions={"calm": {"initial": 0.4}})
    two = EmotionConfig(
        emotions={"calm": {"initial": 0.4}, "alert": {"initial": 0.9}},
    )

    assert EmotionModule(one).emotion_names == ("calm",)
    assert EmotionModule(two).emotion_names == ("calm", "alert")


# ---------------------------------------------------------------------------
# Relationship-key coverage (FEAT-002).
#
# The cases below require numpy + pydantic (EmotionModule/EmotionConfig) and are
# therefore CI-only: they cannot run in the restricted sandbox where those
# packages are unavailable.  The numpy-free portion of this behaviour lives in
# test_emotion_relationship_keys.py, which does run in the sandbox.
# ---------------------------------------------------------------------------


def _relationship_config(event_updates: dict[str, float]) -> EmotionConfig:
    """A single-emotion config with a neutral 0.0 change_rate fallback.

    ``initial_value`` is 0.5 and ``change_rate`` is 0.0 so any applied delta is
    entirely attributable to the matched ``event_updates`` rule (no decay, no
    fallback noise), keeping the assertions unambiguous.
    """

    return EmotionConfig(
        schema_version="relationship-test-v1",
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.5,
                "change_rate": 0.0,
                "decay_rate": 0.0,
                "event_updates": event_updates,
            }
        ],
    )


def _applied_delta(module: EmotionModule, event: dict) -> float:
    """Return the change a single event applies to the (single) emotion value."""

    before = float(module.values[0])
    after = float(module.apply_event(event)[0])
    return after - before


def test_relationship_key_overrides_plain_key_when_target_is_me() -> None:
    seat = 2
    module = EmotionModule(_relationship_config({"bang_at_me": 0.5, "bang": 0.1}), player_idx=seat)
    # bang aimed at our own seat -> the bang_at_me relationship key wins.
    delta = _applied_delta(
        module, {"event_id": 1, "event_type": "bang", "actor_idx": 0, "target_idx": seat}
    )
    assert delta == pytest.approx(0.5)


def test_relationship_key_falls_back_to_plain_key_for_other_target() -> None:
    seat = 2
    module = EmotionModule(_relationship_config({"bang_at_me": 0.5, "bang": 0.1}), player_idx=seat)
    # bang aimed at someone else -> no bang_at_me match, plain "bang" applies.
    delta = _applied_delta(
        module, {"event_id": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 4}
    )
    assert delta == pytest.approx(0.1)


def test_relationship_keys_disabled_when_player_idx_none() -> None:
    module = EmotionModule(_relationship_config({"bang_at_me": 0.5, "bang": 0.1}), player_idx=None)
    # With no seat, relationship keys never fire regardless of target.
    delta_at_me = _applied_delta(
        module, {"event_id": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 2}
    )
    assert delta_at_me == pytest.approx(0.1)
    delta_other = _applied_delta(
        module, {"event_id": 2, "event_type": "bang", "actor_idx": 0, "target_idx": 4}
    )
    assert delta_other == pytest.approx(0.1)


def test_plain_only_config_is_backward_compatible() -> None:
    seat = 2
    module_hit = EmotionModule(_relationship_config({"bang": 0.4}), player_idx=seat)
    module_miss = EmotionModule(_relationship_config({"bang": 0.4}), player_idx=seat)
    # A plain-only config yields the same delta whether or not actor/target
    # match our seat.
    delta_at_me = _applied_delta(
        module_hit, {"event_id": 1, "event_type": "bang", "actor_idx": seat, "target_idx": seat}
    )
    delta_other = _applied_delta(
        module_miss, {"event_id": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 4}
    )
    assert delta_at_me == pytest.approx(0.4)
    assert delta_other == pytest.approx(0.4)


def test_actor_relative_key_beats_target_relative_key() -> None:
    seat = 1
    module = EmotionModule(
        _relationship_config({"bang_by_me": 0.7, "bang_at_me": 0.5}), player_idx=seat
    )
    # actor == target == our seat: both by_me and at_me match, but the
    # actor-relative by_me key is tried first and wins.
    delta = _applied_delta(
        module, {"event_id": 1, "event_type": "bang", "actor_idx": seat, "target_idx": seat}
    )
    assert delta == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Per-opponent relation vector coverage (FEAT-002).
#
# These cases require numpy + pydantic (EmotionModule/EmotionConfig) and are
# therefore CI-only: they cannot run in the restricted sandbox where those
# packages are unavailable.  The numpy-free portion of this behaviour lives in
# test_emotion_relation_vector.py, which does run in the sandbox.
# ---------------------------------------------------------------------------


def _relation_config(relation_updates: dict[str, float]) -> EmotionConfig:
    """A single-emotion config carrying relation deltas but neutral emotions.

    ``change_rate`` is 0.0 and ``event_updates`` empty so applying events never
    moves the flat emotion value; only the relation vector should change, which
    keeps relation assertions unambiguous.
    """

    return EmotionConfig(
        schema_version="relation-test-v1",
        relation_baseline=0.0,
        relation_updates=relation_updates,
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.5,
                "change_rate": 0.0,
                "decay_rate": 0.0,
            }
        ],
    )


def test_relation_increments_toward_actor_seat_when_shot_by() -> None:
    seat = 1
    shooter = 3
    module = EmotionModule(_relation_config({"shot_by": 0.4}), player_idx=seat)
    module.apply_event(
        {"event_id": 1, "event_type": "bang", "actor_idx": shooter, "target_idx": seat}
    )
    relation = module.relation
    assert relation.shape == (5,)
    assert relation[shooter] == pytest.approx(0.4)
    # No other seat moved (self and uninvolved seats stay neutral).
    for other in range(5):
        if other != shooter:
            assert relation[other] == pytest.approx(0.0)


def test_relation_self_slot_never_changes() -> None:
    seat = 2
    module = EmotionModule(
        _relation_config({"shot_by": 0.4, "shot_at": 0.3}), player_idx=seat
    )
    # A self-directed event implicates no seat, so nothing moves.
    module.apply_event(
        {"event_id": 1, "event_type": "bang", "actor_idx": seat, "target_idx": seat}
    )
    assert module.relation[seat] == pytest.approx(0.0)
    # We shoot seat 4; only seat 4 moves, our own slot stays neutral.
    module.apply_event(
        {"event_id": 2, "event_type": "bang", "actor_idx": seat, "target_idx": 4}
    )
    assert module.relation[seat] == pytest.approx(0.0)
    assert module.relation[4] == pytest.approx(0.3)


def test_relation_is_idempotent_on_duplicate_event_id() -> None:
    seat = 0
    shooter = 2
    module = EmotionModule(_relation_config({"shot_by": 0.4}), player_idx=seat)
    event = {"event_id": 7, "event_type": "bang", "actor_idx": shooter, "target_idx": seat}
    module.apply_event(event)
    module.apply_event(event)
    # Duplicate event_id must not double-count relation.
    assert module.relation[shooter] == pytest.approx(0.4)


def test_relation_untouched_when_player_idx_none() -> None:
    module = EmotionModule(_relation_config({"shot_by": 0.4}), player_idx=None)
    module.apply_event(
        {"event_id": 1, "event_type": "bang", "actor_idx": 2, "target_idx": 3}
    )
    np.testing.assert_allclose(module.relation, np.zeros(5, dtype=np.float32))


def test_relation_clamped_to_signed_range() -> None:
    seat = 1
    shooter = 4
    module = EmotionModule(_relation_config({"shot_by": -0.9}), player_idx=seat)
    for event_id in range(3):
        module.apply_event(
            {"event_id": event_id, "event_type": "bang", "actor_idx": shooter, "target_idx": seat}
        )
    # Three -0.9 deltas would reach -2.7 but the relation is clamped at -1.0.
    assert module.relation[shooter] == pytest.approx(-1.0)


def test_relation_updates_default_empty_is_backward_compatible() -> None:
    seat = 1
    # No relation_updates configured -> relation never moves and the flat
    # emotion behaviour is identical to before this feature.
    module = EmotionModule(
        EmotionConfig(
            schema_version="relation-empty-v1",
            emotions=[{"name": "fear", "initial_value": 0.5, "event_updates": {"bang": 0.2}}],
        ),
        player_idx=seat,
    )
    before = module.relation
    delta = _applied_delta(
        module, {"event_id": 1, "event_type": "bang", "actor_idx": 3, "target_idx": seat}
    )
    assert delta == pytest.approx(0.2)
    np.testing.assert_allclose(module.relation, before)


def test_state_dict_serializes_relation_and_round_trips() -> None:
    seat = 2
    shooter = 0
    module = EmotionModule(_relation_config({"shot_by": 0.5}), player_idx=seat)
    module.apply_event(
        {"event_id": 1, "event_type": "bang", "actor_idx": shooter, "target_idx": seat}
    )
    encoded = module.state_dict()
    assert encoded["schema_version"] == "1.1"
    assert encoded["relation"][shooter] == pytest.approx(0.5)
    assert encoded["relation_baseline"] == pytest.approx(0.0)

    restored = EmotionModule.from_json(module.to_json())
    np.testing.assert_allclose(restored.relation, module.relation)
    assert restored.to_json() == module.to_json()


def test_old_1_0_state_loads_with_neutral_relation() -> None:
    seat = 2
    module = EmotionModule(_relation_config({"shot_by": 0.5}), player_idx=seat)
    state = module.state_dict()
    # Simulate a pre-relation "1.0" state: downgrade the version and drop the
    # relation fields entirely.
    state["schema_version"] = "1.0"
    state.pop("relation", None)
    state.pop("relation_baseline", None)

    restored = EmotionModule.from_state_dict(state)
    np.testing.assert_allclose(restored.relation, np.zeros(5, dtype=np.float32))


def test_unsupported_state_version_is_rejected() -> None:
    module = EmotionModule(_relation_config({"shot_by": 0.5}), player_idx=2)
    state = module.state_dict()
    state["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported emotion module state schema"):
        EmotionModule.from_state_dict(state)


# ---------------------------------------------------------------------------
# HP-context bucket + public board snapshot tests (FEAT-003).
#
# These construct an EmotionConfig / EmotionModule / DummyAgent and so import
# numpy + pydantic; they are therefore CI-only and do not run in the restricted
# sandbox.  The numpy-free portion of the HP-bucket derivation lives in
# test_emotion_hp_buckets.py, which does run in the sandbox.
# ---------------------------------------------------------------------------


def _hp_config(
    event_updates: dict[str, float],
    *,
    hp_low_threshold: int | None = None,
    hp_lethal_threshold: int | None = None,
) -> EmotionConfig:
    """A single-emotion config for exercising HP-context bucket keys.

    ``change_rate`` is 0.0 so a plain event with no matching key applies nothing;
    only a configured HP/relationship/event key moves the value, keeping the
    delta assertions unambiguous.
    """

    return EmotionConfig(
        schema_version="hp-test-v1",
        hp_low_threshold=hp_low_threshold,
        hp_lethal_threshold=hp_lethal_threshold,
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.5,
                "change_rate": 0.0,
                "decay_rate": 0.0,
                "event_updates": event_updates,
            }
        ],
    )


def test_hp_bucket_key_fires_only_when_thresholds_configured() -> None:
    seat = 0
    event = {
        "event_id": 1,
        "event_type": "bang",
        "actor_idx": 3,
        "target_idx": 2,
        "hp_after": 1,
    }
    # Without thresholds the HP bucket key is never derived -> plain change_rate 0.
    without = EmotionModule(_hp_config({"target_low_hp": 0.4}), player_idx=seat)
    assert _applied_delta(without, dict(event)) == pytest.approx(0.0)
    # With the low threshold configured the target_low_hp key fires.
    with_threshold = EmotionModule(
        _hp_config({"target_low_hp": 0.4}, hp_low_threshold=1), player_idx=seat
    )
    assert _applied_delta(with_threshold, dict(event)) == pytest.approx(0.4)


def test_hp_lethal_key_wins_over_plain_event_key() -> None:
    seat = 0
    module = EmotionModule(
        _hp_config(
            {"target_lethal": 0.6, "bang": 0.1},
            hp_low_threshold=2,
            hp_lethal_threshold=0,
        ),
        player_idx=seat,
    )
    # hp_after 0 -> target_lethal precedes plain bang.
    delta = _applied_delta(
        module,
        {
            "event_id": 1,
            "event_type": "bang",
            "actor_idx": 3,
            "target_idx": 2,
            "hp_after": 0,
        },
    )
    assert delta == pytest.approx(0.6)


def test_relationship_key_still_wins_over_hp_key() -> None:
    seat = 2
    module = EmotionModule(
        _hp_config(
            {"bang_at_me": 0.5, "self_low_hp": 0.3, "bang": 0.1},
            hp_low_threshold=2,
        ),
        player_idx=seat,
    )
    # A bang aimed at our own low-HP seat matches both bang_at_me and self_low_hp;
    # the relationship key must retain precedence over the HP key.
    delta = _applied_delta(
        module,
        {
            "event_id": 1,
            "event_type": "bang",
            "actor_idx": 0,
            "target_idx": seat,
            "hp_after": 1,
        },
    )
    assert delta == pytest.approx(0.5)


def test_hp_default_config_behaviour_is_unchanged() -> None:
    seat = 1
    # A config that predates HP thresholds (defaults None) with a plain bang rule
    # behaves exactly as before: the plain key applies, no HP key interferes.
    module = EmotionModule(
        EmotionConfig(
            schema_version="hp-default-v1",
            emotions=[{"name": "fear", "initial_value": 0.5, "event_updates": {"bang": 0.2}}],
        ),
        player_idx=seat,
    )
    delta = _applied_delta(
        module,
        {
            "event_id": 1,
            "event_type": "bang",
            "actor_idx": 3,
            "target_idx": seat,
            "hp_after": 0,
        },
    )
    assert delta == pytest.approx(0.2)


def test_hp_thresholds_default_none_and_reject_bool() -> None:
    assert EmotionConfig().hp_low_threshold is None
    assert EmotionConfig().hp_lethal_threshold is None
    with pytest.raises(Exception):
        EmotionConfig(hp_low_threshold=True)


def test_observe_public_state_holds_transient_board_snapshot() -> None:
    module = EmotionModule(_hp_config({}), player_idx=0)
    assert module.board_healths is None
    assert module.board_alive is None
    module.observe_public_state([5, 4, 0, 4, 4], [True, True, False, True, True])
    np.testing.assert_allclose(module.board_healths, np.asarray([5, 4, 0, 4, 4], dtype=np.float32))
    assert module.board_alive.tolist() == [True, True, False, True, True]
    # A returned copy is detached from internal state.
    snapshot = module.board_healths
    snapshot[0] = 99.0
    assert module.board_healths[0] == pytest.approx(5.0)
    # The snapshot is transient: it is not part of the serialized state.
    assert "board_healths" not in module.state_dict()
    assert "board_alive" not in module.state_dict()


def test_observe_public_state_missing_board_is_safe_noop() -> None:
    module = EmotionModule(_hp_config({}), player_idx=0)
    module.observe_public_state([5, 4, 4, 4, 4], [True] * 5)
    # A later fixture that omits the board clears the snapshot rather than raising.
    module.observe_public_state(None, None)
    assert module.board_healths is None
    assert module.board_alive is None


def test_dummy_agent_feeds_public_board_into_emotion_module() -> None:
    observation = {
        "healths": [5, 4, 4, 4, 4],
        "alive": [True, True, True, True, True],
        "environment_step": [0],
        "event_sequence": (),
    }
    agent = DummyAgent(agent_id="player_0", player_idx=0)
    agent._prepare_emotion(observation)
    np.testing.assert_allclose(
        agent.emotion_module.board_healths, np.asarray([5, 4, 4, 4, 4], dtype=np.float32)
    )
    assert agent.emotion_module.board_alive.tolist() == [True] * 5


def test_dummy_agent_missing_board_is_safe_noop() -> None:
    # A tiny fixture with no healths/alive must not crash; the module simply
    # holds no board snapshot.
    observation = {"environment_step": [0], "event_sequence": ()}
    agent = DummyAgent(agent_id="player_0", player_idx=0)
    agent._prepare_emotion(observation)
    assert agent.emotion_module.board_healths is None
    assert agent.emotion_module.board_alive is None
