"""Focused TASK-3-4 tests for public confirmed-role hard overrides."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import IntentInferenceModule, RoleFusion


ROLE_COUNT = 5
HIDDEN = 255


def _public(*, viewer_idx: int = 0) -> dict:
    confirmed = np.zeros((5, 5), dtype=np.float32)
    return {
        "viewer_idx": viewer_idx,
        "role_visibility": np.zeros(5, dtype=np.uint8),
        "revealed_roles": np.full(5, HIDDEN, dtype=np.uint8),
        "confirmed_role_distribution": confirmed,
    }


def test_each_public_override_source_is_hard_and_unresolved_rows_are_preserved() -> None:
    raw = np.asarray(
        [
            [0.05, 0.10, 0.20, 0.25, 0.40],
            [0.40, 0.30, 0.15, 0.10, 0.05],
            [0.10, 0.20, 0.30, 0.25, 0.15],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.45, 0.15, 0.10, 0.10, 0.20],
        ],
        dtype=np.float32,
    )
    public = _public()
    public["self_role"] = np.asarray([0], dtype=np.uint8)
    public["sheriff_idx"] = np.asarray([0], dtype=np.uint8)
    public["revealed_roles"][1] = 1
    public["confirmed_role_distribution"][2, 2] = 1.0
    public["role_visibility"][[0, 1, 2]] = [1, 2, 3]

    fused = RoleFusion().fuse({"raw_role_distribution": raw, **public})

    np.testing.assert_array_equal(fused[0], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(fused[1], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(fused[2], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(fused[3:], raw[3:])
    np.testing.assert_allclose(fused.sum(axis=-1), 1.0, atol=1e-6)


def test_four_target_streams_map_around_viewer_and_full_slot_streams_retain_self() -> None:
    raw_targets = np.tile(np.asarray([[0.1, 0.2, 0.3, 0.15, 0.25]], dtype=np.float32), (4, 1))
    public = _public(viewer_idx=2)
    public["self_role"] = 3
    public["sheriff_idx"] = 0
    public["confirmed_role_distribution"][0, 0] = 1.0
    public["confirmed_role_distribution"][2, 3] = 1.0

    target_fused = RoleFusion().fuse(raw_targets, **public)
    # viewer 2 is omitted: target rows map to slots 0, 1, 3, 4.
    np.testing.assert_array_equal(target_fused[0], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(target_fused[1:], raw_targets[1:])

    full_raw = np.tile(raw_targets[:1], (5, 1))
    full_fused = RoleFusion().fuse(full_raw, **public)
    np.testing.assert_array_equal(full_fused[2], [0, 0, 0, 1, 0])


def test_logits_are_softmaxed_and_explicit_logit_output_keeps_unresolved_logits() -> None:
    logits = np.asarray(
        [[-2.0, 0.0, 1.0, 2.0, -1.0]] * 4,
        dtype=np.float32,
    )
    fusion = RoleFusion()
    probabilities = fusion.fuse(logits, input_type="logits")
    expected = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    np.testing.assert_allclose(probabilities, expected, rtol=1e-6, atol=1e-6)

    logit_output = fusion.fuse(logits, input_type="logits", output_type="logits")
    np.testing.assert_array_equal(logit_output, logits)


def test_batched_streams_apply_per_observation_public_constraints() -> None:
    raw = np.asarray(
        [
            [[0.7, 0.1, 0.1, 0.05, 0.05]] * 4,
            [[0.05, 0.05, 0.1, 0.2, 0.6]] * 4,
        ],
        dtype=np.float32,
    )
    visibility = np.zeros((2, 5), dtype=np.uint8)
    revealed = np.full((2, 5), HIDDEN, dtype=np.uint8)
    confirmed = np.zeros((2, 5, 5), dtype=np.float32)
    # Batch 0 viewer=0, target slots 1..4; batch 1 viewer=1, target slots 0,2..4.
    confirmed[0, 1, 1] = 1.0
    confirmed[1, 4, 4] = 1.0

    fused = RoleFusion().fuse(
        raw,
        role_visibility=visibility,
        revealed_roles=revealed,
        confirmed_role_distribution=confirmed,
        viewer_idx=np.asarray([0, 1], dtype=np.uint8),
    )

    assert fused.shape == (2, 4, 5)
    np.testing.assert_array_equal(fused[0, 0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(fused[1, 3], [0, 0, 0, 0, 1])
    np.testing.assert_allclose(fused.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(fused[0, 1:], raw[0, 1:])


def test_conflicts_malformed_confirmations_and_invalid_shapes_are_rejected() -> None:
    fusion = RoleFusion()
    public = _public()
    public["self_role"] = 1
    public["confirmed_role_distribution"][0, 2] = 1.0
    with pytest.raises(ValueError, match="conflicting confirmed roles"):
        fusion.fuse(np.full((5, 5), 0.2, dtype=np.float32), **public)

    malformed = _public()
    malformed["confirmed_role_distribution"][1, :2] = [0.5, 0.5]
    with pytest.raises(ValueError, match="zero or one-hot"):
        fusion.fuse(np.full((5, 5), 0.2, dtype=np.float32), **malformed)

    with pytest.raises(ValueError, match="role_visibility must have shape"):
        fusion.fuse(
            np.full((4, 5), 0.2, dtype=np.float32),
            role_visibility=np.zeros(4, dtype=np.uint8),
        )

    duplicate = _public()
    duplicate["confirmed_role_distribution"][1, 1] = 1.0
    duplicate["confirmed_role_distribution"][2, 1] = 1.0
    with pytest.raises(ValueError, match="multiple players"):
        fusion.fuse(np.full((5, 5), 0.2, dtype=np.float32), **duplicate)


def test_privileged_role_inputs_are_rejected_even_when_raw_distribution_is_public() -> None:
    fusion = RoleFusion()
    raw = np.full((4, 5), 0.2, dtype=np.float32)
    private = {"raw_role_distribution": raw, **_public(), "true_roles": [0, 1, 2, 3, 4]}
    with pytest.raises(ValueError, match="privileged role inputs"):
        fusion.fuse(private)

    with pytest.raises(ValueError, match="privileged role inputs"):
        fusion.fuse(raw, metadata={"hidden_role_labels": [0, 1, 2, 3, 4]})


def test_role_fusion_serialization_and_intent_artifact_metadata_are_stable(tmp_path) -> None:
    fusion = RoleFusion()
    assert fusion.to_json() == RoleFusion.from_json(fusion.to_json()).to_json()
    assert fusion.metadata()["role_count"] == 5
    assert fusion.metadata()["role_count_projection"] == "not_implemented"

    path = tmp_path / "role_fusion.json"
    fusion.save(path)
    restored = RoleFusion.load(path)
    raw = np.full((4, 5), 0.2, dtype=np.float32)
    np.testing.assert_array_equal(restored.fuse(raw), fusion.fuse(raw))

    metadata = IntentInferenceModule(seed=3).metadata()
    assert metadata["role_fusion"]["module_type"] == "RoleFusion"
    assert metadata["role_fusion"]["role_fusion_format_version"] == "1.0"
    assert metadata["role_fusion"]["role_count_projection"] == "not_implemented"
