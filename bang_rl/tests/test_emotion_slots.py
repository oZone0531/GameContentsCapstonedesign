"""Constant relation entries (self / dead seats) are excluded from emotion training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bang_rl.data import EpisodeSplitter, NPZTensorCacheLoader, NPZTensorCacheWriter
from bang_rl.data.emotion_slots import emotion_value_mask, relation_seats
from bang_rl.data.game_collection import (
    FORMULA_LABEL_SCHEMA,
    collect_games,
    formula_label_emotion_config,
    load_collection_config,
)
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.data import ParquetCanonicalShardWriter
from bang_rl.training.emotion_sl_trainer import (
    EmotionInferenceSLTrainer,
    EmotionSLConfig,
    build_emotion_batch,
)


def test_relation_seats_reads_only_relation_names():
    assert relation_seats(["calm", "relation_0", "anger", "relation_4", "relation_9"]).tolist() == [
        -1, 0, -1, 4, -1,
    ]


def test_mask_drops_own_seat_and_dead_seats_only():
    owners = np.array([2])
    alive = np.array([[True, False, True, True, True]])
    mask = emotion_value_mask(LABEL_NAMES, owners, alive)[0]
    assert mask[:4].all()  # global emotions always count
    assert mask[4:].tolist() == [True, False, False, True, True]  # seat1 dead, seat2 is the owner


def test_mask_requires_matching_alive_shape():
    with pytest.raises(ValueError):
        emotion_value_mask(LABEL_NAMES, np.zeros((2, 4)), np.ones((2, 5)))


@pytest.fixture(scope="module")
def formula_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("slots")
    collect_games(load_collection_config(), 10, root / "canonical")
    episodes = ParquetCanonicalShardWriter.read_dataset(root / "canonical")
    split = EpisodeSplitter(counts=(6, 2, 2), seed=0, manifest_path=root / "split.json").split(episodes)
    artifacts = NPZTensorCacheWriter(
        root / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=len(LABEL_NAMES)
    ).write_all(split)
    return {name: NPZTensorCacheLoader(artifacts[name].tensor_path) for name in ("train", "validation", "test")}


def test_batch_masks_exactly_the_constant_relation_entries(formula_cache):
    loader = formula_cache["train"]
    rows = np.arange(len(loader))
    batch = build_emotion_batch(loader, rows, emotion_names=LABEL_NAMES)
    labels = batch.emotion_labels.numpy()
    valid = batch.emotion_label_mask.numpy()
    entries = batch.emotion_value_mask.numpy()
    assert entries.shape == labels.shape
    assert entries[..., :4].all()
    masked_relations = ~entries[..., 4:] & valid[..., None]
    assert masked_relations.any()  # own seat always, dead seats later in games
    # Every masked relation of a valid label is exactly the neutral constant.
    assert np.allclose(labels[..., 4:][masked_relations], 0.5)
    # Each owner's own seat is always masked.
    viewers = batch.viewer_indices.numpy()
    for row, viewer in enumerate(viewers):
        owners = [seat for seat in range(5) if seat != viewer]
        for stream, owner in enumerate(owners):
            assert not entries[row, stream, 4 + owner]


def test_batch_without_names_keeps_previous_behaviour(formula_cache):
    batch = build_emotion_batch(formula_cache["train"], np.arange(4))
    assert batch.emotion_value_mask is None


def test_loss_ignores_masked_entries(formula_cache):
    trainer = EmotionInferenceSLTrainer(
        formula_cache["train"], formula_cache["validation"], test_cache=formula_cache["test"],
        config=EmotionSLConfig(epochs=1, batch_size=16), emotion_config=formula_label_emotion_config(),
    )
    batch = build_emotion_batch(formula_cache["train"], np.arange(32), emotion_names=LABEL_NAMES)
    with torch.no_grad():
        base, _, count = trainer._loss_and_metrics(batch)
        changed_labels = batch.emotion_labels.clone()
        changed_labels[~batch.emotion_value_mask] = 1.0  # rewrite only the constant entries
        changed = type(batch)(
            model_inputs=batch.model_inputs, emotion_labels=changed_labels,
            emotion_label_mask=batch.emotion_label_mask, viewer_indices=batch.viewer_indices,
            row_indices=batch.row_indices, emotion_value_mask=batch.emotion_value_mask,
        )
        after, _, _ = trainer._loss_and_metrics(changed)
    assert float(after) == pytest.approx(float(base))
    expected = int((batch.emotion_value_mask & batch.emotion_label_mask[..., None]).sum())
    assert count == expected
    result = trainer.fit()
    assert len(result.history) == 1
