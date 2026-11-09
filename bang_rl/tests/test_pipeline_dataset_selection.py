from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.exceptions import DataError
from bang_rl.training import pipeline_runner


def _episode(episode_id: str, *, schema: str, dimension: int, names: tuple[str, ...]) -> SimpleNamespace:
    label = {
        "values": [0.5] * dimension,
        "emotion_schema_version": schema,
        "emotion_names": list(names),
    }
    step = SimpleNamespace(emotion_labels={"0": label})
    return SimpleNamespace(
        episode_id=episode_id,
        steps=(step,),
        to_dict=lambda include_private=False, episode_id=episode_id: {
            "episode_id": episode_id,
            "include_private": include_private,
        },
    )


def test_read_canonical_dataset_selects_formula_and_reports_incompatible_exclusions(tmp_path, monkeypatch) -> None:
    formula_path = tmp_path / "formula" / "shard-000000.parquet"
    legacy_path = tmp_path / "legacy" / "shard-000000.parquet"
    formula_path.parent.mkdir()
    legacy_path.parent.mkdir()
    formula = _episode(
        "formula-episode",
        schema=FORMULA_LABEL_SCHEMA,
        dimension=9,
        names=tuple(LABEL_NAMES),
    )
    legacy = _episode(
        "legacy-episode",
        schema="formula-1.1+emotion6",
        dimension=6,
        names=("anger", "fear", "joy", "sadness", "vigilance", "confidence"),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_canonical_shards",
        lambda _root: (legacy_path, formula_path),
    )
    monkeypatch.setattr(
        pipeline_runner.ParquetCanonicalShardWriter,
        "read_shard",
        lambda path: (legacy,) if path == legacy_path else (formula,),
    )

    episodes, shards, duplicates, selection = pipeline_runner._read_canonical_dataset(tmp_path)

    assert [episode.episode_id for episode in episodes] == ["formula-episode"]
    assert shards == (formula_path,)
    assert duplicates == 0
    assert selection["selected_episode_count"] == 1
    assert selection["excluded_incompatible_episode_count"] == 1
    assert selection["excluded_incompatible_contracts"] == {"formula-1.1+emotion6/6": 1}


def test_read_canonical_dataset_requires_formula_labels(tmp_path, monkeypatch) -> None:
    legacy_path = tmp_path / "legacy" / "shard-000000.parquet"
    legacy_path.parent.mkdir()
    legacy = _episode(
        "legacy-episode",
        schema="formula-1.1+emotion6",
        dimension=6,
        names=("anger", "fear", "joy", "sadness", "vigilance", "confidence"),
    )
    monkeypatch.setattr(pipeline_runner, "_canonical_shards", lambda _root: (legacy_path,))
    monkeypatch.setattr(
        pipeline_runner.ParquetCanonicalShardWriter,
        "read_shard",
        lambda _path: (legacy,),
    )

    with pytest.raises(DataError, match="no nine-emotion formula episodes found") as error:
        pipeline_runner._read_canonical_dataset(tmp_path)

    assert "formula-1.1+emotion6/6" in str(error.value)


def test_publish_module_metric_uses_explicit_emotion_accuracy_not_one_minus_mse() -> None:
    run = SimpleNamespace(lock=RLock(), metrics={})
    pipeline_runner._publish_module_metric(
        run,
        {
            "event": "batch",
            "phase": "phase2_supervised_emotion",
            "accuracy": 0.42,
            "accuracy_by_dimension": [0.2, 0.4, 0.6, 0.3, 0.5, 0.52, 0.1, 0.7, 0.33],
            "mse": 0.01,
        },
    )
    assert run.metrics["emotion_accuracy_points"] == [0.42]
    assert run.metrics["emotion_accuracy_by_dimension_points"] == [[0.2, 0.4, 0.6, 0.3, 0.5, 0.52, 0.1, 0.7, 0.33]]

    pipeline_runner._publish_module_metric(
        run,
        {
            "event": "epoch",
            "phase": "phase2_supervised_emotion",
            "validation_emotion_accuracy": 0.47,
            "validation_emotion_accuracy_by_dimension": [0.4] * 9,
            "validation_mse": 0.99,
        },
    )
    assert run.metrics["emotion_validation_accuracy"] == 0.47
    assert run.metrics["emotion_validation_accuracy_by_dimension"] == [0.4] * 9
    assert run.metrics["emotion_validation_mse"] == 0.99
