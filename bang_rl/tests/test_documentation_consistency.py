"""TASK-10-4 checks for documentation, configuration, and package contracts."""

from __future__ import annotations

import re
from pathlib import Path
import tomllib

from bang_rl.app import (
    ConfigService,
    DatasetService,
    EvaluationService,
    GameService,
    TrainingService,
)
from bang_rl.config import load_config
from bang_rl.data import (
    ACTION_SPACE_VERSION as DATA_ACTION_SPACE_VERSION,
    CACHE_FORMAT_VERSION,
    CANONICAL_DATASET_SCHEMA_VERSION,
)
from bang_rl.env import ACTION_SPACE_VERSION, OBSERVATION_SCHEMA_VERSION
from bang_rl.runtime import LOG_CHANNELS
from bang_rl.training import (
    FINAL_ARTIFACT_EXTENSION,
    FINAL_ARTIFACT_FORMAT_VERSION,
    FINAL_ARTIFACT_TYPE,
    PHASE2_ARTIFACT_EXTENSION,
    PHASE2_ARTIFACT_FORMAT_VERSION,
    PHASE2_ARTIFACT_TYPE,
)


ROOT = Path(__file__).resolve().parents[2]


def _requirement_key(name: str, marker: str = "") -> str:
    """Normalise ``name`` plus an optional environment marker into one key.

    A pin may be split per Python version (e.g. numpy for <3.13 and >=3.13);
    the marker is part of the key so both pins are compared separately.
    Quotes and spaces are dropped so ``'3.13'`` and ``"3.13"`` compare equal.
    """

    key = name.strip().lower().replace("_", "-")
    marker = re.sub(r"[\s'\"]", "", marker)
    return f"{key};{marker}" if marker else key


def _split_requirement(requirement: str) -> tuple[str, str]:
    spec, _, marker = requirement.partition(";")
    name, version = spec.strip().split("==", 1)
    return _requirement_key(name, marker), version.strip()


def _pinned_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "==" not in text:
            continue
        key, version = _split_requirement(text)
        result[key] = version
    return result


def test_readme_and_example_cover_implemented_contracts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = ROOT / "config.example.yaml"
    config = load_config(example)

    for required in (
        "requirements.lock",
        "requirements-gui.lock",
        "max_episode_steps=2000",
        "bang_rl/datasets/canonical/",
        "bang_rl/datasets/cache/",
        ".bangmodel",
        ".phase2model",
        "manifest.json",
        "metadata.json",
        "model.safetensors",
        "terminated=True",
        "truncated=True",
        "MainAgent transitions",
        "estimated emotion",
        "Hardware Benchmark runners",
        "disk checkpoint",

    ):
        assert required in readme

    assert config.env.max_episode_steps == 2_000
    assert config.env.warn_turn_threshold == 200
    assert config.env.critical_turn_threshold == 500
    assert config.env.event_history.max_events is None
    assert config.dataset.canonical_path == "bang_rl/datasets/canonical"
    assert config.dataset.cache_path == "bang_rl/datasets/cache"
    assert config.artifact.output_directory == "bang_rl/models"
    assert config.artifact.extension == ".bangmodel"
    assert config.artifact.weights_file == "model.safetensors"
    assert config.artifact.metadata_file == "metadata.json"
    assert config.artifact.manifest_file == "manifest.json"


def test_lock_files_match_project_extras_and_keep_gui_optional() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = _pinned_requirements(ROOT / "requirements.lock")
    gui_lock = _pinned_requirements(ROOT / "requirements-gui.lock")

    for requirement in project["project"]["dependencies"] + project["project"]["optional-dependencies"]["test"]:
        key, version = _split_requirement(requirement)
        assert lock[key] == version

    for requirement in project["project"]["optional-dependencies"]["gui"]:
        key, version = _split_requirement(requirement)
        assert gui_lock[key] == version
        assert key not in lock

    assert "pyqt6" not in lock
    assert "pyqtgraph" not in lock


def test_public_versions_paths_and_exports_match_documentation_contract() -> None:
    config = load_config(ROOT / "config.example.yaml")
    paths = DatasetService(config).paths()

    assert Path(paths["canonical"]) == Path("bang_rl/datasets/canonical")
    assert Path(paths["cache"]) == Path("bang_rl/datasets/cache")
    assert FINAL_ARTIFACT_TYPE == "inference"
    assert FINAL_ARTIFACT_FORMAT_VERSION == "1.0"
    assert FINAL_ARTIFACT_EXTENSION == ".bangmodel"
    assert PHASE2_ARTIFACT_TYPE == "phase2_inference_module"
    assert PHASE2_ARTIFACT_FORMAT_VERSION == "1.0"
    assert PHASE2_ARTIFACT_EXTENSION == ".phase2model"
    assert CANONICAL_DATASET_SCHEMA_VERSION == "1.0"
    assert CACHE_FORMAT_VERSION == "1.0"
    assert OBSERVATION_SCHEMA_VERSION == "1.0"
    assert ACTION_SPACE_VERSION == "discrete-6-v1"
    assert DATA_ACTION_SPACE_VERSION == "1.0"
    assert set(LOG_CHANNELS) == {
        "episode_events",
        "termination",
        "reward_breakdown",
        "invalid_action",
        "training",
        "evaluation",
        "privileged_diagnostics",
    }

    # These are the documented GUI-independent boundaries, not private state.
    assert ConfigService and DatasetService and EvaluationService is not None
    assert GameService and TrainingService
    for directory in (
        ROOT / "bang_rl/datasets/canonical",
        ROOT / "bang_rl/datasets/cache",
        ROOT / "bang_rl/models",
        ROOT / "bang_rl/logs",
    ):
        assert directory.is_dir()
