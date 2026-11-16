"""Generated data is git-ignored; the user-kept saved/ folder is not."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

IGNORED = [
    "bang_rl/models/intent/20261110-0240_intent_e5_b8ecac.phase2model",
    "bang_rl/models/intent/20261110-0240_intent_e5_b8ecac.phase2model.about.md",
    "bang_rl/models/emotion/x.phase2model",
    "bang_rl/models/main_ai/x.bangmodel",
    "bang_rl/datasets/canonical/x/shard-0000.parquet",
    "bang_rl/datasets/canonical/x/ABOUT.md",
    "bang_rl/datasets/cache/x/cache-train.npz",
    "datasets/canonical/20261115-1432_dataset_dummy5_50000g_3f9a1c/collection_summary.json",
    "datasets/cache/x/ABOUT.md",
    "reports/20261115-1530_report_module-training_7b2d04.json",
    "reports/baselines/20261115-1530_baseline_250g_abcdef.json",
    "outputs/some-run/canonical/shard-0000.parquet",
    "artifacts/20261116-0910_mainai_80000g_c81e55.bangmodel",
    "artifacts/20261116-0910_mainai_80000g_c81e55.bangmodel.about.md",
    "artifacts/migrations/migration-1.json",
    "bang_rl/replays/20261116-1012_replay_main1-dummy4_a1b2c3.json",
    "bang_rl/logs/run.log",
    "runs/iterate1/rounds.json",
]

NOT_IGNORED = [
    "saved/README.md",
    "saved/datasets/더미끼리 5만판/shard-0000.parquet",
    "saved/datasets/더미끼리 5만판/ABOUT.md",
    "saved/modules/my run/intent.phase2model",
    "saved/modules/my run/intent.phase2model.about.md",
    "saved/main_ai/my model.bangmodel",
    "saved/main_ai/my model.bangmodel.about.md",
    "saved/reports/my run.json",
    "bang_rl/tests/data/sample.json",
    "bang_rl/models/.gitkeep",
    "reports/.gitkeep",
]


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {result.stderr.strip()}")
    return result.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _need_git() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git repository not available")


@pytest.mark.parametrize("path", IGNORED)
def test_generated_paths_are_ignored(path: str) -> None:
    assert _ignored(path), f"{path} should be git-ignored"


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_saved_and_source_paths_are_not_ignored(path: str) -> None:
    assert not _ignored(path), f"{path} must not be git-ignored"


def test_gitignore_documents_saved_folder() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "saved/ is DELIBERATELY NOT IGNORED" in text
    assert "!/saved/" in text and "Git LFS" in text
    assert not any(line.strip() in {"saved", "saved/", "/saved", "/saved/", "saved/*", "/saved/*"} for line in text.splitlines())
