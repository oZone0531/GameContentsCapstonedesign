"""Episode-budget and live-progress tests for the default training runner.

DELIVERABLE 3 regression guard.  A default run (episodes=1, workers=1, batch=64,
buffer=256, mode rl, 1 core / 1 GB) used to never visibly finish because
``bang_rl/training/pipeline_runner.py`` forced at least ``_DEFAULT_COLLECTION_
EPISODES`` (then 8) full episodes via ``max(episode_budget, _DEFAULT_COLLECTION_
EPISODES)``, even when the user set episodes=1.  With env max_episode_steps=2000
a first-legal-action standoff can run thousands of environment steps per
episode, so forcing 8 of them on a single core looked like a hang; the GUI also
appeared frozen because counters only updated after each full episode.

The fix honours an explicit episodes value verbatim (episodes=1 collects exactly
1 episode), reduces the unset floor to 1, and mirrors the in-progress episode's
environment-step total onto run during collection.

Test A is PURE: it inspects the runner SOURCE via ast WITHOUT importing it, so
it runs in the restricted sandbox (importing the runner would pull torch / numpy
/ pydantic, none of which are installable here).  It is revert-sensitive: a
revert to the forced-8 behavior fails it.

Tests B and C are CI-ONLY: they need numpy / pydantic / torch and are guarded
with ``pytest.importorskip`` so they SKIP cleanly in-sandbox and run in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _runner_module_path() -> Path:
    return Path(__file__).parents[1] / "training" / "pipeline_runner.py"


# --- Test A: pure, dependency-free source guard ------------------------- #


def test_default_collection_floor_is_one_and_episode_budget_is_not_forced() -> None:
    """Revert-sensitive guard: the forced-8 episode floor is gone.

    Inspects the module SOURCE directly (never imports it, so it runs in the
    restricted sandbox).  It asserts (a) the module-level assignment
    ``_DEFAULT_COLLECTION_EPISODES = 1`` and (b) the old forcing pattern
    ``max(episode_budget, _DEFAULT_COLLECTION_EPISODES)`` is NOT present, so a
    revert that re-forces a minimum episode count fails this test.  Modelled on
    ``test_pipeline_runner_module_exists_is_callable_and_qt_free`` in
    ``test_training_runner_wiring.py``.
    """

    path = _runner_module_path()
    assert path.is_file(), "bang_rl/training/pipeline_runner.py must exist"

    source = path.read_text(encoding="utf-8")

    # (a) Parse the module-level assignment and confirm the floor is 1.
    tree = ast.parse(source)
    floor_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_DEFAULT_COLLECTION_EPISODES":
                    assert isinstance(node.value, ast.Constant), "_DEFAULT_COLLECTION_EPISODES must be a literal"
                    floor_value = node.value.value
    assert floor_value == 1, "_DEFAULT_COLLECTION_EPISODES must be 1 (deliberate reduction from the forced 8)"

    # (b) The old forcing pattern must be gone so episodes=1 is honoured.
    assert (
        "max(episode_budget, _DEFAULT_COLLECTION_EPISODES)" not in source
    ), "the runner must not force a minimum episode count on an explicit episodes value"

    # The runner must stay Qt-free (a cross-layer guard greps for these).
    lowered = source.lower()
    assert "pyqt" not in lowered
    assert "pyqtgraph" not in lowered


# --- Test B: CI-only behavioral end-to-end run with episodes=1 ---------- #


def test_default_run_with_episodes_one_finishes_with_exactly_one_episode(tmp_path: Path) -> None:
    """CI-ONLY: a default episodes=1 run finishes and collects exactly 1 episode.

    Starts ``TrainingService`` with NO injected runner (so the real default
    runner resolves and runs on the daemon thread) and episodes=1, waits with a
    generous timeout, and asserts the run finished, collected exactly one
    episode, produced environment steps, and wrote a ``.bangmodel`` artifact.
    This is the behavioral proof that the forced-8 floor no longer applies.

    CI-ONLY: needs numpy / pydantic / torch, which are NOT installable in the
    restricted sandbox, so the importorskip guards SKIP it there.
    """

    pytest.importorskip("numpy")
    pytest.importorskip("pydantic")
    pytest.importorskip("torch")

    from bang_rl.app import TrainingConfig, TrainingService
    from bang_rl.config import SystemConfig

    service = TrainingService()
    handle = service.start(
        TrainingConfig(
            config=SystemConfig(),
            episodes=1,
            report_path=tmp_path / "report.json",
            artifact_path=tmp_path / "model.bangmodel",
            resume_artifact_path=tmp_path / "no-resume",
        )
    )
    report = service.wait(handle, timeout=600.0)

    assert report.status == "finished", f"default episodes=1 run did not finish: {report.status}"
    assert report.progress.episodes_completed == 1, "episodes=1 must collect exactly one episode"
    assert report.progress.environment_steps > 0
    assert report.progress.artifact_path is not None
    assert Path(report.progress.artifact_path).exists(), "a .bangmodel artifact must be written"


# --- Test C: CI-only short-rollout PPO update ---------------------------- #


def test_ppo_update_tolerates_a_short_rollout(tmp_path: Path) -> None:
    """CI-ONLY: PPOTrainer.update() handles a short (few-transition) rollout.

    Feeds only 3 trainable transitions through ``collect_rollout`` (far fewer
    than the default ``rollout_transitions_per_worker`` of 256) and calls
    ``update()``.  It must return a ``PPOUpdateMetrics`` with samples>0 and not
    raise or hang, proving that a single short episode does not block PPO and
    that there is no buffer-fill floor.  Construction follows the PPOTrainer /
    PPOTransition usage in ``test_ppo_trainer.py``.

    CI-ONLY: needs torch (and numpy), so the importorskip guards SKIP it in the
    restricted sandbox.
    """

    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    import numpy as np

    from bang_rl.agents.main_agent import PPOModule
    from bang_rl.config import PPOConfig
    from bang_rl.training import PPOTrainer, PPOTransition, PPOUpdateMetrics

    module = PPOModule(input_dim=4, seed=5)
    # The default rollout buffer holds rollout_transitions_per_worker (256)
    # transitions; feed only a few to prove update() does not require a full one.
    trainer = PPOTrainer(
        module,
        worker_count=1,
        config=PPOConfig(ppo_epochs=2, minibatch_size=2),
        seed=13,
    )

    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    transitions = [
        PPOTransition(
            features=np.full(4, index / 10, dtype=np.float32),
            action=index % 2,
            action_mask=mask,
            old_log_prob=-0.4,
            value=0.1,
            reward=1.0,
            terminated=True,
            trainable=True,
            worker_id=0,
            episode_id="short-episode",
            transition_id=index,
            environment_step=index + 1,
        )
        for index in range(3)
    ]
    selected = trainer.collect_rollout(transitions)
    assert len(selected) == 3, "all three trainable transitions should be collected"

    metrics = trainer.update()

    assert isinstance(metrics, PPOUpdateMetrics)
    assert metrics.samples > 0, "a short rollout must still produce samples"
    assert metrics.ppo_update == 1
    assert metrics.main_agent_transitions == 3
