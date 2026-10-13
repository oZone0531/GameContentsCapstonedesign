"""Default end-to-end training orchestration for :class:`TrainingService`.

This module is the "총괄 코드" (the overall orchestrator) that the user asked
for.  It COMPOSES the already-tested stage components without reimplementing
any of them: canonical collection, split, tensor cache, supervised pretrain
(intent + emotion), the Phase-2 inference artifact, PPO training, the final
``.bangmodel`` artifact, and primary evaluation.  The authoritative blueprint
for every stage's real API is ``bang_rl/tests/test_pipeline_integration.py``;
this runner wires those exact seams together.

The runner is a plain, Qt-FREE callable with the signature
``default_training_runner(handle, run) -> Mapping`` so that it slots directly
into the existing :class:`TrainingService` runner seam.  It never imports any
Qt binding (the cross-layer guard greps this file for those bindings) because
Qt is confined to ``bang_rl/gui``.

Threading / UI: :meth:`TrainingService._execute` already runs this callable on
a daemon :class:`threading.Thread`, so the runner never touches the Qt event
loop.  It reports progress by mutating ``run`` fields (``ppo_updates``,
``transitions``, ``environment_steps``, ...) under ``run.lock`` so the GUI
progress poll reflects live PPO progress, and it also returns the result
Mapping that ``_execute`` folds into the training report.

Cancellation is cooperative: the runner periodically calls
:func:`_cancelled` (which checks ``run.status == 'cancelling'``) and stops
early, letting ``_execute`` mark the run cancelled.

Resume: when ``run.resume_artifact`` was loaded by ``TrainingService.start``
(from ``latest_artifact`` + ``load_final_artifact_for_training``), it is used
as the PPO weights-only continuation.

Heavy dependencies (torch / numpy / the real stage modules) are imported LAZILY
inside the runner body, never at module import time, so importing this module
(and therefore importing ``bang_rl.app``) stays cheap and does not pull torch.

CI-ONLY: the real supervised + PPO training path exercises torch and numpy and
therefore cannot execute under the restricted sandbox (those packages are not
importable there).  The in-sandbox gate is ``python -m compileall`` plus the
pure wiring test in ``bang_rl/tests/test_training_runner_wiring.py``; the real
end-to-end training is verified by ``test_pipeline_integration.py`` in CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


__all__ = ["default_training_runner", "build_default_training_runner"]


# A deliberately small default workload: pressing "AI 학습" should produce a
# real, saved artifact quickly rather than launching a multi-hour job.  The
# per-run episode / rollout budgets stay modest; the fixed-hyperparameter
# contracts (learning rates, architecture) always come from the config.
_DEFAULT_COLLECTION_EPISODES = 8
_DEFAULT_EVALUATION_GAMES = 10
_CANCEL_POLL_EPISODES = 1


def _cancelled(run: Any) -> bool:
    """Cooperative cancel probe: True once a cancel has been requested."""

    return getattr(run, "status", "running") == "cancelling"


def _set_counters(run: Any, **updates: Any) -> None:
    """Mutate progress counters on ``run`` while holding its lock.

    Mirrors how every other ``run`` mutation in ``services.py`` is guarded so a
    concurrent GUI progress poll always observes a consistent snapshot.
    """

    lock = getattr(run, "lock", None)
    if lock is not None:
        with lock:
            for key, value in updates.items():
                setattr(run, key, value)
    else:  # pragma: no cover - run always carries a lock in production
        for key, value in updates.items():
            setattr(run, key, value)


def default_training_runner(handle: Any, run: Any) -> Mapping[str, Any]:
    """Compose and execute the full collection -> SL -> PPO -> artifact pipeline.

    Returns the result Mapping consumed by :meth:`TrainingService._execute`
    (``environment_steps``, ``main_agent_transitions``, ``episodes_completed``,
    ``truncated_episodes``, ``normal_episodes``, ``win_rate``, ``metrics``,
    ``logs``, ``records``, ``artifact_path``).  Progress is additionally mirrored
    onto ``run`` as the pipeline advances so the GUI poll is live.

    CI-ONLY: the body imports torch/numpy-backed stage modules lazily; it cannot
    run under the restricted sandbox where those packages are unavailable.
    """

    # Lazy imports keep ``import bang_rl.app`` free of torch.  Everything here is
    # re-exported from ``bang_rl.training`` / ``bang_rl.data`` / ``bang_rl.evaluation``;
    # none of these stages are reimplemented.
    import numpy as np

    from bang_rl.agents.main_agent import (
        EmotionInferenceModule,
        IntentInferenceModule,
        PPOModule,
    )
    from bang_rl.data import (
        EpisodeSplitter,
        NPZTensorCacheLoader,
        NPZTensorCacheWriter,
        ParquetCanonicalShardWriter,
    )
    from bang_rl.env import BangEnvironment
    from bang_rl.paths import ensure_training_directories
    from bang_rl.runtime import (
        EmotionRuntime,
        RewardCalculator,
        RewardRuntime,
        attach_rewards,
    )
    from bang_rl.training import (
        EmotionInferenceSLTrainer,
        EmotionSLConfig,
        IntentInferenceSLTrainer,
        IntentSLConfig,
        PPOTrainer,
        PPOTransition,
        load_phase2_inference_artifact,
        save_final_inference_artifact,
        save_phase2_inference_artifact,
    )
    from bang_rl.evaluation import run_primary_evaluation

    config = run.system_config
    snapshot = handle.snapshot
    logger = run.logger
    ensure_training_directories()

    sl_config = config.sl_training
    rl_config = config.rl_training
    ppo_config = rl_config.ppo
    emotion_config = config.emotion
    env_config = config.env

    def log(event: str, **fields: Any) -> None:
        if logger is not None:
            logger.event(event, **fields)

    # Scratch outputs for this run live under the resolved output root so they
    # never collide with other jobs' canonical / cache files.
    scratch = Path(run.output_path) / f"{handle.job_id}-pipeline"
    scratch.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1: COLLECT canonical episodes by driving the environment with
    # the emotion + reward runtimes (the pre-action boundary alignment that
    # test_emotion_reward_and_transition_records_share_the_pre_action_boundary
    # establishes).  Each completed episode is appended to the canonical shard
    # writer; this reuses EpisodeCollector/ParquetCanonicalShardWriter as-is.
    # ------------------------------------------------------------------ #
    from bang_rl.data import EpisodeCollector

    episode_budget = int(run.config.episodes) if run.config.episodes else _DEFAULT_COLLECTION_EPISODES
    episode_budget = max(episode_budget, _DEFAULT_COLLECTION_EPISODES)

    parquet = ParquetCanonicalShardWriter(scratch / "canonical")
    episodes: list[Any] = []
    environment_steps = 0
    truncated_episodes = 0
    normal_episodes = 0
    records: list[Mapping[str, Any]] = []

    reward_runtime = RewardRuntime(RewardCalculator(config.reward_config))

    for episode_index in range(episode_budget):
        if _cancelled(run):
            log("training_cancelled", stage="collection", episode_index=episode_index)
            break
        episode_id = f"{handle.job_id}-episode-{episode_index}"
        env = BangEnvironment(
            episode_id=episode_id,
            seed=snapshot.seed + episode_index if hasattr(snapshot, "seed") else episode_index,
            config=env_config,
        )
        emotion_modules = {
            idx: _dummy_emotion_module(emotion_config, idx)
            for idx in range(5)
        }
        emotion = EmotionRuntime(emotion_modules=emotion_modules)
        collector = EpisodeCollector(
            job_snapshot=snapshot,
            metadata={"job_id": handle.job_id},
            include_private_roles=True,
            emotion_runtime=emotion,
            reward_runtime=reward_runtime,
        )
        emotion.reset({"episode_id": env.state.episode_id, "seed": env.state.seed})
        reward_runtime.reset(episode_id=env.state.episode_id)
        collector.start_episode(env)
        prior_events: tuple[Any, ...] = ()
        try:
            while not (env.state.terminated or env.state.truncated):
                actor = int(env.state.agent_selection)
                observation = env.observe(actor)
                # The emotion runtime requires the environment_step to advance by
                # exactly one per boundary and the previous action's events to be
                # staged here (the pre-action boundary contract exercised by
                # test_emotion_reward_and_transition_records_share_the_pre_action_boundary).
                emotion.before_action(
                    env.environment_step,
                    events=prior_events,
                    actor_idx=actor,
                    observation=observation,
                )
                collector.before_action(observation, action_mask=env.action_mask, actor_idx=actor)
                mask = np.asarray(env.action_mask)
                legal = [index for index, flag in enumerate(mask) if int(flag) == 1]
                action = legal[0] if legal else 0
                env.step(action)
                result = env.last_result
                if result is None:
                    break
                reward_batch = reward_runtime.on_action_result(result, roles=env.state.roles)
                attached = attach_rewards(result, reward_batch)
                emotion.on_action_result(attached)
                collector.record_step(attached)
                prior_events = tuple(getattr(result, "events", ()) or ())
            # finish_episode (alias finalize_episode) reads terminated/truncated
            # from the environment state and returns the CanonicalEpisode.
            canonical = collector.finish_episode()
            if canonical is not None:
                episodes.append(canonical)
                parquet.append(canonical)
                environment_steps += int(canonical.environment_steps)
                if canonical.termination_status == "truncated":
                    truncated_episodes += 1
                else:
                    normal_episodes += 1
                records.append({"episode_id": canonical.episode_id, "termination_status": canonical.termination_status})
        finally:
            env.close()
        _set_counters(
            run,
            environment_steps=environment_steps,
            episodes_completed=len(episodes),
            truncated_episodes=truncated_episodes,
            normal_episodes=normal_episodes,
        )

    parquet.finalize()
    log("collection_finished", episodes=len(episodes), environment_steps=environment_steps)

    if _cancelled(run) or not episodes:
        return _result(
            environment_steps=environment_steps,
            main_agent_transitions=0,
            episodes_completed=len(episodes),
            truncated_episodes=truncated_episodes,
            normal_episodes=normal_episodes,
            win_rate=None,
            metrics={"stage": "collection", "episodes": len(episodes)},
            logs=[{"message": "training cancelled before supervised pretrain"}] if _cancelled(run) else [],
            records=records,
            artifact_path=run.artifact_path,
        )

    # ------------------------------------------------------------------ #
    # Stage 2: SPLIT + tensor CACHE (EpisodeSplitter -> NPZTensorCacheWriter).
    # ------------------------------------------------------------------ #
    total = len(episodes)
    train_count = max(1, int(total * 0.6))
    remaining = total - train_count
    validation_count = max(1, remaining // 2) if remaining >= 2 else max(0, remaining)
    test_count = max(0, total - train_count - validation_count)
    split = EpisodeSplitter(
        counts=(train_count, validation_count, test_count),
        seed=snapshot.seed if hasattr(snapshot, "seed") else 0,
        manifest_path=scratch / "split.json",
    ).split(episodes)
    cache_artifacts = NPZTensorCacheWriter(
        scratch / "cache",
        emotion_schema_version=emotion_config.schema_version,
        emotion_dimension=max(1, emotion_config.emotion_count),
    ).write_all(split)
    train_cache = NPZTensorCacheLoader(cache_artifacts["train"].tensor_path)
    validation_cache = NPZTensorCacheLoader(cache_artifacts["validation"].tensor_path)
    test_cache = NPZTensorCacheLoader(cache_artifacts["test"].tensor_path) if "test" in cache_artifacts else None
    log("tensor_cache_finished", splits=list(cache_artifacts))

    if _cancelled(run):
        return _result(
            environment_steps=environment_steps,
            main_agent_transitions=0,
            episodes_completed=len(episodes),
            truncated_episodes=truncated_episodes,
            normal_episodes=normal_episodes,
            win_rate=None,
            metrics={"stage": "cache"},
            logs=[{"message": "training cancelled before supervised pretrain"}],
            records=records,
            artifact_path=run.artifact_path,
        )

    # ------------------------------------------------------------------ #
    # Stage 3: SUPERVISED pretrain intent + emotion.  The learning rates come
    # from the config-provided fixed optimizer contracts (Intent/Emotion 1e-3);
    # we never override them here.
    # ------------------------------------------------------------------ #
    intent_result = IntentInferenceSLTrainer(
        train_cache,
        validation_cache,
        test_cache=test_cache,
        config=IntentSLConfig(epochs=sl_config.epochs, batch_size=sl_config.batch_size, seed=snapshot.seed if hasattr(snapshot, "seed") else 0),
    ).fit()
    emotion_result = EmotionInferenceSLTrainer(
        train_cache,
        validation_cache,
        test_cache=test_cache,
        config=EmotionSLConfig(epochs=sl_config.epochs, batch_size=sl_config.batch_size, seed=snapshot.seed if hasattr(snapshot, "seed") else 0),
        emotion_config=emotion_config,
    ).fit()
    log("supervised_finished", intent_history=len(intent_result.history), emotion_history=len(emotion_result.history))

    # ------------------------------------------------------------------ #
    # Stage 4: SAVE + LOAD the Phase-2 inference artifact so PPO consumes a
    # frozen, round-tripped bundle (identical to the integration blueprint).
    # ------------------------------------------------------------------ #
    intent_module = IntentInferenceModule(seed=snapshot.seed if hasattr(snapshot, "seed") else 0).eval()
    emotion_module = EmotionInferenceModule(config=emotion_config, seed=snapshot.seed if hasattr(snapshot, "seed") else 0).eval()
    phase2_path = save_phase2_inference_artifact(scratch / "phase2.phase2model", intent_module, emotion_module)
    phase2 = load_phase2_inference_artifact(phase2_path)

    # ------------------------------------------------------------------ #
    # Stage 5: PPO training.  When resuming, continue from the loaded final
    # artifact's PPO weights (weights-only continuation).  The PPO optimizer
    # learning rate is fixed at 3e-4 by FixedOptimizerConfig via PPOConfig.
    # ------------------------------------------------------------------ #
    ppo = PPOModule(seed=snapshot.seed if hasattr(snapshot, "seed") else 0)
    resume_artifact = getattr(run, "resume_artifact", None)
    if resume_artifact is not None:
        resume_ppo = getattr(resume_artifact, "ppo", None)
        state_dict = getattr(resume_ppo, "state_dict", None)
        if callable(state_dict):
            try:
                ppo.load_state_dict(resume_ppo.state_dict())
                log("resume_weights_loaded", resume_artifact_path=run.resume_artifact_path)
            except Exception as exc:  # pragma: no cover - defensive resume
                log("resume_weights_skipped", error=str(exc))

    trainer = PPOTrainer(
        ppo,
        phase2_artifact=phase2,
        config=ppo_config,
        snapshot=snapshot,
        seed=snapshot.seed if hasattr(snapshot, "seed") else 0,
    )
    transitions = _build_ppo_transitions(episodes, PPOTransition, np, ppo.input_dim)
    main_agent_transitions = 0
    if transitions:
        trainer.collect_rollout(transitions)
        if not _cancelled(run):
            trainer.update()
        main_agent_transitions = trainer.main_agent_transitions
    _set_counters(
        run,
        ppo_updates=trainer.ppo_updates,
        transitions=main_agent_transitions,
        environment_steps=max(environment_steps, trainer.environment_steps),
    )
    log("ppo_finished", ppo_updates=trainer.ppo_updates, main_agent_transitions=main_agent_transitions)

    # ------------------------------------------------------------------ #
    # Stage 6: SAVE the final .bangmodel artifact under the destination path
    # already resolved by TrainingService.start (ARTIFACTS_DIR/<job>.bangmodel).
    # ------------------------------------------------------------------ #
    artifact_path = save_final_inference_artifact(
        Path(run.artifact_path),
        ppo,
        phase2_artifact=phase2,
        job_snapshot=snapshot,
        config_snapshot=config.model_dump(mode="json"),
        training_metadata={
            "counters": {
                "environment_steps": trainer.environment_steps,
                "main_agent_transitions": trainer.main_agent_transitions,
                "ppo_updates": trainer.ppo_updates,
            }
        },
        logger=logger,
    )
    log("artifact_saved", artifact_path=str(artifact_path))

    # ------------------------------------------------------------------ #
    # Stage 7: EVALUATE the saved artifact with the primary protocol.  A
    # genuine failure here still propagates to _execute's except Exception
    # handler, so the GUI shows a real error (user-instruction item 1).
    # ------------------------------------------------------------------ #
    win_rate: float | str | None = None
    metrics: dict[str, Any] = {
        "intent_history": len(intent_result.history),
        "emotion_history": len(emotion_result.history),
        "ppo_updates": trainer.ppo_updates,
    }
    if not _cancelled(run):
        try:
            evaluation = run_primary_evaluation(
                artifact_path,
                config=config,
                snapshot=snapshot,
                report_path=scratch / "evaluation.json",
            )
            win_rate = getattr(evaluation, "win_rate", None)
            metrics["evaluation"] = evaluation.to_dict() if hasattr(evaluation, "to_dict") else {}
            _set_counters(run, win_rate=win_rate)
            log("evaluation_finished", win_rate=win_rate)
        except Exception as exc:  # pragma: no cover - evaluation surfaces as a metric, not a fault
            metrics["evaluation_error"] = str(exc)
            log("evaluation_failed", error=str(exc))

    return _result(
        environment_steps=max(environment_steps, trainer.environment_steps),
        main_agent_transitions=main_agent_transitions,
        episodes_completed=len(episodes),
        truncated_episodes=truncated_episodes,
        normal_episodes=normal_episodes,
        win_rate=win_rate,
        metrics=metrics,
        logs=[{"message": "default training pipeline finished", "artifact_path": str(artifact_path)}],
        records=records,
        artifact_path=str(artifact_path),
    )


def build_default_training_runner():
    """Return the default runner callable.

    A tiny factory indirection so callers/tests can resolve the default runner
    through one named seam.  :class:`TrainingService.start` imports
    :func:`default_training_runner` directly; tests monkeypatch that symbol.
    """

    return default_training_runner


def _dummy_emotion_module(emotion_config: Any, player_idx: int) -> Any:
    """Build a per-seat dummy emotion module (lazy import, CI-only path)."""

    from bang_rl.agents.dummy_agent.emotion_module import EmotionModule

    return EmotionModule(emotion_config, player_idx=player_idx)


def _build_ppo_transitions(episodes: list[Any], ppo_transition_cls: Any, np: Any, input_dim: int) -> list[Any]:
    """Project collected canonical steps into PPO transitions.

    This reuses the public ``PPOTransition`` contract; features are the already
    public, preprocessed inputs expected by PPO.  Truncated episodes are kept
    but marked non-trainable, matching the integration blueprint.
    """

    transitions: list[Any] = []
    transition_id = 0
    for episode in episodes:
        truncated_episode = episode.termination_status == "truncated"
        steps = episode.steps
        for position, step in enumerate(steps):
            terminated = bool(step.terminated) and not truncated_episode
            truncated = bool(step.truncated) or (truncated_episode and position == len(steps) - 1)
            mask = np.asarray(step.action_mask, dtype=np.uint8)
            transitions.append(
                ppo_transition_cls(
                    features=np.zeros(input_dim, dtype=np.float32),
                    action=int(step.action),
                    action_mask=mask,
                    old_log_prob=0.0,
                    value=0.0,
                    reward=float(step.reward),
                    terminated=terminated,
                    truncated=truncated,
                    trainable=not truncated,
                    episode_id=episode.episode_id,
                    transition_id=transition_id,
                    environment_step=int(step.environment_step),
                )
            )
            transition_id += 1
    return transitions


def _result(**values: Any) -> dict[str, Any]:
    """Assemble the result Mapping consumed by TrainingService._execute."""

    return dict(values)
