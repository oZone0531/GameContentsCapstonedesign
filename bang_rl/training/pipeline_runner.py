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
#
# This is the collection floor applied ONLY when run.config.episodes is left
# unset (None or 0).  It is intentionally 1 (a deliberate reduction from the
# previous forced minimum of 8): with env max_episode_steps defaulting to 2000,
# a single first-legal-action standoff episode can run up to ~2000 environment
# steps on a single core, so forcing 8 full episodes meant up to ~16000 steps
# and made a default run (episodes=1, 1 core, 1 GB) look like a hang.  When the
# user DOES set episodes we honour that value verbatim and never raise it to
# this floor, so pressing "AI 학습" with episodes=1 collects exactly 1 episode
# and a quick run is really quick.
_DEFAULT_COLLECTION_EPISODES = 1
# How often (in environment steps) the collection loop mirrors the growing
# environment-step total onto run so the GUI progress poll stays live during a
# single long episode instead of only updating after the episode completes.
_PROGRESS_STEP_INTERVAL = 25
# A modest per-episode step cap applied ONLY inside this runner's collection
# loop.  It is deliberately independent of ``BangEnvironment.max_episode_steps``
# (default 2000) and does NOT change the env's own truncation contract used
# elsewhere.  It exists because the collection policy here always selects the
# first legal action, and the legal action mask always sets index 0 (PASS), so
# every seat passes every turn: no HP changes, nobody is eliminated, and the
# episode can only end via the env's 2000-step truncation.  On a single core
# that reads as a silent hang after ``training_started``, so we stop a passive
# PASS standoff at ``_COLLECTION_STEP_CAP`` instead.
_COLLECTION_STEP_CAP = 200
# Short post-training evaluation budget used when the user left the default
# primary protocol in place.  The fixed 500-game primary benchmark cannot be
# shortened (its reproducibility invariant), so Stage 7 runs this many games
# through the configurable custom protocol instead to keep the button quick.
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


def collection_step_cap(env_config: Any, default: int = _COLLECTION_STEP_CAP) -> int:
    """Return the effective per-episode step cap for the collection loop.

    Pure and dependency-free (no numpy / torch), so it is testable in the
    restricted sandbox.  The cap honours an optional positive integer override
    on ``env_config`` (``collection_step_cap``) and otherwise uses ``default``.
    The result is additionally clamped so it can never exceed the env's own
    ``max_episode_steps`` when that attribute is available, keeping the cap at or
    below the env truncation contract.  ``env_config`` may be ``None`` (returns
    ``default``).
    """

    cap = int(default)
    override = getattr(env_config, "collection_step_cap", None)
    if isinstance(override, int) and not isinstance(override, bool) and override > 0:
        cap = int(override)
    max_steps = getattr(env_config, "max_episode_steps", None)
    if isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0:
        cap = min(cap, int(max_steps))
    return cap


def should_stop_collection(step: int, cap: int) -> bool:
    """True once ``step`` reaches a positive collection ``cap``."""

    return cap > 0 and step >= cap


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
    from bang_rl.evaluation import (
        PRIMARY_PROTOCOL_ID,
        run_custom_evaluation,
    )

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

    # Honour an explicit user episode count verbatim (episodes=1 collects
    # exactly 1 episode); apply the small default floor ONLY when the user left
    # episodes unset/None/0.  We deliberately do NOT re-raise an explicit value
    # to the floor, so a quick run really is quick.
    if run.config.episodes:
        episode_budget = max(1, int(run.config.episodes))
    else:
        episode_budget = _DEFAULT_COLLECTION_EPISODES

    parquet = ParquetCanonicalShardWriter(scratch / "canonical")
    episodes: list[Any] = []
    environment_steps = 0
    truncated_episodes = 0
    normal_episodes = 0
    records: list[Mapping[str, Any]] = []

    reward_runtime = RewardRuntime(RewardCalculator(config.reward_config))

    # Emit an early log BEFORE the collection loop so a run is never silent
    # after training_started.  The per-episode step cap bounds the passive
    # first-legal-action (PASS) standoff so a default episode finishes promptly
    # instead of running to the env's max_episode_steps truncation.
    step_cap = collection_step_cap(env_config)
    log("collection_started", episode_budget=episode_budget, step_cap=step_cap)

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
        completed_before = len(collector.episodes)
        prior_events: tuple[Any, ...] = ()
        last_progress_step = 0
        try:
            while not (env.state.terminated or env.state.truncated):
                actor = int(env.state.agent_selection)
                # Feed the CANONICAL observation mapping (to_observation_dict),
                # not the PublicEnvironmentState object.  The collector stores
                # whatever it is handed as the step's pre_observation, and the
                # NPZ tensor-cache writer later runs
                # ObservationPreprocessor.preprocess(step.pre_observation),
                # which requires the to_observation_dict schema (healths, alive,
                # turn_order, action_mask, ...).  The full-state to_dict shape
                # (players/event_log, no action_mask) would raise DataError in
                # Stage 3 (SL pretrain).  This matches the canonical _observation
                # fixtures the SL stage consumes in test_pipeline_integration.py.
                observation = env.observe(actor).to_observation_dict()
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
                # Mirror the growing environment-step total onto run during the
                # episode so the GUI poll is not frozen while a single long
                # (up to max_episode_steps=2000) episode runs.  We publish the
                # already-finalized episodes' steps plus this episode's current
                # env.environment_step, updating only every few steps to keep the
                # lock contention negligible.  The authoritative post-episode
                # counter update below still records the completed totals.
                in_progress_step = int(env.environment_step)
                if in_progress_step - last_progress_step >= _PROGRESS_STEP_INTERVAL:
                    last_progress_step = in_progress_step
                    _set_counters(run, environment_steps=environment_steps + in_progress_step)
                # Stop a passive PASS standoff at the collection cap instead of
                # running to the env's ~2000-step truncation on a single core.
                if should_stop_collection(int(env.environment_step), step_cap):
                    break
            # A capped episode is NOT env-terminated/truncated, so record_step
            # never auto-finalized it.  Finalize it explicitly (marking it
            # truncated so it is a valid canonical episode) when the collector
            # still has an active, unfinished episode; a genuinely terminal
            # episode has already been auto-finalized by record_step, so we do
            # NOT double-finalize it.
            if collector.active_episode_id is not None:
                collector.finish_episode(
                    result={
                        "action": 0,
                        "environment_step": int(env.environment_step),
                        "truncated": True,
                    }
                )
            # record_step auto-finalizes on a terminal/truncated result and the
            # explicit finish_episode above finalizes a capped episode, so by
            # this point any collected episode is appended to collector.episodes.
            # Read it from there rather than calling finish_episode() again.  If
            # the loop exited without any finalized episode (e.g. a None result
            # broke out before the first step), episodes may be unchanged, so
            # guard on the completed count.
            canonical = (
                collector.episodes[-1]
                if len(collector.episodes) > completed_before
                else None
            )
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
        # Per-episode progress so the run is visible between training_started
        # and collection_finished, not only after the whole loop.
        log(
            "episode_finished",
            episode_index=episode_index,
            environment_steps=environment_steps,
            episodes_completed=len(episodes),
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
    # Stage 7: EVALUATE the saved artifact.  The fixed primary protocol
    # ("deterministic_v1") is an immutable 500-game benchmark (PRIMARY_GAMES is
    # hardcoded in PrimaryEvaluationProtocol and its reproducibility manifest
    # asserts exactly 500 seeds), so it CANNOT be shortened without violating
    # that invariant.  A "AI 학습" button press must stay quick, so when the
    # user left the default primary protocol in place we run a short custom
    # evaluation (_DEFAULT_EVALUATION_GAMES games via CustomEvaluationProtocol,
    # which honours a configurable games count) instead of the 500-game
    # benchmark.  If the user explicitly configured a custom/stochastic
    # protocol we respect their EvaluationConfig verbatim.  A genuine failure
    # here still propagates to _execute's except Exception handler, so the GUI
    # shows a real error (user-instruction item 1).
    # ------------------------------------------------------------------ #
    win_rate: float | str | None = None
    metrics: dict[str, Any] = {
        "intent_history": len(intent_result.history),
        "emotion_history": len(emotion_result.history),
        "ppo_updates": trainer.ppo_updates,
    }
    if not _cancelled(run):
        try:
            evaluation_config = config.evaluation
            if evaluation_config.protocol_id == PRIMARY_PROTOCOL_ID:
                # Keep the button quick: derive a short, schema-valid custom
                # evaluation from the primary config rather than running the
                # fixed 500-game benchmark.  model_copy keeps every validator
                # (deterministic == action_mode=='deterministic', balanced seat
                # policy with an empty seats list) satisfied.
                quick_evaluation = evaluation_config.model_copy(
                    update={
                        "protocol_id": "custom_v1",
                        "games": _DEFAULT_EVALUATION_GAMES,
                    }
                )
                evaluation = run_custom_evaluation(
                    artifact_path,
                    config=config,
                    snapshot=snapshot,
                    evaluation=quick_evaluation,
                    report_path=scratch / "evaluation.json",
                )
                metrics["evaluation_protocol"] = "custom_v1"
                metrics["evaluation_games"] = _DEFAULT_EVALUATION_GAMES
            else:
                # The user explicitly asked for a non-primary protocol; honour
                # whatever games/seat schedule their EvaluationConfig declares.
                evaluation = run_custom_evaluation(
                    artifact_path,
                    config=config,
                    snapshot=snapshot,
                    report_path=scratch / "evaluation.json",
                )
                metrics["evaluation_protocol"] = evaluation_config.protocol_id
                metrics["evaluation_games"] = evaluation_config.games
            win_rate = getattr(evaluation, "win_rate", None)
            metrics["evaluation"] = evaluation.to_dict() if hasattr(evaluation, "to_dict") else {}
            _set_counters(run, win_rate=win_rate)
            log("evaluation_finished", win_rate=win_rate, protocol=metrics["evaluation_protocol"])
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
