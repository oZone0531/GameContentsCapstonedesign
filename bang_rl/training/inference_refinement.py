"""Inference refinement ("추론 모듈 개선"): improve the modules under a fixed PPO.

The main AI exists to infer the dummies' roles and emotions while it plays among
four dummies. Dummy-vs-dummy training data does not show the main AI's own
situation, and PPO training optimises the win rate. This workflow keeps the PPO
**fixed** and refines only the intent and emotion modules with games the main AI
plays among four dummies:

* Round 0: evaluate the base model's own modules (reference).
* Each round: (a) collect main1+dummy4 games with the CURRENT best modules and
  the fixed PPO; (b) train the modules, resuming from the current best, on the
  base dummy datasets plus every main1+dummy4 dataset collected so far;
  (c) evaluate the candidate pair under the fixed PPO on the fixed main-AI
  evaluation deals; (d) accept it only if :func:`accept_candidate` says so,
  otherwise keep the previous best.
* Stop after ``max rounds``, a user stop / time budget, or ``patience`` rounds
  in a row without an accepted candidate.

Output: a new ``.bangmodel`` = the base PPO weights (unchanged, checked by a
fingerprint) + the best modules, with an ``ArtifactInfo``. It loads like any
other main-AI model, so the PPO can be trained further from it (main-AI training
with the refined model as the continue path); the refined modules stay frozen
there.

Evaluating with swapped modules changes the PPO's inputs, so its play may shift.
That is the real deployment effect and is intended.
"""
from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import torch

from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.artifact_builders import (
    cache_info,
    info_ref,
    main_ai_info,
    module_infos,
    run_created_at,
)
from bang_rl.artifact_info import (
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    write_info,
)
from bang_rl.config import SystemConfig
from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA, formula_label_emotion_config
from bang_rl.data.splitter import EpisodeSplitter
from bang_rl.data.tensor_cache import NPZTensorCacheLoader, NPZTensorCacheWriter
from bang_rl.exceptions import TrainingError
from bang_rl.paths import PROJECT_ROOT, SAVED_DATASETS_DIR
from bang_rl.runtime.seed import derive_seed

from .dataset_artifacts import DEFAULT_CANONICAL_ROOT, list_datasets
from .emotion_sl_trainer import EmotionInferenceSLTrainer, EmotionSLConfig
from .final_inference_artifact import load_final_inference_artifact, save_final_inference_artifact
from .inference_evaluation import evaluate_inference, inference_scalars
from .main_ai_collection import MAIN_ACTION_MODES, collect_main_ai_games, main_ai_dataset_info
from .main_ai_training import (
    MAIN_AI_EVAL_BASE_SEED,
    MAIN_AI_EVAL_GAMES,
    PolicyModules,
    _SEED_MODULUS,
    _verify_frozen,
    normalized_scores,
    validate_emotion_contract,
)
from .sl_trainer import IntentInferenceSLTrainer, IntentSLConfig

REFINE_DEFAULT_GAMES_PER_ROUND = 5000
REFINE_DEFAULT_MAX_ROUNDS = 5
REFINE_DEFAULT_EPOCHS = 5
REFINE_DEFAULT_PATIENCE = 1

# Acceptance rule (one place). All numbers are on the fixed evaluation deals
# with the PPO fixed. A candidate is accepted when it is not meaningfully worse
# on either primary metric AND meaningfully better on at least one:
#   hidden_role_accuracy >= best - 0.005
#   emotion_mse          <= best_mse * 1.01
#   (hidden_role_accuracy > best + 0.005  OR  emotion_mse < best_mse * 0.99)
ACCEPT_ACCURACY_TOLERANCE = 0.005
ACCEPT_MSE_TOLERANCE = 0.01
ACCEPTANCE_RULE_TEXT = (
    "Pareto with tolerance: accept if hidden_role_accuracy >= best - 0.005 and "
    "emotion_mse <= best * 1.01, and (hidden_role_accuracy > best + 0.005 or "
    "emotion_mse < best * 0.99)"
)

REASON_ACCURACY = "improved_hidden_role_accuracy"
REASON_MSE = "improved_emotion_mse"
REASON_BOTH = "improved_both"
REASON_ACCURACY_WORSE = "hidden_role_accuracy_worse"
REASON_MSE_WORSE = "emotion_mse_worse"
REASON_NO_GAIN = "no_meaningful_improvement"
REASON_MISSING = "metric_missing"
REASON_BASE = "base_reference"

STOP_MAX_ROUNDS = "max_rounds"
STOP_PATIENCE = "patience"
STOP_USER = "user_stop_or_time_budget"


def accept_candidate(best: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, str]:
    """Decide whether ``candidate`` replaces ``best`` (see the module constants).

    Both mappings need ``hidden_role_accuracy`` and ``emotion_mse``. Returns
    ``(accepted, reason)``; a missing metric (no samples) rejects.
    """

    best_acc = best.get("hidden_role_accuracy")
    best_mse = best.get("emotion_mse")
    cand_acc = candidate.get("hidden_role_accuracy")
    cand_mse = candidate.get("emotion_mse")
    if None in (best_acc, best_mse, cand_acc, cand_mse):
        return False, REASON_MISSING
    if cand_acc < best_acc - ACCEPT_ACCURACY_TOLERANCE:
        return False, REASON_ACCURACY_WORSE
    if cand_mse > best_mse * (1.0 + ACCEPT_MSE_TOLERANCE):
        return False, REASON_MSE_WORSE
    accuracy_better = cand_acc > best_acc + ACCEPT_ACCURACY_TOLERANCE
    mse_better = cand_mse < best_mse * (1.0 - ACCEPT_MSE_TOLERANCE)
    if accuracy_better and mse_better:
        return True, REASON_BOTH
    if accuracy_better:
        return True, REASON_ACCURACY
    if mse_better:
        return True, REASON_MSE
    return False, REASON_NO_GAIN


def module_fingerprint(module: torch.nn.Module) -> str:
    """SHA-256 of a module's state (names, dtypes, shapes, bytes)."""

    state = module.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class _Best:
    """The accepted modules so far (frozen) and where they came from."""

    intent: Any
    emotion: Any
    metrics: dict[str, Any]
    round: int = 0
    evaluation: dict[str, Any] = field(default_factory=dict)
    intent_path: Path | None = None
    emotion_path: Path | None = None
    intent_epochs: int = 0
    emotion_epochs: int = 0


def _freeze_pair(*modules: torch.nn.Module) -> None:
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def _trainable_copy(module: torch.nn.Module) -> torch.nn.Module:
    clone = copy.deepcopy(module)
    clone.train()
    for parameter in clone.parameters():
        parameter.requires_grad_(True)
    return clone


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _record_series(run: Any, evaluation_metrics: Mapping[str, Any], accepted: bool) -> None:
    with run.lock:
        for key in ("role_accuracy", "hidden_role_accuracy", "emotion_mse", "win_rate"):
            run.metrics.setdefault(f"refine_{key}_points", []).append(_number(evaluation_metrics.get(key)))
        run.metrics.setdefault("refine_accepted_points", []).append(int(bool(accepted)))


def _project_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scalars(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _number(value) for key, value in inference_scalars(evaluation).items()}


def run_inference_refinement(
    handle: Any,
    run: Any,
    *,
    eval_games: int | None = None,
    eval_base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    dataset_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    model_root: str | Path | None = None,
    environment_factory: Callable[[str, int, Any], Any] | None = None,
) -> Mapping[str, Any]:
    """Run the refinement loop for one training job (see the module docstring).

    Reads from ``run.config``: ``base_model_path`` (the ``.bangmodel``),
    ``episodes`` (games per round), ``refine_rounds``, ``refine_epochs``,
    ``refine_patience``, ``refine_eval_games``, ``main_action_mode``,
    ``dataset_paths`` (base datasets; default every dummy5 dataset) and the
    optional time budget (``train_by_time`` + ``max_training_seconds``, total).
    """

    from .pipeline_runner import (
        TRAINING_MODE_INFERENCE_REFINEMENT,
        _base_model_file,
        _publish_module_stage,
        _read_canonical_dataset,
        _TrainingStopController,
        _training_seed,
    )

    started = time.monotonic()
    options = run.config
    system_config: SystemConfig = run.system_config
    games_per_round = int(options.episodes)
    max_rounds = int(getattr(options, "refine_rounds", REFINE_DEFAULT_MAX_ROUNDS))
    epochs = int(getattr(options, "refine_epochs", REFINE_DEFAULT_EPOCHS))
    patience = int(getattr(options, "refine_patience", REFINE_DEFAULT_PATIENCE))
    eval_games = int(eval_games if eval_games is not None else getattr(options, "refine_eval_games", MAIN_AI_EVAL_GAMES))
    action_mode = str(getattr(options, "main_action_mode", "deterministic") or "deterministic")
    for label, value in (
        ("게임 수/라운드 games per round", games_per_round),
        ("라운드 수 rounds", max_rounds),
        ("라운드당 에폭 epochs per round", epochs),
        ("개선 없음 허용 라운드 patience", patience),
        ("평가 게임 수 eval games", eval_games),
    ):
        if value < 1:
            raise TrainingError(f"{label} must be at least 1 / 1 이상이어야 합니다.")
    if action_mode not in MAIN_ACTION_MODES:
        raise TrainingError(f"main_action_mode must be one of {MAIN_ACTION_MODES}")

    budget = getattr(options, "max_training_seconds", None) if bool(getattr(options, "train_by_time", False)) else None
    try:
        budget = float(budget) if budget is not None and float(budget) > 0 else None
    except (TypeError, ValueError):
        budget = None
    controller = _TrainingStopController(run, budget_seconds=budget)
    run_seed = _training_seed(run)
    job_id = str(handle.job_id)

    def announce(stage: str, message: str, **fields: Any) -> None:
        _publish_module_stage(run, stage, message, **fields)
        with run.lock:
            run.metrics["refine_stage"] = stage

    base_path = _base_model_file(run)
    artifact = load_final_inference_artifact(base_path)  # modules and PPO frozen
    validate_emotion_contract(artifact.emotion)
    ppo = artifact.ppo
    _verify_frozen("ppo", ppo)
    ppo_fingerprint = module_fingerprint(ppo)
    base_ref = info_ref(base_path, KIND_MAIN_AI)
    role_fusion, preprocessor = artifact.role_fusion, artifact.preprocessor

    canonical_root = _project_dir(dataset_root if dataset_root is not None else system_config.dataset.canonical_path)
    cache_base = _project_dir(cache_root if cache_root is not None else system_config.dataset.cache_path)
    models_base = _project_dir(model_root) if model_root is not None else PROJECT_ROOT / "bang_rl" / "models"
    default_root = canonical_root == DEFAULT_CANONICAL_ROOT
    chosen = [_project_dir(item) for item in (getattr(options, "dataset_paths", None) or ())]
    if chosen:
        base_folders = chosen
    else:
        base_folders = [
            item.path
            for item in list_datasets(canonical_root, saved_root=SAVED_DATASETS_DIR if default_root else None)
            if not item.is_main_ai
        ]
    if not base_folders:
        raise TrainingError(
            "개선에 쓸 더미 데이터셋이 없습니다. 먼저 데이터 수집을 하거나 데이터셋을 선택하세요. "
            "No dummy datasets to refine with; collect data first or select datasets."
        )
    extra_roots = [SAVED_DATASETS_DIR] if SAVED_DATASETS_DIR.is_dir() else []

    with run.lock:
        run.metrics.update(
            {
                "refine_total_rounds": max_rounds,
                "refine_round": 0,
                "refine_best_round": 0,
                "refine_games_per_round": games_per_round,
                "refine_epochs_per_round": epochs,
                "refine_patience": patience,
                "refine_eval_games": eval_games,
                "refine_action_mode": action_mode,
                "refine_acceptance_rule": ACCEPTANCE_RULE_TEXT,
                "refine_base_model": str(base_path),
                "refine_ppo_fingerprint": ppo_fingerprint,
                "refine_history": [],
            }
        )

    history: list[dict[str, Any]] = []
    main_folders: list[Path] = []
    saved_pairs: list[dict[str, Any]] = []
    totals = {"games": 0, "environment_steps": 0, "normal": 0, "truncated": 0, "rows": 0}
    stop_reason = STOP_MAX_ROUNDS

    def publish_history() -> None:
        with run.lock:
            run.metrics["refine_history"] = [dict(item) for item in history]

    # --- round 0: the base modules are the reference ---------------------------
    announce("refine_base_eval", f"기준(0라운드): 기존 모듈 평가 중 ({eval_games}판)", refine_round=0
    )
    base_policy = PolicyModules(artifact.intent, artifact.emotion, role_fusion, preprocessor, ppo)
    started_round = time.monotonic()
    base_eval = evaluate_inference(
        base_policy, system_config, games=eval_games, base_seed=eval_base_seed,
        environment_factory=environment_factory, should_stop=controller.should_stop,
    )
    if base_eval is None:
        return _stopped_before_start(run, stop_reason=STOP_USER)
    best = _Best(
        intent=artifact.intent,
        emotion=artifact.emotion,
        metrics=_scalars(base_eval),
        round=0,
        evaluation=base_eval,
    )
    history.append(
        {
            "round": 0,
            "accepted": True,
            "reason": REASON_BASE,
            "metrics": dict(best.metrics),
            "best_round_after": 0,
            "seconds": round(time.monotonic() - started_round, 1),
        }
    )
    _record_series(run, best.metrics, True)
    publish_history()
    run.logger.event("inference_refinement_base", job_id=job_id, **{k: v for k, v in best.metrics.items()})

    artifact_path = Path(run.artifact_path) if getattr(run, "artifact_path", None) else None
    saved_round = {"value": None}

    def save_model(final: bool) -> str | None:
        """Write the base PPO + best modules (atomic); unchanged PPO is asserted first."""

        if artifact_path is None:
            return None
        if module_fingerprint(ppo) != ppo_fingerprint:
            raise TrainingError("PPO weights changed during inference refinement / 개선 중 PPO 가중치가 바뀌었습니다.")
        _verify_frozen("ppo", ppo)
        if not final and saved_round["value"] == best.round:
            return str(artifact_path)
        sources = [base_ref]
        for folder in (*base_folders, *main_folders):
            sources.append(info_ref(folder, KIND_DATASET))
        for path, kind in ((best.intent_path, KIND_INTENT), (best.emotion_path, KIND_EMOTION)):
            if path is not None and path.exists():
                sources.append(info_ref(path, kind))
        base_main = dict(artifact.metadata.get("main_ai_training") or {})
        baselines = base_main.get("baselines") or {}
        try:
            scores = normalized_scores(best.evaluation, baselines) if baselines else base_main.get("normalized_scores", {})
        except Exception:  # noqa: BLE001 - baselines of an older model may not match
            scores = base_main.get("normalized_scores", {})
        info = main_ai_info(
            run,
            handle,
            name=artifact_path.stem,
            training_games=int(base_main.get("training_games", 0) or 0),
            sources=sources,
            summary={
                "refined": True,
                "best_round": best.round,
                "rounds_run": len(history) - 1,
                "role_accuracy": best.metrics.get("role_accuracy"),
                "hidden_role_accuracy": best.metrics.get("hidden_role_accuracy"),
                "emotion_mse": best.metrics.get("emotion_mse"),
                "win_rate": best.metrics.get("win_rate"),
                "base_hidden_role_accuracy": history[0]["metrics"].get("hidden_role_accuracy"),
                "base_emotion_mse": history[0]["metrics"].get("emotion_mse"),
                "rounds": [
                    {
                        "round": item["round"],
                        "accepted": item["accepted"],
                        "reason": item["reason"],
                        "hidden_role_accuracy": item["metrics"].get("hidden_role_accuracy"),
                        "emotion_mse": item["metrics"].get("emotion_mse"),
                        "win_rate": item["metrics"].get("win_rate"),
                    }
                    for item in history
                ],
            },
            extra_conditions={
                "base_model": str(base_path),
                "ppo_fingerprint": ppo_fingerprint,
                "ppo_updated": False,
                "games_per_round": games_per_round,
                "max_rounds": max_rounds,
                "epochs_per_round": epochs,
                "patience": patience,
                "eval_games": eval_games,
                "evaluation_base_seed": int(eval_base_seed),
                "main_action_mode": action_mode,
                "acceptance_rule": ACCEPTANCE_RULE_TEXT,
                "stop_reason": stop_reason if final else "in_progress",
            },
            created_by="inference_refinement",
        )
        base_training = artifact.metadata.get("training") if isinstance(artifact.metadata.get("training"), Mapping) else {}
        counters = dict(base_training.get("counters") or {})
        save_final_inference_artifact(
            artifact_path,
            ppo,
            info=info,
            intent=best.intent,
            emotion=best.emotion,
            role_fusion=role_fusion,
            preprocessor=preprocessor,
            job_snapshot=handle.snapshot,
            training_metadata={
                "counters": counters,
                "total_training_time_seconds": time.monotonic() - started,
                "opponent": "dummy",
                "seed": run_seed,
                "evaluation": {
                    "protocol": "inference_refinement_eval",
                    "games": best.evaluation["games"],
                    "base_seed": best.evaluation["base_seed"],
                    "win_rate": best.evaluation["win_rate"],
                    "result": dict(best.evaluation),
                },
            },
            metadata={
                "main_ai_training": {
                    **base_main,
                    "source": {
                        "origin": "inference_refinement",
                        "base_model": str(base_path),
                        "intent_path": None if best.intent_path is None else str(best.intent_path),
                        "emotion_path": None if best.emotion_path is None else str(best.emotion_path),
                    },
                    "evaluation": dict(best.evaluation),
                    "normalized_scores": dict(scores) if isinstance(scores, Mapping) else {},
                },
                "inference_refinement": {
                    "base_model": str(base_path),
                    "ppo_fingerprint": ppo_fingerprint,
                    "ppo_updated": False,
                    "best_round": best.round,
                    "acceptance_rule": ACCEPTANCE_RULE_TEXT,
                    "history": [dict(item) for item in history],
                    "main_datasets": [str(item) for item in main_folders],
                    "evaluation_base_seed": int(eval_base_seed),
                    "eval_games": eval_games,
                },
            },
        )
        saved_round["value"] = best.round
        return str(artifact_path)

    # --- refinement rounds ------------------------------------------------------
    bad_rounds = 0
    for round_no in range(1, max_rounds + 1):
        if controller.should_stop():
            stop_reason = STOP_USER
            break
        round_started = time.monotonic()
        with run.lock:
            run.metrics["refine_round"] = round_no
        # (a) collect with the current best modules and the fixed PPO.
        announce("refine_collecting",
            f"{round_no}라운드: 메인 1 + 더미 4 게임 수집 중 ({games_per_round}판)",
            refine_round=round_no,
        )
        collect_seed = derive_seed(run_seed, "inference-refinement", "collect", round_no) % _SEED_MODULUS
        dataset_meta = main_ai_dataset_info(
            games=games_per_round,
            base_seed=collect_seed,
            created_by=TRAINING_MODE_INFERENCE_REFINEMENT,
            created_at=run_created_at(run),
            action_mode=action_mode,
            model_path=base_path,
            extra_conditions={"refinement_job": job_id, "round": round_no, "modules_from_round": best.round},
        )
        for path, kind in ((best.intent_path, KIND_INTENT), (best.emotion_path, KIND_EMOTION)):
            if path is not None and path.exists():
                dataset_meta.sources.append(info_ref(path, kind))
        folder = canonical_root / dataset_meta.name
        games_before = totals["games"]

        def collect_progress(done: int, total: int, *, before: int = games_before) -> None:
            with run.lock:
                run.episodes_completed = before + int(done)
                run.metrics["collection_completed"] = int(done)
                run.metrics["collection_total"] = int(total)

        collected = collect_main_ai_games(
            system_config,
            PolicyModules(best.intent, best.emotion, role_fusion, preprocessor, ppo),
            games_per_round,
            folder,
            base_seed=collect_seed,
            action_mode=action_mode,
            progress=collect_progress,
            should_stop=controller.should_stop,
            info=dataset_meta,
            source_model_path=base_path,
            episode_id_prefix=f"refine-{job_id}-r{round_no}",
            environment_factory=environment_factory,
        )
        main_folders.append(folder)
        status = collected.get("termination_status", {})
        totals["games"] += int(collected["games"])
        totals["environment_steps"] += int(collected.get("total_steps", 0))
        totals["normal"] += int(status.get("normal", 0))
        totals["truncated"] += int(status.get("truncated", 0))
        totals["rows"] += int(collected["rows"])
        with run.lock:
            run.episodes_completed = totals["games"]
            run.environment_steps = totals["environment_steps"]
            run.normal_episodes = totals["normal"]
            run.truncated_episodes = totals["truncated"]
            run.metrics["refine_datasets"] = [str(item) for item in main_folders]
            run.metrics["refine_main_rows"] = totals["rows"]
        collect_seconds = time.monotonic() - round_started
        if collected["stopped"] or controller.should_stop():
            stop_reason = STOP_USER
            break

        # (b) train the modules on dummy data + every main1+dummy4 dataset so far.
        train_started = time.monotonic()
        announce("refine_training",
            f"{round_no}라운드: 모듈 학습 중 (에폭 {epochs}, 데이터셋 {len(base_folders)}+{len(main_folders)}개)",
            refine_round=round_no,
        )
        episodes, shards, duplicates, selection = _read_canonical_dataset(
            canonical_root, extra_roots, only_folders=[*base_folders, *main_folders]
        )
        round_handle = SimpleNamespace(job_id=str(uuid4()), snapshot=handle.snapshot)
        source_refs = [info_ref(item, KIND_DATASET) for item in (*base_folders, *main_folders)]
        cache_meta = cache_info(run, round_handle, sources=source_refs, summary={})
        cache_dir = cache_base / cache_meta.name
        split_seed = int(options.split_seed if getattr(options, "split_seed", None) is not None else run_seed)
        split = EpisodeSplitter(seed=split_seed, manifest_path=cache_dir / "split_manifest.json").split(
            episodes, source_metadata={"source_shard_ids": [str(path) for path in shards]}
        )
        cache_artifacts = NPZTensorCacheWriter(
            cache_dir,
            emotion_schema_version=FORMULA_LABEL_SCHEMA,
            emotion_dimension=len(LABEL_NAMES),
            split_manifest=split.manifest,
            config=system_config,
        ).write_all(split)
        cache_meta.summary = {
            "episodes": len(episodes),
            "episodes_by_players": selection.get("selected_episodes_by_players"),
            "splits": {name: len(item.manifest.get("episode_ids", ())) for name, item in cache_artifacts.items()},
        }
        write_info(cache_dir, cache_meta)
        loaders = {
            name: NPZTensorCacheLoader(item.tensor_path, manifest_path=item.manifest_path)
            for name, item in cache_artifacts.items()
        }
        batch = int(system_config.sl_training.batch_size)
        round_seed = int(derive_seed(run_seed, "inference-refinement", "train", round_no) % _SEED_MODULUS)
        intent_trainer = IntentInferenceSLTrainer(
            loaders["train"], loaders["validation"], test_cache=loaders["test"],
            config=IntentSLConfig(epochs=epochs, batch_size=batch, seed=round_seed, role_count=4),
            model=_trainable_copy(best.intent),
            job_snapshot=handle.snapshot, config_snapshot=system_config,
        )
        emotion_trainer = EmotionInferenceSLTrainer(
            loaders["train"], loaders["validation"], test_cache=loaders["test"],
            config=EmotionSLConfig(epochs=epochs, batch_size=batch, seed=round_seed),
            model=_trainable_copy(best.emotion),
        )
        for trainer, prior in ((intent_trainer, best.intent_epochs), (emotion_trainer, best.emotion_epochs)):
            trainer.should_stop = controller.should_stop
            trainer.prior_epochs = prior
            resumed = best.intent_path if trainer is intent_trainer else best.emotion_path
            if resumed is not None:
                trainer.resumed_from = str(resumed)
            # The current best modules are the starting "best so far": a round
            # only replaces them with weights that lower validation loss.
            trainer.adopt_current_weights_as_baseline()
        intent_result = intent_trainer.fit()
        emotion_result = emotion_trainer.fit()
        train_seconds = time.monotonic() - train_started
        if controller.should_stop():
            stop_reason = STOP_USER
            break
        candidate_intent, candidate_emotion = intent_trainer.model, emotion_trainer.model
        _freeze_pair(candidate_intent, candidate_emotion)

        # (c) evaluate the candidate pair under the fixed PPO on the fixed deals.
        eval_started = time.monotonic()
        announce("refine_evaluating", f"{round_no}라운드: 새 모듈 평가 중 ({eval_games}판)", refine_round=round_no
        )
        candidate_eval = evaluate_inference(
            PolicyModules(candidate_intent, candidate_emotion, role_fusion, preprocessor, ppo),
            system_config, games=eval_games, base_seed=eval_base_seed,
            environment_factory=environment_factory, should_stop=controller.should_stop,
        )
        if candidate_eval is None:
            stop_reason = STOP_USER
            break
        candidate_metrics = _scalars(candidate_eval)

        # (d) accept or keep the previous best.
        accepted, reason = accept_candidate(best.metrics, candidate_metrics)
        record: dict[str, Any] = {
            "round": round_no,
            "accepted": accepted,
            "reason": reason,
            "metrics": candidate_metrics,
            "previous_best_round": best.round,
            "best_round_after": round_no if accepted else best.round,
            "dataset": str(folder),
            "games": int(collected["games"]),
            "rows": int(collected["rows"]),
            "train_epochs": {"intent": len(intent_result.history), "emotion": len(emotion_result.history)},
            "seconds": round(time.monotonic() - round_started, 1),
            "seconds_collect": round(collect_seconds, 1),
            "seconds_train": round(train_seconds, 1),
            "seconds_eval": round(time.monotonic() - eval_started, 1),
        }
        history.append(record)
        _record_series(run, candidate_metrics, accepted)
        if accepted:
            intent_meta = emotion_meta = None
            try:
                intent_meta, emotion_meta = module_infos(
                    run,
                    round_handle,
                    sources=[*source_refs, base_ref],
                    intent_total_epochs=best.intent_epochs + len(intent_result.history),
                    emotion_total_epochs=best.emotion_epochs + len(emotion_result.history),
                    intent_summary={
                        "epochs_trained": len(intent_result.history),
                        "best_epoch": intent_result.best_epoch,
                        "test_accuracy": (intent_result.test_metrics or {}).get("accuracy"),
                        **{f"main_setting_{k}": v for k, v in candidate_metrics.items()},
                    },
                    emotion_summary={
                        "epochs_trained": len(emotion_result.history),
                        "best_epoch": emotion_result.best_epoch,
                        "test_mse": (emotion_result.test_metrics or {}).get("mse"),
                        **{f"main_setting_{k}": v for k, v in candidate_metrics.items()},
                    },
                    extra_conditions={
                        "emotion_schema": FORMULA_LABEL_SCHEMA,
                        "batch_size": batch,
                        "dataset_episodes": len(episodes),
                        "dataset_episodes_by_players": selection.get("selected_episodes_by_players"),
                        "refinement_job": job_id,
                        "refinement_round": round_no,
                    },
                    resumed_from={
                        **({KIND_INTENT: info_ref(best.intent_path, KIND_INTENT)} if best.intent_path else {}),
                        **({KIND_EMOTION: info_ref(best.emotion_path, KIND_EMOTION)} if best.emotion_path else {}),
                    },
                )
                intent_meta.created_by = emotion_meta.created_by = TRAINING_MODE_INFERENCE_REFINEMENT
            except Exception as exc:  # noqa: BLE001 - never lose trained modules over their description
                run.logger.event("module_info_failed", error=str(exc))
            stem = round_handle.job_id
            intent_path = models_base / "intent" / (f"{intent_meta.name}.phase2model" if intent_meta else f"intent-{stem}.phase2model")
            emotion_path = models_base / "emotion" / (f"{emotion_meta.name}.phase2model" if emotion_meta else f"emotion-{stem}.phase2model")
            intent_trainer.save_artifact(intent_path, test_metrics=intent_result.test_metrics, info=intent_meta)
            emotion_trainer.save_artifact(emotion_path, test_metrics=emotion_result.test_metrics, info=emotion_meta)
            best = _Best(
                intent=candidate_intent,
                emotion=candidate_emotion,
                metrics=candidate_metrics,
                round=round_no,
                evaluation=candidate_eval,
                intent_path=intent_path,
                emotion_path=emotion_path,
                intent_epochs=best.intent_epochs + len(intent_result.history),
                emotion_epochs=best.emotion_epochs + len(emotion_result.history),
            )
            saved_pairs.append({"round": round_no, "intent": str(intent_path), "emotion": str(emotion_path)})
            bad_rounds = 0
            with run.lock:
                run.metrics["intent_artifact_path"] = str(intent_path)
                run.metrics["emotion_artifact_path"] = str(emotion_path)
        else:
            bad_rounds += 1
        with run.lock:
            run.metrics["refine_best_round"] = best.round
            run.win_rate = best.metrics.get("win_rate")
        publish_history()
        run.logger.event(
            "inference_refinement_round",
            job_id=job_id,
            round=round_no,
            accepted=accepted,
            reason=reason,
            **candidate_metrics,
        )
        if accepted:
            save_model(final=False)  # an interrupted run keeps the best model so far
        if not accepted and bad_rounds >= patience:
            stop_reason = STOP_PATIENCE
            break
    else:
        stop_reason = STOP_MAX_ROUNDS

    # --- finish ----------------------------------------------------------------
    model_path = save_model(final=True)
    with run.lock:
        run.metrics["refine_stage"] = "done"
    publish_history()
    metrics_out = {
        "workflow": TRAINING_MODE_INFERENCE_REFINEMENT,
        "base_model": str(base_path),
        "stop_reason": stop_reason,
        "stopped_early": stop_reason == STOP_USER,
        "rounds_run": len(history) - 1,
        "best_round": best.round,
        "base_metrics": dict(history[0]["metrics"]),
        "best_metrics": dict(best.metrics),
        "history": [dict(item) for item in history],
        "acceptance_rule": ACCEPTANCE_RULE_TEXT,
        "ppo_fingerprint": ppo_fingerprint,
        "ppo_updated": False,
        "base_datasets": [str(item) for item in base_folders],
        "main_datasets": [str(item) for item in main_folders],
        "saved_module_pairs": saved_pairs,
        "intent_artifact_path": None if best.intent_path is None else str(best.intent_path),
        "emotion_artifact_path": None if best.emotion_path is None else str(best.emotion_path),
        "main_rows": totals["rows"],
        "emotion_dimension": len(LABEL_NAMES),
        "emotion_schema": FORMULA_LABEL_SCHEMA,
        "role_class_count": 4,
        "raw_role_count": 5,
        "seconds": round(time.monotonic() - started, 1),
    }
    return {
        "output_path": str(main_folders[-1]) if main_folders else str(canonical_root),
        "artifact_path": model_path,
        "environment_steps": totals["environment_steps"],
        "episodes_completed": totals["games"],
        "normal_episodes": totals["normal"],
        "truncated_episodes": totals["truncated"],
        "win_rate": best.metrics.get("win_rate"),
        "metrics": metrics_out,
        "logs": (
            {
                "message": (
                    f"inference refinement finished: {len(history) - 1} rounds, best round {best.round}, "
                    f"stop reason {stop_reason}"
                )
            },
        ),
    }


def _stopped_before_start(run: Any, *, stop_reason: str) -> Mapping[str, Any]:
    """Stopped during the round-0 evaluation: nothing to save."""

    return {
        "output_path": None,
        "artifact_path": None,
        "environment_steps": 0,
        "episodes_completed": 0,
        "metrics": {"workflow": "inference_refinement", "stop_reason": stop_reason, "stopped_early": True, "rounds_run": 0},
        "logs": ({"message": "inference refinement stopped before the first round"},),
    }


__all__ = [
    "ACCEPTANCE_RULE_TEXT",
    "ACCEPT_ACCURACY_TOLERANCE",
    "ACCEPT_MSE_TOLERANCE",
    "REFINE_DEFAULT_EPOCHS",
    "REFINE_DEFAULT_GAMES_PER_ROUND",
    "REFINE_DEFAULT_MAX_ROUNDS",
    "REFINE_DEFAULT_PATIENCE",
    "accept_candidate",
    "module_fingerprint",
    "run_inference_refinement",
]
