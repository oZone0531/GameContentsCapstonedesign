"""Main-AI (Phase 3) PPO training against DummyAgents.

The main agent is frozen intent + emotion inference, RoleFusion, the
observation preprocessor and a trainable :class:`PPOModule`. Games are played
in this process, or with ``hardware.workers`` > 1 in that many worker processes
(:mod:`bang_rl.training.parallel_games`, a batch of ``workers`` games at a
time). The main seat rotates (``game_index % 5``) and the four
other seats are DummyAgents built exactly like the evaluation protocol builds
them (:func:`bang_rl.evaluation.protocol.build_dummy_opponents`).

Reward
------
Training uses the terminal team result only (:func:`main_ai_reward_config`):
+1 when the main agent's team wins, -1 when it loses, no shaping. Being
eliminated does not change this; the team result decides.

Truncation
----------
A game cut by the step limit is a time limit, not a loss. Its final transition
gets reward 0 and is bootstrapped from the critic's value of the state the
agent was left in; earlier transitions of that game are ordinary. A warning is
logged when more than 1% of games are truncated.

Evaluation
----------
Every ``eval_every_updates`` PPO updates (and at the end) the deterministic
policy plays a fixed deal set (``MAIN_AI_EVAL_GAMES`` games, seeds independent
of training). Reference "dummy" and "random" agents play the same deals once at
the start (cached on disk); per-role scores are normalized so 0 is random
level and 1 is dummy level. Only the best model (highest overall eval win rate,
ties broken by mean normalized score) is saved, atomically, to the run's
``.bangmodel`` path. The last model is not saved separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bang_rl.agents.main_agent import (
    ObservationPreprocessor,
    PPOModule,
    RoleFusion,
    build_policy_inputs,
    policy_emotion_dim,
)
from bang_rl.agents.main_agent.policy_inputs import PolicyInputs
from bang_rl.config import PPOConfig, RewardConfig, SystemConfig
from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.env import BangEnvironment
from bang_rl.env.types import TerminationStatus
from bang_rl.evaluation.baseline import baseline_factory, run_seat_evaluation
from bang_rl.evaluation.protocol import (
    PRIMARY_SEATS,
    SEMANTIC_ROLE_NAMES,
    SEMANTIC_ROLE_OF,
    InferenceMetrics,
    _wilson_interval,
    build_dummy_opponents,
    observe_boundaries,
)
from bang_rl.artifact_builders import baseline_info, info_ref, main_ai_info
from bang_rl.artifact_info import (
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    dump_json_with_info,
    read_info,
)
from bang_rl.exceptions import TrainingError
from bang_rl.paths import REPORTS_DIR
from bang_rl.runtime.reward import RewardCalculator
from bang_rl.runtime.seed import derive_seed

from . import module_artifacts
from .final_inference_artifact import (
    load_final_artifact_for_training,
    save_final_inference_artifact,
)
from .parallel_games import BrokenProcessPool, open_game_pool, usable_workers
from .pending_transition import PendingTransitionLedger
from .ppo_trainer import PPOTrainer, PPOTransition

MAIN_AI_EVAL_GAMES = 250
MAIN_AI_EVAL_EVERY_UPDATES = 10
# Evaluation deals use these seeds (``base + index``); training game seeds are
# derived by hash and never fall in the reserved range below.
MAIN_AI_EVAL_BASE_SEED = 1_000_000_000
_EVAL_SEED_RESERVED = 1_000_000
MAIN_AI_MOVING_AVERAGE_GAMES = 200
MAIN_AI_TRUNCATION_WARNING_RATE = 0.01
NORMALIZED_SCORE_EPS = 0.01
BASELINE_NAMES = ("dummy", "random")
BASELINE_CACHE_VERSION = 1
MAIN_AI_REWARD_VERSION = "main-ai-terminal-team-1.0"
_ENVIRONMENT_VERSION = "1.0"
_SEED_MODULUS = 2**31

EnvironmentFactory = Callable[[str, int, Any], Any]


# --- reward -----------------------------------------------------------------


def main_ai_reward_config() -> RewardConfig:
    """Reward definition for main-AI training: team win +1, team loss -1.

    Terminal only, no shaping and no clipping. Change the reward here (one
    place); the id of this config is stored in the saved model's metadata.
    """

    return RewardConfig.model_validate(
        {
            "version": MAIN_AI_REWARD_VERSION,
            "terminal": {
                "components": [
                    {
                        "id": "terminal_win",
                        "enabled": True,
                        "condition": {"type": "terminal_result", "result": "win"},
                        "value": 1.0,
                    },
                    {
                        "id": "terminal_loss",
                        "enabled": True,
                        "condition": {"type": "terminal_result", "result": "loss"},
                        "value": -1.0,
                    },
                ]
            },
            "shaping": {"components": []},
            "clipping": {"enabled": False},
        }
    )


# --- policy container --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyModules:
    """Everything needed to turn a public observation into an action."""

    intent: Any
    emotion: Any
    role_fusion: Any
    preprocessor: Any
    ppo: PPOModule

    def inputs(self, observation: Any, seat: int) -> PolicyInputs:
        return build_policy_inputs(
            observation,
            preprocessor=self.preprocessor,
            intent=self.intent,
            emotion=self.emotion,
            role_fusion=self.role_fusion,
            viewer_idx=seat,
        )


def _scalar(value: Any) -> float:
    return float(torch.as_tensor(value).detach().reshape(-1)[0])


def _selected_player(env: Any) -> int:
    selection = env.agent_selection
    index = getattr(selection, "index", None)
    if isinstance(index, (int, np.integer)):
        return int(index)
    return int(str(selection).removeprefix("player_"))


def _default_environment(episode_id: str, seed: int, env_config: Any) -> BangEnvironment:
    return BangEnvironment(episode_id=episode_id, seed=seed, config=env_config)


def _critic_value(ppo: PPOModule, inputs: PolicyInputs) -> float:
    """Critic value of a state; the value does not depend on the action mask."""

    features = {**inputs.features, "action_mask": np.ones(6, dtype=np.uint8)}
    with torch.no_grad():
        return _scalar(ppo(features).value)


# --- seeds and seats ---------------------------------------------------------


def main_seat_for_game(game_index: int) -> int:
    """The main agent rotates through the five seats."""

    return int(game_index) % 5


def training_game_seed(run_seed: int, game_index: int) -> int:
    """Deterministic environment seed of one training game.

    Derived with :func:`derive_seed`; seeds inside the evaluation range are
    skipped so training never plays an evaluation deal.
    """

    attempt = 0
    while True:
        seed = derive_seed(run_seed, "main-ai-training", "game", int(game_index), attempt) % _SEED_MODULUS
        if not MAIN_AI_EVAL_BASE_SEED <= seed < MAIN_AI_EVAL_BASE_SEED + _EVAL_SEED_RESERVED:
            return seed
        attempt += 1


# --- one training game -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingGameSummary:
    """Plain-data outcome of one training game (safe to ship between processes)."""

    game_index: int
    episode_id: str
    seed: int
    seat: int
    role: str
    termination_status: str
    won: bool | None
    environment_steps: int
    main_decisions: int
    main_agent_eliminated: bool
    total_reward: float

    @property
    def truncated(self) -> bool:
        return self.termination_status != TerminationStatus.NORMAL.value


@dataclass(frozen=True, slots=True)
class TrainingGameResult:
    transitions: tuple[PPOTransition, ...]
    summary: TrainingGameSummary


def play_training_game(
    game_index: int,
    *,
    run_seed: int,
    config: SystemConfig,
    policy: PolicyModules,
    reward_calculator: RewardCalculator | None = None,
    environment_factory: EnvironmentFactory | None = None,
    worker_id: int = 0,
) -> TrainingGameResult:
    """Play one game with the main agent sampling from its policy.

    All inputs are plain values and modules, and the result is plain data plus
    transitions, so this function can later run unchanged in a worker process.
    The ledger accumulates each environment step's reward between the main
    agent's actions into one transition per decision.
    """

    calculator = reward_calculator or RewardCalculator(main_ai_reward_config())
    factory = environment_factory or _default_environment
    seat = main_seat_for_game(game_index)
    seed = training_game_seed(run_seed, game_index)
    episode_id = f"train-g{int(game_index):09d}"
    env = factory(episode_id, seed, config.env)
    try:
        env.reset(seed=seed, episode_id=episode_id)
        opponents = build_dummy_opponents(seat, seed, config.preset)
        for agent in opponents.values():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(run_seed, "main-ai-training", "action", int(game_index)) & 0x7FFFFFFFFFFFFFFF)
        roles = tuple(int(role) for role in env.state.roles)

        ledger = PendingTransitionLedger()
        ledger.start_episode(worker_id, episode_id)
        transitions: list[PPOTransition] = []
        pending_open = False
        eliminated = False
        decisions = 0
        last_mask = np.ones(6, dtype=np.uint8)
        while not env.state.terminated and not env.state.truncated:
            selected = _selected_player(env)
            observe_boundaries(opponents, env, selected)
            observation = env.observe(selected)
            mask = np.asarray(env.action_mask, dtype=np.uint8)
            if selected == seat:
                inputs = policy.inputs(observation, seat)
                with torch.no_grad():
                    chosen = policy.ppo.act(
                        inputs.ppo_inputs(), inputs.action_mask, deterministic=False, generator=generator
                    )
                action = int(chosen.action.reshape(-1)[0])
                value = _scalar(chosen.value)
                log_prob = _scalar(chosen.log_prob)
                features = {key: np.asarray(block) for key, block in inputs.features.items()}
                finished = ledger.begin_action(
                    worker_id,
                    episode_id,
                    env.environment_step,
                    seat,
                    features,
                    inputs.action_mask,
                    action,
                    log_prob=log_prob,
                    value=value,
                )
                if finished is not None:
                    # The critic's value of this new decision state is the
                    # bootstrap of the previous transition.
                    transitions.append(replace(PPOTransition.from_committed(finished), next_value=value))
                pending_open = True
                decisions += 1
                last_mask = inputs.action_mask
            else:
                action = opponents[selected].act(observation, mask)
            if mask[action] != 1:
                raise RuntimeError("training agent produced an action rejected by the public action mask")
            result = env.step(action)
            if pending_open:
                reward = calculator.calculate(
                    result,
                    agent_idx=seat,
                    roles=roles,
                    terminal_result=env.state.terminal_result,
                ).total
                ledger.commit_result(
                    worker_id,
                    episode_id,
                    {
                        "environment_step": env.environment_step,
                        "actor_idx": seat,
                        "reward": reward,
                        "terminated": bool(env.state.terminated),
                        "truncated": bool(env.state.truncated),
                        "events": (),
                    },
                )
            if (
                not eliminated
                and pending_open
                and not bool(env.state.alive[seat])
                and not (env.state.terminated or env.state.truncated)
            ):
                ledger.eliminate_agent(worker_id, episode_id)
                eliminated = True

        terminated = bool(env.state.terminated)
        truncated = bool(env.state.truncated)
        eliminated = eliminated or not bool(env.state.alive[seat])
        for finished in ledger.finish_episode(
            worker_id,
            episode_id,
            terminated=terminated,
            truncated=truncated,
            main_agent_eliminated=eliminated,
        ):
            transition = PPOTransition.from_committed(finished)
            if finished.truncated:
                # Time limit, not a loss: bootstrap from the critic. A player
                # who is already out has no decision state left to value.
                bootstrap = 0.0
                if bool(env.state.alive[seat]):
                    final_inputs = policy.inputs(env.observe(seat), seat)
                    bootstrap = _critic_value(policy.ppo, final_inputs)
                transition = replace(
                    transition, trainable=True, time_limit_bootstrap=True, next_value=bootstrap
                )
            transitions.append(transition)

        status = env.state.termination_status.value
        terminal = env.state.terminal_result
        normal = status == TerminationStatus.NORMAL.value
        summary = TrainingGameSummary(
            game_index=int(game_index),
            episode_id=episode_id,
            seed=seed,
            seat=seat,
            role=SEMANTIC_ROLE_NAMES[SEMANTIC_ROLE_OF[roles[seat]]],
            termination_status=status,
            won=bool(terminal is not None and seat in terminal.winners) if normal else None,
            environment_steps=int(env.environment_step),
            main_decisions=decisions,
            main_agent_eliminated=eliminated,
            total_reward=float(sum(item.reward for item in transitions)),
        )
        for agent in opponents.values():
            agent.on_episode_end({"episode_id": episode_id, "terminated": terminated, "truncated": truncated})
        return TrainingGameResult(tuple(transitions), summary)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


# --- evaluation --------------------------------------------------------------


def _role_table() -> dict[str, dict[str, Any]]:
    return {name: {"games": 0, "wins": 0} for name in SEMANTIC_ROLE_NAMES}


def _finish_role_table(table: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            **counts,
            "win_rate": (counts["wins"] / counts["games"]) if counts["games"] else None,
        }
        for name, counts in table.items()
    }


def _play_evaluation_game(
    policy: PolicyModules,
    config: SystemConfig,
    *,
    seed: int,
    seat: int,
    index: int,
    environment_factory: EnvironmentFactory | None,
    metrics: InferenceMetrics,
) -> tuple[bool | None, str, int]:
    episode_id = f"main-ai-eval-g{index:05d}"
    env = (environment_factory or _default_environment)(episode_id, seed, config.env)
    try:
        env.reset(seed=seed, episode_id=episode_id)
        roles = tuple(int(role) for role in env.state.roles)
        opponents = build_dummy_opponents(seat, seed, config.preset)
        for agent in opponents.values():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})
        while not env.state.terminated and not env.state.truncated:
            selected = _selected_player(env)
            observe_boundaries(opponents, env, selected)
            observation = env.observe(selected)
            mask = np.asarray(env.action_mask, dtype=np.uint8)
            if selected == seat:
                inputs = policy.inputs(observation, seat)
                with torch.inference_mode():
                    chosen = policy.ppo.act(inputs.ppo_inputs(), inputs.action_mask, deterministic=True)
                action = int(chosen.action.reshape(-1)[0])
                metrics.observe_step(
                    env=env,
                    observation=observation,
                    seat=seat,
                    fused_intent=inputs.fused_intent,
                    estimated_emotion=inputs.estimated_emotion,
                    opponents=opponents,
                )
            else:
                action = opponents[selected].act(observation, mask)
            if mask[action] != 1:
                raise RuntimeError("evaluation agent produced an action rejected by the public action mask")
            env.step(action)
        for agent in opponents.values():
            agent.on_episode_end(
                {"episode_id": episode_id, "terminated": env.state.terminated, "truncated": env.state.truncated}
            )
        role = SEMANTIC_ROLE_NAMES[SEMANTIC_ROLE_OF[roles[seat]]]
        steps = int(env.environment_step)
        if env.state.termination_status.value != TerminationStatus.NORMAL.value:
            return None, role, steps
        result = env.state.terminal_result
        return bool(result is not None and seat in result.winners), role, steps
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def evaluate_policy(
    policy: PolicyModules,
    config: SystemConfig,
    *,
    games: int = MAIN_AI_EVAL_GAMES,
    base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    seats: Sequence[int] = PRIMARY_SEATS,
    environment_factory: EnvironmentFactory | None = None,
    should_stop: Callable[[], bool] | None = None,
    metrics_factory: Callable[[], InferenceMetrics] | None = None,
) -> dict[str, Any] | None:
    """Deterministic evaluation on the fixed deal set; None if stopped midway.

    Game ``i`` uses seed ``base_seed + i`` and seat ``seats[i % len(seats)]``,
    the deals the baselines play. Besides win rates it records role
    classification, hidden-role accuracy and emotion MSE (privileged labels are
    read for these metrics only) so inference drift is visible.

    ``metrics_factory`` may build an :class:`InferenceMetrics` subclass; when it
    has a ``detail()`` method its result is stored as ``inference_detail``.
    """

    games = int(games)
    if games < 1:
        raise ValueError("evaluation games must be at least 1")
    seats = tuple(int(seat) for seat in seats)
    metrics = (metrics_factory or InferenceMetrics)()
    rows: list[tuple[bool | None, str, int]] = []
    for index in range(games):
        if should_stop is not None and index % 10 == 0 and should_stop():
            return None
        rows.append(
            _play_evaluation_game(
                policy,
                config,
                seed=int(base_seed) + index,
                seat=seats[index % len(seats)],
                index=index,
                environment_factory=environment_factory,
                metrics=metrics,
            )
        )
    return evaluation_summary(rows, metrics, games=games, base_seed=base_seed)


def evaluation_summary(
    rows: Sequence[tuple[bool | None, str, int]],
    metrics: InferenceMetrics,
    *,
    games: int,
    base_seed: int,
) -> dict[str, Any]:
    """The evaluation result from per-game ``(won, role, steps)`` rows (``won`` None = truncated)."""

    per_role = _role_table()
    wins = normal = truncated = 0
    steps: list[int] = []
    for won, role, length in rows:
        steps.append(length)
        if won is None:
            truncated += 1
            continue
        normal += 1
        wins += int(won)
        per_role[role]["games"] += 1
        per_role[role]["wins"] += int(won)
    interval = _wilson_interval(wins, normal)
    detail = getattr(metrics, "detail", None)
    return {
        **({"inference_detail": detail()} if callable(detail) else {}),
        "games": games,
        "base_seed": int(base_seed),
        "normal_games": normal,
        "truncated_games": truncated,
        "wins": wins,
        "win_rate": (wins / normal) if normal else None,
        "win_rate_ci": interval.to_dict() if hasattr(interval, "to_dict") else None,
        "by_role": _finish_role_table(per_role),
        "role_accuracy": metrics.role_accuracy,
        "hidden_role_accuracy": metrics.hidden_role_accuracy,
        "emotion_mse": metrics.emotion_mse,
        "mean_steps": float(np.mean(steps)),
    }


# --- baselines and scores ----------------------------------------------------


def baseline_cache_key(config: SystemConfig, *, games: int, base_seed: int, seats: Sequence[int]) -> str:
    """Key of the cached baselines: the deals, opponent preset and environment."""

    payload = {
        "cache_version": BASELINE_CACHE_VERSION,
        "games": int(games),
        "base_seed": int(base_seed),
        "seats": [int(seat) for seat in seats],
        "preset": config.preset.model_dump(mode="json"),
        "env": config.env.model_dump(mode="json"),
        "environment_version": _ENVIRONMENT_VERSION,
        "baselines": list(BASELINE_NAMES),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compact_baseline(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "agent": summary["agent"],
        "games": summary["games"],
        "normal_games": summary["normal_games"],
        "truncated_games": summary["truncated_games"],
        "win_rate": summary["win_rate"],
        "win_rate_ci": summary["win_rate_ci"],
        "by_role": {
            name: {key: values[key] for key in ("games", "wins", "win_rate")}
            for name, values in summary["by_role"].items()
        },
    }


def load_or_compute_baselines(
    config: SystemConfig,
    *,
    games: int = MAIN_AI_EVAL_GAMES,
    base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    seats: Sequence[int] = PRIMARY_SEATS,
    cache_dir: str | Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], bool] | None:
    """Dummy and random baselines on the evaluation deals, reused when cached.

    Returns ``(baselines, from_cache)``, or None when stopped before both
    finished (nothing is cached then).
    """

    directory = Path(cache_dir) if cache_dir is not None else REPORTS_DIR / "baselines"
    key = baseline_cache_key(config, games=games, base_seed=base_seed, seats=seats)
    # Cache files are matched by the ``key`` stored inside them, never by file
    # name, so a renamed baseline file is still found.
    path = None
    if directory.is_dir():
        for candidate in sorted(directory.glob("*.json")):
            try:
                cached = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # unreadable cache: skip it
            if (
                isinstance(cached, Mapping)
                and cached.get("key") == key
                and set(cached.get("baselines", {})) == set(BASELINE_NAMES)
            ):
                return dict(cached["baselines"]), True
    created = baseline_info(
        id=key,
        games=games,
        label="dummy+random",
        conditions={"base_seed": int(base_seed), "seats": list(seats)},
    )
    path = directory / f"{created.name}.json"
    baselines: dict[str, Any] = {}
    for name in BASELINE_NAMES:
        if should_stop is not None and should_stop():
            return None
        summary = run_seat_evaluation(
            baseline_factory(name, config),
            config,
            games=games,
            base_seed=base_seed,
            seats=seats,
            label=f"main-ai-baseline-{name}",
        )
        baselines[name] = _compact_baseline(summary)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    created.summary = {
        name: {"win_rate": values.get("win_rate")} for name, values in baselines.items() if isinstance(values, Mapping)
    }
    temporary.write_text(
        dump_json_with_info(
            {"key": key, "games": int(games), "base_seed": int(base_seed), "baselines": baselines},
            created,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return baselines, False


def normalized_scores(
    evaluation: Mapping[str, Any], baselines: Mapping[str, Any], *, eps: float = NORMALIZED_SCORE_EPS
) -> dict[str, Any]:
    """Per-role ``(main - random) / max(dummy - random, eps)``.

    0 is random level, 1 is dummy level, above 1 beats the dummy. Roles with no
    games on the deal set are None and left out of the mean.
    """

    by_role: dict[str, float | None] = {}
    for name in SEMANTIC_ROLE_NAMES:
        main = evaluation["by_role"][name]["win_rate"]
        dummy = baselines["dummy"]["by_role"][name]["win_rate"]
        random = baselines["random"]["by_role"][name]["win_rate"]
        if main is None or dummy is None or random is None:
            by_role[name] = None
        else:
            by_role[name] = float((main - random) / max(dummy - random, eps))
    available = [value for value in by_role.values() if value is not None]
    return {"by_role": by_role, "mean": float(np.mean(available)) if available else None}


def _is_better(candidate: tuple[float, float], best: tuple[float, float] | None) -> bool:
    """Overall win rate first, ties by mean normalized score."""

    return best is None or candidate > best


def _rank(evaluation: Mapping[str, Any], scores: Mapping[str, Any]) -> tuple[float, float]:
    win_rate = evaluation["win_rate"]
    mean_score = scores["mean"]
    return (float("-inf") if win_rate is None else float(win_rate), float("-inf") if mean_score is None else float(mean_score))


# --- module sources ----------------------------------------------------------


@dataclass(slots=True)
class MainAiSource:
    """Frozen inference modules plus the PPO module to train."""

    policy: PolicyModules
    origin: str
    description: dict[str, Any] = field(default_factory=dict)
    prior_training_games: int = 0
    # Evaluation stored in the .bangmodel being continued (its best evaluation).
    prior_evaluation: dict[str, Any] | None = None


def prior_evaluation_row(metadata: Any) -> dict[str, Any] | None:
    """The saved evaluation of a ``.bangmodel`` as an evaluation row, or None.

    Main-AI training stores the evaluation of the model it saved (the best one)
    under ``main_ai_training.evaluation``; other producers (refinement) put it
    in the top-level ``evaluation``.
    """

    if not isinstance(metadata, Mapping):
        return None
    training = metadata.get("main_ai_training") if isinstance(metadata.get("main_ai_training"), Mapping) else {}
    evaluation = training.get("evaluation") if isinstance(training.get("evaluation"), Mapping) else None
    if evaluation is None and isinstance(metadata.get("evaluation"), Mapping):
        evaluation = metadata["evaluation"]
    if not evaluation:
        return None
    win_rate = evaluation.get("win_rate")
    if isinstance(win_rate, bool) or not isinstance(win_rate, (int, float)):
        return None
    scores = training.get("normalized_scores") if isinstance(training.get("normalized_scores"), Mapping) else {}
    return {
        "win_rate": float(win_rate),
        "normalized_score": scores.get("mean"),
        "role_accuracy": evaluation.get("role_accuracy"),
        "hidden_role_accuracy": evaluation.get("hidden_role_accuracy"),
        "emotion_mse": evaluation.get("emotion_mse"),
        "by_role": {
            name: (values.get("win_rate") if isinstance(values, Mapping) else values)
            for name, values in dict(evaluation.get("by_role") or {}).items()
        },
        "ppo_update": training.get("ppo_updates"),
        "training_games": training.get("training_games"),
    }


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _verify_frozen(name: str, module: torch.nn.Module) -> None:
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise TrainingError(f"{name} module must be frozen during main AI training / {name} 모듈이 고정되지 않았습니다.")


def validate_emotion_contract(emotion: Any) -> None:
    """The emotion module must predict the dummies' nine-entry formula label."""

    count = int(emotion.emotion_count)
    version = str(emotion.emotion_schema_version)
    if count != len(LABEL_NAMES) or version != FORMULA_LABEL_SCHEMA:
        raise TrainingError(
            f"감정 모듈의 스키마가 더미 AI와 맞지 않습니다 (필요: {FORMULA_LABEL_SCHEMA}, {len(LABEL_NAMES)}개 / "
            f"현재: {version}, {count}개). 감정 모듈을 다시 학습하세요. "
            f"The emotion module schema does not match the dummies (need {FORMULA_LABEL_SCHEMA} with "
            f"{len(LABEL_NAMES)} entries, got {version} with {count}); retrain the emotion module."
        )


def _missing_modules_error(korean: str, english: str) -> TrainingError:
    return TrainingError(
        f"{korean} 먼저 '의도·감정 추론 모듈 학습'을 실행해 모듈을 저장하세요. "
        f"{english} Train the intent and emotion inference modules first."
    )


def _main_ai_source_refs(source: "MainAiSource") -> list[dict[str, Any]]:
    """Info references for the modules / model a main-AI run started from."""

    refs: list[dict[str, Any]] = []
    description = source.description or {}
    for key, kind in (("intent_path", KIND_INTENT), ("emotion_path", KIND_EMOTION), ("resume_artifact_path", KIND_MAIN_AI)):
        value = description.get(key)
        if value and Path(str(value)).exists():
            refs.append(info_ref(value, kind))
    return refs


def resolve_main_ai_source(run: Any, *, ppo_seed: int) -> MainAiSource:
    """Choose the frozen modules (and PPO weights) to train from.

    Order: an explicit intent + emotion ``.phase2model`` pair (fresh PPO), then a
    ``.bangmodel`` to continue (PPO weights kept, optimizer fresh), then the most
    recent saved module run that has both modules (fresh PPO).
    """

    config = run.config
    intent_path = getattr(config, "resume_intent_artifact_path", None) or None
    emotion_path = getattr(config, "resume_emotion_artifact_path", None) or None
    ppo_config = run.system_config.rl_training.ppo
    if (intent_path is None) != (emotion_path is None):
        raise _missing_modules_error(
            "의도 모듈과 감정 모듈 파일을 둘 다 지정해야 합니다.",
            "Both an intent and an emotion module file are required.",
        )
    if intent_path is not None and emotion_path is not None:
        origin = "phase2_pair"
        description: dict[str, Any] = {
            "intent_path": str(intent_path),
            "emotion_path": str(emotion_path),
        }
    elif getattr(run, "resume_artifact", None) is not None:
        artifact = run.resume_artifact
        validate_emotion_contract(artifact.emotion)
        _freeze(artifact.intent)
        _freeze(artifact.emotion)
        prior = 0
        metadata = getattr(artifact, "metadata", {})
        if isinstance(metadata, Mapping) and isinstance(metadata.get("main_ai_training"), Mapping):
            prior = int(metadata["main_ai_training"].get("training_games", 0) or 0)
        return MainAiSource(
            policy=PolicyModules(artifact.intent, artifact.emotion, artifact.role_fusion, artifact.preprocessor, artifact.ppo),
            origin="bangmodel",
            description={"resume_artifact_path": getattr(run, "resume_artifact_path", None)},
            prior_training_games=prior,
            prior_evaluation=prior_evaluation_row(metadata),
        )
    else:
        candidates = [
            item for item in module_artifacts.list_saved_module_runs() if item.intent_path and item.emotion_path
        ]
        if not candidates:
            raise _missing_modules_error(
                "저장된 의도·감정 추론 모듈이 없습니다.",
                "No saved intent/emotion inference modules were found.",
            )
        # Newest first; a saved run from an older, incompatible format is skipped.
        skipped: list[str] = []
        for saved in candidates:
            try:
                intent, _ = module_artifacts.load_intent_for_resume(saved.intent_path)
                emotion, _ = module_artifacts.load_emotion_for_resume(saved.emotion_path)
                validate_emotion_contract(emotion)
            except Exception as exc:  # noqa: BLE001 - any unreadable/incompatible file is skipped
                skipped.append(f"{saved.job_id}: {exc}")
                continue
            origin = "latest_saved_modules"
            intent_path, emotion_path = saved.intent_path, saved.emotion_path
            description = {
                "intent_path": str(intent_path),
                "emotion_path": str(emotion_path),
                "module_job_id": saved.job_id,
                "skipped_incompatible_runs": len(skipped),
            }
            break
        else:
            raise _missing_modules_error(
                f"저장된 모듈 {len(candidates)}개가 모두 호환되지 않습니다 (가장 최근: {skipped[0][:200]}).",
                f"All {len(candidates)} saved module runs are incompatible (newest: {skipped[0][:200]}).",
            )
    if origin != "latest_saved_modules":
        intent, _ = module_artifacts.load_intent_for_resume(intent_path)
        emotion, _ = module_artifacts.load_emotion_for_resume(emotion_path)
        validate_emotion_contract(emotion)
    _freeze(intent)
    _freeze(emotion)
    ppo = PPOModule(emotion_dim=policy_emotion_dim(emotion), ppo_config=ppo_config, seed=ppo_seed)
    return MainAiSource(
        policy=PolicyModules(intent, emotion, RoleFusion(), ObservationPreprocessor(), ppo),
        origin=origin,
        description=description,
    )


# --- orchestration -----------------------------------------------------------


def _set_progress(run: Any, **attributes: Any) -> None:
    with run.lock:
        for name, value in attributes.items():
            setattr(run, name, value)


def _append_point(run: Any, key: str, value: float | None) -> None:
    if value is None:
        return
    with run.lock:
        run.metrics.setdefault(key, []).append(float(value))


def _summary_row(evaluation: Mapping[str, Any], scores: Mapping[str, Any], *, update: int, games: int) -> dict[str, Any]:
    return {
        "ppo_update": int(update),
        "training_games": int(games),
        "win_rate": evaluation["win_rate"],
        "normalized_score": scores["mean"],
        "normalized_by_role": dict(scores["by_role"]),
        "by_role": {name: values["win_rate"] for name, values in evaluation["by_role"].items()},
        "role_accuracy": evaluation["role_accuracy"],
        "hidden_role_accuracy": evaluation["hidden_role_accuracy"],
        "emotion_mse": evaluation["emotion_mse"],
        "truncated_games": evaluation["truncated_games"],
    }


def train_main_ai(
    handle: Any,
    run: Any,
    *,
    eval_games: int = MAIN_AI_EVAL_GAMES,
    eval_every_updates: int = MAIN_AI_EVAL_EVERY_UPDATES,
    eval_base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    baseline_cache_dir: str | Path | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> Mapping[str, Any]:
    """Train the PPO policy against dummies and save the best evaluated model."""

    from .pipeline_runner import (
        TRAINING_MODE_MAIN_AI,
        _keep_graph_point,
        _publish_module_stage,
        _TrainingStopController,
        _training_seed,
    )

    started = time.monotonic()
    system_config: SystemConfig = run.system_config
    run_seed = _training_seed(run)
    eval_games = int(eval_games)
    eval_every_updates = max(1, int(eval_every_updates))

    train_by_time = bool(getattr(run.config, "train_by_time", False))
    budget = getattr(run.config, "max_training_seconds", None) if train_by_time else None
    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    if budget is not None and budget <= 0:
        budget = None
    total_games = 0 if train_by_time else int(getattr(run.config, "episodes", 0) or 0)
    if not train_by_time and total_games < 1:
        raise TrainingError(
            "학습할 게임 수(에피소드)를 1 이상으로 지정하거나 시간 기준 학습을 사용하세요. "
            "Set episodes to at least 1 or train by time."
        )
    controller = _TrainingStopController(run, budget_seconds=budget)
    try:
        graph_interval = max(1, int(getattr(run.config, "graph_point_interval", 1)))
    except (TypeError, ValueError):
        graph_interval = 1
    with run.lock:
        # Lets the monitor place the sampled graph points on a games/updates axis.
        run.metrics["main_graph_point_interval"] = graph_interval
        run.metrics["main_games_total"] = total_games

    source = resolve_main_ai_source(run, ppo_seed=run_seed)
    policy = source.policy
    if source.prior_evaluation is not None:
        # A continued model starts from its saved evaluation, not from 0: the
        # monitor draws it as the point at game 0 until this run evaluates.
        with run.lock:
            run.metrics["main_prior_eval"] = dict(source.prior_evaluation)
    _verify_frozen("intent", policy.intent)
    _verify_frozen("emotion", policy.emotion)
    _publish_module_stage(
        run,
        "main_ai_modules_ready",
        f"추론 모듈 준비 완료 ({source.origin})",
        origin=source.origin,
        **source.description,
    )

    ppo_config: PPOConfig = system_config.rl_training.ppo
    trainer = PPOTrainer(
        policy.ppo,
        config=ppo_config,
        seed=run_seed,
        intent_module=policy.intent,
        emotion_module=policy.emotion,
        normalize_advantages=True,
        bootstrap_truncated=True,
    )
    reward_config = main_ai_reward_config()
    calculator = RewardCalculator(reward_config)
    seats = PRIMARY_SEATS

    _publish_module_stage(
        run,
        "main_ai_baselines",
        f"기준선(dummy/random) 평가 중 ({eval_games}판, 캐시 사용 가능)",
        eval_games=eval_games,
    )
    loaded = load_or_compute_baselines(
        system_config,
        games=eval_games,
        base_seed=eval_base_seed,
        seats=seats,
        cache_dir=baseline_cache_dir,
        should_stop=controller.should_stop,
    )
    baselines: dict[str, Any] = {}
    baselines_cached = False
    if loaded is not None:
        baselines, baselines_cached = loaded
        with run.lock:
            run.metrics["main_baselines"] = baselines
        run.logger.event(
            "main_ai_baselines_ready",
            job_id=handle.job_id,
            cached=baselines_cached,
            dummy_win_rate=baselines["dummy"]["win_rate"],
            random_win_rate=baselines["random"]["win_rate"],
        )

    rollout_size = int(ppo_config.rollout_transitions_per_worker)
    buffer: list[PPOTransition] = []
    recent_wins: deque[int] = deque(maxlen=MAIN_AI_MOVING_AVERAGE_GAMES)
    recent_rewards: deque[float] = deque(maxlen=MAIN_AI_MOVING_AVERAGE_GAMES)
    games_done = normal_games = truncated_games = wins_total = 0
    environment_steps = transitions_total = updates = 0
    last_truncation_warning = -MAIN_AI_MOVING_AVERAGE_GAMES
    eval_history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_rank: tuple[float, float] | None = None
    last_eval_update = 0
    game_offset = source.prior_training_games
    artifact_path = Path(run.artifact_path) if getattr(run, "artifact_path", None) else None

    def reward_definition() -> dict[str, Any]:
        return {
            "version": reward_config.version,
            "config_id": reward_config.config_identifier,
            "terminal_win": 1.0,
            "terminal_loss": -1.0,
            "shaping": "none",
            "truncated": "reward 0, bootstrapped from the critic (time limit)",
        }

    def save_best(evaluation: Mapping[str, Any], scores: Mapping[str, Any]) -> None:
        if artifact_path is None:
            return
        counters = {
            "environment_steps": environment_steps,
            "main_agent_transitions": transitions_total,
            "ppo_updates": updates,
        }
        # ``save_final_inference_artifact`` freezes the module it is given, so
        # it gets a copy and training keeps its trainable PPO module.
        snapshot_ppo = copy.deepcopy(policy.ppo)
        try:
            previous = read_info(artifact_path) if artifact_path.is_file() else None
            artifact_meta = main_ai_info(
                run,
                handle,
                name=artifact_path.stem,
                training_games=game_offset + games_done,
                sources=_main_ai_source_refs(source),
                summary={
                    "win_rate": evaluation["win_rate"],
                    "normalized_score": scores.get("mean"),
                    "evaluation_games": evaluation["games"],
                    "ppo_updates": updates,
                    "baselines": {
                        name: values.get("win_rate")
                        for name, values in dict(baselines or {}).items()
                        if isinstance(values, Mapping)
                    },
                },
                extra_conditions={
                    "source_origin": source.origin,
                    "prior_training_games": game_offset,
                    "evaluation_base_seed": int(eval_base_seed),
                    "seats": [int(seat) for seat in seats],
                },
                notes=previous.notes if previous is not None else "",
            )
        except Exception:  # noqa: BLE001 - describing the model must never stop training
            artifact_meta = None
        save_final_inference_artifact(
            artifact_path,
            snapshot_ppo,
            info=artifact_meta,
            intent=policy.intent,
            emotion=policy.emotion,
            role_fusion=policy.role_fusion,
            preprocessor=policy.preprocessor,
            job_snapshot=handle.snapshot,
            training_metadata={
                "counters": counters,
                "total_training_time_seconds": time.monotonic() - started,
                "opponent": "dummy",
                "seed": run_seed,
                "evaluation": {
                    "protocol": "main_ai_training_eval",
                    "games": evaluation["games"],
                    "base_seed": evaluation["base_seed"],
                    "win_rate": evaluation["win_rate"],
                    "result": dict(evaluation),
                    "baselines": baselines,
                    "normalized_scores": dict(scores),
                },
            },
            metadata={
                "main_ai_training": {
                    "training_games": game_offset + games_done,
                    "ppo_updates": updates,
                    "source": {"origin": source.origin, **source.description},
                    "reward": reward_definition(),
                    "seeds": {
                        "run_seed": run_seed,
                        "evaluation_base_seed": int(eval_base_seed),
                        "training_game_seed": "derive_seed(run_seed, 'main-ai-training', 'game', index) % 2**31",
                    },
                    "evaluation": dict(evaluation),
                    "baselines": baselines,
                    "normalized_scores": dict(scores),
                    "best_rule": "overall win rate, ties by mean normalized score",
                }
            },
        )

    # The "training ..." stage text, restored after each evaluation.
    stage_message = {"training": ""}

    def run_evaluation(should_stop: Callable[[], bool] | None) -> dict[str, Any] | None:
        """Evaluate in the worker pool when there is one, else in this process."""

        nonlocal pool
        if pool is not None:
            try:
                played = pool.evaluate(
                    policy.ppo, games=eval_games, base_seed=eval_base_seed, seats=seats, should_stop=should_stop
                )
            except BrokenProcessPool as exc:
                drop_pool(exc)
            else:
                if played is None:
                    return None
                rows, totals = played
                return evaluation_summary(
                    [row[1:] for row in rows], InferenceMetrics(**totals), games=eval_games, base_seed=eval_base_seed
                )
        return evaluate_policy(
            policy,
            system_config,
            games=eval_games,
            base_seed=eval_base_seed,
            seats=seats,
            environment_factory=environment_factory,
            should_stop=should_stop,
        )

    def drop_pool(exc: BaseException) -> None:
        """A worker died: continue in this process instead of failing the run."""

        nonlocal pool
        log = getattr(run.logger, "error", None) or run.logger.event
        log("main_ai_parallel_failed", job_id=handle.job_id, error=f"{type(exc).__name__}: {exc}")
        if pool is not None:
            try:
                pool.close()
            except Exception:  # noqa: BLE001 - the pool is already broken
                pass
        pool = None
        with run.lock:
            run.metrics["main_parallel_workers"] = 1

    def evaluate(final: bool) -> None:
        nonlocal best, best_rank, last_eval_update
        if not baselines:
            return
        _publish_module_stage(
            run,
            "main_ai_evaluating",
            f"평가 중 (PPO 업데이트 {updates}회, {eval_games}판)",
            ppo_update=updates,
        )
        evaluation = run_evaluation(None if final else controller.should_stop)
        if not final and stage_message["training"]:
            # Back to training: the banner must not keep saying "evaluating".
            with run.lock:
                run.metrics["stage_message"] = stage_message["training"]
        if evaluation is None:
            return
        last_eval_update = updates
        scores = normalized_scores(evaluation, baselines)
        row = _summary_row(evaluation, scores, update=updates, games=game_offset + games_done)
        eval_history.append(row)
        _append_point(run, "main_eval_win_rate_points", evaluation["win_rate"])
        _append_point(run, "main_eval_score_points", scores["mean"])
        with run.lock:
            run.metrics.setdefault("main_eval_updates", []).append(updates)
            run.metrics.setdefault("main_eval_games", []).append(games_done)
            run.metrics["main_eval_latest"] = row
        rank = _rank(evaluation, scores)
        improved = _is_better(rank, best_rank)
        if improved:
            best, best_rank = row, rank
            save_best(evaluation, scores)
            with run.lock:
                run.metrics["main_best_eval"] = row
                run.win_rate = evaluation["win_rate"]
        run.logger.event(
            "main_ai_evaluation",
            job_id=handle.job_id,
            ppo_update=updates,
            win_rate=evaluation["win_rate"],
            normalized_score=scores["mean"],
            role_accuracy=evaluation["role_accuracy"],
            emotion_mse=evaluation["emotion_mse"],
            improved=improved,
        )

    def run_update(batch: list[PPOTransition]) -> None:
        nonlocal updates
        trainer.main_agent_transitions += len(batch)
        trainer.environment_steps += sum(item.environment_steps_spanned for item in batch)
        metrics = trainer.update(batch)
        updates = trainer.ppo_updates
        if _keep_graph_point(run, "main_update"):
            for key, value in (
                ("main_policy_loss_points", metrics.policy_loss),
                ("main_value_loss_points", metrics.value_loss),
                ("main_entropy_points", metrics.entropy),
                ("main_approx_kl_points", metrics.approx_kl),
            ):
                _append_point(run, key, value)
        _set_progress(run, ppo_updates=updates)
        run.logger.event(
            "main_ai_ppo_update",
            job_id=handle.job_id,
            ppo_update=updates,
            samples=metrics.samples,
            policy_loss=metrics.policy_loss,
            value_loss=metrics.value_loss,
            entropy=metrics.entropy,
            approx_kl=metrics.approx_kl,
        )

    stopped_early = False
    discarded_transitions = 0

    def play_next() -> list[TrainingGameResult]:
        """The next games: one in this process, or a batch of ``workers`` games in the pool."""

        first = game_offset + games_done
        if pool is not None:
            count = pool.batch_games if train_by_time else min(pool.batch_games, total_games - games_done)
            try:
                return pool.play_training_games(range(first, first + count), policy.ppo)
            except BrokenProcessPool as exc:
                drop_pool(exc)
        return [
            play_training_game(
                first,
                run_seed=run_seed,
                config=system_config,
                policy=policy,
                reward_calculator=calculator,
                environment_factory=environment_factory,
            )
        ]

    def record(game: TrainingGameResult) -> bool:
        """Count one finished game and update/evaluate when a rollout is full; True = stop."""

        nonlocal games_done, normal_games, truncated_games, wins_total, transitions_total
        nonlocal environment_steps, last_truncation_warning, buffer
        summary = game.summary
        games_done += 1
        buffer.extend(game.transitions)
        transitions_total += len(game.transitions)
        environment_steps += summary.environment_steps
        if summary.truncated:
            truncated_games += 1
        else:
            normal_games += 1
            wins_total += int(bool(summary.won))
            recent_wins.append(int(bool(summary.won)))
        recent_rewards.append(summary.total_reward)
        truncated_rate = truncated_games / games_done
        if summary.truncated and truncated_rate > MAIN_AI_TRUNCATION_WARNING_RATE and games_done - last_truncation_warning >= 50:
            last_truncation_warning = games_done
            warn = getattr(run.logger, "warning", None) or run.logger.event
            warn(
                "main_ai_truncation_rate_high",
                job_id=handle.job_id,
                truncated_rate=truncated_rate,
                truncated_games=truncated_games,
                games=games_done,
                threshold=MAIN_AI_TRUNCATION_WARNING_RATE,
            )
        _set_progress(
            run,
            episodes_completed=games_done,
            normal_episodes=normal_games,
            truncated_episodes=truncated_games,
            environment_steps=environment_steps,
            transitions=transitions_total,
        )
        with run.lock:
            run.metrics["main_games_completed"] = games_done
            run.metrics["truncated_rate"] = truncated_rate
            run.metrics["main_training_win_rate"] = (sum(recent_wins) / len(recent_wins)) if recent_wins else None
        if _keep_graph_point(run, "main_game"):
            _append_point(run, "main_win_rate_points", (sum(recent_wins) / len(recent_wins)) if recent_wins else None)
            _append_point(run, "main_reward_points", sum(recent_rewards) / len(recent_rewards))
        if len(buffer) >= rollout_size:
            if controller.should_stop():
                return True
            run_update(buffer)
            buffer = []
            if updates % eval_every_updates == 0:
                evaluate(final=False)
        return False

    # Hardware "workers" > 1 plays games in that many processes. A custom
    # environment factory (tests) cannot be sent to other processes.
    requested_workers = getattr(getattr(system_config, "hardware", None), "workers", 1)
    pool = None
    if environment_factory is None and usable_workers(requested_workers) > 1:
        _publish_module_stage(run, "main_ai_workers", f"게임 워커 준비 중 ({usable_workers(requested_workers)}개)")
        pool = open_game_pool(
            requested_workers,
            system_config,
            policy,
            run_seed,
            log=lambda message: run.logger.event("main_ai_parallel_unavailable", job_id=handle.job_id, reason=message),
        )
    with run.lock:
        run.metrics["main_parallel_workers"] = pool.workers if pool is not None else 1
    run.logger.event(
        "main_ai_parallel_workers",
        job_id=handle.job_id,
        requested=requested_workers,
        workers=pool.workers if pool is not None else 1,
    )
    stage_message["training"] = (
        "학습 게임 진행 중 ("
        + (
            "시간 기준: "
            + (f"{budget / 60.0:g}분" if budget is not None else "무제한, 중지를 누를 때까지")
            if train_by_time
            else f"{total_games}판"
        )
        + ")"
    )
    _publish_module_stage(
        run,
        "main_ai_training",
        stage_message["training"],
        train_by_time=train_by_time,
        total_games=total_games,
    )
    try:
        while True:
            if controller.should_stop():
                stopped_early = True
                break
            if not train_by_time and games_done >= total_games:
                break
            for game in play_next():
                if record(game):
                    stopped_early = True
                    break
            if stopped_early:
                break
        if buffer and updates == 0:
            # A short run that never reached a full rollout still trains once so it
            # leaves a model; otherwise the partial rollout is dropped.
            run_update(buffer)
            buffer = []
        discarded_transitions = len(buffer)
        if updates > 0 and last_eval_update != updates:
            evaluate(final=True)
    finally:
        if pool is not None:
            pool.close()

    truncated_rate = (truncated_games / games_done) if games_done else 0.0
    best_path = str(artifact_path) if (best is not None and artifact_path is not None) else None
    logs = [{"message": f"main AI training finished: {games_done} games, {updates} PPO updates"}]
    if best is None:
        logs.append({"message": "no model saved (no completed evaluation)"})
    metrics_out = {
        "workflow": TRAINING_MODE_MAIN_AI,
        "source": {"origin": source.origin, **source.description},
        "reward": reward_definition(),
        "train_by_time": train_by_time,
        "stopped_early": stopped_early,
        "training_games": games_done,
        "total_training_games": game_offset + games_done,
        "truncated_rate": truncated_rate,
        "discarded_transitions": discarded_transitions,
        "baselines": baselines,
        "baselines_from_cache": baselines_cached,
        "eval_games": eval_games,
        "eval_every_updates": eval_every_updates,
        "eval_history": eval_history,
        "best_eval": best,
        "emotion_dimension": len(LABEL_NAMES),
        "emotion_schema": FORMULA_LABEL_SCHEMA,
        "role_class_count": 4,
        "raw_role_count": 5,
        "saved_model": "best evaluated model only",
    }
    return {
        "output_path": run.output_path if getattr(run, "output_path", None) else None,
        "artifact_path": best_path,
        "environment_steps": environment_steps,
        "main_agent_transitions": transitions_total,
        "ppo_updates": updates,
        "episodes_completed": games_done,
        "normal_episodes": normal_games,
        "truncated_episodes": truncated_games,
        "win_rate": None if best is None else best["win_rate"],
        "metrics": metrics_out,
        "logs": tuple(logs),
    }


__all__ = [
    "BASELINE_NAMES",
    "MAIN_AI_EVAL_BASE_SEED",
    "MAIN_AI_EVAL_EVERY_UPDATES",
    "MAIN_AI_EVAL_GAMES",
    "MAIN_AI_REWARD_VERSION",
    "MainAiSource",
    "PolicyModules",
    "TrainingGameResult",
    "TrainingGameSummary",
    "baseline_cache_key",
    "evaluate_policy",
    "load_or_compute_baselines",
    "main_ai_reward_config",
    "main_seat_for_game",
    "normalized_scores",
    "play_training_game",
    "resolve_main_ai_source",
    "train_main_ai",
    "training_game_seed",
    "validate_emotion_contract",
]
