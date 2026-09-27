"""Deterministic primary evaluation protocol.

The evaluator is deliberately an execution boundary, not a training runtime.
It loads a frozen inference artifact, composes MainAgent features from the
public observation only, and keeps privileged game state on a metrics-only
path.  The primary protocol is fixed at 500 games (100 games per seat).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bang_rl.agents.dummy_agent import DummyAgent
from bang_rl.agents.random_agent import RandomAgent
from bang_rl.config import EvaluationConfig, EnvConfig, PresetConfig, SystemConfig
from bang_rl.config.snapshot import JobSnapshot
from bang_rl.env import BangEnvironment
from bang_rl.env.types import Role, TerminationStatus
from bang_rl.exceptions import ArtifactCompatibilityError
from bang_rl.runtime.logging import LogChannel, StructuredLogger
from bang_rl.runtime.metadata import build_run_metadata
from bang_rl.runtime.seed import derive_seed
from bang_rl.training.final_inference_artifact import load_final_inference_artifact

PRIMARY_PROTOCOL_ID = "deterministic_v1"
PRIMARY_GAMES = 500
PRIMARY_GAMES_PER_SEAT = 100
PRIMARY_SEATS = (0, 1, 2, 3, 4)
PRIMARY_ACTION_MODE = "deterministic"
PRIMARY_SEED_FORMULA = "base_seed + evaluation_index"
REPORT_FORMAT_VERSION = "1.0"
CONFIDENCE_LEVEL = 0.95
_WILSON_Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided 95% Wilson binomial confidence interval."""

    confidence_level: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence_level": self.confidence_level,
            "lower": self.lower,
            "upper": self.upper,
        }


def _rate(wins: int, normal_games: int) -> float | str:
    return float(wins / normal_games) if normal_games else "N/A"


def _wilson_interval(wins: int, trials: int) -> ConfidenceInterval | str:
    if trials <= 0:
        return "N/A"
    p = wins / trials
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    margin = (_WILSON_Z / denominator) * np.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    )
    return ConfidenceInterval(CONFIDENCE_LEVEL, max(0.0, centre - margin), min(1.0, centre + margin))


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (Role, TerminationStatus)):
        return value.name if isinstance(value, Role) else value.value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _safe(value.model_dump(mode="json"))
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, JobSnapshot):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("snapshot must be a JobSnapshot or mapping")


def _artifact_id(artifact: Any) -> str:
    path = getattr(artifact, "path", None)
    if path is not None and Path(path).is_file():
        digest = sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    metadata = getattr(artifact, "metadata", {})
    declared = metadata.get("artifact_id") if isinstance(metadata, Mapping) else None
    if declared:
        return str(declared)
    return sha256(_canonical_json(metadata)).hexdigest()


def _version_expectations(
    config: SystemConfig | None,
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot_versions: dict[str, Any] = {}
    if snapshot is not None:
        for source in (snapshot.get("artifact_compatibility", {}), snapshot.get("versions", {})):
            if isinstance(source, Mapping):
                for field in (
                    "environment_version",
                    "observation_schema_version",
                    "action_space_version",
                    "reward_config_version",
                    "emotion_schema_version",
                    "inference_model_artifact_format_version",
                ):
                    if field in source:
                        snapshot_versions.setdefault(field, source[field])
    config_versions: dict[str, Any] = {}
    if config is not None:
        config_versions = {
            "environment_version": "1.0",
            "observation_schema_version": "1.0",
            "action_space_version": "1.0",
            "reward_config_version": config.reward_config.version,
            "emotion_schema_version": config.emotion.schema_version,
            "inference_model_artifact_format_version": config.artifact.inference_model_artifact_format_version,
        }
    for field, snapshot_value in snapshot_versions.items():
        config_value = config_versions.get(field)
        if config_value is not None and str(config_value) != str(snapshot_value):
            raise ArtifactCompatibilityError(
                f"snapshot.{field}", snapshot_value, config_value,
                context={"reason": "config and snapshot version mismatch"},
            )
    result = dict(config_versions)
    result.update(snapshot_versions)
    return result


def _validate_metadata_compatibility(
    artifact: Any,
    expected_versions: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
) -> None:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ArtifactCompatibilityError("metadata", type(metadata).__name__, "mapping")
    required = {
        "artifact_type": "inference",
        "artifact_format_version": "1.0",
        "inference_model_artifact_format_version": "1.0",
    }
    required.update({f"versions.{field}": value for field, value in expected_versions.items()})
    for field, expected in required.items():
        actual = _path_value(metadata, field)
        if actual != expected:
            raise ArtifactCompatibilityError(field, actual, expected, artifact=str(getattr(artifact, "path", "")) or None)
    modules = metadata.get("modules")
    if not isinstance(modules, Mapping) or any(not isinstance(modules.get(name), Mapping) for name in ("intent", "emotion", "ppo")):
        raise ArtifactCompatibilityError("modules", modules, "intent/emotion/ppo")
    identifiers = {"intent": "IntentInferenceModule", "emotion": "EmotionInferenceModule", "ppo": "PPOModule"}
    for name, expected in identifiers.items():
        actual = modules[name].get("identifier")
        if actual != expected:
            raise ArtifactCompatibilityError(f"modules.{name}.identifier", actual, expected)
    if snapshot is not None and isinstance(snapshot.get("model_architecture"), Mapping):
        artifact_architecture = metadata.get("model_architecture")
        if artifact_architecture is not None and artifact_architecture != snapshot["model_architecture"]:
            raise ArtifactCompatibilityError("model_architecture", artifact_architecture, snapshot["model_architecture"])


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """JSON-serializable result of one primary evaluation run."""

    evaluation_run_id: str
    model_artifact_id: str
    protocol: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    opponent: Mapping[str, Any]
    seed_set: tuple[int, ...]
    seat_counts: Mapping[str, int]
    total_games: int
    normal_games: int
    truncated_games: int
    normal_wins: int
    normal_losses: int
    win_rate: float | str
    win_rate_ci: ConfidenceInterval | str
    truncated_rate: float
    role_specific_results: Mapping[str, Mapping[str, Any]]
    role_classification_accuracy: float | str
    role_classification_ci: ConfidenceInterval | str
    hidden_role_accuracy: float | str
    hidden_role_ci: ConfidenceInterval | str
    estimated_emotion_mse: float | str
    metric_samples: Mapping[str, int]
    game_results: tuple[Mapping[str, Any], ...]
    comparison: Mapping[str, Any] = field(default_factory=dict)
    output_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, ConfidenceInterval):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return _safe(value)

        return {
            "report_format_version": REPORT_FORMAT_VERSION,
            "evaluation_run_id": self.evaluation_run_id,
            "model_artifact_id": self.model_artifact_id,
            "protocol": convert(self.protocol),
            "compatibility": convert(self.compatibility),
            "opponent": convert(self.opponent),
            "seed_set": list(self.seed_set),
            "seat_counts": convert(self.seat_counts),
            "total_games": self.total_games,
            "normal_games": self.normal_games,
            "truncated_games": self.truncated_games,
            "normal_wins": self.normal_wins,
            "normal_losses": self.normal_losses,
            "win_rate": convert(self.win_rate),
            "win_rate_ci": convert(self.win_rate_ci),
            "truncated_rate": self.truncated_rate,
            "role_specific_results": convert(self.role_specific_results),
            "role_classification_accuracy": convert(self.role_classification_accuracy),
            "role_classification_ci": convert(self.role_classification_ci),
            "hidden_role_accuracy": convert(self.hidden_role_accuracy),
            "hidden_role_ci": convert(self.hidden_role_ci),
            "estimated_emotion_mse": convert(self.estimated_emotion_mse),
            "metric_samples": convert(self.metric_samples),
            "game_results": convert(self.game_results),
            "output_metadata": convert(self.output_metadata),
            "comparison": convert(self.comparison),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination


EnvironmentFactory = Callable[[str, int, EnvConfig], BangEnvironment]


def _cleanup_explicit_owned_loggers(factory: Any, environments: list[Any]) -> None:
    """Run only explicit factory/environment logger ownership contracts.

    ``BangEnvironment.close`` remains responsible for the environment's own
    resources.  This helper handles logger resources whose ownership is
    declared by a factory or environment, without guessing from arbitrary
    logger attributes supplied by callers.
    """

    handled_owners: set[int] = set()
    closed_loggers: set[int] = set()
    for owner in (*reversed(environments), factory):
        owner_id = id(owner)
        if owner_id in handled_owners:
            continue
        handled_owners.add(owner_id)

        close_owned = getattr(owner, "close_owned_loggers", None)
        if callable(close_owned):
            close_owned()
            continue

        if not hasattr(owner, "owned_loggers"):
            continue
        owned_loggers = getattr(owner, "owned_loggers")
        if owned_loggers is None:
            continue
        try:
            iterator = iter(owned_loggers)
        except TypeError as exc:
            raise TypeError("owned_loggers must be an iterable of closeable loggers") from exc
        for logger in iterator:
            logger_id = id(logger)
            if logger_id in closed_loggers:
                continue
            closed_loggers.add(logger_id)
            close = getattr(logger, "close", None)
            if not callable(close):
                raise TypeError("owned_loggers must contain closeable loggers")
            close()


class PrimaryEvaluationProtocol:
    """Run the fixed deterministic_v1 evaluation against a frozen artifact."""

    def __init__(
        self,
        artifact: Any,
        *,
        config: SystemConfig | None = None,
        snapshot: JobSnapshot | Mapping[str, Any] | None = None,
        logger: StructuredLogger | None = None,
        report_path: str | Path | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ) -> None:
        if config is not None and not isinstance(config, SystemConfig):
            raise TypeError("config must be a SystemConfig or None")
        self.config = config or SystemConfig()
        self.snapshot = _as_mapping(snapshot)
        self.report_path = Path(report_path) if report_path is not None else None
        self._owns_logger = logger is None
        self._logger_closed = False
        self.logger = logger or StructuredLogger(LogChannel.EVALUATION, component="primary-evaluation")
        self.environment_factory = environment_factory or self._default_environment
        try:
            expected_versions = _version_expectations(config, self.snapshot)
            self.compatibility = {
                "expected_versions": dict(expected_versions),
                "config_version": self.config.config_version,
                "snapshot_id": None if self.snapshot is None else self.snapshot.get("job_id"),
            }
            self.artifact = self._resolve_artifact(artifact, expected_versions)
            self.artifact_id = _artifact_id(self.artifact)
            _validate_metadata_compatibility(self.artifact, expected_versions, self.snapshot)
            self.compatibility["validated"] = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close only a logger created by this protocol, at most once."""

        if self._logger_closed:
            return
        self._logger_closed = True
        if self._owns_logger:
            self.logger.close()

    def _resolve_artifact(self, artifact: Any, expected_versions: Mapping[str, Any]) -> Any:
        if isinstance(artifact, (str, Path)):
            expected = {f"versions.{field}": value for field, value in expected_versions.items()}
            return load_final_inference_artifact(
                artifact,
                expected=expected,
                expected_snapshot=self.snapshot,
            )
        required = ("intent", "emotion", "ppo", "role_fusion", "preprocessor", "metadata")
        if any(not hasattr(artifact, name) for name in required):
            raise TypeError("artifact must be a .bangmodel path or loaded inference artifact")
        return artifact

    def _default_environment(self, episode_id: str, seed: int, env_config: EnvConfig) -> BangEnvironment:
        return BangEnvironment(episode_id=episode_id, seed=seed, config=env_config)

    def _preset(self, evaluation: EvaluationConfig) -> PresetConfig:
        if evaluation.opponent not in {"default", "dummy", "random"}:
            raise ValueError(f"unsupported evaluation opponent: {evaluation.opponent!r}")
        configured = evaluation.opponent_presets.get(evaluation.opponent_preset)
        if configured is not None:
            return configured
        if evaluation.opponent_preset in {"default", self.config.preset.preset_id}:
            return self.config.preset
        raise ValueError(f"unknown opponent preset: {evaluation.opponent_preset!r}")

    @staticmethod
    def _validate_primary_config(evaluation: EvaluationConfig) -> None:
        expected = {
            "protocol_id": PRIMARY_PROTOCOL_ID,
            "games": PRIMARY_GAMES,
            "games_per_seat": PRIMARY_GAMES_PER_SEAT,
            "seed_formula": "base_seed_plus_evaluation_index",
            "deterministic": True,
            "action_mode": PRIMARY_ACTION_MODE,
            "seat_policy": "balanced",
        }
        for field, required in expected.items():
            if getattr(evaluation, field) != required:
                raise ValueError(f"Primary protocol requires evaluation.{field}={required!r}")
        if evaluation.seeds:
            expected_seeds = [evaluation.base_seed + index for index in range(PRIMARY_GAMES)]
            if evaluation.seeds != expected_seeds:
                raise ValueError("Primary protocol seeds must equal base_seed + evaluation_index")

    def _main_action(
        self,
        observation: Any,
        seat: int,
        *,
        action_mode: str = PRIMARY_ACTION_MODE,
        action_seed: int | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Every object passed below is derived from PublicEnvironmentState.  In
        # particular, no true role or DummyAgent emotion target reaches a model.
        processed = self.artifact.preprocessor.preprocess(observation)
        with torch.inference_mode():
            raw_intent = self.artifact.intent.predict_proba(processed)
            estimated_emotion = self.artifact.emotion.predict_proba(processed)
        raw_intent_array = raw_intent.detach().cpu().numpy() if torch.is_tensor(raw_intent) else np.asarray(raw_intent)
        estimated_emotion_array = estimated_emotion.detach().cpu().numpy() if torch.is_tensor(estimated_emotion) else np.asarray(estimated_emotion)
        values = {key: observation[key] for key in observation.keys()}
        fused = self.artifact.role_fusion.fuse(
            raw_intent_array,
            role_visibility=values["role_visibility"],
            revealed_roles=values["revealed_roles"],
            confirmed_role_distribution=values["confirmed_role_distribution"],
            self_role=values["self_role"],
            sheriff_idx=values["sheriff_idx"],
            viewer_idx=seat,
        )
        fused_array = np.asarray(fused, dtype=np.float32)
        emotion_features = estimated_emotion_array.reshape(-1).astype(np.float32)
        policy_inputs = {
            "self_state": processed.state_vector,
            "board_state": processed.player_state_matrix,
            "confirmed_role_features": processed.role_beliefs,
            "fused_intent": fused_array,
            "emotion_features": emotion_features,
            "action_mask": processed.action_mask,
        }
        with torch.inference_mode():
            if action_mode == "stochastic":
                # A private generator isolates stochastic evaluation from the
                # process-wide Torch RNG and from every other game stream.
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(0 if action_seed is None else action_seed))
                try:
                    selected = self.artifact.ppo.act(
                        policy_inputs,
                        processed.action_mask,
                        deterministic=False,
                        generator=generator,
                    )
                except TypeError as exc:
                    # Lightweight integrations may expose the older policy
                    # signature.  fork_rng keeps their fallback isolated too.
                    if "generator" not in str(exc):
                        raise
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(int(0 if action_seed is None else action_seed))
                        selected = self.artifact.ppo.act(
                            policy_inputs,
                            processed.action_mask,
                            deterministic=False,
                        )
            else:
                selected = self.artifact.ppo.act(
                    policy_inputs,
                    processed.action_mask,
                    deterministic=True,
                )
        action = int(selected.action.item())
        return action, raw_intent_array, fused_array, estimated_emotion_array, processed.role_beliefs

    @staticmethod
    def _fingerprint_module(module: Any) -> str:
        state = module.state_dict()
        digest = sha256()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _run_game(
        self,
        evaluation_index: int,
        seat: int,
        seed: int,
        preset: PresetConfig,
        *,
        action_mode: str = PRIMARY_ACTION_MODE,
        opponent: str = "dummy",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one game and close its environment on every execution path."""

        created: list[Any] = []
        factory = self.environment_factory

        def tracked_factory(episode_id: str, game_seed: int, env_config: EnvConfig) -> Any:
            environment = factory(episode_id, game_seed, env_config)
            created.append(environment)
            return environment

        self.environment_factory = tracked_factory
        try:
            return self._run_game_impl(
                evaluation_index,
                seat,
                seed,
                preset,
                action_mode=action_mode,
                opponent=opponent,
            )
        finally:
            self.environment_factory = factory
            closed: set[int] = set()
            try:
                for environment in reversed(created):
                    identity = id(environment)
                    if identity in closed:
                        continue
                    closed.add(identity)
                    close = getattr(environment, "close", None)
                    if close is not None:
                        close()
            finally:
                _cleanup_explicit_owned_loggers(factory, created)

    def _run_game_impl(
        self,
        evaluation_index: int,
        seat: int,
        seed: int,
        preset: PresetConfig,
        *,
        action_mode: str = PRIMARY_ACTION_MODE,
        opponent: str = "dummy",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        episode_id = f"{self.evaluation_run_id}-game-{evaluation_index:03d}"
        env = self.environment_factory(episode_id, seed, self.config.env)
        first_observation = env.reset(seed=seed, episode_id=episode_id)
        opponents = {
            index: (
                RandomAgent(
                    agent_id=f"player_{index}",
                    seed=seed + index + 1,
                )
                if opponent == "random"
                else DummyAgent(
                    agent_id=f"player_{index}",
                    player_idx=index,
                    seed=seed + index + 1,
                    policy_config=preset.intent_policy,
                    emotion_config=preset.emotion,
                    action_mode=action_mode,
                )
            )
            for index in range(5)
            if index != seat
        }
        for agent in opponents.values():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})

        role_correct = role_total = hidden_correct = hidden_total = 0
        emotion_squared_error = 0.0
        emotion_dimensions = 0
        while not env.state.terminated and not env.state.truncated:
            selection = env.agent_selection
            selected_attribute = getattr(selection, "index", None)
            if isinstance(selected_attribute, (int, np.integer)):
                selected = int(selected_attribute)
            else:
                selected = int(str(selection).removeprefix("player_"))
            observation = env.observe(selected)
            mask = np.asarray(observation["action_mask"], dtype=np.uint8)
            if selected == seat:
                action, raw_intent, fused_intent, estimated_emotion, _ = self._main_action(
                    observation,
                    seat,
                    action_mode=action_mode,
                    action_seed=derive_seed(
                        seed,
                        "evaluation",
                        self.evaluation_run_id,
                        "main-action",
                        int(np.asarray(observation["environment_step"]).reshape(-1)[0]),
                    ),
                )
                target_slots = [index for index in range(5) if index != seat]
                actual_roles = env.state.roles  # metrics-only privileged read
                visibility = np.asarray(observation["role_visibility"]).reshape(-1)
                for stream, target in enumerate(target_slots):
                    predicted = int(np.argmax(fused_intent[stream]))
                    actual = int(actual_roles[target])
                    role_total += 1
                    role_correct += int(predicted == actual)
                    is_hidden = int(visibility[target]) == 0
                    if is_hidden:
                        hidden_total += 1
                        hidden_correct += int(predicted == actual)
                for stream, target in enumerate(target_slots):
                    target_emotion = getattr(opponents[target], "emotion_state", None)
                    if target_emotion is None:
                        continue
                    target_values = np.asarray(target_emotion, dtype=np.float64).reshape(-1)
                    predicted_values = np.asarray(estimated_emotion[stream], dtype=np.float64).reshape(-1)
                    if target_values.shape == predicted_values.shape:
                        emotion_squared_error += float(np.square(predicted_values - target_values).sum())
                        emotion_dimensions += int(predicted_values.size)
            else:
                action = opponents[selected].act(observation, mask)
            if mask[action] != 1:
                raise RuntimeError("evaluation agent produced an action rejected by its public action mask")
            env.step(action)

        result = env.state.terminal_result
        main_won = bool(result is not None and seat in result.winners)
        status = env.state.termination_status.value
        for agent in opponents.values():
            agent.on_episode_end({"episode_id": episode_id, "terminated": env.state.terminated, "truncated": env.state.truncated})
        game = {
            "evaluation_index": evaluation_index,
            "episode_id": episode_id,
            "seed": seed,
            "main_agent_seat": seat,
            "main_agent_role": Role(env.state.roles[seat]).name,
            "termination_status": status,
            "terminated": bool(env.state.terminated),
            "truncated": bool(env.state.truncated),
            "environment_steps": int(env.environment_step),
            "normal_win": main_won if status == TerminationStatus.NORMAL.value else None,
            "normal_loss": (not main_won) if status == TerminationStatus.NORMAL.value else None,
            "winning_team": None if result is None or result.winning_team is None else result.winning_team.name,
            "winners": [] if result is None else list(result.winners),
        }
        metrics = {
            "role_correct": role_correct,
            "role_total": role_total,
            "hidden_correct": hidden_correct,
            "hidden_total": hidden_total,
            "emotion_squared_error": emotion_squared_error,
            "emotion_dimensions": emotion_dimensions,
        }
        return game, metrics

    def run(self, evaluation: EvaluationConfig | None = None) -> EvaluationReport:
        try:
            return self._run(evaluation)
        finally:
            self.close()

    def _run(self, evaluation: EvaluationConfig | None = None) -> EvaluationReport:
        evaluation = evaluation or self.config.evaluation
        self._validate_primary_config(evaluation)
        self.evaluation_run_id = "evaluation-" + sha256(_canonical_json({
            "protocol": PRIMARY_PROTOCOL_ID,
            "artifact": self.artifact_id,
            "base_seed": evaluation.base_seed,
            "opponent": evaluation.opponent,
            "preset": evaluation.opponent_preset,
        })).hexdigest()[:20]
        self.logger.info(
            "primary evaluation started",
            protocol_id=PRIMARY_PROTOCOL_ID,
            evaluation_run_id=self.evaluation_run_id,
            total_games=PRIMARY_GAMES,
            games_per_seat=PRIMARY_GAMES_PER_SEAT,
            base_seed=evaluation.base_seed,
        )
        fingerprints = tuple(self._fingerprint_module(module) for module in (self.artifact.intent, self.artifact.emotion, self.artifact.ppo))
        preset = self._preset(evaluation)
        game_results: list[Mapping[str, Any]] = []
        role_counts = {role.name: {"normal_games": 0, "normal_wins": 0, "normal_losses": 0} for role in Role}
        seat_counts = {str(seat): 0 for seat in PRIMARY_SEATS}
        normal_games = truncated_games = normal_wins = normal_losses = 0
        role_correct = role_total = hidden_correct = hidden_total = 0
        emotion_squared_error = 0.0
        emotion_dimensions = 0
        for evaluation_index in range(PRIMARY_GAMES):
            seat = PRIMARY_SEATS[evaluation_index % len(PRIMARY_SEATS)]
            seed = evaluation.base_seed + evaluation_index
            seat_counts[str(seat)] += 1
            game, metrics = self._run_game(
                evaluation_index,
                seat,
                seed,
                preset,
                opponent=evaluation.opponent,
            )
            game_results.append(game)
            status = game["termination_status"]
            role = str(game["main_agent_role"])
            if status == TerminationStatus.NORMAL.value:
                normal_games += 1
                normal_wins += int(bool(game["normal_win"]))
                normal_losses += int(bool(game["normal_loss"]))
                role_counts[role]["normal_games"] += 1
                role_counts[role]["normal_wins"] += int(bool(game["normal_win"]))
                role_counts[role]["normal_losses"] += int(bool(game["normal_loss"]))
            else:
                truncated_games += 1
            role_correct += metrics["role_correct"]
            role_total += metrics["role_total"]
            hidden_correct += metrics["hidden_correct"]
            hidden_total += metrics["hidden_total"]
            emotion_squared_error += metrics["emotion_squared_error"]
            emotion_dimensions += metrics["emotion_dimensions"]
            self.logger.event(
                "evaluation_game",
                "primary evaluation game complete",
                step=game["environment_steps"],
                evaluation_index=evaluation_index,
                seat=seat,
                seed=seed,
                termination_status=status,
                main_agent_role=role,
                main_agent_win=game["normal_win"],
            )
        after = tuple(self._fingerprint_module(module) for module in (self.artifact.intent, self.artifact.emotion, self.artifact.ppo))
        if after != fingerprints:
            raise RuntimeError("primary evaluation mutated inference artifact weights")

        role_results: dict[str, Mapping[str, Any]] = {}
        for role, counts in role_counts.items():
            role_results[role] = {
                **counts,
                "win_rate": _rate(counts["normal_wins"], counts["normal_games"]),
                "win_rate_ci": _wilson_interval(counts["normal_wins"], counts["normal_games"]),
            }
        protocol = {
            "id": PRIMARY_PROTOCOL_ID,
            "games": PRIMARY_GAMES,
            "games_per_seat": PRIMARY_GAMES_PER_SEAT,
            "seat_assignment_policy": "balanced_cyclic_0_1_2_3_4",
            "action_selection_mode": PRIMARY_ACTION_MODE,
            "seed_formula": PRIMARY_SEED_FORMULA,
            "base_seed": evaluation.base_seed,
        }
        opponent = {
            "id": evaluation.opponent,
            "preset_id": preset.preset_id,
            "preset_version": preset.version,
            "configured_preset": evaluation.opponent_preset,
        }
        report = EvaluationReport(
            evaluation_run_id=self.evaluation_run_id,
            model_artifact_id=self.artifact_id,
            protocol=protocol,
            compatibility=self.compatibility,
            opponent=opponent,
            seed_set=tuple(evaluation.base_seed + index for index in range(PRIMARY_GAMES)),
            seat_counts=seat_counts,
            total_games=PRIMARY_GAMES,
            normal_games=normal_games,
            truncated_games=truncated_games,
            normal_wins=normal_wins,
            normal_losses=normal_losses,
            win_rate=_rate(normal_wins, normal_games),
            win_rate_ci=_wilson_interval(normal_wins, normal_games),
            truncated_rate=float(truncated_games / PRIMARY_GAMES),
            role_specific_results=role_results,
            role_classification_accuracy=_rate(role_correct, role_total),
            role_classification_ci=_wilson_interval(role_correct, role_total),
            hidden_role_accuracy=_rate(hidden_correct, hidden_total),
            hidden_role_ci=_wilson_interval(hidden_correct, hidden_total),
            estimated_emotion_mse=(float(emotion_squared_error / emotion_dimensions) if emotion_dimensions else "N/A"),
            metric_samples={
                "role_classification": role_total,
                "hidden_role_classification": hidden_total,
                "estimated_emotion": emotion_dimensions,
            },
            game_results=tuple(game_results),
            output_metadata={
                "report_format_version": REPORT_FORMAT_VERSION,
                "artifact_compatibility_checked": True,
                "privileged_evaluation_data": "metrics_and_analysis_only",
                "main_agent_inputs": "public_observation_only",
                "training_mutation": False,
                "report_path": None if self.report_path is None else str(self.report_path),
                "run_metadata": build_run_metadata(
                    run_id=self.evaluation_run_id,
                    component="primary-evaluation",
                    snapshot=self.snapshot,
                    config=self.config,
                    artifact=self.artifact,
                    counters={
                        "environment_steps": sum(int(item.get("environment_steps", 0)) for item in game_results),
                        "evaluation_games": PRIMARY_GAMES,
                        "normal_games": normal_games,
                        "truncated_games": truncated_games,
                    },
                ),
            },
        )
        if self.report_path is not None:
            report.save_json(self.report_path)
        self.logger.info(
            "primary evaluation completed",
            evaluation_run_id=self.evaluation_run_id,
            total_games=PRIMARY_GAMES,
            normal_games=normal_games,
            truncated_games=truncated_games,
            normal_wins=normal_wins,
            normal_losses=normal_losses,
            win_rate=report.win_rate,
            truncated_rate=report.truncated_rate,
        )
        return report


def run_primary_evaluation(
    artifact: Any,
    *,
    config: SystemConfig | None = None,
    snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    report_path: str | Path | None = None,
    logger: StructuredLogger | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> EvaluationReport:
    """Convenience function for the fixed primary protocol."""

    return PrimaryEvaluationProtocol(
        artifact,
        config=config,
        snapshot=snapshot,
        report_path=report_path,
        logger=logger,
        environment_factory=environment_factory,
    ).run()


class CustomEvaluationProtocol(PrimaryEvaluationProtocol):
    """Run an explicitly configured evaluation separate from ``deterministic_v1``.

    The execution and metric path is shared with the primary protocol, but the
    protocol identity, seed set, seat schedule, opponent preset, and action
    mode are all recorded in the report.  This prevents custom/stochastic
    observations from being merged into the fixed benchmark.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        supplied_logger = kwargs.get("logger")
        super().__init__(*args, **kwargs)
        if supplied_logger is None:
            previous_logger = self.logger
            try:
                replacement = StructuredLogger(LogChannel.EVALUATION, component="custom-evaluation")
            except BaseException:
                self.close()
                raise
            if self._owns_logger and not self._logger_closed:
                previous_logger.close()
            self.logger = replacement
            self._logger_closed = False

    @staticmethod
    def _validate_custom_config(evaluation: EvaluationConfig) -> None:
        if evaluation.protocol_id == PRIMARY_PROTOCOL_ID:
            raise ValueError("custom evaluation requires protocol_id='custom_v1' or 'stochastic_v1'")
        if evaluation.games < 1:
            raise ValueError("custom evaluation games must be positive")
        if evaluation.action_mode not in {"deterministic", "stochastic"}:
            raise ValueError("evaluation.action_mode must be deterministic or stochastic")
        if evaluation.deterministic != (evaluation.action_mode == "deterministic"):
            raise ValueError("deterministic must agree with action_mode")
        if evaluation.seeds and len(evaluation.seeds) != evaluation.games:
            raise ValueError("custom evaluation seeds must contain one seed per game")
        if evaluation.seat_policy == "fixed" and len(evaluation.seats) != 1:
            raise ValueError("fixed seat_policy requires exactly one seat")

    @staticmethod
    def _seat_schedule(evaluation: EvaluationConfig) -> tuple[int, ...]:
        seats = tuple(evaluation.seats) or PRIMARY_SEATS
        if any(seat < 0 or seat >= 5 for seat in seats):
            raise ValueError("evaluation seats must be player indices in [0, 4]")
        if evaluation.seat_policy == "fixed":
            return tuple(seats[0] for _ in range(evaluation.games))
        if evaluation.seat_policy == "custom" and not evaluation.seats:
            raise ValueError("custom seat_policy requires an explicit seats list")
        if evaluation.seats and evaluation.games_per_seat != PRIMARY_GAMES_PER_SEAT:
            expected_games = len(seats) * evaluation.games_per_seat
            if expected_games != evaluation.games:
                raise ValueError(
                    "games must equal seats * games_per_seat for an explicit seat schedule"
                )
        return tuple(seats[index % len(seats)] for index in range(evaluation.games))

    @staticmethod
    def _comparison(
        report: EvaluationReport,
        primary_report: EvaluationReport | None,
    ) -> Mapping[str, Any]:
        base: dict[str, Any] = {
            "primary_protocol_id": PRIMARY_PROTOCOL_ID,
            "available": primary_report is not None,
            "same_conditions": False,
            "display_as_continuous_performance": False,
        }
        if primary_report is None:
            base["reason"] = "primary_report_not_supplied"
            return base
        base.update(
            {
                "primary_evaluation_run_id": primary_report.evaluation_run_id,
                "primary_win_rate": primary_report.win_rate,
                "primary_truncated_rate": primary_report.truncated_rate,
                "win_rate_delta": (
                    report.win_rate - primary_report.win_rate
                    if isinstance(report.win_rate, float)
                    and isinstance(primary_report.win_rate, float)
                    else "N/A"
                ),
                "truncated_rate_delta": report.truncated_rate - primary_report.truncated_rate,
                "normal_games_delta": report.normal_games - primary_report.normal_games,
                "total_games_delta": report.total_games - primary_report.total_games,
            }
        )
        return base

    def run(
        self,
        evaluation: EvaluationConfig | None = None,
        *,
        primary_report: EvaluationReport | None = None,
    ) -> EvaluationReport:
        try:
            return self._run(evaluation, primary_report=primary_report)
        finally:
            self.close()

    def _run(
        self,
        evaluation: EvaluationConfig | None = None,
        *,
        primary_report: EvaluationReport | None = None,
    ) -> EvaluationReport:
        evaluation = evaluation or self.config.evaluation
        self._validate_custom_config(evaluation)
        seats = self._seat_schedule(evaluation)
        seeds = tuple(evaluation.seeds) if evaluation.seeds else tuple(
            evaluation.base_seed + index for index in range(evaluation.games)
        )
        preset = self._preset(evaluation)
        protocol_id = evaluation.protocol_id
        self.evaluation_run_id = "evaluation-" + sha256(_canonical_json({
            "protocol": protocol_id,
            "artifact": self.artifact_id,
            "opponent": evaluation.opponent,
            "preset": evaluation.opponent_preset,
            "preset_version": preset.version,
            "seeds": seeds,
            "seats": seats,
            "action_mode": evaluation.action_mode,
        })).hexdigest()[:20]
        self.logger.info(
            "custom evaluation started",
            protocol_id=protocol_id,
            evaluation_run_id=self.evaluation_run_id,
            total_games=evaluation.games,
            action_mode=evaluation.action_mode,
            opponent=evaluation.opponent,
            opponent_preset=evaluation.opponent_preset,
        )
        fingerprints = tuple(self._fingerprint_module(module) for module in (self.artifact.intent, self.artifact.emotion, self.artifact.ppo))
        game_results: list[Mapping[str, Any]] = []
        role_counts = {role.name: {"normal_games": 0, "normal_wins": 0, "normal_losses": 0} for role in Role}
        seat_counts = {str(seat): 0 for seat in PRIMARY_SEATS}
        normal_games = truncated_games = normal_wins = normal_losses = 0
        role_correct = role_total = hidden_correct = hidden_total = 0
        emotion_squared_error = 0.0
        emotion_dimensions = 0
        for evaluation_index, (seat, seed) in enumerate(zip(seats, seeds)):
            seat_counts[str(seat)] += 1
            game, metrics = self._run_game(
                evaluation_index,
                seat,
                int(seed),
                preset,
                action_mode=evaluation.action_mode,
                opponent=evaluation.opponent,
            )
            game_results.append(game)
            status = game["termination_status"]
            role = str(game["main_agent_role"])
            if status == TerminationStatus.NORMAL.value:
                normal_games += 1
                normal_wins += int(bool(game["normal_win"]))
                normal_losses += int(bool(game["normal_loss"]))
                role_counts[role]["normal_games"] += 1
                role_counts[role]["normal_wins"] += int(bool(game["normal_win"]))
                role_counts[role]["normal_losses"] += int(bool(game["normal_loss"]))
            else:
                truncated_games += 1
            role_correct += metrics["role_correct"]
            role_total += metrics["role_total"]
            hidden_correct += metrics["hidden_correct"]
            hidden_total += metrics["hidden_total"]
            emotion_squared_error += metrics["emotion_squared_error"]
            emotion_dimensions += metrics["emotion_dimensions"]
            self.logger.event(
                "evaluation_game",
                "custom evaluation game complete",
                evaluation_run_id=self.evaluation_run_id,
                evaluation_index=evaluation_index,
                seat=seat,
                seed=int(seed),
                protocol_id=protocol_id,
                action_mode=evaluation.action_mode,
                termination_status=status,
                main_agent_role=role,
                main_agent_win=game["normal_win"],
            )
        after = tuple(self._fingerprint_module(module) for module in (self.artifact.intent, self.artifact.emotion, self.artifact.ppo))
        if after != fingerprints:
            raise RuntimeError("custom evaluation mutated inference artifact weights")
        role_results = {
            role: {
                **counts,
                "win_rate": _rate(counts["normal_wins"], counts["normal_games"]),
                "win_rate_ci": _wilson_interval(counts["normal_wins"], counts["normal_games"]),
            }
            for role, counts in role_counts.items()
        }
        report = EvaluationReport(
            evaluation_run_id=self.evaluation_run_id,
            model_artifact_id=self.artifact_id,
            protocol={
                "id": protocol_id,
                "kind": "stochastic" if evaluation.action_mode == "stochastic" else "custom",
                "games": evaluation.games,
                "games_per_seat": evaluation.games_per_seat,
                "seat_assignment_policy": evaluation.seat_policy,
                "seats": list(seats),
                "action_selection_mode": evaluation.action_mode,
                "seed_formula": evaluation.seed_formula,
                "base_seed": evaluation.base_seed,
                "primary_comparable": False,
            },
            compatibility=self.compatibility,
            opponent={
                "id": evaluation.opponent,
                "preset_id": preset.preset_id,
                "preset_version": preset.version,
                "configured_preset": evaluation.opponent_preset,
            },
            seed_set=seeds,
            seat_counts=seat_counts,
            total_games=evaluation.games,
            normal_games=normal_games,
            truncated_games=truncated_games,
            normal_wins=normal_wins,
            normal_losses=normal_losses,
            win_rate=_rate(normal_wins, normal_games),
            win_rate_ci=_wilson_interval(normal_wins, normal_games),
            truncated_rate=float(truncated_games / evaluation.games),
            role_specific_results=role_results,
            role_classification_accuracy=_rate(role_correct, role_total),
            role_classification_ci=_wilson_interval(role_correct, role_total),
            hidden_role_accuracy=_rate(hidden_correct, hidden_total),
            hidden_role_ci=_wilson_interval(hidden_correct, hidden_total),
            estimated_emotion_mse=(float(emotion_squared_error / emotion_dimensions) if emotion_dimensions else "N/A"),
            metric_samples={
                "role_classification": role_total,
                "hidden_role_classification": hidden_total,
                "estimated_emotion": emotion_dimensions,
            },
            game_results=tuple(game_results),
            output_metadata={
                "report_format_version": REPORT_FORMAT_VERSION,
                "run_kind": "stochastic" if evaluation.action_mode == "stochastic" else "custom",
                "protocol_id": protocol_id,
                "artifact_compatibility_checked": True,
                "privileged_evaluation_data": "metrics_and_analysis_only",
                "main_agent_inputs": "public_observation_only",
                "training_mutation": False,
                "rng_isolation": "per_game_environment_and_agent_streams",
                "report_path": None if self.report_path is None else str(self.report_path),
                "run_metadata": build_run_metadata(
                    run_id=self.evaluation_run_id,
                    component="custom-evaluation",
                    snapshot=self.snapshot,
                    config=self.config,
                    artifact=self.artifact,
                    counters={
                        "environment_steps": sum(int(item.get("environment_steps", 0)) for item in game_results),
                        "evaluation_games": evaluation.games,
                        "normal_games": normal_games,
                        "truncated_games": truncated_games,
                    },
                ),
            },
        )
        comparison = self._comparison(report, primary_report)
        object.__setattr__(report, "comparison", comparison)
        if self.report_path is not None:
            report.save_json(self.report_path)
        self.logger.info(
            "custom evaluation completed",
            protocol_id=protocol_id,
            evaluation_run_id=self.evaluation_run_id,
            total_games=evaluation.games,
            normal_games=normal_games,
            truncated_games=truncated_games,
            normal_wins=normal_wins,
            normal_losses=normal_losses,
            win_rate=report.win_rate,
            truncated_rate=report.truncated_rate,
            training_mutation=False,
        )
        return report


def run_custom_evaluation(
    artifact: Any,
    *,
    config: SystemConfig | None = None,
    snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    evaluation: EvaluationConfig | None = None,
    primary_report: EvaluationReport | None = None,
    report_path: str | Path | None = None,
    logger: StructuredLogger | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> EvaluationReport:
    """Run a custom or stochastic evaluation without changing the primary run."""

    return CustomEvaluationProtocol(
        artifact,
        config=config,
        snapshot=snapshot,
        report_path=report_path,
        logger=logger,
        environment_factory=environment_factory,
    ).run(evaluation, primary_report=primary_report)


# Service spelling from Design §14.1; the service remains evaluation-only.
class EvaluationService:
    def __init__(self, artifact: Any, *, config: SystemConfig | None = None, snapshot: JobSnapshot | Mapping[str, Any] | None = None) -> None:
        self.artifact = artifact
        self.config = config
        self.snapshot = snapshot

    def run(
        self,
        config: EvaluationConfig | None = None,
        *,
        primary_report: EvaluationReport | None = None,
    ) -> EvaluationReport:
        if config is not None:
            evaluation = config
        elif self.config is not None:
            evaluation = self.config.evaluation
        else:
            evaluation = None
        if evaluation is not None and evaluation.protocol_id != PRIMARY_PROTOCOL_ID:
            return CustomEvaluationProtocol(
                self.artifact,
                config=self.config,
                snapshot=self.snapshot,
            ).run(evaluation, primary_report=primary_report)
        return PrimaryEvaluationProtocol(
            self.artifact,
            config=self.config,
            snapshot=self.snapshot,
        ).run(evaluation)


__all__ = [
    "CONFIDENCE_LEVEL",
    "ConfidenceInterval",
    "CustomEvaluationProtocol",
    "EvaluationReport",
    "EvaluationService",
    "PRIMARY_ACTION_MODE",
    "PRIMARY_GAMES",
    "PRIMARY_GAMES_PER_SEAT",
    "PRIMARY_PROTOCOL_ID",
    "PRIMARY_SEATS",
    "PRIMARY_SEED_FORMULA",
    "PrimaryEvaluationProtocol",
    "run_custom_evaluation",
    "run_primary_evaluation",
]
