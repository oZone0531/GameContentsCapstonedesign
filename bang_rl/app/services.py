"""GUI-independent application orchestration services.

The application layer owns sessions and jobs, while environment, agents, and
training runtime remain independent modules.  DTOs exposed from this module are
immutable snapshots: callers never receive environment arrays, agent objects,
worker controls, or private role state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from threading import RLock, Thread
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from bang_rl.agents import DummyAgent, HumanAgent, RandomAgent
from bang_rl.config import (
    ConfigLoader,
    ConfigService as SchemaConfigService,
    JobSnapshot,
    PresetConfig,
    SystemConfig,
    TrainingJob,
)
from bang_rl.env import BangEnvironment
from bang_rl.exceptions import BangRLError, CompatibilityError, TrainingError
from bang_rl.runtime.logging import LogChannel, StructuredLogger, get_component_logger
from bang_rl.runtime.metadata import build_run_metadata
from bang_rl.runtime.pause_coordinator import PauseState, WorkerCancelled
from bang_rl.runtime.seed import SeedManager
from bang_rl.paths import (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    ensure_training_directories,
    latest_artifact,
    resolve_training_file,
)
from bang_rl.training import (
    DummyVecEnv,
    EpisodeRecord,
    ResourceGuard,
    WorkerFault,
    load_final_inference_artifact,
    load_final_artifact_for_training,
)


SERVICE_VERSION = "1.0"


def _json_value(value: Any) -> Any:
    """Convert a public value to detached JSON-compatible data."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_observation_dict"):
        return _json_value(value.to_observation_dict())
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)):
        try:
            return _json_value(value.to_dict())
        except TypeError:
            pass
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_value(value.value)
    return value


def _freeze(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping_config(value: SystemConfig | Mapping[str, Any] | None) -> SystemConfig:
    if value is None:
        return SystemConfig()
    if isinstance(value, SystemConfig):
        return value
    if isinstance(value, Mapping):
        return SystemConfig.model_validate(dict(value))
    raise TypeError("config must be a SystemConfig, mapping, or None")


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class ApplicationServiceError(BangRLError):
    """Structured application-layer failure with a stable error payload."""

    error_code = "application_service_error"


class UnknownHandleError(ApplicationServiceError):
    error_code = "unknown_handle"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Input for one interactive game session.

    ``agents`` is an optional test/integration seam.  When absent, the service
    constructs the requested human/random/dummy/main composition itself.
    """

    mode: str = "human_random"
    config: SystemConfig | Mapping[str, Any] | None = None
    seed: int | None = None
    artifact_path: str | Path | None = None
    agents: Sequence[Any] | Mapping[Any, Any] | None = None
    action_mode: str = "deterministic"


@dataclass(frozen=True, slots=True)
class SessionHandle:
    session_id: str
    mode: str
    seed: int
    snapshot: JobSnapshot


@dataclass(frozen=True, slots=True)
class PublicGameState:
    session_id: str
    episode_id: str
    observation: Mapping[str, Any]
    action_mask: tuple[int, ...]
    environment_step: int
    current_player: int | None
    terminated: bool
    truncated: bool
    termination_status: str
    winning_team: str | None = None
    winners: tuple[int, ...] = ()
    # Every seat's true role index, populated ONLY once the episode is over
    # (terminated or truncated) so the GUI can reveal all players at game end.
    # Empty during play so no living role ever leaks. Never persisted to a
    # recorded observation or replay record (replays stay public-projection).
    final_roles: tuple[int, ...] = ()
    # Item 1: True only for spectator (all-AI) sessions, where there is no human
    # whose hidden info must be protected, so every seat's true role is revealed
    # DURING play. Defaults False so a human session keeps the sheriff-only-
    # until-terminal privacy rule unchanged. The GUI reads this to pass
    # reveal_all into board_layout.role_for_display.
    reveal_all_roles: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "observation": _thaw(self.observation),
            "action_mask": list(self.action_mask),
            "environment_step": self.environment_step,
            "current_player": self.current_player,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_status": self.termination_status,
            "winning_team": self.winning_team,
            "winners": list(self.winners),
            "final_roles": list(self.final_roles),
            "reveal_all_roles": self.reveal_all_roles,
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    state: PublicGameState
    reward: float
    action: int | None
    info: Mapping[str, Any]
    cleanup: bool = False

    @property
    def observation(self) -> Mapping[str, Any]:
        return self.state.observation

    @property
    def terminated(self) -> bool:
        return self.state.terminated

    @property
    def truncated(self) -> bool:
        return self.state.truncated


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Immutable training-start request; the validated snapshot is made at start."""

    config: SystemConfig | Mapping[str, Any] | None = None
    mode: str = "rl"
    episodes: int = 1
    target_transitions: int | None = None
    seed: int | None = None
    split_seed: int | None = None
    agents: Sequence[Any] | None = None
    action_provider: Any = None
    runner: Callable[..., Any] | None = None
    report_path: str | Path | None = REPORTS_DIR
    output_path: str | Path | None = OUTPUTS_DIR
    artifact_path: str | Path | None = ARTIFACTS_DIR
    resume_artifact_path: str | Path | None = ARTIFACTS_DIR


@dataclass(frozen=True, slots=True)
class TrainingHandle:
    job_id: str
    snapshot: JobSnapshot


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    job_id: str
    status: str
    environment_steps: int
    main_agent_transitions: int
    ppo_updates: int
    episodes_completed: int
    worker_count: int
    active_training_seconds: float
    artifact_path: str | None = None
    report_path: str | None = None
    error: Mapping[str, Any] | None = None
    truncated_episodes: int = 0
    normal_episodes: int = 0
    win_rate: float | str | None = None
    truncated_rate: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    logs: tuple[Mapping[str, Any], ...] = ()
    output_path: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingReport:
    job_id: str
    status: str
    snapshot: JobSnapshot
    progress: TrainingProgress
    output_path: str | None
    report_path: str | None
    records: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_version": SERVICE_VERSION,
            "job_id": self.job_id,
            "status": self.status,
            "snapshot": self.snapshot.to_dict(),
            "progress": {
                "job_id": self.progress.job_id,
                "status": self.progress.status,
                "environment_steps": self.progress.environment_steps,
                "main_agent_transitions": self.progress.main_agent_transitions,
                "ppo_updates": self.progress.ppo_updates,
                "episodes_completed": self.progress.episodes_completed,
                "worker_count": self.progress.worker_count,
                "active_training_seconds": self.progress.active_training_seconds,
                "artifact_path": self.progress.artifact_path,
                "report_path": self.progress.report_path,
                "output_path": self.progress.output_path,
                "normal_episodes": self.progress.normal_episodes,
                "win_rate": self.progress.win_rate,
                "truncated_rate": self.progress.truncated_rate,
                "metrics": _thaw(_freeze(self.progress.metrics)),
                "logs": [_thaw(_freeze(item)) for item in self.progress.logs],
                "error": _thaw(_freeze(self.progress.error)) if self.progress.error is not None else None,
            },
            "output_path": self.output_path,
            "report_path": self.report_path,
            "records": [_thaw(_freeze(record)) for record in self.records],
            "metadata": _thaw(_freeze(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRunConfig:
    config: SystemConfig | Mapping[str, Any] | None = None
    games: int | None = None
    seed: int | None = None
    agents: Sequence[Any] | None = None
    action_provider: Any = None
    runner: Callable[..., Any] | None = None
    artifact: Any = None
    artifact_path: str | Path | None = None
    report_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    run_id: str
    protocol_id: str
    total_games: int
    normal_games: int
    truncated_games: int
    normal_wins: int
    normal_losses: int
    win_rate: float | str
    truncated_rate: float
    seed_set: tuple[int, ...]
    report_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "protocol_id": self.protocol_id,
            "total_games": self.total_games,
            "normal_games": self.normal_games,
            "truncated_games": self.truncated_games,
            "normal_wins": self.normal_wins,
            "normal_losses": self.normal_losses,
            "win_rate": self.win_rate,
            "truncated_rate": self.truncated_rate,
            "seed_set": list(self.seed_set),
            "report_path": self.report_path,
            "metadata": _thaw(_freeze(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    compatible: bool
    artifact_path: str
    errors: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Session:
    handle: SessionHandle
    config: SystemConfig
    environment: BangEnvironment
    agents: tuple[Any, ...]
    artifact: Any = None
    owned_loggers: tuple[StructuredLogger, ...] = ()


@dataclass
class _TrainingRun:
    handle: TrainingHandle
    config: TrainingConfig
    system_config: SystemConfig
    vector: Any | None
    logger: StructuredLogger
    lock: RLock = field(default_factory=RLock)
    status: str = "running"
    environment_steps: int = 0
    transitions: int = 0
    ppo_updates: int = 0
    episodes_completed: int = 0
    active_training_seconds: float = 0.0
    paused_seconds: float = 0.0
    pause_started_at: datetime | None = None
    artifact_path: str | None = None
    report_path: str | None = None
    output_path: str | None = None
    resume_artifact_path: str | None = None
    resume_artifact: Any = None
    error: Mapping[str, Any] | None = None
    truncated_episodes: int = 0
    normal_episodes: int = 0
    win_rate: float | str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[Mapping[str, Any]] = field(default_factory=list)
    records: list[Mapping[str, Any]] = field(default_factory=list)
    thread: Thread | None = None
    # The runner actually invoked by ``_execute``: an injected ``config.runner``
    # when provided, otherwise the default pipeline runner resolved in
    # ``TrainingService.start``.
    resolved_runner: Callable[..., Any] | None = None


class _ArtifactAgent:
    """Adapter from public observations to a frozen final inference artifact."""

    def __init__(self, artifact: Any, agent_id: str, seed: int) -> None:
        self.artifact = artifact
        self.agent_id = agent_id
        self.seed = seed

    def on_episode_start(self, context: Any) -> None:
        for module_name in ("intent", "emotion"):
            module = getattr(self.artifact, module_name, None)
            reset = getattr(module, "reset_state", None)
            if reset is not None:
                reset()

    def on_episode_end(self, result: Any) -> None:
        return None

    def act(self, observation: Any, action_mask: Any) -> int:
        processed = self.artifact.preprocessor.preprocess(observation)
        intent = self.artifact.intent.predict_proba(processed).detach().cpu().numpy()
        emotion = self.artifact.emotion.predict_proba(processed).detach().cpu().numpy()
        values = observation if isinstance(observation, Mapping) else observation.to_observation_dict()
        fused = self.artifact.role_fusion.fuse(
            intent,
            values.get("role_visibility"),
            values.get("revealed_roles"),
            values.get("confirmed_role_distribution"),
        )
        output = self.artifact.ppo.act(
            {
                "self_state": processed.state_vector,
                "board_state": processed.player_state_matrix,
                "confirmed_role_features": processed.role_beliefs,
                "fused_intent": fused,
                "emotion_features": emotion,
            },
            action_mask,
            deterministic=True,
        )
        return int(output.action.detach().cpu().item())


class GameService:
    """Own interactive game sessions and expose only public state DTOs."""

    def __init__(
        self,
        config_service: SchemaConfigService | None = None,
        compatibility_service: "CompatibilityService | None" = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.config_service = config_service or SchemaConfigService()
        self.compatibility_service = compatibility_service or CompatibilityService()
        self.logger = logger
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    def create_session(self, request: SessionConfig | SystemConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> SessionHandle:
        if isinstance(request, SessionConfig):
            options = request
        elif isinstance(request, SystemConfig) or isinstance(request, Mapping) or request is None:
            options = SessionConfig(config=request, **kwargs)
        else:
            raise TypeError("create_session expects SessionConfig, config mapping, or None")
        config = _mapping_config(options.config)
        seed = config.seed if options.seed is None else int(options.seed)
        snapshot = TrainingJob.start(config, mode="game", seed=seed).snapshot
        seed_manager = SeedManager(seed, snapshot=snapshot)
        seed_manager.seed_all()
        session_id = _id("session")
        episode_id = f"{session_id}-episode-0"
        artifact = None
        if options.artifact_path is not None:
            artifact = self.compatibility_service.load(options.artifact_path, config=config)
        agents = self._build_agents(options, config, seed_manager, artifact)
        owned_loggers: tuple[StructuredLogger, ...] = ()
        if self.logger is None:
            # Never let an environment close pytest/UI stderr during service
            # cleanup.  Application-owned buffers are closed with the session.
            streams = [io.StringIO() for _ in range(4)]
            env_logger = StructuredLogger(LogChannel.INVALID_ACTION, stream=streams[0], component="app-environment")
            event_logger = StructuredLogger(LogChannel.EPISODE_EVENTS, stream=streams[1], component="app-episode-events")
            termination_logger = StructuredLogger(LogChannel.TERMINATION, stream=streams[2], component="app-termination")
            diagnostics_logger = StructuredLogger(LogChannel.PRIVILEGED_DIAGNOSTICS, stream=streams[3], component="app-diagnostics")
            owned_loggers = (env_logger, event_logger, termination_logger, diagnostics_logger)
        else:
            env_logger = event_logger = termination_logger = diagnostics_logger = self.logger
        environment = BangEnvironment(
            episode_id=episode_id,
            seed=seed_manager.environment_seed(episode_id),
            config=config.env,
            logger=env_logger,
            event_logger=event_logger,
            termination_logger=termination_logger,
            diagnostics_logger=diagnostics_logger,
            diagnostics_metadata={"run_id": session_id, "job_id": snapshot.job_id, "config_version": config.config_version},
        )
        # Initialize every seat's episode so each agent folds its OWN role and
        # the public sheriff into its internal state from the very first turn.
        # Without this, a dummy that has not yet taken a turn still holds its
        # RoleBeliefEstimator constructor default (a uniform 1/5 matrix), which
        # reads as the naive 40/20/20 outlaw/traitor/deputy prior in the GUI and
        # replay.  environment.observe(index) hands each seat its OWN per-seat
        # public projection (self_role is that seat's role, sheriff is public),
        # so the eliminated prior falls out of the existing belief math.  The
        # per-agent guard mirrors the project's defensive export style: a single
        # agent failing here must never abort session creation.  Do NOT mirror
        # this into restore_session, which re-applies saved per-seat internals.
        #
        # Intended, deterministic side effect: running the normal episode-start
        # lifecycle here also reseeds each stochastic agent's action RNG from the
        # episode seed.  Interactive game sessions never called on_episode_start
        # before, so such agents used to run the whole game on their
        # constructor-time RNG.  The reseeded stream is still fully determined by
        # the session seed (reproducible), just a different stream than the
        # pre-fix behavior; this is a conscious decision, not a silent change.
        for index, agent in enumerate(agents):
            on_episode_start = getattr(agent, "on_episode_start", None)
            if on_episode_start is None:
                continue
            try:
                on_episode_start(
                    {
                        "episode_id": episode_id,
                        "seed": seed_manager.environment_seed(episode_id),
                        "observation": environment.observe(index),
                    }
                )
            except Exception:  # noqa: BLE001 - defensive: never abort session setup
                self._log("agent_episode_start_failed", session_id=session_id, seat=index)
        handle = SessionHandle(session_id, self._normalize_mode(options.mode), seed, snapshot)
        with self._lock:
            self._sessions[session_id] = _Session(handle, config, environment, agents, artifact, owned_loggers)
        self._log("game_session_started", session_id=session_id, mode=handle.mode, seed=seed, job_id=snapshot.job_id)
        return handle

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "human_vs_random": "human_random",
            "human_random_ai": "human_random",
            "human_vs_dummy": "human_dummy",
            "human_vs_main": "human_main_dummy",
            "human_main": "human_main_dummy",
            # Item 6: all-AI spectator modes and their GUI-string variants.
            "random_ai_5": "random_five",
            "random5": "random_five",
            "dummy_ai_5": "dummy_five",
            "dummy5": "dummy_five",
            "main_1_dummy_4": "main_one_dummy_four",
        }
        return aliases.get(normalized, normalized)

    # Modes the human plays a seat in (seat 0 is a HumanAgent).
    _HUMAN_MODES = frozenset({"human_random", "human_dummy", "human_main_dummy"})
    # Item 6: all-AI spectator modes; every seat is AI-driven.
    _SPECTATOR_MODES = frozenset({"random_five", "dummy_five", "main_one_dummy_four"})

    def _build_agents(self, options: SessionConfig, config: SystemConfig, seeds: SeedManager, artifact: Any) -> tuple[Any, ...]:
        if options.agents is not None:
            if isinstance(options.agents, Mapping):
                values = tuple(options.agents.get(index, options.agents.get(f"player_{index}")) for index in range(5))
            else:
                values = tuple(options.agents)
            if len(values) != 5 or any(agent is None for agent in values):
                raise ValueError("agents must provide exactly five non-null player agents")
            return values
        mode = self._normalize_mode(options.mode)
        if mode not in (self._HUMAN_MODES | self._SPECTATOR_MODES):
            raise ValueError(
                "unsupported game mode; use human_random, human_dummy, human_main_dummy, "
                "random_five, dummy_five, or main_one_dummy_four"
            )

        def _dummy(index: int) -> Any:
            return DummyAgent(
                f"player_{index}", seed=seeds.agent_seed(f"player_{index}"), player_idx=index,
                policy_config=config.preset.intent_policy,
                emotion_config=config.preset.emotion,
                action_mode=options.action_mode,
            )

        # Item 6: all-AI spectator compositions -- every one of the five seats
        # is AI-driven so the human only watches.
        if mode == "random_five":
            return tuple(
                RandomAgent(f"player_{index}", seed=seeds.agent_seed(f"player_{index}")) for index in range(5)
            )
        if mode == "dummy_five":
            return tuple(_dummy(index) for index in range(5))
        if mode == "main_one_dummy_four":
            if artifact is None:
                raise ValueError("main_one_dummy_four mode requires an inference artifact")
            values: list[Any] = [_ArtifactAgent(artifact, "player_0", seeds.agent_seed("player_0"))]
            values.extend(_dummy(index) for index in range(1, 5))
            return tuple(values)

        # Human-played compositions: seat 0 is the human, seats 1..4 are AI.
        values = [HumanAgent("player_0", seed=seeds.agent_seed("player_0"))]
        for index in range(1, 5):
            agent_seed = seeds.agent_seed(f"player_{index}")
            if mode == "human_random":
                values.append(RandomAgent(f"player_{index}", seed=agent_seed))
            elif mode == "human_dummy" or (mode == "human_main_dummy" and index != 1):
                values.append(_dummy(index))
            else:
                if artifact is None:
                    raise ValueError("human_main_dummy mode requires an inference artifact")
                values.append(_ArtifactAgent(artifact, f"player_{index}", agent_seed))
        return tuple(values)

    def _session(self, handle: SessionHandle | str) -> _Session:
        key = handle.session_id if isinstance(handle, SessionHandle) else str(handle)
        with self._lock:
            session = self._sessions.get(key)
        if session is None:
            raise UnknownHandleError("game session was not found", field="session_id", current_value=key)
        return session

    @staticmethod
    def _state(session: _Session) -> PublicGameState:
        env = session.environment
        # Item 7 (flicker fix) + item 1 (spectator reveal): project the public
        # observation from a STABLE viewer, not from state.agent_selection (the
        # current player). env.observe() with no argument defaults the viewer to
        # whoever's turn it is, so self_role and the SELF/REVEALED visibility
        # bits (which the GUI reads for seat 0) rotated with the current player
        # and made the human seat's role text flicker frame-to-frame. Always
        # project from the fixed human seat (index 0) so self_role/visibility are
        # seat-0-relative every turn. For a spectator session index 0 is also a
        # stable, arbitrary geometry viewer and all roles are revealed below.
        mode = getattr(session.handle, "mode", None)
        is_spectator = mode in GameService._SPECTATOR_MODES
        observation = _json_value(env.observe("player_0"))
        selected = getattr(env.agent_selection, "index", None)
        status = "normal" if bool(env.state.terminated) else "truncated" if bool(env.state.truncated) else "running"
        # Surface the winning faction as a stable string id (e.g. "sheriff_team")
        # only when the episode terminated normally. Never expose living roles.
        winning_team: str | None = None
        winners: tuple[int, ...] = ()
        terminal = env.state.terminal_result if bool(env.state.terminated) else None
        if terminal is not None:
            team = getattr(terminal, "winning_team", None)
            if team is not None:
                winning_team = str(getattr(team, "name", team)).lower()
            winners = tuple(int(index) for index in getattr(terminal, "winners", ()))
        # Reveal every seat's true role ONLY once the episode is over so the
        # GUI can show all players at game end. During play this stays empty so
        # no living role leaks. This is the only GUI-facing read of the private
        # env.state.roles, guarded so a fake/mock session that lacks it never
        # raises.
        final_roles: tuple[int, ...] = ()
        if bool(env.state.terminated) or bool(env.state.truncated):
            true_roles = getattr(getattr(env, "state", None), "roles", None)
            if true_roles is not None:
                final_roles = tuple(int(getattr(role, "value", role)) for role in true_roles)
        # Item 1: for a SPECTATOR (all-AI) session there is no human whose
        # hidden role must be protected, so reveal every living seat's true role
        # DURING play. Rewrite the public observation's revealed_roles from the
        # authoritative env.state.roles (the same guarded private read used for
        # final_roles) and mark every seat REVEALED (visibility >= 2). This is a
        # strict superset of the sheriff/eliminated reveal, so it never hides
        # anything the public projection already showed. Human sessions never
        # take this path, so their sheriff-only-until-terminal privacy is
        # unchanged.
        reveal_all_roles = bool(is_spectator)
        if reveal_all_roles:
            true_roles = getattr(getattr(env, "state", None), "roles", None)
            if true_roles is not None and isinstance(observation, dict):
                roles = [int(getattr(role, "value", role)) for role in true_roles]
                observation["revealed_roles"] = list(roles)
                # RoleVisibility.REVEALED == 2; keep every seat visible.
                observation["role_visibility"] = [2 for _ in roles]
        return PublicGameState(
            session.handle.session_id,
            env.state.episode_id,
            _freeze(observation),
            tuple(int(item) for item in np.asarray(env.action_mask).tolist()),
            int(env.environment_step),
            None if selected is None else int(selected),
            bool(env.state.terminated),
            bool(env.state.truncated),
            status,
            winning_team,
            winners,
            final_roles,
            reveal_all_roles,
        )

    def get_state(self, handle: SessionHandle | str) -> PublicGameState:
        return self._state(self._session(handle))

    def get_full_state(self, handle: SessionHandle | str) -> dict[str, Any]:
        """Return the authoritative :class:`EnvironmentState` as a JSON-safe dict.

        Item 5/6: the GUI controller records this per turn for human and
        AI-spectator games so a moment can later be restored to a live,
        continuable session (rewind-and-continue). Unlike :meth:`get_state`
        this intentionally exposes the full reproduction state (hidden roles,
        deck, seat order); it is therefore ONLY persisted into the ``full``
        replay format for human-played and spectator games, never into the
        public training-safe format.
        """

        session = self._session(handle)
        return _json_value(session.environment.state.to_dict())

    # The controller probes these method names for the authoritative state
    # export; expose the same behaviour under each so either name works.
    full_state = get_full_state
    authoritative_state = get_full_state
    export_state = get_full_state

    def export_agent_internals(self, handle: SessionHandle | str) -> dict[str, Any]:
        """Return a JSON-safe per-seat snapshot of each AI seat's internal state.

        Item 4: for human-played and AI-spectator games the GUI controller
        records this per turn so a moment can later be resumed with AI internal
        state, and so every player's HP is always present in the per-step record
        even when a seat has no AI internals. The fidelity of that resume
        differs by seat type: a DummyAgent seat round-trips its FULL internal
        state (emotion module + role beliefs / guess via ``state_dict`` /
        ``load_state_dict``), while the Main-AI seat is BEST-EFFORT only. The
        Main-AI inference modules expose ``get_state`` / ``set_state`` for their
        stream buffers, but ``get_state`` returns ``None`` when no active stream
        buffer is held, so the main-AI descriptor records an active inference
        stream buffer only when one is present (see the ``captured`` flag on the
        main descriptor); it does NOT reconstruct the main AI's full inference
        state. Do not read the "resume with AI internals" wording as a promise
        that the main AI resumes byte-for-byte.

        The returned mapping has two top-level keys:

        * ``health``: a list of every player's current HP (from
          ``env.state.health``), so HP is always carried per step.
        * ``seats``: a seat -> descriptor mapping. A DummyAgent seat reuses
          ``agent.state_dict()`` (which already serializes its emotion module
          via ``EmotionModule.state_dict`` and its role beliefs / guess via
          ``RoleBeliefEstimator.state_dict``). A RandomAgent seat records a
          minimal ``{"kind": "random"}`` descriptor (it has no belief / emotion
          state). The Main-AI seat (``_ArtifactAgent``) has no single
          ``state_dict``, so this captures a best-effort descriptor: the
          inference modules' ``get_state()`` stream buffers where present, all
          converted JSON-safe.

        This method is DEFENSIVE and never raises: a seat whose serialization
        fails is recorded as ``{"kind": "unknown"}`` rather than aborting the
        whole export. It is intentionally kept out of :meth:`get_state`,
        ``PublicGameState`` and every observation so AI internals never leak
        into public / training data.
        """

        session = self._session(handle)
        env = session.environment
        health: list[int] = []
        try:
            health = [int(value) for value in getattr(env.state, "health", ()) or ()]
        except Exception:
            health = []
        seats: dict[str, Any] = {}
        agents = getattr(session, "agents", ()) or ()
        for index, agent in enumerate(agents):
            seats[str(index)] = self._agent_internal_descriptor(agent)
        return {"health": _json_value(health), "seats": seats}

    @staticmethod
    def _agent_internal_descriptor(agent: Any) -> dict[str, Any]:
        """Return a JSON-safe internal-state descriptor for one AI seat.

        Never raises: any serialization failure degrades to a minimal
        ``{"kind": "unknown"}`` descriptor. Kept Qt-free.
        """

        # A DummyAgent carries the emotion module + role belief / guess we want.
        state_dict = getattr(agent, "state_dict", None)
        if isinstance(agent, DummyAgent) and callable(state_dict):
            try:
                return {"kind": "dummy", "state": _json_value(state_dict())}
            except Exception:
                return {"kind": "dummy"}
        if isinstance(agent, RandomAgent):
            return {"kind": "random"}
        if isinstance(agent, HumanAgent):
            return {"kind": "human"}
        # Main-AI (_ArtifactAgent): no single state_dict. Capture the inference
        # modules' stateful stream buffers where available, best effort. This is
        # genuinely opportunistic: an inference module's ``get_state()`` returns
        # ``None`` when it holds no active stream buffer, so ``module_states``
        # can legitimately be empty and a later restore is then a no-op for that
        # seat. Record a ``captured`` boolean so a caller (and a resume) can tell
        # whether any non-empty main-AI internal state was actually recorded,
        # rather than silently assuming it was.
        artifact = getattr(agent, "artifact", None)
        if artifact is not None:
            descriptor: dict[str, Any] = {"kind": "main"}
            module_states: dict[str, Any] = {}
            for module_name in ("intent", "emotion"):
                module = getattr(artifact, module_name, None)
                get_state = getattr(module, "get_state", None)
                if callable(get_state):
                    try:
                        value = get_state()
                    except Exception:
                        value = None
                    if value is not None:
                        module_states[module_name] = GameService._json_safe_state(value)
            descriptor["captured"] = bool(module_states)
            if module_states:
                descriptor["module_states"] = module_states
            return descriptor
        # Fall back to a generic state_dict probe for any other agent type.
        if callable(state_dict):
            try:
                return {"kind": "unknown", "state": _json_value(state_dict())}
            except Exception:
                return {"kind": "unknown"}
        return {"kind": "unknown"}

    @staticmethod
    def _json_safe_state(value: Any) -> Any:
        """Recursively convert a value (incl. torch / numpy tensors) JSON-safe.

        The main-AI inference modules' ``get_state()`` returns tensors, which
        ``_json_value`` alone leaves as raw objects. Any object exposing
        ``tolist`` (torch Tensor / numpy array) is converted to nested lists;
        everything else defers to :func:`_json_value`. Never raises: an
        un-convertible value degrades to its ``str`` form. Kept Qt-free.
        """

        if isinstance(value, Mapping):
            return {str(key): GameService._json_safe_state(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [GameService._json_safe_state(item) for item in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist) and not isinstance(value, (str, bytes)):
            try:
                return tolist()
            except Exception:
                return str(value)
        return _json_value(value)

    def restore_session(
        self,
        state_dict: Mapping[str, Any],
        *,
        mode: str | None = None,
        seed: int | None = None,
        artifact_path: str | Path | None = None,
        config: SystemConfig | Mapping[str, Any] | None = None,
        internals: Mapping[str, Any] | None = None,
    ) -> SessionHandle:
        """Rebuild a live, continuable session from an authoritative state dict.

        Item 5: constructs a fresh :class:`BangEnvironment`, applies
        ``load_state(EnvironmentState.from_dict(state_dict))`` so the game
        resumes from exactly the recorded moment, rebuilds the agents for the
        (optional) mode, registers the session, and returns a
        :class:`SessionHandle`. When ``mode`` is omitted it defaults to the
        human-vs-random composition so the human can continue playing seat 0.
        The GUI controller calls this with only the state dict, so every keyword
        is optional. Privacy is unchanged: the returned handle still projects
        public state via :meth:`get_state` (``final_roles`` only at terminal).
        """

        from bang_rl.env import EnvironmentState

        restored_state = EnvironmentState.from_dict(dict(state_dict))
        system_config = _mapping_config(config)
        resolved_seed = restored_state.seed if seed is None else int(seed)
        resolved_mode = self._normalize_mode(mode) if mode is not None else "human_random"
        snapshot = TrainingJob.start(system_config, mode="game", seed=resolved_seed).snapshot
        seed_manager = SeedManager(resolved_seed, snapshot=snapshot)
        seed_manager.seed_all()
        session_id = _id("session")
        episode_id = restored_state.episode_id or f"{session_id}-episode-0"
        artifact = None
        if artifact_path is not None:
            artifact = self.compatibility_service.load(artifact_path, config=system_config)
        options = SessionConfig(
            mode=resolved_mode, config=system_config, seed=resolved_seed, artifact_path=artifact_path
        )
        agents = self._build_agents(options, system_config, seed_manager, artifact)
        # Item 4: optionally restore each AI seat's internal state so a resumed
        # session continues with recorded AI internals rather than fresh agents.
        # A DummyAgent seat is restored in full (emotion / role belief); the
        # Main-AI seat is best-effort (only an active inference stream buffer is
        # reapplied, when one was captured). Every step here is defensive: a
        # missing / malformed entry leaves that seat's freshly built agent
        # untouched.
        if internals is not None:
            agents = self._restore_agent_internals(agents, internals)
        owned_loggers: tuple[StructuredLogger, ...] = ()
        if self.logger is None:
            streams = [io.StringIO() for _ in range(4)]
            env_logger = StructuredLogger(LogChannel.INVALID_ACTION, stream=streams[0], component="app-environment")
            event_logger = StructuredLogger(LogChannel.EPISODE_EVENTS, stream=streams[1], component="app-episode-events")
            termination_logger = StructuredLogger(LogChannel.TERMINATION, stream=streams[2], component="app-termination")
            diagnostics_logger = StructuredLogger(LogChannel.PRIVILEGED_DIAGNOSTICS, stream=streams[3], component="app-diagnostics")
            owned_loggers = (env_logger, event_logger, termination_logger, diagnostics_logger)
        else:
            env_logger = event_logger = termination_logger = diagnostics_logger = self.logger
        environment = BangEnvironment(
            episode_id=episode_id,
            seed=seed_manager.environment_seed(episode_id),
            config=system_config.env,
            logger=env_logger,
            event_logger=event_logger,
            termination_logger=termination_logger,
            diagnostics_logger=diagnostics_logger,
            diagnostics_metadata={"run_id": session_id, "job_id": snapshot.job_id, "config_version": system_config.config_version},
        )
        # Apply the authoritative recorded state so play resumes from the moment.
        environment.load_state(restored_state)
        handle = SessionHandle(session_id, resolved_mode, resolved_seed, snapshot)
        with self._lock:
            self._sessions[session_id] = _Session(handle, system_config, environment, agents, artifact, owned_loggers)
        self._log("game_session_restored", session_id=session_id, mode=handle.mode, seed=resolved_seed, job_id=snapshot.job_id)
        return handle

    # The controller also probes ``create_session_from_state``; expose the same.
    create_session_from_state = restore_session

    @staticmethod
    def _restore_agent_internals(
        agents: tuple[Any, ...], internals: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        """Apply a recorded per-seat internals mapping onto freshly built agents.

        Item 4: ``internals`` is the mapping produced by
        :meth:`export_agent_internals` (``{"health": [...], "seats": {seat: descriptor}}``).
        A DummyAgent seat is restored in FULL via ``load_state_dict`` (emotion +
        role belief / guess), so a resumed dummy keeps its internal state. A
        main-AI seat is BEST-EFFORT: its inference modules' ``set_state`` is only
        called for a captured active stream buffer, so when no buffer was
        recorded (``get_state`` returned ``None`` at capture, i.e. ``captured``
        is False / ``module_states`` absent) the main seat simply keeps its
        freshly built agent. Defensive: any per-seat failure is swallowed so a
        resume never raises and simply keeps that seat's fresh agent.
        """

        if not isinstance(internals, Mapping):
            return agents
        seats = internals.get("seats")
        if not isinstance(seats, Mapping):
            return agents
        for index, agent in enumerate(agents):
            descriptor = seats.get(str(index))
            if not isinstance(descriptor, Mapping):
                continue
            kind = str(descriptor.get("kind", "")).strip().lower()
            try:
                if kind == "dummy" and isinstance(agent, DummyAgent):
                    state = descriptor.get("state")
                    if isinstance(state, Mapping):
                        agent.load_state_dict(dict(state))
                elif kind == "main":
                    module_states = descriptor.get("module_states")
                    artifact = getattr(agent, "artifact", None)
                    if isinstance(module_states, Mapping) and artifact is not None:
                        for module_name in ("intent", "emotion"):
                            value = module_states.get(module_name)
                            module = getattr(artifact, module_name, None)
                            set_state = getattr(module, "set_state", None)
                            if value is not None and callable(set_state):
                                set_state(value)
            except Exception:
                # Keep the freshly built agent for this seat on any failure.
                continue
        return agents

    def step(self, handle: SessionHandle | str, action: Any | None = None) -> StepResult:
        session = self._session(handle)
        env = session.environment
        if action is None and (env.state.terminated or env.state.truncated or not env.state.alive[env.state.agent_selection]):
            env.step(None)
            return StepResult(self._state(session), 0.0, None, _freeze({"cleanup": True}), cleanup=True)
        selected = int(env.state.agent_selection)
        # Emotion formula: every living dummy (not only the actor) advances its
        # emotion state at every step, each from its OWN public observation.
        # No-op for dummies on the rule-table EmotionModule (the default).
        for index, seat_agent in enumerate(session.agents):
            if (
                index != selected
                and isinstance(seat_agent, DummyAgent)
                and seat_agent.uses_emotion_formula
                and env.state.alive[index]
            ):
                seat_agent.observe_public_boundary(env.observe(index))
        selected_action = action
        if selected_action is None:
            agent = session.agents[selected]
            selected_action = agent.act(env.observe(), env.action_mask)
        normalized = env.validate_action(selected_action)
        env.step(normalized)
        result = env.last_result
        info = {} if result is None else _json_value(getattr(result, "info", {}))
        reward = 0.0 if result is None else float(getattr(result, "reward", 0.0))
        state = self._state(session)
        self._log("game_step", session_id=session.handle.session_id, environment_step=state.environment_step, action=normalized, termination_status=state.termination_status)
        return StepResult(state, reward, normalized, _freeze(info))

    def submit_human_action(self, handle: SessionHandle | str, action: Any) -> None:
        session = self._session(handle)
        agent = session.agents[0]
        submit = getattr(agent, "submit_action", None)
        if submit is None:
            raise ApplicationServiceError("session player 0 is not a HumanAgent", field="player_0")
        submit(action)

    def run(self, handle: SessionHandle | str, *, max_steps: int | None = None) -> PublicGameState:
        session = self._session(handle)
        limit = max_steps or session.config.env.max_episode_steps
        for _ in range(int(limit)):
            state = self.get_state(handle)
            if state.terminated or state.truncated:
                break
            self.step(handle)
        return self.get_state(handle)

    def close_session(self, handle: SessionHandle | str) -> None:
        session = self._session(handle)
        session.environment.close()
        for logger in session.owned_loggers:
            logger.close()
        with self._lock:
            self._sessions.pop(session.handle.session_id, None)
        self._log("game_session_closed", session_id=session.handle.session_id)

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(event, **fields)


class TrainingService:
    """Start headless jobs and coordinate cooperative pause/resume/cancel."""

    def __init__(self, config_service: SchemaConfigService | None = None, logger: StructuredLogger | None = None) -> None:
        self.config_service = config_service or SchemaConfigService()
        self.logger = logger
        self._runs: dict[str, _TrainingRun] = {}
        self._lock = RLock()

    def start(self, request: TrainingConfig | SystemConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> TrainingHandle:
        if isinstance(request, TrainingConfig):
            options = request
        else:
            options = TrainingConfig(config=request, **kwargs)
        config = _mapping_config(options.config)
        if isinstance(options.episodes, bool) or options.episodes < 0:
            raise ValueError("episodes must be a non-negative integer")
        if options.target_transitions is not None and (isinstance(options.target_transitions, bool) or options.target_transitions < 1):
            raise ValueError("target_transitions must be positive when provided")
        # Resolve the runner: an injected runner is used verbatim; otherwise the
        # default end-to-end pipeline runner is wired in so that pressing "AI 학습"
        # actually trains and saves an artifact.  The pipeline runner is imported
        # lazily (not at module top) so importing ``bang_rl.app`` never pulls torch.
        if options.runner is not None:
            resolved_runner = options.runner
        else:
            from bang_rl.training.pipeline_runner import default_training_runner

            resolved_runner = default_training_runner
        seed = config.seed if options.seed is None else int(options.seed)
        ensure_training_directories()
        job = TrainingJob.start(config, mode=options.mode, seed=seed, split_seed=options.split_seed)
        resume_source = latest_artifact(options.resume_artifact_path)
        resume_artifact = None
        if resume_source is not None:
            # This is a weights-only continuation.  The loader verifies the
            # bundle and leaves PPO trainable with a fresh optimizer created by
            # the caller's PPOTrainer.
            resume_artifact = load_final_artifact_for_training(resume_source)
        logger = self.logger or get_component_logger("training-service", channel=LogChannel.TRAINING, job_id=job.job_id)
        # The default pipeline runner builds and owns its own vectorized env /
        # collector from ``run.system_config``; the service no longer builds a
        # DummyVecEnv up front.  ``run.vector`` stays None so pause/resume route
        # through the cooperative status-based cancel in ``_execute``.
        vector = None
        handle = TrainingHandle(job.job_id, job.snapshot)
        output_root = OUTPUTS_DIR if options.output_path is None else Path(options.output_path).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        run = _TrainingRun(
            handle,
            options,
            config,
            vector,
            logger,
            output_path=str(output_root),
            resume_artifact_path=None if resume_source is None else str(resume_source),
            resume_artifact=resume_artifact,
            resolved_runner=resolved_runner,
        )
        # Paths are public job metadata and should be visible before completion.
        run.report_path = str(resolve_training_file(options.report_path, root=REPORTS_DIR, filename=f"{job.job_id}.json", suffix=".json"))
        run.artifact_path = str(resolve_training_file(options.artifact_path, root=ARTIFACTS_DIR, filename=f"{job.job_id}.bangmodel", suffix=".bangmodel"))
        with self._lock:
            self._runs[handle.job_id] = run
        run.thread = Thread(target=self._execute, args=(run,), name=f"bang-rl-{handle.job_id}", daemon=True)
        run.thread.start()
        logger.event("training_started", mode=options.mode, seed=seed, worker_count=config.hardware.workers)
        return handle

    def _run(self, handle: TrainingHandle | str) -> _TrainingRun:
        key = handle.job_id if isinstance(handle, TrainingHandle) else str(handle)
        with self._lock:
            run = self._runs.get(key)
        if run is None:
            raise UnknownHandleError("training job was not found", field="job_id", current_value=key)
        return run

    def _execute(self, run: _TrainingRun) -> None:
        started = datetime.now(timezone.utc)
        try:
            # ``resolved_runner`` is the injected runner when provided, otherwise
            # the default end-to-end pipeline runner wired in by ``start``.  Both
            # follow the same ``runner(handle, run) -> Mapping`` seam; the default
            # runner also mutates ``run`` progress counters as it advances so the
            # GUI poll stays live.
            runner = run.resolved_runner if run.resolved_runner is not None else run.config.runner
            result = runner(run.handle, run)
            if isinstance(result, Mapping):
                run.records.extend(_json_value(result.get("records", ())))
                run.artifact_path = _path_string(result.get("artifact_path", run.artifact_path))
                run.environment_steps = int(result.get("environment_steps", run.environment_steps))
                run.transitions = int(result.get("main_agent_transitions", result.get("transitions", run.transitions)))
                run.episodes_completed = int(result.get("episodes_completed", run.episodes_completed))
                run.truncated_episodes = int(result.get("truncated_episodes", result.get("truncated_count", run.truncated_episodes)))
                run.normal_episodes = int(result.get("normal_episodes", result.get("normal_count", run.normal_episodes)))
                run.win_rate = result.get("win_rate", run.win_rate)
                run.metrics.update(_json_value(result.get("metrics", {})))
                raw_logs = _json_value(result.get("logs", result.get("public_logs", ())))
                if isinstance(raw_logs, Mapping):
                    raw_logs = (raw_logs,)
                elif isinstance(raw_logs, str):
                    raw_logs = ({"message": raw_logs},)
                run.logs.extend(raw_logs or ())
            if run.status == "running":
                run.status = "finished"
            elif run.status == "cancelling":
                run.status = "cancelled"
        except (WorkerCancelled, ):
            run.status = "cancelled"
        except Exception as exc:
            run.status = "faulted"
            run.error = _error_dict(exc)
            run.logger.error("training fault", error_type=type(exc).__name__, error=str(exc))
        finally:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            paused = run.paused_seconds
            if run.pause_started_at is not None:
                paused += max(0.0, (datetime.now(timezone.utc) - run.pause_started_at).total_seconds())
            run.paused_seconds = paused
            run.active_training_seconds += max(0.0, elapsed - paused)
            if run.vector is not None:
                try:
                    run.vector.close()
                except Exception as exc:
                    run.logger.error("training vector close failed", error=str(exc))
            progress = self._progress(run)
            report_metadata = build_run_metadata(
                run_id=run.handle.job_id,
                job_id=run.handle.job_id,
                component="training-service",
                snapshot=run.handle.snapshot,
                counters={
                    "environment_steps": run.environment_steps,
                    "main_agent_transitions": run.transitions,
                    "ppo_updates": run.ppo_updates,
                    "episodes_completed": run.episodes_completed,
                },
                active_training_seconds=run.active_training_seconds,
                paused_seconds=run.paused_seconds,
                hardware=run.system_config.hardware.model_dump(mode="json"),
            )
            report = TrainingReport(
                run.handle.job_id, run.status, run.handle.snapshot, progress,
                run.output_path, run.report_path,
                tuple(run.records), report_metadata,
            )
            report_path = run.report_path
            if report_path is not None:
                _write_json(Path(report_path), report.to_dict())
            run.logger.event("training_finished", status=run.status, environment_steps=run.environment_steps, main_agent_transitions=run.transitions)

    def pause(self, handle: TrainingHandle | str, timeout: float | None = None) -> None:
        run = self._run(handle)
        if run.vector is None:
            with run.lock:
                run.status = "paused"
            return
        acknowledged = run.vector.pause(timeout)
        with run.lock:
            if acknowledged:
                run.status = "paused"
                run.pause_started_at = datetime.now(timezone.utc)
            else:
                run.status = "pause_failed"
                run.error = _freeze({"error_type": "PauseTimeout", "message": "not all workers acknowledged pause"})
        if not acknowledged:
            run.vector.cancel()
            run.logger.error("pause failed", state=PauseState.PAUSE_FAILED.value)

    def resume(self, handle: TrainingHandle | str) -> None:
        run = self._run(handle)
        if run.status not in {"paused", "running"}:
            raise ApplicationServiceError("training job is not resumable in its current state", field="status", current_value=run.status, expected_value="paused or running")
        if run.vector is not None:
            run.vector.resume()
        with run.lock:
            if run.pause_started_at is not None:
                run.paused_seconds += max(0.0, (datetime.now(timezone.utc) - run.pause_started_at).total_seconds())
                run.pause_started_at = None
            run.status = "running"
        run.logger.event("training_resumed", job_id=run.handle.job_id)

    def cancel(self, handle: TrainingHandle | str) -> None:
        run = self._run(handle)
        if run.vector is not None:
            run.vector.cancel()
        with run.lock:
            if run.status not in {"finished", "faulted", "cancelled"}:
                run.status = "cancelling"
        run.logger.event("training_cancel_requested", job_id=run.handle.job_id)

    stop = cancel

    def progress(self, handle: TrainingHandle | str) -> TrainingProgress:
        return self._progress(self._run(handle))

    def wait(self, handle: TrainingHandle | str, timeout: float | None = None) -> TrainingReport:
        run = self._run(handle)
        if run.thread is not None:
            run.thread.join(timeout)
        if run.thread is not None and run.thread.is_alive():
            raise TimeoutError("training job did not finish before timeout")
        return TrainingReport(run.handle.job_id, run.status, run.handle.snapshot, self._progress(run), run.output_path, run.report_path, tuple(run.records))

    @staticmethod
    def _progress(run: _TrainingRun) -> TrainingProgress:
        completed = run.truncated_episodes + run.normal_episodes
        truncated_rate = None if completed == 0 else run.truncated_episodes / completed
        return TrainingProgress(
            job_id=run.handle.job_id,
            status=run.status,
            environment_steps=run.environment_steps,
            main_agent_transitions=run.transitions,
            ppo_updates=run.ppo_updates,
            episodes_completed=run.episodes_completed,
            worker_count=run.system_config.hardware.workers,
            active_training_seconds=run.active_training_seconds,
            artifact_path=run.artifact_path,
            report_path=run.report_path,
            error=run.error,
            truncated_episodes=run.truncated_episodes,
            normal_episodes=run.normal_episodes,
            win_rate=run.win_rate,
            truncated_rate=truncated_rate,
            metrics=_freeze({"episodes_completed": run.episodes_completed, **run.metrics}),
            logs=tuple(_freeze(_json_value(item)) for item in run.logs),
            output_path=run.output_path,
        )


class EvaluationService:
    """Application adapter for the artifact-backed evaluation protocols.

    ``run`` is reserved for real model evaluation.  The former lightweight
    agent simulation remains available as ``run_diagnostic`` so it cannot be
    mistaken for the deterministic primary/custom protocol.
    """

    def __init__(
        self,
        config_service: SchemaConfigService | None = None,
        logger: StructuredLogger | None = None,
        *,
        artifact: Any = None,
        environment_factory: Any = None,
    ) -> None:
        self.config_service = config_service or SchemaConfigService()
        self.logger = logger
        self.artifact = artifact
        self.environment_factory = environment_factory

    def run(self, request: EvaluationRunConfig | SystemConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        if isinstance(request, EvaluationRunConfig):
            options = request
        else:
            options = EvaluationRunConfig(config=request, **kwargs)
        if options.runner is not None:
            raise ApplicationServiceError(
                "diagnostic simulation is not the public artifact evaluation path; use run_diagnostic"
            )
        config = _mapping_config(options.config)
        evaluation = config.evaluation
        updates: dict[str, Any] = {}
        if options.games is not None:
            updates["games"] = int(options.games)
        if options.seed is not None:
            updates["base_seed"] = int(options.seed)
        if updates:
            evaluation = evaluation.model_copy(update=updates)
            config = config.model_copy(update={"evaluation": evaluation})
        artifact = options.artifact if options.artifact is not None else self.artifact
        if options.artifact_path is not None:
            artifact = options.artifact_path
        if artifact is None:
            raise ApplicationServiceError(
                "artifact evaluation requires a loaded inference artifact or artifact_path"
            )
        from bang_rl.evaluation.protocol import (
            PRIMARY_PROTOCOL_ID,
            CustomEvaluationProtocol,
            PrimaryEvaluationProtocol,
        )

        protocol_type = PrimaryEvaluationProtocol if evaluation.protocol_id == PRIMARY_PROTOCOL_ID else CustomEvaluationProtocol
        protocol = protocol_type(
            artifact,
            config=config,
            report_path=options.report_path,
            logger=self.logger,
            environment_factory=self.environment_factory,
        )
        return protocol.run(evaluation)

    def run_diagnostic(
        self,
        request: EvaluationRunConfig | SystemConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Run the old lightweight simulation under an explicitly diagnostic API."""

        if isinstance(request, EvaluationRunConfig):
            options = request
        else:
            options = EvaluationRunConfig(config=request, **kwargs)
        config = _mapping_config(options.config)
        evaluation = config.evaluation
        games = evaluation.games if options.games is None else int(options.games)
        if games < 1:
            raise ValueError("games must be positive")
        base_seed = evaluation.base_seed if options.seed is None else int(options.seed)
        seeds = tuple(base_seed + index for index in range(games))
        run_id = _id("evaluation-diagnostic")
        if options.runner is not None:
            result = options.runner(options, seeds)
            if isinstance(result, EvaluationReport):
                return result
            values = dict(result)
            normal = int(values.get("normal_games", 0))
            truncated = int(values.get("truncated_games", games - normal))
            wins = int(values.get("normal_wins", 0))
        else:
            agents = tuple(options.agents) if options.agents is not None else tuple(RandomAgent(f"player_{i}", seed=base_seed + i) for i in range(5))
            vector = DummyVecEnv(num_workers=1, environment_factory=_safe_environment_factory(config), seed=base_seed, config=config, agents=agents, hardware=config.hardware, job_id=run_id)
            try:
                records = vector.run_episodes(games, action_provider=options.action_provider)
            finally:
                vector.close()
            normal = sum(1 for record in records if record.terminated)
            truncated = sum(1 for record in records if record.truncated)
            wins = sum(1 for record in records if record.terminated and record.terminal_result and record.terminal_result.get("winning_team") is not None)
        losses = max(0, normal - wins)
        report_metadata = {
            "diagnostic_simulation": True,
            "seed_formula": evaluation.seed_formula,
            "seat_policy": evaluation.seat_policy,
            "action_mode": evaluation.action_mode,
            "run_metadata": build_run_metadata(
                run_id=run_id,
                component="evaluation-diagnostic",
                config=config,
                counters={"environment_steps": 0, "main_agent_transitions": 0, "evaluation_games": games, "normal_games": normal, "truncated_games": truncated},
            ),
        }
        report = EvaluationReport(run_id, evaluation.protocol_id, games, normal, truncated, wins, losses, "N/A" if normal == 0 else wins / normal, truncated / games, seeds, _path_string(options.report_path), _freeze(report_metadata))
        if options.report_path is not None:
            _write_json(Path(options.report_path), report.to_dict())
        if self.logger is not None:
            self.logger.event("evaluation_diagnostic_finished", run_id=run_id, total_games=games, normal_games=normal, truncated_games=truncated)
        return report


class ConfigService(SchemaConfigService):
    """Application alias for the validated YAML/JSON config facade."""

    def snapshot(self, value: SystemConfig | Mapping[str, Any], *, mode: str = "rl", seed: int | None = None, split_seed: int | None = None, job_id: str | None = None) -> JobSnapshot:
        config = _mapping_config(value)
        return TrainingJob.start(config, mode=mode, seed=seed, split_seed=split_seed, job_id=job_id).snapshot


class PresetService:
    """Import/export versioned DummyAgent presets without GUI dependencies."""

    def __init__(self, config_service: SchemaConfigService | None = None) -> None:
        self.config_service = config_service or SchemaConfigService()

    def load(self, path: str | Path) -> PresetConfig:
        return self.config_service.load(path, PresetConfig)

    def validate(self, value: PresetConfig | Mapping[str, Any]) -> PresetConfig:
        if isinstance(value, PresetConfig):
            return value
        return self.config_service.validate(value, PresetConfig)

    def save(self, path: str | Path, value: PresetConfig | Mapping[str, Any]) -> Path:
        return self.config_service.save(path, self.validate(value), PresetConfig)

    export = save
    import_preset = load


class DatasetService:
    """Application boundary for canonical dataset writers and path metadata."""

    def __init__(self, config: SystemConfig | Mapping[str, Any] | None = None) -> None:
        self.config = _mapping_config(config)

    @property
    def canonical_path(self) -> Path:
        return Path(self.config.dataset.canonical_path)

    @property
    def cache_path(self) -> Path:
        return Path(self.config.dataset.cache_path)

    def writer(self, output_path: str | Path | None = None, *, include_private_roles: bool = True) -> Any:
        from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
        return ParquetCanonicalShardWriter(output_path or self.canonical_path, dataset_config=self.config.dataset, include_private_roles=include_private_roles)

    def collector(self, *, snapshot: JobSnapshot | None = None, metadata: Mapping[str, Any] | None = None, include_private_roles: bool = True) -> Any:
        from bang_rl.data.collector import EpisodeCollector
        return EpisodeCollector(job_snapshot=snapshot, metadata=metadata, include_private_roles=include_private_roles)

    def paths(self) -> Mapping[str, str]:
        return _freeze({"canonical": str(self.canonical_path), "cache": str(self.cache_path)})


class CompatibilityService:
    """Check and load inference-only artifacts against a current snapshot."""

    def check(self, path: str | Path, *, config: SystemConfig | Mapping[str, Any] | None = None, snapshot: JobSnapshot | None = None) -> CompatibilityResult:
        source = Path(path)
        runtime_config = _mapping_config(config) if config is not None else None
        try:
            if snapshot is not None:
                artifact = load_final_inference_artifact(source, config=runtime_config, expected_snapshot=snapshot)
            else:
                artifact = load_final_inference_artifact(source, config=runtime_config)
        except Exception as exc:
            return CompatibilityResult(False, str(source), (_freeze(_error_dict(exc)),), {})
        return CompatibilityResult(True, str(source), (), _freeze(getattr(artifact, "metadata", {})))

    def load(self, path: str | Path, *, config: SystemConfig | Mapping[str, Any] | None = None, snapshot: JobSnapshot | None = None) -> Any:
        runtime_config = _mapping_config(config) if config is not None else None
        try:
            if snapshot is not None:
                return load_final_inference_artifact(path, config=runtime_config, expected_snapshot=snapshot)
            return load_final_inference_artifact(path, config=runtime_config)
        except Exception as exc:
            if isinstance(exc, BangRLError):
                raise
            raise CompatibilityError(str(exc)) from exc


def _safe_environment_factory(config: SystemConfig) -> Callable[..., BangEnvironment]:
    """Build environments with application-owned log streams."""
    def factory(*, worker_id: int = 0, seed: int = 0, episode_id: str = "episode-0", **_: Any) -> BangEnvironment:
        del worker_id
        streams = [io.StringIO() for _ in range(4)]
        return BangEnvironment(
            episode_id=episode_id,
            seed=seed,
            config=config.env,
            logger=StructuredLogger(LogChannel.INVALID_ACTION, stream=streams[0], component="worker-environment"),
            event_logger=StructuredLogger(LogChannel.EPISODE_EVENTS, stream=streams[1], component="worker-episode-events"),
            termination_logger=StructuredLogger(LogChannel.TERMINATION, stream=streams[2], component="worker-termination"),
            diagnostics_logger=StructuredLogger(LogChannel.PRIVILEGED_DIAGNOSTICS, stream=streams[3], component="worker-diagnostics"),
        )
    return factory


def _error_dict(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, BangRLError):
        return exc.to_dict()
    return {"error_type": type(exc).__name__, "code": "runtime_error", "message": str(exc)}


def _path_string(value: str | Path | None) -> str | None:
    return None if value is None else str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ApplicationServiceError",
    "CompatibilityResult",
    "CompatibilityService",
    "ConfigService",
    "DatasetService",
    "EvaluationReport",
    "EvaluationRunConfig",
    "EvaluationService",
    "GameService",
    "PresetService",
    "PublicGameState",
    "SessionConfig",
    "SessionHandle",
    "StepResult",
    "TrainingConfig",
    "TrainingHandle",
    "TrainingProgress",
    "TrainingReport",
    "TrainingService",
    "UnknownHandleError",
]
