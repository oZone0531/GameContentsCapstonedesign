"""Phase 1 dataset collection from real all-dummy games.

``build_game`` creates the same five DummyAgents and environment as
``GameService.create_session`` in ``dummy_five`` mode, but derives every seed
from the game seed alone, so a collected game is reproducible (GameService also
mixes in a random session id). ``drive_game`` then plays it in the same order as
``GameService.step``:

1. every living non-actor dummy advances its emotion formula at the boundary
   (``observe_public_boundary`` with its own public observation);
2. the actor chooses with ``act`` (which advances its own emotion first);
3. the pre-action emotion of every living dummy is read (never written);
4. the action is validated and applied, and the step is recorded.

Labels are only read, never updated, so collecting does not change any
decision: from the same starting state a collected game and a game stepped
through ``GameService.step`` are move-for-move identical.

Emotion labels
--------------
With the emotion formula (a ``formula`` block) each label has 9 entries,
``[calm, confidence, aggression, anxiety, relation_0..relation_4]`` in [0, 1].
These are deterministic projections of the formula's public latent state. The
schema version is ``FORMULA_LABEL_SCHEMA``; the module-training workflow uses
:func:`formula_label_emotion_config`.

With the rule-table EmotionModule (no ``formula`` block) the label is the
module's own value vector and its configured schema version.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bang_rl.agents import DummyAgent
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.config import EmotionConfig, SystemConfig
from bang_rl.data.collector import CanonicalEpisode, EpisodeCollector
from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
from bang_rl.runtime.reward import RewardCalculator

COLLECTION_FORMAT_VERSION = "game-collection-1.0"
FORMULA_LABEL_SCHEMA = "formula-1.0+label9"
DEFAULT_EMOTION_FILE = "emotion_formula.example.json"
ACTION_MODES = ("deterministic", "stochastic")
ROLE_NAMES = ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")
# Safety net against a game that never ends; real games end in ~20 steps and the
# environment's own max_episode_steps truncation normally fires first.
_GUARD_STEPS = 10_000

__all__ = [
    "ACTION_MODES",
    "COLLECTION_FORMAT_VERSION",
    "DEFAULT_EMOTION_FILE",
    "FORMULA_LABEL_SCHEMA",
    "build_game",
    "collect_games",
    "drive_game",
    "emotion_labels",
    "emotion_mode",
    "formula_label_emotion_config",
    "game_seed",
    "load_collection_config",
    "play_and_collect_game",
]


def formula_label_emotion_config() -> EmotionConfig:
    """Emotion schema for data collected from emotion-formula dummies.

    The main agent's emotion inference module (``rl_training.emotion_inference``
    and the ``emotion_config`` given to ``EmotionInferenceSLTrainer``) must use
    this schema to train on formula data; the trainer rejects a cache whose
    emotion schema version differs.
    """

    return EmotionConfig(
        schema_version=FORMULA_LABEL_SCHEMA,
        emotions=[{"name": name} for name in LABEL_NAMES],
    )


def load_collection_config(
    config: SystemConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    emotion_file: str | Path | None = DEFAULT_EMOTION_FILE,
) -> SystemConfig:
    """Build the SystemConfig used for collection.

    ``config`` may be a SystemConfig, a mapping, or a YAML/JSON file path.
    ``emotion_file`` (a JSON emotion block with a ``formula`` section) replaces
    ``preset.emotion``; pass ``None`` to keep the config's own emotion block.
    """

    if isinstance(config, SystemConfig):
        values = config.model_dump(mode="json")
    elif isinstance(config, Mapping):
        values = json.loads(json.dumps(config))
    elif config is not None:
        path = Path(config)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            values = yaml.safe_load(text) or {}
        else:
            values = json.loads(text)
    else:
        values = {}
    if emotion_file is not None:
        path = Path(emotion_file)
        if not path.is_absolute() and not path.exists():
            # Fall back to the repository root (next to the bang_rl folder) so
            # the default works from any working directory.
            path = Path(__file__).resolve().parents[2] / path
        block = json.loads(path.read_text(encoding="utf-8"))
        values.setdefault("preset", {})["emotion"] = block
    return SystemConfig.model_validate(values)


def emotion_mode(agent: Any) -> str:
    return "formula" if isinstance(agent, DummyAgent) and agent.uses_emotion_formula else "rule_table"


def game_seed(base_seed: int, index: int) -> int:
    return int(base_seed) + int(index)


def _label_layout(agent: DummyAgent, config: SystemConfig) -> tuple[str, tuple[str, ...]]:
    if agent.uses_emotion_formula:
        return FORMULA_LABEL_SCHEMA, tuple(LABEL_NAMES)
    module = agent.emotion_module
    names = tuple(getattr(module, "emotion_names", ()) or ())
    return str(config.preset.emotion.schema_version), names


def emotion_labels(
    agents: Any, env: Any, config: SystemConfig, *, episode_id: str
) -> dict[str, dict[str, Any]]:
    """Read (never update) the current emotion label of every living dummy."""

    step = int(env.state.environment_step)
    labels: dict[str, dict[str, Any]] = {}
    for index, agent in enumerate(agents):
        if not isinstance(agent, DummyAgent) or not bool(env.state.alive[index]):
            continue
        if agent.uses_emotion_formula:
            values = np.asarray(agent.emotion_module.label(), dtype=np.float64)
        else:
            values = np.asarray(agent.emotion_module.values, dtype=np.float64)
        schema, names = _label_layout(agent, config)
        labels[str(index)] = {
            "episode_id": episode_id,
            "environment_step": step,
            "player_idx": index,
            "values": [float(item) for item in np.clip(values, 0.0, 1.0)],
            "emotion_names": list(names),
            "emotion_schema_version": schema,
            "input_event_ids": [],
        }
    return labels


def build_game(
    config: SystemConfig,
    *,
    seed: int,
    action_mode: str = "deterministic",
    episode_id: str | None = None,
) -> tuple[Any, tuple[DummyAgent, ...], str]:
    """Create the environment and five dummies for one game, reproducibly.

    Mirrors ``GameService.create_session`` in ``dummy_five`` mode (global
    seeding, agent construction, environment, per-seat ``on_episode_start``).
    The difference is that every seed is derived from ``seed`` and the episode
    id alone: GameService also mixes in a random session id, so a GUI game can
    never be replayed from its seed, while a collected game can.
    """

    from bang_rl.env import BangEnvironment
    from bang_rl.runtime.seed import SeedManager

    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")
    resolved_id = episode_id or f"game-{int(seed):08d}"
    seeds = SeedManager(int(seed))
    seeds.seed_all()
    agents = tuple(
        DummyAgent(
            f"player_{index}",
            seed=seeds.agent_seed(f"player_{index}", episode_id=resolved_id),
            player_idx=index,
            policy_config=config.preset.intent_policy,
            emotion_config=config.preset.emotion,
            action_mode=action_mode,
        )
        for index in range(5)
    )
    env_seed = seeds.environment_seed(resolved_id)
    env = BangEnvironment(episode_id=resolved_id, seed=env_seed, config=config.env)
    for index, agent in enumerate(agents):
        agent.on_episode_start(
            {"episode_id": resolved_id, "seed": env_seed, "observation": env.observe(index)}
        )
    return env, agents, resolved_id


def drive_game(
    env: Any,
    agents: tuple[Any, ...],
    config: SystemConfig,
    collector: EpisodeCollector,
    *,
    episode_id: str,
    seed: int,
) -> CanonicalEpisode:
    """Play ``env`` to the end with ``agents`` and record every valid step."""

    modes = sorted({emotion_mode(agent) for agent in agents})
    collector.start_episode(
        env,
        episode_id=episode_id,
        seed=int(seed),
        metadata={"game_seed": int(seed), "emotion_mode": ",".join(modes)},
    )
    guard = 0
    while not (env.state.terminated or env.state.truncated):
        guard += 1
        if guard > _GUARD_STEPS:
            raise RuntimeError(f"game {episode_id} did not end within {_GUARD_STEPS} steps")
        actor = int(env.state.agent_selection)
        if not bool(env.state.alive[actor]):
            env.step(None)  # dead-agent cleanup, same as GameService.step
            continue
        # (1) non-actors observe the boundary (GameService.step order).
        for index, agent in enumerate(agents):
            if (
                index != actor
                and isinstance(agent, DummyAgent)
                and agent.uses_emotion_formula
                and bool(env.state.alive[index])
            ):
                agent.observe_public_boundary(env.observe(index))
        # (2) the actor decides; act() advances its own emotion first.
        observation = env.observe()
        mask = np.asarray(env.action_mask, dtype=np.uint8).copy()
        chosen = agents[actor].act(observation, mask)
        # (3) read every living dummy's pre-action label.
        labels = emotion_labels(agents, env, config, episode_id=episode_id)
        collector.before_action(
            observation,
            action_mask=mask,
            emotion_record={"labels": labels},
            actor_idx=actor,
        )
        # (4) apply and record.
        env.step(env.validate_action(chosen))
        collector.record_step(env.last_result)
    if collector.active_episode_id is not None:
        collector.finish_episode()
    return collector.episodes[-1]


def play_and_collect_game(
    config: SystemConfig,
    collector: EpisodeCollector,
    *,
    seed: int,
    action_mode: str = "deterministic",
    episode_id: str | None = None,
) -> tuple[CanonicalEpisode, Any, tuple[Any, ...]]:
    """Play one all-dummy game and record it. Returns (episode, env, agents)."""

    env, agents, resolved_id = build_game(
        config, seed=seed, action_mode=action_mode, episode_id=episode_id
    )
    episode = drive_game(env, agents, config, collector, episode_id=resolved_id, seed=seed)
    return episode, env, agents


def _check_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{path} exists and is not a folder")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; use a new or empty folder")


def collect_games(
    config: SystemConfig,
    games: int,
    output_dir: str | Path,
    *,
    base_seed: int = 0,
    action_mode: str = "deterministic",
    progress: Callable[[int, int], None] | None = None,
    keep_episodes: bool = False,
    shard_games: int = 20,
    episode_id_prefix: str | None = None,
) -> dict[str, Any]:
    """Play ``games`` all-dummy games and write canonical Parquet shards.

    Writes ``collection_summary.json`` next to the shards and returns the same
    summary (plus ``episodes`` when ``keep_episodes`` is true).

    A shard is committed every ``shard_games`` games: the Parquet writer
    re-serializes its whole pending shard on every append, so one large shard
    makes collection time grow quadratically with the number of games.
    """

    games = int(games)
    if games < 1:
        raise ValueError("games must be at least 1")
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")
    if int(shard_games) < 1:
        raise ValueError("shard_games must be at least 1")
    out = Path(output_dir)
    _check_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    reward = RewardCalculator(config.reward_config)
    metadata = {
        "collection_format_version": COLLECTION_FORMAT_VERSION,
        "collection_base_seed": int(base_seed),
        "action_mode": action_mode,
        "reward_config_version": config.reward_config.version,
    }
    writer = ParquetCanonicalShardWriter(out, config, include_private_roles=True)
    wins: Counter = Counter()
    status: Counter = Counter()
    lengths: list[int] = []
    label_schemas: set[str] = set()
    label_dims: set[int] = set()
    modes: set[str] = set()
    kept: list[CanonicalEpisode] = []
    shards: list[Path] = []
    started = time.perf_counter()
    try:
        for index in range(games):
            # One collector per game: a collector retains every completed
            # episode, so reusing one would keep the whole run in memory.
            collector = EpisodeCollector(
                include_private_roles=True, reward_calculator=reward, metadata=metadata
            )
            episode_id = (
                None
                if episode_id_prefix is None
                else f"{episode_id_prefix}-game-{game_seed(base_seed, index):08d}"
            )
            episode, env, agents = play_and_collect_game(
                config,
                collector,
                seed=game_seed(base_seed, index),
                action_mode=action_mode,
                episode_id=episode_id,
            )
            writer.write_episode(episode)
            if (index + 1) % int(shard_games) == 0:
                shards.extend(writer.flush())
            if keep_episodes:
                kept.append(episode)
            status[episode.termination_status] += 1
            lengths.append(int(episode.environment_steps))
            result = env.state.terminal_result
            wins[result.winning_team.name if result is not None else "TRUNCATED"] += 1
            modes.update(emotion_mode(agent) for agent in agents)
            for step in episode.steps:
                for label in step.emotion_labels.values():
                    label_schemas.add(str(label.get("emotion_schema_version")))
                    label_dims.add(len(label["values"]))
            if progress is not None:
                progress(index + 1, games)
    finally:
        shards.extend(writer.finalize())
    label_dim = next(iter(label_dims)) if len(label_dims) == 1 else sorted(label_dims) or None
    summary: dict[str, Any] = {
        "collection_format_version": COLLECTION_FORMAT_VERSION,
        "games": games,
        "base_seed": int(base_seed),
        "seeds": [game_seed(base_seed, 0), game_seed(base_seed, games - 1)],
        "action_mode": action_mode,
        "emotion_mode": sorted(modes),
        "emotion_label_schema": sorted(label_schemas),
        "emotion_label_dimension": label_dim,
        "emotion_label_names": (
            list(LABEL_NAMES) if FORMULA_LABEL_SCHEMA in label_schemas else None
        ),
        "termination_status": dict(sorted(status.items())),
        "win_rate": {key: round(value / games, 4) for key, value in sorted(wins.items())},
        "mean_steps": round(float(np.mean(lengths)), 2),
        "total_steps": int(sum(lengths)),
        "shards": [path.name for path in shards],
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if keep_episodes:
        summary["episodes"] = kept
    return summary
