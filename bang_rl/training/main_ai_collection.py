"""Collect "main1+dummy4" games: one main AI among four dummies.

This is the deployment situation the main AI infers in, so it is the data the
inference modules should also learn from. A ``.bangmodel`` plays one seat (the
seat rotates, ``game_index % 5``) and four DummyAgents fill the others, built and
driven exactly like in evaluation and main-AI training
(:func:`bang_rl.evaluation.protocol.build_dummy_opponents`,
:func:`bang_rl.evaluation.protocol.observe_boundaries`).

Viewpoint
---------
Only the main seat's rows are stored (``viewpoint = "main_only"``): one row per
main-AI decision, with the observation the main AI saw, the nine-entry emotion
label of every living dummy and all role labels. The main AI's own label is never
a target (it is the viewer), exactly as in deployment. The collector requires
contiguous environment steps, so the whole game is recorded and then reduced to
the main seat's steps; the canonical shards, tensor cache and trainers need no
change because every row already names its viewer (the acting seat).

Seeds
-----
Game seeds are derived from the base seed and skip the evaluation deal range, so
a collected game is never an evaluation deal. Episode ids contain the dataset id,
so datasets of different rounds never collide.
"""
from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.artifact_builders import dataset_info, dataset_summary, info_ref
from bang_rl.artifact_info import (
    KIND_MAIN_AI,
    ArtifactInfo,
    dump_json_with_info,
    write_info,
)
from bang_rl.config import SystemConfig
from bang_rl.data.collector import CanonicalEpisode, EpisodeCollector
from bang_rl.data.game_collection import (
    ACTION_MODES,
    COLLECTION_FORMAT_VERSION,
    FORMULA_LABEL_SCHEMA,
    emotion_labels,
)
from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
from bang_rl.env import BangEnvironment
from bang_rl.env.types import TerminationStatus
from bang_rl.evaluation.protocol import (
    SEMANTIC_ROLE_NAMES,
    SEMANTIC_ROLE_OF,
    build_dummy_opponents,
    observe_boundaries,
)
from bang_rl.runtime.seed import derive_seed

from .main_ai_training import (
    MAIN_AI_EVAL_BASE_SEED,
    PolicyModules,
    _EVAL_SEED_RESERVED,
    _SEED_MODULUS,
    _selected_player,
    main_seat_for_game,
)

PLAYERS_DUMMY_FIVE = "dummy5"
PLAYERS_MAIN_DUMMY_FOUR = "main1-dummy4"
COLLECTION_PLAYER_MODES = (PLAYERS_DUMMY_FIVE, PLAYERS_MAIN_DUMMY_FOUR)
VIEWPOINT_MAIN_ONLY = "main_only"
MAIN_ACTION_MODES = ACTION_MODES  # ("deterministic", "stochastic")
_GUARD_STEPS = 10_000


def collection_game_seed(base_seed: int, game_index: int) -> int:
    """Environment seed of one collected game; never inside the evaluation deal range."""

    attempt = 0
    while True:
        seed = derive_seed(base_seed, "main-ai-collection", "game", int(game_index), attempt) % _SEED_MODULUS
        if not MAIN_AI_EVAL_BASE_SEED <= seed < MAIN_AI_EVAL_BASE_SEED + _EVAL_SEED_RESERVED:
            return seed
        attempt += 1


def policy_from_artifact(artifact: Any) -> PolicyModules:
    """The frozen, inference-only policy of a loaded ``.bangmodel``."""

    artifact.ppo.eval()
    return PolicyModules(artifact.intent, artifact.emotion, artifact.role_fusion, artifact.preprocessor, artifact.ppo)


def play_collection_game(
    config: SystemConfig,
    policy: PolicyModules,
    *,
    seed: int,
    seat: int,
    episode_id: str,
    action_mode: str = "deterministic",
    generator: torch.Generator | None = None,
    environment_factory: Callable[[str, int, Any], Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[CanonicalEpisode, dict[str, Any]]:
    """Play one main1+dummy4 game; return the main-seat-only episode and a summary.

    The dummies observe the step boundary before every decision (as in
    evaluation) and their pre-action emotion labels are read, never updated.
    """

    if action_mode not in MAIN_ACTION_MODES:
        raise ValueError(f"action_mode must be one of {MAIN_ACTION_MODES}")
    deterministic = action_mode == "deterministic"
    factory = environment_factory or (lambda eid, s, env_config: BangEnvironment(episode_id=eid, seed=s, config=env_config))
    env = factory(episode_id, seed, config.env)
    try:
        env.reset(seed=seed, episode_id=episode_id)
        roles = tuple(int(role) for role in env.state.roles)
        opponents = build_dummy_opponents(seat, seed, config.preset)
        for agent in opponents.values():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})
        seat_agents = tuple(opponents.get(index) for index in range(5))  # None at the main seat
        collector = EpisodeCollector(
            include_private_roles=True,
            metadata={
                "collection_format_version": COLLECTION_FORMAT_VERSION,
                "game_seed": int(seed),
                "reward_config_version": config.reward_config.version,
                "players": PLAYERS_MAIN_DUMMY_FOUR,
                "viewpoint": VIEWPOINT_MAIN_ONLY,
                "viewpoint_seat": int(seat),
                "main_action_mode": action_mode,
                **dict(extra_metadata or {}),
            },
        )
        collector.start_episode(env, episode_id=episode_id, seed=int(seed))
        guard = 0
        while not (env.state.terminated or env.state.truncated):
            guard += 1
            if guard > _GUARD_STEPS:
                raise RuntimeError(f"game {episode_id} did not end within {_GUARD_STEPS} steps")
            actor = _selected_player(env)
            if not bool(env.state.alive[actor]):
                env.step(None)  # dead-agent cleanup, as in drive_game
                continue
            observe_boundaries(opponents, env, actor)
            observation = env.observe(actor)
            mask = np.asarray(env.action_mask, dtype=np.uint8).copy()
            if actor == seat:
                inputs = policy.inputs(observation, seat)
                with torch.inference_mode():
                    chosen = policy.ppo.act(
                        inputs.ppo_inputs(), inputs.action_mask, deterministic=deterministic, generator=generator
                    )
                action = int(chosen.action.reshape(-1)[0])
                labels = emotion_labels(seat_agents, env, config, episode_id=episode_id)
                collector.before_action(
                    observation, action_mask=mask, emotion_record={"labels": labels}, actor_idx=actor
                )
            else:
                action = opponents[actor].act(observation, mask)
                collector.before_action(observation, action_mask=mask, actor_idx=actor)
            if mask[action] != 1:
                raise RuntimeError("collection agent produced an action rejected by the public action mask")
            env.step(env.validate_action(action))
            collector.record_step(env.last_result)
        if collector.active_episode_id is not None:
            collector.finish_episode()
        for agent in opponents.values():
            agent.on_episode_end(
                {"episode_id": episode_id, "terminated": env.state.terminated, "truncated": env.state.truncated}
            )
        full = collector.episodes[-1]
        main_steps = tuple(step for step in full.steps if int(step.actor_idx) == seat)
        metadata = dict(full.metadata)
        metadata.update(
            {
                "environment_steps_total": len(full.steps),
                "main_decisions": len(main_steps),
                "main_role": SEMANTIC_ROLE_NAMES[SEMANTIC_ROLE_OF[roles[seat]]],
            }
        )
        episode = dataclasses.replace(full, steps=main_steps, metadata=metadata)
        result = env.state.terminal_result
        normal = full.termination_status == TerminationStatus.NORMAL.value
        summary = {
            "episode_id": episode_id,
            "seed": int(seed),
            "seat": int(seat),
            "role": metadata["main_role"],
            "termination_status": full.termination_status,
            "environment_steps": len(full.steps),
            "main_decisions": len(main_steps),
            "won": bool(result is not None and seat in result.winners) if normal else None,
        }
        return episode, summary
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def collect_main_ai_games(
    config: SystemConfig,
    policy: PolicyModules,
    games: int,
    output_dir: str | Path,
    *,
    base_seed: int = 0,
    action_mode: str = "deterministic",
    first_game_index: int = 0,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    shard_games: int = 20,
    info: ArtifactInfo | None = None,
    source_model_path: str | Path | None = None,
    episode_id_prefix: str | None = None,
    created_by: str = "collect_data",
    environment_factory: Callable[[str, int, Any], Any] | None = None,
) -> dict[str, Any]:
    """Play ``games`` main1+dummy4 games and write main-seat-only canonical shards.

    Writes ``collection_summary.json`` and the folder's ABOUT.md (an
    :class:`ArtifactInfo` of kind ``dataset``, ``players = "main1-dummy4"``, with
    the model, action mode, seeds and viewpoint recorded). Returns the summary;
    ``summary["stopped"]`` is true when ``should_stop`` ended the run early (the
    games played so far are kept).
    """

    games = int(games)
    if games < 1:
        raise ValueError("games must be at least 1")
    if action_mode not in MAIN_ACTION_MODES:
        raise ValueError(f"action_mode must be one of {MAIN_ACTION_MODES}")
    if int(shard_games) < 1:
        raise ValueError("shard_games must be at least 1")
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not an empty folder; use a new folder")
    out.mkdir(parents=True, exist_ok=True)

    identifier = info.id if info is not None else str(uuid4())
    prefix = episode_id_prefix or f"main1-{identifier}"
    writer = ParquetCanonicalShardWriter(out, config, include_private_roles=True)
    status: Counter = Counter()
    seats: Counter = Counter()
    roles: Counter = Counter()
    wins = normal = rows = environment_steps = played = 0
    seeds: list[int] = []
    label_schemas: set[str] = set()
    shards: list[Path] = []
    stopped = False
    started = time.perf_counter()
    try:
        for index in range(games):
            if should_stop is not None and should_stop():
                stopped = True
                break
            game_index = int(first_game_index) + index
            seat = main_seat_for_game(game_index)
            seed = collection_game_seed(base_seed, game_index)
            generator = None
            if action_mode == "stochastic":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(derive_seed(base_seed, "main-ai-collection", "action", game_index) & 0x7FFFFFFFFFFFFFFF)
            episode, summary = play_collection_game(
                config,
                policy,
                seed=seed,
                seat=seat,
                episode_id=f"{prefix}-g{game_index:06d}",
                action_mode=action_mode,
                generator=generator,
                environment_factory=environment_factory,
                extra_metadata={"collection_base_seed": int(base_seed), "game_index": game_index},
            )
            played += 1
            seeds.append(seed)
            status[summary["termination_status"]] += 1
            seats[summary["seat"]] += 1
            environment_steps += int(summary["environment_steps"])
            if episode.steps:
                writer.write_episode(episode)
                rows += len(episode.steps)
                for step in episode.steps:
                    for label in step.emotion_labels.values():
                        label_schemas.add(str(label.get("emotion_schema_version")))
            if summary["won"] is not None:
                normal += 1
                wins += int(summary["won"])
                roles[summary["role"]] += 1
            if played % int(shard_games) == 0:
                shards.extend(writer.flush())
            if progress is not None:
                progress(played, games)
    finally:
        shards.extend(writer.finalize())

    summary_out: dict[str, Any] = {
        "collection_format_version": COLLECTION_FORMAT_VERSION,
        "players": PLAYERS_MAIN_DUMMY_FOUR,
        "viewpoint": VIEWPOINT_MAIN_ONLY,
        "games": played,
        "requested_games": games,
        "stopped": stopped,
        "base_seed": int(base_seed),
        "seed_rule": "derive_seed(base_seed, 'main-ai-collection', 'game', index) % 2**31, evaluation deal range skipped",
        "seeds": [min(seeds), max(seeds)] if seeds else [],
        "action_mode": action_mode,
        "rows": int(rows),
        "main_decisions": int(rows),
        "total_environment_steps": int(environment_steps),
        "total_steps": int(environment_steps),
        "main_seat_games": {str(seat): int(seats.get(seat, 0)) for seat in range(5)},
        "main_role_games": dict(sorted(roles.items())),
        "main_win_rate": (wins / normal) if normal else None,
        "win_rate": (wins / normal) if normal else None,
        "termination_status": dict(sorted(status.items())),
        "emotion_label_schema": sorted(label_schemas),
        "emotion_label_dimension": len(LABEL_NAMES) if FORMULA_LABEL_SCHEMA in label_schemas else None,
        "mean_steps": round(environment_steps / played, 2) if played else 0.0,
        "shards": [path.name for path in shards],
        "seconds": round(time.perf_counter() - started, 1),
    }
    dataset_meta = info
    if dataset_meta is None:
        dataset_meta = dataset_info(
            id=identifier,
            games=played,
            base_seed=base_seed,
            created_by=created_by,
            players=PLAYERS_MAIN_DUMMY_FOUR,
            action_mode=action_mode,
            extra_conditions=_dataset_conditions(action_mode, base_seed),
        )
        if source_model_path is not None:
            dataset_meta.sources = [info_ref(source_model_path, KIND_MAIN_AI)]
    dataset_meta.summary = {
        **dataset_summary(summary_out),
        "rows": int(rows),
        "main_decisions": int(rows),
        "main_win_rate": summary_out["main_win_rate"],
        "stopped": stopped,
    }
    dataset_meta.name = out.name
    (out / "collection_summary.json").write_text(
        dump_json_with_info(summary_out, dataset_meta, indent=2), encoding="utf-8"
    )
    write_info(out, dataset_meta)
    return summary_out


def _dataset_conditions(action_mode: str, base_seed: int) -> dict[str, Any]:
    return {
        "viewpoint": VIEWPOINT_MAIN_ONLY,
        "main_action_mode": action_mode,
        "main_seat_rule": "game_index % 5",
        "eval_seed_range_excluded": [MAIN_AI_EVAL_BASE_SEED, MAIN_AI_EVAL_BASE_SEED + _EVAL_SEED_RESERVED],
        "base_seed": int(base_seed),
    }


def main_ai_dataset_info(
    *,
    games: int,
    base_seed: int,
    created_by: str,
    created_at: str | None,
    action_mode: str,
    model_path: str | Path | None,
    identifier: str | None = None,
    extra_conditions: Mapping[str, Any] | None = None,
) -> ArtifactInfo:
    """Info (naming rule, players ``main1-dummy4``, source = the ``.bangmodel``) of a collected dataset."""

    info = dataset_info(
        id=identifier or str(uuid4()),
        games=games,
        base_seed=base_seed,
        created_by=created_by,
        created_at=created_at,
        players=PLAYERS_MAIN_DUMMY_FOUR,
        action_mode=action_mode,
        extra_conditions={**_dataset_conditions(action_mode, base_seed), **dict(extra_conditions or {})},
    )
    if model_path is not None:
        info.sources = [info_ref(model_path, KIND_MAIN_AI)]
    return info


__all__ = [
    "COLLECTION_PLAYER_MODES",
    "MAIN_ACTION_MODES",
    "PLAYERS_DUMMY_FIVE",
    "PLAYERS_MAIN_DUMMY_FOUR",
    "VIEWPOINT_MAIN_ONLY",
    "collect_main_ai_games",
    "collection_game_seed",
    "main_ai_dataset_info",
    "play_collection_game",
    "policy_from_artifact",
]
