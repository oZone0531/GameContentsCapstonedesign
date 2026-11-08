"""Baseline seat agents evaluated under the same games as the main agent.

The main agent's evaluation (``PrimaryEvaluationProtocol`` /
``CustomEvaluationProtocol``) puts the main agent in one seat against four
DummyAgents. This module plays exactly the same games with a *baseline* in that
seat instead, so a main-agent win rate can be read against reference points:

* ``random`` - uniformly random legal actions (floor);
* ``dummy``  - the rule-based dummy itself (what the main agent must beat:
  a main agent at this level has learned nothing beyond the dummies);
* ``hint:Q`` / ``oracle:A`` - a dummy given privileged role information
  (see ``belief_source``). These cheat and are reference ceilings only.

Game construction mirrors ``PrimaryEvaluationProtocol._run_game_impl``:
seed = ``base_seed + index``, seat = ``seats[index % len(seats)]``, environment
``BangEnvironment(seed=seed)``, opponent ``i`` a DummyAgent with seed
``seed + i + 1``. Roles depend only on the seed, so game ``i`` deals the same
roles to every evaluated agent. With the ``dummy`` baseline, game ``i`` is
exactly the all-dummy game the main agent replaces one dummy in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from bang_rl.agents.dummy_agent import DummyAgent
from bang_rl.agents.dummy_agent.belief_source import make_belief_source, parse_belief_spec
from bang_rl.agents.random_agent import RandomAgent
from bang_rl.config import SystemConfig
from bang_rl.env import BangEnvironment
from bang_rl.env.types import TerminationStatus

from .protocol import (
    PRIMARY_SEATS,
    SEMANTIC_ROLE_NAMES,
    SEMANTIC_ROLE_OF,
    _wilson_interval,
    observe_boundaries,
)

SeatAgentFactory = Callable[[int, int, tuple[int, ...]], Any]

__all__ = [
    "BASELINE_NAMES",
    "SeatAgentFactory",
    "baseline_factory",
    "run_seat_evaluation",
    "summarize_protocol_report",
]

BASELINE_NAMES = ("random", "dummy")


def _dummy(config: SystemConfig, seat: int, seed: int) -> DummyAgent:
    return DummyAgent(
        agent_id=f"player_{seat}",
        player_idx=seat,
        seed=seed + seat + 1,  # the seed this seat's dummy has in the protocol
        policy_config=config.preset.intent_policy,
        emotion_config=config.preset.emotion,
        action_mode="deterministic",
    )


def baseline_factory(name: str, config: SystemConfig) -> SeatAgentFactory:
    """Seat-agent factory for a named baseline (see module docstring)."""

    key = str(name).strip().lower()
    if key == "random":
        return lambda seat, seed, roles: RandomAgent(agent_id=f"player_{seat}", seed=seed + seat + 1)
    if key == "dummy":
        return lambda seat, seed, roles: _dummy(config, seat, seed)
    kind, _ = parse_belief_spec(key)  # raises for unknown names
    if kind in ("hint", "oracle"):
        def factory(seat: int, seed: int, roles: tuple[int, ...]) -> DummyAgent:
            agent = _dummy(config, seat, seed)
            agent.set_belief_source(make_belief_source(key, roles, seed=seed * 7919 + seat))
            return agent

        return factory
    raise ValueError(f"unknown baseline {name!r}")


def _rate(wins: int, games: int) -> float | None:
    return None if games == 0 else wins / games


def _ci(wins: int, games: int) -> dict[str, float] | None:
    interval = _wilson_interval(wins, games)
    return interval.to_dict() if hasattr(interval, "to_dict") else None


def run_seat_evaluation(
    factory: SeatAgentFactory,
    config: SystemConfig,
    *,
    games: int = 500,
    base_seed: int = 0,
    seats: Sequence[int] = PRIMARY_SEATS,
    label: str = "baseline",
) -> dict[str, Any]:
    """Play ``games`` evaluation games with ``factory``'s agent in the seat."""

    games = int(games)
    if games < 1:
        raise ValueError("games must be at least 1")
    seats = tuple(int(seat) for seat in seats)
    if not seats or any(not 0 <= seat < 5 for seat in seats):
        raise ValueError("seats must be player indices in [0, 4]")
    per_role = {name: {"games": 0, "wins": 0} for name in SEMANTIC_ROLE_NAMES}
    normal = wins = truncated = friendly = bangs = 0
    lengths: list[int] = []
    game_rows: list[dict[str, Any]] = []
    for index in range(games):
        seed = int(base_seed) + index
        seat = seats[index % len(seats)]
        episode_id = f"{label}-game-{index:03d}"
        env = BangEnvironment(episode_id=episode_id, seed=seed, config=config.env)
        env.reset(seed=seed, episode_id=episode_id)
        roles = tuple(int(role) for role in env.state.roles)
        agents: dict[int, Any] = {
            index_: _dummy(config, index_, seed) for index_ in range(5) if index_ != seat
        }
        agents[seat] = factory(seat, seed, roles)
        for player, agent in agents.items():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})
        while not env.state.terminated and not env.state.truncated:
            selected = int(str(env.agent_selection).removeprefix("player_"))
            observe_boundaries({k: v for k, v in agents.items() if k != seat}, env, selected)
            observation = env.observe(selected)
            mask = np.asarray(observation["action_mask"], dtype=np.uint8)
            action = int(agents[selected].act(observation, mask))
            if mask[action] != 1:
                raise RuntimeError("seat agent produced an action rejected by its action mask")
            if selected == seat and action > 0:
                bangs += 1
                target_role = SEMANTIC_ROLE_OF[roles[action - 1]]
                own_role = SEMANTIC_ROLE_OF[roles[seat]]
                same_side = {0: 0, 1: 0, 2: 1, 3: 2}
                friendly += int(own_role != 3 and same_side[own_role] == same_side[target_role])
            env.step(action)
        for agent in agents.values():
            agent.on_episode_end({"episode_id": episode_id, "terminated": env.state.terminated, "truncated": env.state.truncated})
        lengths.append(int(env.environment_step))
        role_name = SEMANTIC_ROLE_NAMES[SEMANTIC_ROLE_OF[roles[seat]]]
        if env.state.termination_status.value != TerminationStatus.NORMAL.value:
            truncated += 1
            game_rows.append({"seed": seed, "seat": seat, "role": role_name, "won": None})
            continue
        result = env.state.terminal_result
        won = bool(result is not None and seat in result.winners)
        game_rows.append({"seed": seed, "seat": seat, "role": role_name, "won": won})
        normal += 1
        wins += int(won)
        per_role[role_name]["games"] += 1
        per_role[role_name]["wins"] += int(won)
    return {
        "agent": label,
        "games": games,
        "base_seed": int(base_seed),
        "seats": list(seats),
        "normal_games": normal,
        "truncated_games": truncated,
        "win_rate": _rate(wins, normal),
        "win_rate_ci": _ci(wins, normal),
        "by_role": {
            name: {**counts, "win_rate": _rate(counts["wins"], counts["games"]), "win_rate_ci": _ci(counts["wins"], counts["games"])}
            for name, counts in per_role.items()
        },
        "seat_friendly_fire": _rate(friendly, bangs),
        "mean_steps": float(np.mean(lengths)),
        "game_results": game_rows,
    }


def summarize_protocol_report(report: Any, label: str = "main") -> dict[str, Any]:
    """Main-agent EvaluationReport in the same shape as ``run_seat_evaluation``."""

    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    by_canonical = data.get("role_specific_results", {})
    per_role = {name: {"games": 0, "wins": 0} for name in SEMANTIC_ROLE_NAMES}
    for canonical, counts in by_canonical.items():
        name = {"SHERIFF": "sheriff", "DEPUTY": "deputy", "OUTLAW_A": "outlaw", "OUTLAW_B": "outlaw", "TRAITOR": "traitor"}[canonical]
        per_role[name]["games"] += int(counts.get("normal_games", 0))
        per_role[name]["wins"] += int(counts.get("normal_wins", 0))
    normal, wins = int(data["normal_games"]), int(data["normal_wins"])
    return {
        "agent": label,
        "games": int(data["total_games"]),
        "normal_games": normal,
        "truncated_games": int(data["truncated_games"]),
        "win_rate": _rate(wins, normal),
        "win_rate_ci": _ci(wins, normal),
        "by_role": {
            name: {**counts, "win_rate": _rate(counts["wins"], counts["games"]), "win_rate_ci": _ci(counts["wins"], counts["games"])}
            for name, counts in per_role.items()
        },
        "role_accuracy": data.get("role_classification_accuracy"),
        "hidden_role_accuracy": data.get("hidden_role_accuracy"),
        "emotion_mse": data.get("estimated_emotion_mse"),
    }
