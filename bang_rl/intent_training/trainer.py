"""Role-balanced, terminal-outcome-only intent learning and evaluation.

Four separate policies are trained, one agent/role per game; the other four
players are a reproducible mixture of the existing rule IntentModule (with
NO emotion) and RandomAgent. "100 episodes" means 100 episodes *per role*
(400 total). Evaluations are extra, seeded and never used for gradient updates.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from bang_rl.agents import RandomAgent
from bang_rl.agents.base import BaseAgent
from bang_rl.agents.dummy_agent.intent_policy import ActionSelector, IntentModule
from bang_rl.agents.dummy_agent.role_belief import RoleBeliefEstimator
from bang_rl.env.environment import BangEnvironment
from bang_rl.env.state import EnvironmentState
from bang_rl.env.types import Role

from .policy import LearnedIntentAgent, LearnedIntentPolicy, ROLE_KEYS, encode_public, role_key


@dataclass(frozen=True)
class IntentTrainingConfig:
    episodes_per_role: int = 100
    max_steps: int = 200
    seed: int = 12345
    random_opponent_probability: float = 0.35
    learning_rate: float = 0.0006
    batch_games: int = 16
    eval_games_per_role: int = 20
    checkpoint_every: int = 500
    output_path: str = "bang_rl/models/intent/trained.intentmodel"
    initial_model_path: str | None = None

    def __post_init__(self) -> None:
        if self.episodes_per_role < 1 or self.max_steps < 2:
            raise ValueError("episodes_per_role >= 1 and max_steps >= 2 required")
        if not 0 <= self.random_opponent_probability <= 1:
            raise ValueError("random_opponent_probability must be within [0,1]")
        if not 0 < self.learning_rate <= 0.1 or self.batch_games < 1:
            raise ValueError("invalid optimizer hyperparameters")
        if self.eval_games_per_role < 1 or self.checkpoint_every < 0:
            raise ValueError("eval_games_per_role >= 1 and checkpoint_every >= 0 required")
        if not self.output_path.lower().endswith(".intentmodel"):
            raise ValueError("output_path must end in .intentmodel")
        if self.initial_model_path:
            if not self.initial_model_path.lower().endswith(".intentmodel"):
                raise ValueError("initial_model_path must end in .intentmodel")
            if Path(self.initial_model_path).resolve() == Path(self.output_path).resolve():
                raise ValueError("Choose a different output path to preserve the original model")


class RuleIntentAgent(BaseAgent):
    """Unmodified rule IntentModule, but no EmotionModule and no emotion factor."""

    def __init__(self, player_idx: int, seed: int, *, policy_config: Any = None) -> None:
        super().__init__(f"player_{player_idx}", seed=seed)
        self.estimator = RoleBeliefEstimator(player_idx=player_idx)
        self.intent = IntentModule(player_idx=player_idx, config=policy_config)
        self.rng = np.random.default_rng(seed)

    def on_episode_start(self, context: Any) -> None:
        super().on_episode_start(context)
        self.estimator.reset()

    def act(self, observation: Any, action_mask: Any) -> int:
        mask = self._validate_act_inputs(observation, action_mask)
        beliefs = self.estimator.update(observation)
        scores = self.intent.decision(observation, beliefs).scores
        action = ActionSelector.select_with_info(scores, mask, rng=self.rng).action
        return self._validate_selected_action(action, mask)


class _TrainingAgent(LearnedIntentAgent):
    def __init__(self, policy: LearnedIntentPolicy, *, player_idx: int, seed: int) -> None:
        super().__init__(policy, player_idx=player_idx, seed=seed, stochastic=True)
        self.trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

    def act(self, observation: Any, action_mask: Any) -> int:
        mask = self._validate_act_inputs(observation, action_mask)
        beliefs = self.belief_estimator.update(observation)
        features = encode_public(observation, beliefs)
        key = role_key(int(np.asarray(observation["self_role"]).reshape(-1)[0]))
        with torch.no_grad():
            logits, _ = self.policy.networks[key](torch.from_numpy(features))
            logits = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -1e9)
            probs = torch.softmax(logits, dim=-1).numpy()
        action = int(self.rng.choice(6, p=probs))
        self.trajectory.append((features, mask.copy(), action))
        return self._validate_selected_action(action, mask)


def _episode_seed(base_seed: int, role_index: int, episode: int, *, eval_mode: bool = False) -> int:

    return int((base_seed + 3456789 * int(eval_mode) + role_index * 102013 + episode * 7919) % (2**31 - 1))


def _target_seat(state: EnvironmentState, key: str, episode: int) -> int:
    if key == "outlaw":
        matches = [i for i, r in enumerate(state.roles) if r in (Role.OUTLAW_A, Role.OUTLAW_B)]
        return matches[episode % 2]
    role = {"sheriff": Role.SHERIFF, "deputy": Role.DEPUTY, "traitor": Role.TRAITOR}[key]
    return state.roles.index(role)


def play_intent_game(
    *, policy: LearnedIntentPolicy | None, role: str, game_seed: int,
    episode: int, max_steps: int, random_probability: float,
    training: bool = False, rule_policy_config: Any = None,
) -> tuple[bool | None, list[tuple[np.ndarray, np.ndarray, int]], dict[str, Any]]:
    """Run a *real* game. True/False is target-role victory/defeat; None is truncation.

    The terminal winning team is read AFTER the game ends solely by the trainer;
    it is never passed to the decision policy (only public observations are).
    """
    if role not in ROLE_KEYS:
        raise ValueError("unknown role")
    state = EnvironmentState.initial(f"intent-{game_seed}", game_seed)
    target = _target_seat(state, role, episode)
    env = BangEnvironment(state=state, max_episode_steps=max_steps)
    rng = np.random.default_rng(game_seed ^ 0x1950A)
    opponents: list[Any] = []
    trainee: _TrainingAgent | None = None
    for i in range(5):
        if i == target:
            if policy is None:
                agent: BaseAgent = RuleIntentAgent(i, seed=game_seed + i, policy_config=rule_policy_config)
            elif training:
                trainee = _TrainingAgent(policy, player_idx=i, seed=game_seed + i)
                agent = trainee
            else:
                agent = LearnedIntentAgent(policy, player_idx=i, seed=game_seed + i)
        elif rng.random() < random_probability:
            agent = RandomAgent(agent_id=f"player_{i}", seed=game_seed + i)
        else:
            agent = RuleIntentAgent(i, seed=game_seed + i, policy_config=rule_policy_config)
        opponents.append(agent)
    for i, agent in enumerate(opponents):
        agent.on_episode_start({"episode_id": state.episode_id, "seed": game_seed, "observation": env.observe(i)})
    while not env.state.terminated and not env.state.truncated:
        actor = int(env.state.agent_selection)
        observation = env.observe(actor)
        action = opponents[actor].act(observation, env.action_mask.copy())
        env.step(action)
    result = env.state.terminal_result
    win: bool | None = None if result is None else bool(target in result.winners)
    return win, ([] if trainee is None else trainee.trajectory), {
        "steps": int(env.environment_step),
        "winner_team": None if result is None or result.winning_team is None else result.winning_team.name,
        "target_seat": target,
        "target_role": role,
    }


def evaluate_intent(
    policy: LearnedIntentPolicy | None,
    *, config: IntentTrainingConfig,
    callback: Callable[[dict[str, Any]], None] | None = None,
    stop_event: threading.Event | None = None,
    rule_policy_config: Any = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate identical role/seed/opponent draws, never update network weights."""
    results: dict[str, dict[str, Any]] = {}
    for role_index, role in enumerate(ROLE_KEYS):
        wins = losses = truncated = 0
        steps = 0
        for episode in range(config.eval_games_per_role):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("intent evaluation cancelled")
            seed = _episode_seed(config.seed, role_index, episode, eval_mode=True)
            win, _, details = play_intent_game(
                policy=policy, role=role, game_seed=seed, episode=episode,
                max_steps=config.max_steps,
                random_probability=config.random_opponent_probability,
                rule_policy_config=rule_policy_config,
            )
            wins += int(win is True)
            losses += int(win is False)
            truncated += int(win is None)
            steps += details["steps"]
        normal = wins + losses
        results[role] = {
            "wins": wins, "losses": losses, "truncated": truncated,
            "normal_games": normal,
            "win_rate": None if normal == 0 else wins / normal,
            "mean_steps": steps / config.eval_games_per_role,
        }
        if callback:
            callback({"phase": "evaluation", "role": role, "results": results[role]})
    return results



def evaluate_five_agents(
    policy: LearnedIntentPolicy | None,
    *, config: IntentTrainingConfig,
    rule_policy_config: Any = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Extra benchmark: five learned agents against each other, or five rule agents.

    This is the direct measurement of the all-AI Traitor team win rate. It is
    evaluated on separate fixed seeds and never contributes training gradients.
    """
    games = max(40, config.eval_games_per_role * 4)
    wins: dict[str, int] = {"SHERIFF_TEAM": 0, "OUTLAW_TEAM": 0, "TRAITOR_TEAM": 0}
    truncations = 0
    for i in range(games):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("five-player evaluation cancelled")
        seed = int((config.seed + 7_222_111 + i * 15427) % (2**31 - 1))
        state = EnvironmentState.initial(f"intent-five-{seed}", seed)
        env = BangEnvironment(state=state, max_episode_steps=config.max_steps)
        agents: list[BaseAgent] = []
        for seat in range(5):
            if policy is None:
                agent: BaseAgent = RuleIntentAgent(
                    seat, seed + seat, policy_config=rule_policy_config
                )
            else:
                agent = LearnedIntentAgent(policy, player_idx=seat, seed=seed + seat)
            agent.on_episode_start({"episode_id": state.episode_id, "seed": seed,
                                    "observation": env.observe(seat)})
            agents.append(agent)
        while not env.state.terminated and not env.state.truncated:
            actor = int(env.state.agent_selection)
            action = agents[actor].act(env.observe(actor), env.action_mask.copy())
            env.step(action)
        result = env.state.terminal_result
        if result is None or result.winning_team is None:
            truncations += 1
        else:
            wins[result.winning_team.name] += 1
    normal = sum(wins.values())
    return {
        "games": games, "normal_games": normal, "truncated": truncations,
        "winning_teams": wins,
        "team_win_rates": {
            team: None if normal == 0 else count / normal
            for team, count in wins.items()
        },
    }



def _update_policy(
    net: nn.Module, optimizer: torch.optim.Optimizer,
    trajectories: list[tuple[list[tuple[np.ndarray, np.ndarray, int]], float]],
) -> dict[str, float]:
    rows = [(feature, mask, action, reward)
            for episode, reward in trajectories
            for feature, mask, action in episode]
    if not rows:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    xs = torch.from_numpy(np.stack([row[0] for row in rows]))
    masks = torch.from_numpy(np.stack([row[1] for row in rows])).bool()
    actions = torch.as_tensor([row[2] for row in rows], dtype=torch.long)
    returns = torch.as_tensor([row[3] for row in rows], dtype=torch.float32)
    logits, values = net(xs)
    dist = torch.distributions.Categorical(logits=logits.masked_fill(~masks, -1e9))
    advantage = returns - values.detach()


    policy_loss = -(dist.log_prob(actions) * advantage).mean()
    value_loss = torch.square(values - returns).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + 0.5 * value_loss - 0.015 * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()), "entropy": float(entropy.item()),
    }


class IntentTrainer:
    def __init__(self, config: IntentTrainingConfig,
                 *, callback: Callable[[dict[str, Any]], None] | None = None,
                 stop_event: threading.Event | None = None,
                 rule_policy_config: Any = None) -> None:
        self.config = config
        self.callback = callback
        self.stop_event = stop_event or threading.Event()
        self.policy = (LearnedIntentPolicy.load(config.initial_model_path)
                       if config.initial_model_path else LearnedIntentPolicy(seed=config.seed))
        old = self.policy.metadata

        prior = old.get("cumulative_episodes_per_role", old.get("training_config", {}).get("episodes_per_role", 0))
        self.prior_episodes_per_role = int(prior or 0) if config.initial_model_path else 0
        self.parent_training_episodes = int(old.get("cumulative_training_episodes", old.get("training_episodes", 0))) if config.initial_model_path else 0
        self.rule_policy_config = rule_policy_config

    def _emit(self, **fields: Any) -> None:
        if self.callback:
            self.callback(fields)

    def run(self) -> dict[str, Any]:
        """Train exactly episodes_per_role per role, then evaluate/save artifact."""
        cfg = self.config
        self._emit(phase="baseline", message="Evaluating existing rule intent (without emotion)")
        baseline = evaluate_intent(None, config=cfg, stop_event=self.stop_event,
                                   rule_policy_config=self.rule_policy_config)

        initial_evaluation = (evaluate_intent(self.policy, config=cfg, stop_event=self.stop_event,
                                              rule_policy_config=self.rule_policy_config)
                              if cfg.initial_model_path else None)
        optimizers = {role: torch.optim.Adam(self.policy.networks[role].parameters(), lr=cfg.learning_rate)
                      for role in ROLE_KEYS}
        pending: dict[str, list[Any]] = defaultdict(list)
        played: dict[str, int] = {role: 0 for role in ROLE_KEYS}
        wins: dict[str, int] = {role: 0 for role in ROLE_KEYS}
        losses: dict[str, int] = {role: 0 for role in ROLE_KEYS}
        truncated: dict[str, int] = {role: 0 for role in ROLE_KEYS}
        updates: dict[str, int] = {role: 0 for role in ROLE_KEYS}
        recent_losses: dict[str, float] = {}
        total = len(ROLE_KEYS) * cfg.episodes_per_role
        completed = 0
        for episode in range(cfg.episodes_per_role):
            for index, role in enumerate(ROLE_KEYS):
                if self.stop_event.is_set():
                    raise InterruptedError("intent learning cancelled")

                global_episode = self.prior_episodes_per_role + episode
                seed = _episode_seed(cfg.seed, index, global_episode)
                win, trajectory, details = play_intent_game(
                    policy=self.policy, role=role, game_seed=seed, episode=global_episode,
                    max_steps=cfg.max_steps,
                    random_probability=cfg.random_opponent_probability, training=True,
                    rule_policy_config=self.rule_policy_config,
                )
                played[role] += 1
                wins[role] += int(win is True)
                losses[role] += int(win is False)
                truncated[role] += int(win is None)

                if win is not None and trajectory:
                    pending[role].append((trajectory, 1.0 if win else -1.0))
                if len(pending[role]) >= cfg.batch_games:
                    recent_losses[role] = _update_policy(
                        self.policy.networks[role], optimizers[role], pending[role]
                    )["loss"]
                    pending[role].clear()
                    updates[role] += 1
                completed += 1
                if completed % max(1, total // 200) == 0 or completed == total:
                    self._emit(
                        phase="training", completed=completed, total=total,
                        played=dict(played), wins=dict(wins), losses=dict(losses),
                        truncated=dict(truncated), updates=dict(updates),
                        latest_loss=recent_losses.get(role), role=role, steps=details["steps"],
                    )
                if cfg.checkpoint_every > 0 and completed % cfg.checkpoint_every == 0:
                    checkpoint = str(Path(cfg.output_path).with_name(Path(cfg.output_path).stem + ".checkpoint.intentmodel"))
                    self.policy.save(checkpoint, metadata={
                        "partial": True, "played": dict(played), "configuration": asdict(cfg),
                        "training_episodes": self.parent_training_episodes + completed,
                        "cumulative_training_episodes": self.parent_training_episodes + completed,
                        "cumulative_episodes_per_role": self.prior_episodes_per_role + min(played.values()),
                        "warm_started_from": cfg.initial_model_path,
                    })
                    self._emit(phase="checkpoint", path=checkpoint, completed=completed)
        for role in ROLE_KEYS:
            if pending[role]:
                recent_losses[role] = _update_policy(
                    self.policy.networks[role], optimizers[role], pending[role]
                )["loss"]
                updates[role] += 1
        for net in self.policy.networks.values():
            net.eval()
        self._emit(phase="evaluation", message="Evaluating learned intent against the same seeds")
        learned = evaluate_intent(self.policy, config=cfg, stop_event=self.stop_event,
                                  rule_policy_config=self.rule_policy_config)
        self._emit(phase="evaluation", message="Evaluating all five rule agents and all five learned agents")
        five_rule = evaluate_five_agents(
            None, config=cfg, rule_policy_config=self.rule_policy_config,
            stop_event=self.stop_event,
        )
        five_learned = evaluate_five_agents(
            self.policy, config=cfg, rule_policy_config=self.rule_policy_config,
            stop_event=self.stop_event,
        )
        outcome = {
            "all_five_rule_evaluation": five_rule,
            "all_five_learned_evaluation": five_learned,
            "training_config": asdict(cfg),
            "training_episodes": total,
            "initial_model_path": cfg.initial_model_path,
            "initial_model_evaluation": initial_evaluation,
            "prior_episodes_per_role": self.prior_episodes_per_role,
            "cumulative_episodes_per_role": self.prior_episodes_per_role + cfg.episodes_per_role,
            "cumulative_training_episodes": self.parent_training_episodes + total,
            "optimizer_resumed": False,
            "episodes_by_role": played, "wins_by_role": wins,
            "losses_by_role": losses, "truncated_by_role": truncated,
            "updates_by_role": updates,
            "baseline_evaluation": baseline,
            "learned_evaluation": learned,
            "improved_roles": [role for role in ROLE_KEYS if baseline[role]["win_rate"] is not None
                               and learned[role]["win_rate"] is not None
                               and learned[role]["win_rate"] > baseline[role]["win_rate"]],
            "artifact_path": str(Path(cfg.output_path)),
        }
        self.policy.metadata = {
            "algorithm": "episodic_actor_critic_terminal_outcome_only",
            "emotions_enabled": False,
            "roles": list(ROLE_KEYS),
            "training_config": asdict(cfg),
            "training_episodes": self.parent_training_episodes + total,
            "cumulative_training_episodes": self.parent_training_episodes + total,
            "cumulative_episodes_per_role": self.prior_episodes_per_role + cfg.episodes_per_role,
            "warm_started_from": cfg.initial_model_path,
            "optimizer_resumed": False,
        }
        output_path = self.policy.save(cfg.output_path)
        report_path = output_path.with_suffix(".report.json")
        report_path.write_text(json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf-8")
        self._emit(phase="done", path=str(output_path), report_path=str(report_path),
                   results=outcome, completed=total, total=total)
        return outcome


__all__ = ["IntentTrainingConfig", "IntentTrainer", "evaluate_intent", "evaluate_five_agents", "play_intent_game", "RuleIntentAgent"]
