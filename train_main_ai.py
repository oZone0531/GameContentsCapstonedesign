"""Train the main agent's action policy (PPO) on top of frozen inference modules.

Usage (from the repository root, next to the bang_rl folder):
    python train_main_ai.py --phase2 runs/iter1/round_2/model.phase2model --updates 200 --out outputs/main.bangmodel
    python train_main_ai.py --intent bang_rl/models/intent/intent-<job>.phase2model \
        --emotion bang_rl/models/emotion/emotion-<job>.phase2model --updates 200 --out outputs/main.bangmodel
    python evaluate_baselines.py --games 500 --artifact outputs/main.bangmodel --ablations no_intent,no_emotion,no_inference

What it does
------------
The role inference and emotion inference modules are loaded and frozen. Only
the PPO policy learns. Each update:

  1. plays --games-per-update games: the main agent sits in one seat (cycling
     0..4), the other four seats are emotion-formula dummies;
  2. at every main-agent turn builds the policy inputs exactly like
     bang_rl.evaluation.protocol (preprocess -> role inference -> role fusion
     -> emotion inference) and samples an action;
  3. rewards the last main-agent decision of each game with +1 (its team won)
     or -1 (lost); every other decision gets 0 (terminal-only reward);
  4. runs PPOTrainer.update (GAE, clipped surrogate, value loss, entropy).

Training seeds start at --seed (default 1,000,000), far from the evaluation
seeds (0..), so evaluation games are never seen in training. The output is a
normal .bangmodel that evaluate_baselines.py and the GUI evaluation load.

Optional imitation warm start (--bc-games N): before PPO, the policy first
learns to copy the rule-based dummy. N all-dummy games are played; at every
decision the acting dummy's own observation is turned into main-agent inputs
(same inference path) and its chosen action becomes the target. The value head
learns the game outcome (+1 / -1) of the acting seat. PPO then starts from a
policy that already plays like the dummy (about 35% win rate) instead of from
random play.

Progress lines show the win rate of the training games of each update
(stochastic policy) and every --eval-every updates a deterministic check on
held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.agents import DummyAgent  # noqa: E402
from bang_rl.agents.main_agent import PPOModule  # noqa: E402
from bang_rl.config import PPOConfig, ensure_formula_emotion  # noqa: E402
from bang_rl.data.game_collection import DEFAULT_EMOTION_FILE, load_collection_config  # noqa: E402
from bang_rl.env import BangEnvironment  # noqa: E402
from bang_rl.evaluation.protocol import _version_expectations, observe_boundaries  # noqa: E402
from bang_rl.training import PPOTrainer, PPOTransition  # noqa: E402
from bang_rl.training.final_inference_artifact import save_final_inference_artifact  # noqa: E402
from bang_rl.training.inference_artifact import (  # noqa: E402
    load_phase2_inference_artifact,
    save_phase2_inference_artifact,
)

ROLE_NAMES = ("sheriff", "deputy", "outlaw", "outlaw", "traitor")
TARGET_PLAYERS = 4


# ---------------------------------------------------------------------------
# inference modules
# ---------------------------------------------------------------------------
def load_phase2(args: argparse.Namespace, work: Path) -> Any:
    """One Phase-2 artifact from --phase2, or built from --intent + --emotion."""

    if args.phase2:
        return load_phase2_inference_artifact(args.phase2)
    if not (args.intent and args.emotion):
        raise SystemExit("give --phase2, or both --intent and --emotion")
    from bang_rl.training.module_artifacts import load_emotion_for_resume, load_intent_for_resume

    intent, _ = load_intent_for_resume(args.intent)
    emotion, _ = load_emotion_for_resume(args.emotion)
    combined = save_phase2_inference_artifact(work / "combined.phase2model", intent, emotion)
    return load_phase2_inference_artifact(combined)


def policy_inputs(phase2: Any, observation: Any, seat: int) -> dict[str, np.ndarray]:
    """Same inputs as PrimaryEvaluationProtocol._main_action (public data only)."""

    processed = phase2.preprocessor.preprocess(observation)
    with torch.inference_mode():
        raw_intent = phase2.intent.predict_proba(processed)
        estimated_emotion = phase2.emotion.predict_proba(processed)
    raw = raw_intent.detach().cpu().numpy() if torch.is_tensor(raw_intent) else np.asarray(raw_intent)
    emotion = (
        estimated_emotion.detach().cpu().numpy()
        if torch.is_tensor(estimated_emotion)
        else np.asarray(estimated_emotion)
    )
    values = {key: observation[key] for key in observation.keys()}
    fused = phase2.role_fusion.fuse(
        raw,
        role_visibility=values["role_visibility"],
        revealed_roles=values["revealed_roles"],
        confirmed_role_distribution=values["confirmed_role_distribution"],
        self_role=values["self_role"],
        sheriff_idx=values["sheriff_idx"],
        viewer_idx=seat,
    )
    return {
        "self_state": np.asarray(processed.state_vector, dtype=np.float32),
        "board_state": np.asarray(processed.player_state_matrix, dtype=np.float32),
        "confirmed_role_features": np.asarray(processed.role_beliefs, dtype=np.float32),
        "fused_intent": np.asarray(fused, dtype=np.float32),
        "emotion_features": emotion.reshape(-1).astype(np.float32),
        "action_mask": np.asarray(processed.action_mask, dtype=np.float32),
    }


def _to_tensors(inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in inputs.items()}


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------
def _dummy(config: Any, seat: int, seed: int) -> DummyAgent:
    return DummyAgent(
        agent_id=f"player_{seat}",
        player_idx=seat,
        seed=seed + seat + 1,
        policy_config=config.preset.intent_policy,
        emotion_config=ensure_formula_emotion(config.preset.emotion),
        action_mode="deterministic",
    )


def play_game(
    config: Any,
    phase2: Any,
    ppo: PPOModule,
    *,
    seed: int,
    seat: int,
    episode_id: str,
    sample: bool,
    generator: torch.Generator | None,
) -> tuple[list[PPOTransition], bool | None, int]:
    """Play one game. Returns (transitions, main team won or None if truncated, role)."""

    env = BangEnvironment(episode_id=episode_id, seed=seed, config=config.env)
    env.reset(seed=seed, episode_id=episode_id)
    role = int(env.state.roles[seat])
    opponents = {index: _dummy(config, index, seed) for index in range(5) if index != seat}
    for agent in opponents.values():
        agent.on_episode_start({"episode_id": episode_id, "seed": seed})
    records: list[dict[str, Any]] = []
    while not env.state.terminated and not env.state.truncated:
        selected = int(str(env.agent_selection).removeprefix("player_"))
        observe_boundaries(opponents, env, selected)
        observation = env.observe(selected)
        mask = np.asarray(observation["action_mask"], dtype=np.uint8)
        if selected == seat:
            inputs = policy_inputs(phase2, observation, seat)
            with torch.inference_mode():
                result = ppo.act(_to_tensors(inputs), mask, deterministic=not sample, generator=generator)
            action = int(result.action.reshape(-1)[0].item())
            records.append(
                {
                    "features": inputs,
                    "mask": mask.copy(),
                    "action": action,
                    "log_prob": float(result.log_prob.reshape(-1)[0].item()),
                    "value": float(result.value.reshape(-1)[0].item()),
                    "step": int(np.asarray(observation["environment_step"]).reshape(-1)[0]),
                }
            )
        else:
            action = int(opponents[selected].act(observation, mask))
        if mask[action] != 1:
            raise RuntimeError("an agent produced an action rejected by its action mask")
        env.step(action)
    for agent in opponents.values():
        agent.on_episode_end({"episode_id": episode_id})
    truncated = bool(env.state.truncated) and not bool(env.state.terminated)
    result = env.state.terminal_result
    won = None if truncated or result is None else bool(seat in result.winners)
    transitions: list[PPOTransition] = []
    for index, record in enumerate(records):
        last = index == len(records) - 1
        nxt = None if last else records[index + 1]
        transitions.append(
            PPOTransition(
                features=record["features"],
                action=record["action"],
                action_mask=record["mask"],
                old_log_prob=record["log_prob"],
                value=record["value"],
                reward=(0.0 if not last or won is None else (1.0 if won else -1.0)),
                next_features=None if nxt is None else nxt["features"],
                next_action_mask=None if nxt is None else nxt["mask"],
                terminated=last and won is not None,
                truncated=last and won is None,
                episode_id=episode_id,
                transition_id=index,
                environment_step=record["step"],
            )
        )
    return transitions, won, role


def imitation_data(config: Any, phase2: Any, games: int, base_seed: int) -> list[dict[str, Any]]:
    """(inputs, mask, dummy action, outcome) for every decision of all-dummy games."""

    rows: list[dict[str, Any]] = []
    for index in range(games):
        seed = base_seed + index
        episode_id = f"imitate-{seed}"
        env = BangEnvironment(episode_id=episode_id, seed=seed, config=config.env)
        env.reset(seed=seed, episode_id=episode_id)
        agents = {seat: _dummy(config, seat, seed) for seat in range(5)}
        for agent in agents.values():
            agent.on_episode_start({"episode_id": episode_id, "seed": seed})
        start = len(rows)
        while not env.state.terminated and not env.state.truncated:
            selected = int(str(env.agent_selection).removeprefix("player_"))
            observe_boundaries({k: v for k, v in agents.items() if k != selected}, env, selected)
            observation = env.observe(selected)
            mask = np.asarray(observation["action_mask"], dtype=np.uint8)
            action = int(agents[selected].act(observation, mask))
            if env.state.alive[selected]:
                rows.append(
                    {"inputs": policy_inputs(phase2, observation, selected), "mask": mask.copy(),
                     "action": action, "seat": selected}
                )
            env.step(action)
        result = env.state.terminal_result
        winners = set(result.winners) if result is not None and env.state.terminated else None
        for row in rows[start:]:
            row["outcome"] = 0.0 if winners is None else (1.0 if row["seat"] in winners else -1.0)
    return rows


def imitate(ppo: PPOModule, rows: list[dict[str, Any]], *, epochs: int, batch: int, seed: int) -> list[dict[str, float]]:
    """Cross-entropy on the dummy's actions + value regression on outcomes."""

    optimizer = ppo.create_optimizer()
    rng = np.random.default_rng(seed)
    keys = tuple(rows[0]["inputs"].keys())
    inputs = {key: torch.as_tensor(np.stack([row["inputs"][key] for row in rows])) for key in keys}
    masks = torch.as_tensor(np.stack([row["mask"] for row in rows]), dtype=torch.uint8)
    actions = torch.as_tensor([row["action"] for row in rows], dtype=torch.long)
    outcomes = torch.as_tensor([row["outcome"] for row in rows], dtype=torch.float32)
    history = []
    ppo.train()
    for epoch in range(epochs):
        order = rng.permutation(len(rows))
        total = correct = 0
        loss_sum = 0.0
        for start in range(0, len(rows), batch):
            idx = torch.as_tensor(order[start:start + batch], dtype=torch.long)
            log_prob, _, value = ppo.evaluate_actions(
                {key: value[idx] for key, value in inputs.items()}, actions[idx], masks[idx]
            )
            loss = -log_prob.mean() + 0.5 * torch.nn.functional.mse_loss(value.reshape(-1), outcomes[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                result = ppo.forward({key: value[idx] for key, value in inputs.items()}, masks[idx])
                correct += int((result.logits.argmax(dim=-1) == actions[idx]).sum())
            total += len(idx)
            loss_sum += float(loss.detach()) * len(idx)
        history.append({"epoch": epoch + 1, "loss": round(loss_sum / total, 4), "match": round(correct / total, 3)})
        print(f"imitation epoch {epoch + 1}: loss {history[-1]['loss']:.4f}  matches dummy {history[-1]['match']:.3f}", flush=True)
    return history


def quick_eval(config: Any, phase2: Any, ppo: PPOModule, games: int, base_seed: int) -> dict[str, Any]:
    wins = normal = 0
    by_role: dict[str, Counter] = {}
    for index in range(games):
        seat = index % 5
        _, won, role = play_game(
            config, phase2, ppo, seed=base_seed + index, seat=seat,
            episode_id=f"check-{base_seed + index}", sample=False, generator=None,
        )
        if won is None:
            continue
        normal += 1
        wins += int(won)
        counter = by_role.setdefault(ROLE_NAMES[role], Counter())
        counter["games"] += 1
        counter["wins"] += int(won)
    return {
        "win_rate": round(wins / max(normal, 1), 3),
        "games": normal,
        "by_role": {name: round(c["wins"] / max(c["games"], 1), 3) for name, c in sorted(by_role.items())},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase2", default=None, help="combined .phase2model (intent + emotion)")
    parser.add_argument("--intent", default=None, help="intent .phase2model saved by GUI module training")
    parser.add_argument("--emotion", default=None, help="emotion .phase2model saved by GUI module training")
    parser.add_argument("--out", required=True, help="output .bangmodel path")
    parser.add_argument("--updates", type=int, default=200, help="PPO updates")
    parser.add_argument("--games-per-update", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_000_000, help="first training seed (keep away from eval seeds)")
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE, help="dummy emotion formula file")
    parser.add_argument("--lr", type=float, default=3e-4, choices=(3e-4, 1e-3), help="Adam learning rate (fixed choices)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=25, help="deterministic check every N updates (0 = off)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--resume", default=None, help="continue from a .bangmodel's PPO weights")
    parser.add_argument("--bc-games", type=int, default=0, help="imitation warm start: all-dummy games to copy (0 = off)")
    parser.add_argument("--bc-epochs", type=int, default=5)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / f"{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    config = load_collection_config(emotion_file=args.emotion_file)
    phase2 = load_phase2(args, work)

    emotion_dim = TARGET_PLAYERS * int(phase2.emotion.emotion_count)
    ppo_config = PPOConfig.model_validate(
        {
            **PPOConfig().model_dump(mode="json"),
            "optimizer": {**PPOConfig().optimizer.model_dump(mode="json"), "learning_rate": args.lr},
            "ppo_epochs": args.epochs,
            "minibatch_size": args.minibatch,
            "entropy_coef": args.entropy,
            "gamma": args.gamma,
            "rollout_transitions_per_worker": 100_000,
        }
    )
    ppo = PPOModule(emotion_dim=emotion_dim, config=ppo_config, seed=args.seed % (2**31))
    if args.resume:
        from bang_rl.training.final_inference_artifact import load_final_inference_artifact

        ppo.load_state_dict(load_final_inference_artifact(args.resume).ppo.state_dict())
    imitation: list[dict[str, float]] = []
    if args.bc_games > 0:
        print(f"imitation warm start: playing {args.bc_games} all-dummy games", flush=True)
        rows = imitation_data(config, phase2, args.bc_games, base_seed=3_000_000)
        imitation = imitate(ppo, rows, epochs=args.bc_epochs, batch=256, seed=args.seed % (2**31))
        if args.eval_every:
            print(f"after imitation: check {quick_eval(config, phase2, ppo, args.eval_games, base_seed=2_000_000)}", flush=True)
    trainer = PPOTrainer(ppo, config=ppo_config, phase2_artifact=phase2, seed=args.seed % (2**31))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed % (2**31))

    log: list[dict[str, Any]] = []
    game_index = 0
    started = time.perf_counter()
    print(
        f"training PPO: {args.updates} updates x {args.games_per_update} games, "
        f"emotion dim {emotion_dim}, dummies from {args.emotion_file}",
        flush=True,
    )
    for update in range(1, args.updates + 1):
        transitions: list[PPOTransition] = []
        wins = normal = 0
        for _ in range(args.games_per_update):
            seed = args.seed + game_index
            seat = game_index % 5
            game_index += 1
            items, won, _ = play_game(
                config, phase2, ppo, seed=seed, seat=seat,
                episode_id=f"train-{seed}", sample=True, generator=generator,
            )
            transitions.extend(items)
            if won is not None:
                normal += 1
                wins += int(won)
        trainer.collect_rollout(transitions, max_transitions=max(1, len(transitions)))
        metrics = trainer.update().to_dict()
        row = {
            "update": update,
            "games": game_index,
            "train_win_rate": round(wins / max(normal, 1), 3),
            "transitions": len(transitions),
            "policy_loss": metrics.get("policy_loss"),
            "value_loss": metrics.get("value_loss"),
            "entropy": metrics.get("entropy"),
            "approx_kl": metrics.get("approx_kl"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        if args.eval_every and (update % args.eval_every == 0 or update == args.updates):
            row["check"] = quick_eval(config, phase2, ppo, args.eval_games, base_seed=2_000_000)
        log.append(row)
        line = (
            f"update {update:4d}  games {game_index:6d}  train win {row['train_win_rate']:.3f}  "
            f"entropy {row['entropy']:.3f}  kl {row['approx_kl']:.4f}  {row['seconds']:.0f}s"
        )
        if "check" in row:
            line += f"  | check win {row['check']['win_rate']:.3f} {row['check']['by_role']}"
        print(line, flush=True)

    versions = dict(_version_expectations(config, None))
    path = save_final_inference_artifact(
        out,
        ppo,
        phase2_artifact=phase2,
        job_snapshot={
            "job_id": f"train-main-ai-{args.seed}",
            "seed": int(args.seed),
            "split_seed": 0,
            "config_snapshot": {},
            "versions": versions,
        },
        config_snapshot={},
        training_metadata={
            "counters": {
                "environment_steps": trainer.environment_steps,
                "main_agent_transitions": trainer.main_agent_transitions,
            },
            "script": "train_main_ai.py",
            "imitation": imitation,
            "args": vars(args),
        },
    )
    (work / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved {path}\nlog {work / 'train_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
