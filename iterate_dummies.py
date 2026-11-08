"""Iterate: dummies make data -> main inference learns -> dummies use it -> repeat.

Usage (from the repository root, next to the bang_rl folder):
    python iterate_dummies.py --rounds 3 --games 1000 --out runs/iterate1
    python iterate_dummies.py --rounds 3 --games 1000 --start hint:0.7 --out runs/iterate_hint

Round r:
  1. collect --games all-dummy games; the dummies' role beliefs come from
     --start in round 0 and from the previous round's trained model after that;
  2. split 80/10/10, build the tensor cache, train role (intent) and emotion
     inference for --epochs, save round_r/model.phase2model;
  3. measure the new model as dummy beliefs on --eval-games fresh games
     (seeds after every collected game): how often the dummies' beliefs pick
     the right role for living hidden players, team win rates, friendly fire.

rounds.json in --out collects every round's numbers; each round's folder keeps
its dataset, cache and model so any round can be reused or inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from bang_rl.data import (  # noqa: E402
    EpisodeCollector,
    EpisodeSplitter,
    NPZTensorCacheLoader,
    NPZTensorCacheWriter,
    ParquetCanonicalShardWriter,
)
from bang_rl.data.game_collection import (  # noqa: E402
    DEFAULT_EMOTION_FILE,
    FORMULA_LABEL_SCHEMA,
    build_game,
    collect_games,
    drive_game,
    formula_label_emotion_config,
    load_collection_config,
)
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES  # noqa: E402
from bang_rl.training.emotion_sl_trainer import EmotionInferenceSLTrainer, EmotionSLConfig  # noqa: E402
from bang_rl.training.inference_artifact import save_phase2_inference_artifact  # noqa: E402
from bang_rl.training.sl_trainer import IntentInferenceSLTrainer, IntentSLConfig  # noqa: E402

TEAM = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
SEMANTIC = (0, 1, 2, 2, 3)


def measure_dummies(config, belief: str, games: int, base_seed: int) -> dict:
    """Play all-dummy games with ``belief`` and measure belief quality and play."""

    correct = total = 0
    wins: Counter = Counter()
    friendly = bangs = 0
    for index in range(games):
        seed = base_seed + index
        env, agents, episode_id = build_game(config, seed=seed, belief=belief)
        roles = [int(role) for role in env.state.roles]
        for seat, agent in enumerate(agents):
            original = agent.act

            def act(observation, mask, _agent=agent, _act=original, _seat=seat, **kwargs):
                nonlocal correct, total
                action = _act(observation, mask, **kwargs)
                beliefs = _agent.role_beliefs
                for other in range(5):
                    if other == _seat or roles[other] == 0 or not env.state.alive[other]:
                        continue  # self, public sheriff, dead (public role)
                    row = beliefs[other]
                    semantic = np.array([row[0], row[1], row[2] + row[3], row[4]])
                    correct += int(np.argmax(semantic) == SEMANTIC[roles[other]])
                    total += 1
                return action

            agent.act = act
        episode = drive_game(env, agents, config, EpisodeCollector(), episode_id=episode_id, seed=seed)
        result = env.state.terminal_result
        wins[result.winning_team.name if result is not None else "TRUNCATED"] += 1
        for step in episode.steps:
            if step.action > 0:
                bangs += 1
                actor, target = roles[step.actor_idx], roles[step.action - 1]
                friendly += int(actor != 4 and TEAM[actor] == TEAM[target])
    return {
        "belief_accuracy": correct / total if total else None,
        "win_rate": {key: round(value / games, 4) for key, value in sorted(wins.items())},
        "friendly_fire": friendly / bangs if bangs else None,
    }


def train_round(round_dir: Path, epochs: int, batch_size: int, seed: int) -> tuple[Path, dict]:
    episodes = ParquetCanonicalShardWriter.read_dataset(round_dir / "canonical")
    count = len(episodes)
    test = max(1, count // 10)
    validation = max(1, count // 10)
    split = EpisodeSplitter(
        counts=(count - test - validation, validation, test), seed=seed, manifest_path=round_dir / "split.json"
    ).split(episodes)
    artifacts = NPZTensorCacheWriter(
        round_dir / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=len(LABEL_NAMES)
    ).write_all(split)
    caches = {name: NPZTensorCacheLoader(artifacts[name].tensor_path) for name in ("train", "validation", "test")}
    intent = IntentInferenceSLTrainer(
        caches["train"], caches["validation"], test_cache=caches["test"],
        config=IntentSLConfig(epochs=epochs, batch_size=batch_size, seed=seed),
    ).fit()
    emotion = EmotionInferenceSLTrainer(
        caches["train"], caches["validation"], test_cache=caches["test"],
        config=EmotionSLConfig(epochs=epochs, batch_size=batch_size, seed=seed),
        emotion_config=formula_label_emotion_config(),
    ).fit()
    model_path = save_phase2_inference_artifact(round_dir / "model.phase2model", intent.model, emotion.model)
    metrics = {
        "intent_test": {k: v for k, v in (intent.test_metrics or {}).items() if isinstance(v, (int, float))},
        "emotion_test": {k: v for k, v in (emotion.test_metrics or {}).items() if isinstance(v, (int, float))},
    }
    return Path(model_path), metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iterate dummy data and main-agent role inference.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--games", type=int, default=1000, help="games collected per round")
    parser.add_argument("--eval-games", type=int, default=300, help="fresh games to measure each round's model")
    parser.add_argument("--start", default="rule", help="round-0 dummy beliefs: rule, hint:Q, ...")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE)
    parser.add_argument("--out", required=True, help="new or empty output folder")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} is not empty", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    config = load_collection_config(emotion_file=args.emotion_file)
    eval_seed = args.seed + args.rounds * args.games  # never overlaps collected seeds

    rounds = []
    print(f"round -: measuring start beliefs {args.start!r}", flush=True)
    start = measure_dummies(config, args.start, args.eval_games, eval_seed)
    rounds.append({"round": "start", "beliefs": args.start, "dummies": start})
    belief = args.start
    for index in range(args.rounds):
        began = time.perf_counter()
        round_dir = out / f"round_{index}"
        print(f"round {index}: collecting {args.games} games with beliefs {belief!r}", flush=True)
        summary = collect_games(
            config, args.games, round_dir / "canonical",
            base_seed=args.seed + index * args.games, belief=belief,
        )
        print(f"round {index}: training {args.epochs} epochs", flush=True)
        model_path, metrics = train_round(round_dir, args.epochs, args.batch_size, args.seed)
        belief = f"learned:{model_path}"
        print(f"round {index}: measuring the new model as dummy beliefs", flush=True)
        measured = measure_dummies(config, belief, args.eval_games, eval_seed)
        row = {
            "round": index,
            "collected_with": summary["belief_source"],
            "collected_win_rate": summary["win_rate"],
            "model": str(model_path),
            "training": metrics,
            "dummies_with_new_model": measured,
            "seconds": round(time.perf_counter() - began, 1),
        }
        rounds.append(row)
        (out / "rounds.json").write_text(json.dumps(rounds, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(
            f"round {index}: belief accuracy {measured['belief_accuracy']:.3f}, "
            f"win {measured['win_rate']}, friendly fire {measured['friendly_fire']:.3f}",
            flush=True,
        )
    (out / "rounds.json").write_text(json.dumps(rounds, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"start beliefs {args.start!r}: accuracy {start['belief_accuracy']:.3f}, win {start['win_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
