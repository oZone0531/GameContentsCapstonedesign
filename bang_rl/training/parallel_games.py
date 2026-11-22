"""Play main-AI training and evaluation games in worker processes.

Each worker process gets one copy of the whole policy (frozen intent/emotion
modules, role fusion, preprocessor and the PPO module) when it starts. Every
task then carries only the current PPO weights, so after a PPO update the
workers play with the new policy. A game is played by the same functions the
serial path uses (:func:`play_training_game` and ``_play_evaluation_game``);
only where it runs changes.

Game ``i`` keeps its seed, seat and action-sampling seed, so a game played with
given weights is identical in a worker and in the main process. What differs
from serial training is when the PPO update happens: games are played in
batches of ``workers`` games, so a few games of a batch may have been played
with the weights from before the update that the batch triggers.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import os
import pickle
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from bang_rl.config import SystemConfig

# Worker-process state, set once by ``_init_worker``.
_WORKER: dict[str, Any] = {}


def usable_workers(requested: Any) -> int:
    """Requested worker count, at least 1 and at most the machine's CPU count."""

    try:
        value = int(requested)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, os.cpu_count() or 1))


def _init_worker(policy_blob: bytes, config: SystemConfig, run_seed: int) -> None:
    # One Torch thread per process: N workers must not each start N threads.
    from bang_rl.data.parallel_collection import _worker_setup

    _worker_setup()
    from bang_rl.runtime.reward import RewardCalculator

    from .main_ai_training import main_ai_reward_config

    _WORKER.clear()
    _WORKER.update(
        policy=pickle.loads(policy_blob),
        config=config,
        run_seed=int(run_seed),
        calculator=RewardCalculator(main_ai_reward_config()),
    )


def _load_weights(ppo_state: dict[str, Any]) -> Any:
    policy = _WORKER["policy"]
    policy.ppo.load_state_dict(ppo_state)
    return policy


def _play_training(game_index: int, ppo_state: dict[str, Any]) -> Any:
    from .main_ai_training import play_training_game

    policy = _load_weights(ppo_state)
    return play_training_game(
        int(game_index),
        run_seed=_WORKER["run_seed"],
        config=_WORKER["config"],
        policy=policy,
        reward_calculator=_WORKER["calculator"],
    )


def _play_evaluation(
    indices: Sequence[int], ppo_state: dict[str, Any], base_seed: int, seats: Sequence[int]
) -> tuple[list[tuple[int, bool | None, str, int]], dict[str, Any]]:
    from bang_rl.evaluation.protocol import InferenceMetrics

    from .main_ai_training import _play_evaluation_game

    policy = _load_weights(ppo_state)
    metrics = InferenceMetrics()
    rows = []
    for index in indices:
        won, role, steps = _play_evaluation_game(
            policy,
            _WORKER["config"],
            seed=int(base_seed) + int(index),
            seat=int(seats[int(index) % len(seats)]),
            index=int(index),
            environment_factory=None,
            metrics=metrics,
        )
        rows.append((int(index), won, role, steps))
    return rows, metrics.to_dict()


def _ppo_state(ppo: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in ppo.state_dict().items()}


class GamePool:
    """A pool of worker processes that play games with the current policy."""

    def __init__(self, workers: int, config: SystemConfig, policy: Any, run_seed: int) -> None:
        if workers < 2:
            raise ValueError("a game pool needs at least 2 workers")
        # Pickling here (not in the worker) reports an unpicklable policy at once.
        blob = pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL)
        self.workers = int(workers)
        # One game per worker per batch keeps the policy used by a batch fresh.
        self.batch_games = self.workers
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(blob, config, int(run_seed)),
        )

    def play_training_games(self, game_indices: Iterable[int], ppo: Any) -> list[Any]:
        """Play the given training games; results come back in game order."""

        state = _ppo_state(ppo)
        futures = [self._executor.submit(_play_training, int(index), state) for index in game_indices]
        return [future.result() for future in futures]

    def evaluate(
        self,
        ppo: Any,
        *,
        games: int,
        base_seed: int,
        seats: Sequence[int],
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[list[tuple[int, bool | None, str, int]], dict[str, Any]] | None:
        """Play the evaluation deals; per-game rows in game order and summed metrics, or None if stopped."""

        state = _ppo_state(ppo)
        seats = tuple(int(seat) for seat in seats)
        # A few chunks per worker balance the load without much dispatch overhead.
        chunk = max(1, -(-int(games) // (self.workers * 4)))
        pending: set[Future] = {
            self._executor.submit(
                _play_evaluation, list(range(start, min(start + chunk, int(games)))), state, int(base_seed), seats
            )
            for start in range(0, int(games), chunk)
        }
        rows: list[tuple[int, bool | None, str, int]] = []
        totals: dict[str, Any] = {}
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                part, metrics = future.result()
                rows.extend(part)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value
            if pending and should_stop is not None and should_stop():
                for future in pending:
                    future.cancel()
                return None
        rows.sort(key=lambda row: row[0])
        return rows, totals

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def open_game_pool(
    workers: Any, config: SystemConfig, policy: Any, run_seed: int, *, log: Callable[[str], None] | None = None
) -> GamePool | None:
    """A started pool, or None to play in this process (1 worker or the pool cannot start)."""

    count = usable_workers(workers)
    if count < 2:
        return None
    try:
        return GamePool(count, config, policy, run_seed)
    except Exception as exc:  # noqa: BLE001 - parallel play is an optimization; fall back to serial
        if log is not None:
            log(f"parallel games unavailable, playing in one process: {type(exc).__name__}: {exc}")
        return None


__all__ = ["BrokenProcessPool", "GamePool", "open_game_pool", "usable_workers"]
