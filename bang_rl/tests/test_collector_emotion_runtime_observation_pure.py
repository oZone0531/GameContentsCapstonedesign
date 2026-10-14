"""Pure, dependency-light regression test for the collection TypeError.

Reproduces the exact fault the user hit when pressing "AI 학습" -> 시작: the
training runner builds ``EpisodeCollector(..., emotion_runtime=EmotionRuntime)``
and then calls ``collector.before_action(observation, ...)``.  The collector's
``before_action`` falls back to ``self.emotion_runtime`` and hands it to the
module-private ``_emotion_record`` helper.  ``EmotionRuntime`` exposes
``public_observation`` as a *method*, so a bare ``getattr(runtime,
"public_observation", None)`` returned the bound method, which was then passed to
``_observation`` -> ``_as_mapping`` and raised

    TypeError: observation must be a mapping or serializable domain value

at ``bang_rl/data/collector.py`` (``_as_mapping``).  The fix teaches
``_emotion_record`` to ignore a callable / non-serializable ``public_observation``
so a runtime handed straight to the collector drives collection cleanly.

The real collector depends on numpy and the full env package, which are not
installable in the restricted sandbox.  To stay Qt-/torch-/numpy-free we stub a
*minimal* ``numpy`` and ``bang_rl.env.types`` (only the handful of names the
collector module touches) and load ``bang_rl.data.collector`` by file path, then
drive ``before_action`` exactly as the runner does.  The real end-to-end torch
collection stays CI-only; this test pins the observation-shape contract and is
revert-sensitive (reverting the ``_emotion_record`` fix makes it raise the exact
TypeError again).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_PKG_ROOT = Path(__file__).parents[1]


class _FakeArray:
    """A tiny stand-in for the 1-D numpy arrays the collector reshapes."""

    def __init__(self, data) -> None:
        if isinstance(data, _FakeArray):
            data = list(data._data)
        elif isinstance(data, (list, tuple)):
            data = list(data)
        else:
            data = [data]
        self._data = data

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self._data)

    @property
    def size(self):
        return len(self._data)

    def item(self):
        return self._data[0]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def _install_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return
    stub = types.ModuleType("numpy")

    def asarray(value, *_args, **_kwargs):
        return _FakeArray(value)

    stub.asarray = asarray
    stub.ndarray = _FakeArray
    # ``generic`` / ``integer`` are only used in isinstance() guards; a dummy
    # class that nothing in this test is an instance of is sufficient.
    stub.generic = type("generic", (), {})
    stub.integer = type("integer", (), {})
    sys.modules["numpy"] = stub


def _install_env_types_stub() -> None:
    # The collector imports only two names from bang_rl.env.types.
    if "bang_rl.env.types" in sys.modules:
        return
    for name in ("bang_rl", "bang_rl.env"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as package
            sys.modules[name] = pkg
    module = types.ModuleType("bang_rl.env.types")
    module.CANONICAL_PLAYER_ORDER = (0, 1, 2, 3, 4)
    module.PASS_ACTION = 0
    sys.modules["bang_rl.env.types"] = module


def _load_collector() -> types.ModuleType:
    _install_numpy_stub()
    _install_env_types_stub()
    dotted = "bang_rl.data.collector"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "bang_rl.data" not in sys.modules:
        pkg = types.ModuleType("bang_rl.data")
        pkg.__path__ = [str(_PKG_ROOT / "data")]
        sys.modules["bang_rl.data"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, _PKG_ROOT / "data" / "collector.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


collector_module = _load_collector()
EpisodeCollector = collector_module.EpisodeCollector


class _RuntimeLikeEmotion:
    """Mirror the EmotionRuntime surface the collector probes.

    Crucially ``public_observation`` is a *method* (as on the real runtime),
    which is exactly what tripped ``_emotion_record`` before the fix.
    """

    def __init__(self) -> None:
        self.labels: dict[int, object] = {}

    def public_observation(self, *, observation=None):  # noqa: ANN001 - stub
        return {"episode_id": "episode", "environment_step": 0}

    # Alias the real runtime also exposes.
    observation = public_observation


def _public_observation(step: int = 0) -> dict:
    """A representative public observation MAPPING (to_observation_dict shape)."""

    return {
        "current_player_idx": [0],
        "environment_step": step,
        "healths": [5, 4, 4, 4, 4],
        "alive": [True, True, True, True, True],
        "turn_order": [0, 1, 2, 3, 4],
        "action_mask": [1, 1, 1, 1, 1, 1],
        "observation_schema_version": "1.0",
    }


def test_before_action_with_emotion_runtime_does_not_raise_observation_typeerror() -> None:
    runtime = _RuntimeLikeEmotion()
    collector = EpisodeCollector(emotion_runtime=runtime)
    collector.start_episode(episode_id="episode", seed=7)

    observation = _public_observation()
    # Drive the collector exactly like default_training_runner does: pass the
    # public observation mapping plus the action mask, and rely on the
    # constructor-provided emotion_runtime fallback inside before_action.
    stored = collector.before_action(
        observation,
        action_mask=[1, 1, 1, 1, 1, 1],
        actor_idx=0,
    )

    assert stored["environment_step"] == 0
    assert stored["action_mask"] == [1, 1, 1, 1, 1, 1]


def test_emotion_record_ignores_callable_public_observation() -> None:
    # Focused guard on the helper itself: a value whose ``public_observation`` is
    # a bound method must yield label_observation=None, never a TypeError.
    runtime = _RuntimeLikeEmotion()
    labels, schema, label_observation = collector_module._emotion_record(runtime)
    assert labels == {}
    assert schema is None
    assert label_observation is None


def test_emotion_record_still_accepts_mapping_public_observation() -> None:
    # A boundary record that stores public_observation as a real mapping must
    # continue to flow through unchanged (the fix must not over-reject).
    boundary = type(
        "Boundary",
        (),
        {"labels": {}, "public_observation": {"environment_step": 0, "healths": [5, 4, 4, 4, 4]}},
    )()
    _labels, _schema, label_observation = collector_module._emotion_record(boundary)
    assert label_observation is not None
    assert label_observation["environment_step"] == 0


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
