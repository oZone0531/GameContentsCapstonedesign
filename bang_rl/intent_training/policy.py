"""Independent, emotion-free learned action policy for the existing six-action game.

The feature encoder reads *only* the current player's public observation and
an independently maintained public-information role belief. It never reads
EnvironmentState.roles or other private fields. The network learns action logits
from terminal outcomes; the old IntentModule's hand-written scores are unused.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as load_tensors, save as save_tensors
from torch import nn

from bang_rl.agents.base import BaseAgent
from bang_rl.agents.dummy_agent.role_belief import RoleBeliefEstimator
from bang_rl.env.types import ACTION_SPACE_VERSION, OBSERVATION_SCHEMA_VERSION

ROLE_KEYS = ("sheriff", "deputy", "outlaw", "traitor")
MODEL_FORMAT = "learned-intent-v1"
FEATURE_NAMES = (
    [f"role_{name}" for name in ROLE_KEYS]
    + [f"hp_{i}" for i in range(5)]
    + [f"alive_{i}" for i in range(5)]
    + [f"belief_{i}_{r}" for i in range(5) for r in range(5)]
    + [f"attacks_{i}_to_{j}" for i in range(5) for j in range(5)]
    + [f"seat_{i}" for i in range(5)]
    + ["alive_fraction", "step_fraction"]
)
FEATURE_SIZE = len(FEATURE_NAMES)


def role_key(role: Any) -> str:
    index = int(role)
    if not 0 <= index <= 4:
        raise ValueError("role must be in 0..4")
    return ROLE_KEYS[(0, 1, 2, 2, 3)[index]]


def _field(observation: Any, key: str) -> Any:
    if isinstance(observation, dict):
        return observation[key]
    if hasattr(observation, key):
        return getattr(observation, key)
    return observation[key]


def encode_public(observation: Any, beliefs: Any) -> np.ndarray:
    """Encode viewer-visible state, including public action history, into float32."""

    from bang_rl.agents.base import _validate_public_observation
    _validate_public_observation(observation)
    own_role = int(np.asarray(_field(observation, "self_role")).reshape(-1)[0])
    seat = int(getattr(observation, "viewer_idx", -1))
    if seat < 0:

        seat = int(np.asarray(_field(observation, "current_player_idx")).reshape(-1)[0])
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    pos = 0
    features[pos + ROLE_KEYS.index(role_key(own_role))] = 1.0
    pos += 4
    hp = np.asarray(_field(observation, "healths"), dtype=np.float32).reshape(5)
    alive = np.asarray(_field(observation, "alive"), dtype=np.float32).reshape(5)
    features[pos:pos + 5] = hp / 5.0
    pos += 5
    features[pos:pos + 5] = alive
    pos += 5
    beliefs = np.asarray(beliefs, dtype=np.float32).reshape(5, 5)
    features[pos:pos + 25] = np.clip(beliefs, 0.0, 1.0).reshape(-1)
    pos += 25
    attacks = np.zeros((5, 5), dtype=np.float32)
    events = _field(observation, "event_sequence")
    for event in events:
        if isinstance(event, dict):
            data = event
        elif hasattr(event, "to_public_dict"):
            data = event.to_public_dict()
        else:
            continue
        kind = data.get("event_type", data.get("type"))
        if str(kind).lower() in ("bang", "eventtype.bang"):
            actor, target = data.get("actor_idx"), data.get("target_idx")
            if actor is not None and target is not None and 0 <= int(actor) < 5 and 0 <= int(target) < 5:
                attacks[int(actor), int(target)] += 1.0
    features[pos:pos + 25] = np.minimum(attacks.reshape(-1), 10.0) / 10.0
    pos += 25
    features[pos + seat] = 1.0
    pos += 5
    features[pos] = float(alive.sum()) / 5.0
    features[pos + 1] = min(float(np.asarray(_field(observation, "environment_step")).reshape(-1)[0]), 200.0) / 200.0
    return features


class PolicyNet(nn.Module):
    """Compact per-role actor/critic, with raw action logits and state value."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(FEATURE_SIZE, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 6)
        self.critic = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(features)
        return self.actor(h), self.critic(h).squeeze(-1)


class LearnedIntentPolicy:
    """Reusable, independent bundle of four policies (outlaws share one)."""

    def __init__(self, *, seed: int = 12345) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.networks = {name: PolicyNet().cpu() for name in ROLE_KEYS}
        self.metadata: dict[str, Any] = {}

    def action_logits(self, observation: Any, beliefs: Any) -> np.ndarray:
        own_role = int(np.asarray(_field(observation, "self_role")).reshape(-1)[0])
        x = torch.from_numpy(encode_public(observation, beliefs))
        net = self.networks[role_key(own_role)]
        with torch.no_grad():
            values, _ = net(x)
        return values.numpy().copy()

    def action_probabilities(self, observation: Any, beliefs: Any, action_mask: Any) -> np.ndarray:
        mask = np.asarray(action_mask, dtype=bool).reshape(6)
        if not mask.any():
            raise ValueError("no legal action")
        values = self.action_logits(observation, beliefs)
        values[~mask] = -1e9
        values -= values.max()
        probabilities = np.exp(values) * mask
        return probabilities / probabilities.sum()

    def explain(self, observation: Any, beliefs: Any, action_mask: Any) -> dict[str, Any]:
        """Return action probabilities and local gradient attribution, not a fictitious global formula."""
        role = role_key(int(np.asarray(_field(observation, "self_role")).reshape(-1)[0]))
        feature = encode_public(observation, beliefs)
        x = torch.from_numpy(feature.copy()).requires_grad_(True)
        logits, _ = self.networks[role](x)
        mask = np.asarray(action_mask, dtype=bool).reshape(6)
        valid = torch.as_tensor(mask)
        chosen = int(torch.argmax(logits.masked_fill(~valid, -1e9)).item())
        logits[chosen].backward()
        gradient = x.grad.detach().numpy()
        ranks = np.argsort(-np.abs(gradient * feature))[:8]
        return {
            "role": role, "chosen_action": chosen,
            "probabilities": self.action_probabilities(observation, beliefs, action_mask).tolist(),
            "local_contributions": [
                {"feature": FEATURE_NAMES[i], "value": float(feature[i]), "gradient_times_input": float(gradient[i] * feature[i])}
                for i in ranks
            ],
            "note": "Local first-order approximation only; not an exact decision formula.",
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Use a portable ZIP + safetensors archive; no pickle/deserialization code."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"{name}.{key}": value.detach().cpu().contiguous()
            for name, net in self.networks.items()
            for key, value in net.state_dict().items()
        }
        raw = save_tensors(tensors)
        manifest = {
            "format": MODEL_FORMAT,
            "feature_size": FEATURE_SIZE,
            "action_space_version": ACTION_SPACE_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "weights_sha256": hashlib.sha256(raw).hexdigest(),
            "roles": list(ROLE_KEYS),
            "metadata": dict(self.metadata if metadata is None else metadata),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
                output.writestr("weights.safetensors", raw)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "LearnedIntentPolicy":
        with zipfile.ZipFile(path, "r") as archive:
            if set(archive.namelist()) != {"manifest.json", "weights.safetensors"}:
                raise ValueError("invalid learned-intent archive members")
            manifest = json.loads(archive.read("manifest.json"))
            raw = archive.read("weights.safetensors")
        if (manifest.get("format") != MODEL_FORMAT
                or manifest.get("feature_size") != FEATURE_SIZE
                or manifest.get("action_space_version") != ACTION_SPACE_VERSION
                or manifest.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION
                or manifest.get("roles") != list(ROLE_KEYS)
                or manifest.get("weights_sha256") != hashlib.sha256(raw).hexdigest()):
            raise ValueError("incompatible/corrupt learned-intent model")
        weights = load_tensors(raw)
        result = cls()
        for name, net in result.networks.items():
            prefix = name + "."
            expected = set(net.state_dict())
            fields = {key[len(prefix):]: tensor for key, tensor in weights.items() if key.startswith(prefix)}
            if set(fields) != expected:
                raise ValueError(f"missing or unexpected tensor for {name}")
            net.load_state_dict(fields, strict=True)
            net.eval()
        result.metadata = dict(manifest.get("metadata") or {})
        return result


class LearnedIntentAgent(BaseAgent):
    """GameService-compatible action-only agent; emotion is never consulted."""

    def __init__(self, policy: LearnedIntentPolicy, *, player_idx: int, seed: int | None = None,
                 stochastic: bool = False) -> None:
        super().__init__(f"player_{player_idx}", seed=seed)
        self.player_idx = int(player_idx)
        self.policy = policy
        self.belief_estimator = RoleBeliefEstimator(player_idx=self.player_idx)
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

    def on_episode_start(self, context: Any) -> None:
        super().on_episode_start(context)
        self.belief_estimator.reset()
        self.rng = np.random.default_rng(self.action_seed(self.episode_seed)) if self.episode_seed is not None else self.rng

    def act(self, observation: Any, action_mask: Any) -> int:
        mask = self._validate_act_inputs(observation, action_mask)
        belief = self.belief_estimator.update(observation)
        p = self.policy.action_probabilities(observation, belief, mask)
        action = int(self.rng.choice(6, p=p)) if self.stochastic else int(np.argmax(p))
        return self._validate_selected_action(action, mask)

    def explain(self, observation: Any, action_mask: Any) -> dict[str, Any]:
        belief = self.belief_estimator.update(observation)
        return self.policy.explain(observation, belief, action_mask)


__all__ = ["LearnedIntentPolicy", "LearnedIntentAgent", "PolicyNet", "ROLE_KEYS", "role_key", "encode_public"]
