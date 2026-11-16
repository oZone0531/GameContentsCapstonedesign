"""Input ablation for the main agent: how much does the policy use its inferences?

The trained policy receives two inferred inputs every turn: the role inference
(``artifact.intent``) and the emotion inference (``artifact.emotion``). An
ablation replaces one or both with a constant "no information" value at
evaluation time and replays the same games:

    full          nothing replaced (same as the plain main-agent row)
    no_intent     role inference -> uniform over roles (public facts are still
                  applied afterwards by role fusion, e.g. the revealed sheriff)
    no_emotion    emotion inference -> the formula's resting state
                  (calm 0.9, confidence / aggression / anxiety 0.1, relations 0.5)
    no_inference  both of the above

The win-rate drop from ``full`` to an ablation is how much that inference
contributes to winning *for this trained policy*. It is a test-time ablation:
the policy was trained with real inferences, so a drop also includes the shift
in its inputs; retraining without the input would be the stricter comparison.

Role-accuracy and emotion-error metrics of an ablated row describe the
constant replacement, not the trained modules.
"""

from __future__ import annotations

from typing import Any

import torch

from .protocol import CustomEvaluationProtocol

ABLATIONS = ("full", "no_intent", "no_emotion", "no_inference")

# Formula resting state: baselines b_S = b_A = b_F = 0.1, calm = 1 - max(...).
_FORMULA_REST = (0.9, 0.1, 0.1, 0.1)


def neutral_emotion(like: torch.Tensor) -> torch.Tensor:
    """Constant emotion estimate with the shape of ``like`` (last dim = emotions)."""

    width = int(like.shape[-1])
    values = torch.full_like(like, 0.5)
    if width >= len(_FORMULA_REST):
        values[..., : len(_FORMULA_REST)] = torch.tensor(_FORMULA_REST, dtype=like.dtype, device=like.device)
    return values


def uniform_roles(like: torch.Tensor) -> torch.Tensor:
    """Uniform role distribution with the shape of ``like`` (last dim = roles)."""

    return torch.full_like(like, 1.0 / float(like.shape[-1]))


def _keep_empty_rows(original: torch.Tensor, replaced: torch.Tensor) -> torch.Tensor:
    """Rows the module leaves at zero (own seat, masked seats) stay zero."""

    if original.ndim == 0:
        return replaced
    filled = original.abs().sum(dim=-1, keepdim=True) > 0
    return torch.where(filled, replaced, original)


class _ConstantModule:
    """Delegates everything to ``module`` except ``predict_proba``."""

    def __init__(self, module: Any, replace) -> None:
        self._module = module
        self._replace = replace

    def predict_proba(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self._module.predict_proba(*args, **kwargs)
        tensor = out if torch.is_tensor(out) else torch.as_tensor(out)
        return _keep_empty_rows(tensor, self._replace(tensor))

    predict = predict_proba
    inference = predict_proba

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class _AblatedArtifact:
    def __init__(self, artifact: Any, *, intent: Any, emotion: Any) -> None:
        self._artifact = artifact
        self.intent = intent
        self.emotion = emotion

    def __getattr__(self, name: str) -> Any:
        return getattr(self._artifact, name)


def ablated_protocol(artifact: Any, ablation: str, **kwargs: Any) -> CustomEvaluationProtocol:
    """A CustomEvaluationProtocol whose main agent sees ``ablation``'s inputs."""

    if ablation not in ABLATIONS:
        raise ValueError(f"ablation must be one of {ABLATIONS}, got {ablation!r}")
    protocol = CustomEvaluationProtocol(artifact, **kwargs)
    if ablation == "full":
        return protocol
    base = protocol.artifact
    intent = base.intent
    emotion = base.emotion
    if ablation in ("no_intent", "no_inference"):
        intent = _ConstantModule(intent, uniform_roles)
    if ablation in ("no_emotion", "no_inference"):
        emotion = _ConstantModule(emotion, neutral_emotion)
    protocol.artifact = _AblatedArtifact(base, intent=intent, emotion=emotion)
    return protocol


__all__ = ["ABLATIONS", "ablated_protocol", "neutral_emotion", "uniform_roles"]
