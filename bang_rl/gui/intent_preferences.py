"""Per-seat GUI choice for a dummy's intent, independent of model/game config.

A learned choice runs the emotion-free LearnedIntentAgent for this seat, rather
than pretending the current DummyAgent supports neural intent/emotion fusion.
Only dummy seats are eligible; human/random/main seats are unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".bang_rl_intent_seats.json"
OPTIONS = ("default", "rule", "learned")


@dataclass(frozen=True)
class IntentSeatPreferences:
    default: str = "rule"
    overrides: tuple[str, str, str, str, str] = ("default",) * 5
    model_path: str = "bang_rl/models/intent/trained.intentmodel"

    def __post_init__(self) -> None:
        if self.default not in ("rule", "learned"):
            raise ValueError("default intent selection must be rule or learned")
        if len(self.overrides) != 5 or any(x not in OPTIONS for x in self.overrides):
            raise ValueError("Provide five seat settings: default/rule/learned")

    @property
    def resolved(self) -> tuple[str, ...]:
        return tuple(self.default if entry == "default" else entry for entry in self.overrides)

    def validate_model(self) -> None:
        if "learned" in self.resolved:
            from bang_rl.intent_training import LearnedIntentPolicy
            if not self.model_path.strip():
                raise ValueError("Select a learned intent model file")
            LearnedIntentPolicy.load(self.model_path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "IntentSeatPreferences":
        file = Path(path)
        if not file.is_file():
            return cls()
        data: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return cls(default=str(data.get("default", "rule")),
                   overrides=tuple(data.get("overrides", ["default"]*5)),
                   model_path=str(data.get("model_path", cls.model_path)))

    def save(self, path: str | Path = DEFAULT_PATH) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temp.write_text(json.dumps({"default": self.default, "overrides": list(self.overrides),
                                        "model_path": self.model_path}, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
