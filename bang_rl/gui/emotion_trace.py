"""Turn-by-turn emotion decision trace built from a FULL replay record (Qt-free).

For every dummy decision it writes: the emotion state used, then per legal action
``intent + emotion = total``, the action intent alone would pick, the action that
was picked, and (for the emotion formula) which emotion term flipped it.

The GUI writes ``<replay>.txt`` next to every saved full replay. It can also be
run by hand on saved replays::

    python -m bang_rl.gui.emotion_trace bang_rl/replays/replay-xxx.json [...]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROLE_NAMES = ("sheriff", "deputy", "outlaw", "outlaw", "traitor")
TEAM_NAMES = ("S-team", "S-team", "O-team", "O-team", "traitor")


def _action_name(action: int) -> str:
    return "pass" if action == 0 else f"bang seat{action - 1}"


def _intent_only_action(intent: np.ndarray, mask: np.ndarray) -> int:
    """Deterministic pick from intent scores alone (same rule as ActionSelector)."""

    masked = np.where(mask.astype(bool), intent, -np.inf)
    return int(np.argmax(masked))


def _why(base: int, chosen: int, intent: np.ndarray, emo: np.ndarray,
         relation: np.ndarray | None, params: Mapping[str, Any] | None) -> str:
    intent_gap = float(intent[base] - intent[chosen])
    emo_gap = float(emo[chosen] - emo[base])
    tie = "intent tie" if abs(intent_gap) < 1e-9 else f"intent gap {intent_gap:.3f}"
    if params is None or relation is None:
        return f"{tie}; emotion contribution favoured it by {emo_gap:+.3f}"
    if base > 0 and chosen > 0:
        rc, rb = float(relation[chosen - 1]), float(relation[base - 1])
        rel_term = -float(params.get("kappa_R", 1.0)) * float(params.get("lambda_E", 1.0)) * (rc - rb)
        if abs(rel_term) >= 0.02:
            feel = "grudge" if rc < rb else "liking"
            return (f"{tie}; relation decided it ({feel}: seat{chosen - 1} R={rc:+.3f} "
                    f"vs seat{base - 1} R={rb:+.3f}, +{emo_gap:.3f})")
        return f"{tie}; small noise in relation broke it (+{emo_gap:.3f})"
    if chosen == 0:
        return f"{tie}; anxiety pushed PASS over shooting (+{emo_gap:.3f})"
    return f"{tie}; confidence/aggression pushed shooting over PASS (+{emo_gap:.3f})"


def _seat_state(internals: Any, seat: int) -> Mapping[str, Any] | None:
    if not isinstance(internals, Mapping):
        return None
    seats = internals.get("seats")
    if not isinstance(seats, Mapping):
        return None
    descriptor = seats.get(str(seat), seats.get(seat))
    if not isinstance(descriptor, Mapping) or descriptor.get("kind") != "dummy":
        return None
    state = descriptor.get("state")
    return state if isinstance(state, Mapping) else None


def build_trace_lines(record: Mapping[str, Any], *, only_changed: bool = False,
                      title: str | None = None) -> list[str]:
    """Return the trace as text lines. Empty list when the record has no internals."""

    states: Sequence[Any] = record.get("states") or ()
    internals: Sequence[Any] = record.get("internals") or ()
    if not states or not internals or len(states) != len(internals):
        return []
    roles = list(record.get("final_roles") or ())
    if len(roles) != 5 and isinstance(states[0], Mapping):
        roles = list(states[0].get("roles") or ())
    roles = [int(r) for r in roles] if len(roles) == 5 else [None] * 5

    lines: list[str] = []
    out = lines.append
    out(f"Emotion decision trace{(' - ' + title) if title else ''}")
    out("Roles (hidden in game): " + ", ".join(
        f"seat{i}={ROLE_NAMES[r] if r is not None else '?'}" for i, r in enumerate(roles)))
    kinds = []
    for seat in range(5):
        state = _seat_state(internals[0], seat)
        module = (state or {}).get("emotion_module") or {}
        kinds.append(module.get("kind", "rule_table") if state else "-")
    out("Emotion module per seat: " + ", ".join(f"seat{i}={k}" for i, k in enumerate(kinds)))
    out("")

    decisions = changed = 0
    for k in range(1, len(states)):
        prev = states[k - 1]
        if not isinstance(prev, Mapping):
            continue
        actor = int(prev.get("agent_selection", -1))
        state = _seat_state(internals[k], actor)
        if state is None:
            continue
        decision = state.get("last_decision")
        if not isinstance(decision, Mapping):
            continue
        intent = np.asarray(decision.get("intent_scores"), dtype=float)
        emo = np.asarray(decision.get("emotion_contribution"), dtype=float)
        total = np.asarray(decision.get("combined_scores"), dtype=float)
        mask = np.asarray(decision.get("action_mask"), dtype=int)
        if intent.shape != (6,) or emo.shape != (6,) or mask.shape != (6,):
            continue
        chosen = int(decision.get("action", -1))
        base = _intent_only_action(intent, mask)
        flipped = chosen != base
        decisions += 1
        changed += int(flipped)
        if only_changed and not flipped:
            continue

        module = state.get("emotion_module") or {}
        names = list(module.get("emotion_names") or [])
        values = list(module.get("values") or [])
        relation = module.get("relation")
        relation = None if relation is None else np.asarray(relation, dtype=float)
        params = module.get("params") if module.get("kind") == "emotion_formula" else None

        role = roles[actor]
        role_text = ROLE_NAMES[role] if role is not None else "?"
        hp = list(prev.get("health") or [])
        tag = "  <== CHANGED BY EMOTION" if flipped else ""
        out(f"step {int(prev.get('environment_step', k - 1)):>2} | seat{actor} {role_text:<7} | HP {hp}{tag}")
        if names and values:
            out("  emotion  " + "  ".join(f"{n} {float(v):.2f}" for n, v in zip(names, values)))
        if relation is not None and relation.shape == (5,):
            out("  relation " + "  ".join(f"seat{j} {relation[j]:+.3f}" for j in range(5) if j != actor))
        out("  action        intent   +emotion   =total")
        for a in range(6):
            if not mask[a]:
                continue
            marks = ("  <- intent alone" if a == base else "") + ("  <- picked" if a == chosen else "")
            target = ""
            if a > 0 and roles[a - 1] is not None:
                target = f" ({TEAM_NAMES[roles[a - 1]]})"
            out(f"  {_action_name(a):<11}{target:<10} {intent[a]:7.3f}   {emo[a]:+7.3f}   {total[a]:7.3f}{marks}")
        if flipped:
            out("  why: " + _why(base, chosen, intent, emo, relation, params))
        out("")

    winner = record.get("winning_team")
    last = states[-1] if isinstance(states[-1], Mapping) else {}
    out(f"Game over at step {last.get('environment_step', '?')}: {winner}. "
        f"Emotion changed {changed} of {decisions} dummy decisions ({changed / max(decisions, 1):.0%}).")
    return lines


def write_trace(record: Mapping[str, Any], path: str | Path) -> Path | None:
    """Write the trace next to a replay. Returns the path, or None when there is nothing to write."""

    target = Path(path)
    lines = build_trace_lines(record, title=target.stem)
    if not lines:
        return None
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print("usage: python -m bang_rl.gui.emotion_trace <replay.json> [...]")
        return 1
    for raw in paths:
        source = Path(raw)
        record = json.loads(source.read_text(encoding="utf-8"))
        written = write_trace(record, source.with_suffix(".txt"))
        print(f"{source.name}: " + (f"trace saved to {written}" if written else "no AI internals (public replay)"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_trace_lines", "write_trace"]
