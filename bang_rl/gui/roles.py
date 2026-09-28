"""Human-friendly, localized role labels for the GUI.

The public observation encodes roles as ``uint8`` indices in the canonical
:class:`bang_rl.config.schema.Role` order and uses ``255`` as the hidden-role
sentinel.  These helpers translate those integers into readable, localized words
such as "Sheriff" / "보안관" so the game screen never shows a bare number.
"""
from __future__ import annotations

from typing import Any

from .i18n import tr


HIDDEN_ROLE_SENTINEL = 255

# Canonical role order (matches bang_rl.config.schema.Role and the uint8
# ``role_index_uint8`` label schema used by the environment/dataset), mapped to
# translation keys resolved at render time.
_ROLE_KEYS: tuple[str, ...] = (
    "role.sheriff",
    "role.deputy",
    "role.outlaw_a",
    "role.outlaw_b",
    "role.traitor",
)

_SCHEMA_TO_KEY = {
    "sheriff": "role.sheriff",
    "deputy": "role.deputy",
    "outlaw_a": "role.outlaw_a",
    "outlaw_b": "role.outlaw_b",
    "traitor": "role.traitor",
}


def hidden_role_label() -> str:
    """Localized label for an unknown/hidden role."""

    return tr("role.hidden")


def role_name(role: Any) -> str:
    """Return a readable, localized role label for an index, enum, or string."""

    if role is None:
        return hidden_role_label()
    if hasattr(role, "value") and not isinstance(role, (int, str, bytes)):
        role = role.value
    if isinstance(role, str):
        text = role.strip()
        if not text:
            return hidden_role_label()
        key = _SCHEMA_TO_KEY.get(text.lower())
        if key is not None:
            return tr(key)
        try:
            return role_name(int(text))
        except ValueError:
            return text
    try:
        index = int(role)
    except (TypeError, ValueError):
        return hidden_role_label()
    if index == HIDDEN_ROLE_SENTINEL or index < 0 or index >= len(_ROLE_KEYS):
        return hidden_role_label()
    return tr(_ROLE_KEYS[index])


__all__ = ["HIDDEN_ROLE_SENTINEL", "hidden_role_label", "role_name"]
