"""Role taxonomy used by MainAgent inference targets.

The environment keeps ``outlaw_a`` and ``outlaw_b`` as distinct authoritative
role identities because seat assignment, replay restoration, and rule logic
need two outlaw seats.  MainAgent role inference has a semantic taxonomy,
where both seats are the same ``outlaw`` class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SEMANTIC_ROLE_COUNT = 4
SEMANTIC_ROLE_NAMES = ("sheriff", "deputy", "outlaw", "traitor")
HIDDEN_ROLE_SENTINEL = 255


def semantic_role_index(value: Any) -> int:
    """Map an environment role index/name to the four inference classes."""

    if isinstance(value, str):
        token = value.strip().lower()
        if token == "sheriff":
            return 0
        if token == "deputy":
            return 1
        if token in {"outlaw", "outlaw_a", "outlaw_b"}:
            return 2
        if token == "traitor":
            return 3
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"unknown role value: {value!r}") from exc
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("role value must not be boolean")
    index = int(value)
    if index == HIDDEN_ROLE_SENTINEL:
        return HIDDEN_ROLE_SENTINEL
    if index == 0:
        return 0
    if index == 1:
        return 1
    if index in {2, 3}:
        return 2
    if index == 4:
        return 3
    raise ValueError(f"environment role index must be in [0, 4] or 255, got {index}")


def semantic_role_labels(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(semantic_role_index(value) for value in values)


def collapse_role_distribution(value: Any) -> np.ndarray:
    """Collapse a five-class distribution into Sheriff/Deputy/Outlaw/Traitor."""

    array = np.asarray(value)
    if array.shape[-1] == SEMANTIC_ROLE_COUNT:
        return array.astype(np.float64, copy=False)
    if array.shape[-1] != 5:
        raise ValueError("role distribution must end in four or five classes")
    result = np.empty(array.shape[:-1] + (SEMANTIC_ROLE_COUNT,), dtype=np.float64)
    result[..., 0] = array[..., 0]
    result[..., 1] = array[..., 1]
    result[..., 2] = array[..., 2] + array[..., 3]
    result[..., 3] = array[..., 4]
    return result


__all__ = [
    "HIDDEN_ROLE_SENTINEL",
    "SEMANTIC_ROLE_COUNT",
    "SEMANTIC_ROLE_NAMES",
    "collapse_role_distribution",
    "semantic_role_index",
    "semantic_role_labels",
]
