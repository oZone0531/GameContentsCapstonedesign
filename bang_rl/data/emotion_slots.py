"""Which entries of an emotion label carry information.

The emotion-formula label stores one relation entry per absolute seat
(``relation_0`` .. ``relation_4``). Two kinds of entry are fixed at the neutral
value 0.5 and say nothing about the dummy:

* the owner's own seat (a dummy has no relation to itself);
* seats of players who are already dead (relations are cleared on death).

Training or scoring those entries rewards a model for predicting a constant,
so emotion losses and errors should ignore them. Global emotions
(calm, confidence, ...) and any non-relation entry are always informative.

The mask is derived from the label names, the owner seat and the public alive
flags, so no stored data format has to change.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

PLAYER_COUNT = 5
_RELATION_NAME = re.compile(r"^relation_(\d+)$")

__all__ = ["relation_seats", "emotion_value_mask"]


def relation_seats(names: Sequence[str]) -> np.ndarray:
    """Seat index for each label entry, or -1 for a non-relation entry."""

    seats = np.full(len(names), -1, dtype=np.int64)
    for index, name in enumerate(names):
        match = _RELATION_NAME.match(str(name))
        if match is not None:
            seat = int(match.group(1))
            if 0 <= seat < PLAYER_COUNT:
                seats[index] = seat
    return seats


def emotion_value_mask(names: Sequence[str], owners: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Boolean mask of informative label entries.

    ``owners`` holds the seat whose emotion each label describes, with any
    leading shape ``S``; ``alive`` holds the matching public alive flags with
    shape ``S + (5,)``. The result has shape ``S + (len(names),)``.
    """

    seats = relation_seats(names)
    owners = np.asarray(owners, dtype=np.int64)
    alive = np.asarray(alive).astype(bool)
    if alive.shape != owners.shape + (PLAYER_COUNT,):
        raise ValueError("alive must have shape owners.shape + (5,)")
    mask = np.ones(owners.shape + (len(seats),), dtype=bool)
    for entry, seat in enumerate(seats):
        if seat < 0:
            continue
        mask[..., entry] = (owners != seat) & alive[..., seat]
    return mask
