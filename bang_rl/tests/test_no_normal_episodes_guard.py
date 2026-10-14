"""Pure tests for the no-normal-episodes guard (FEAT-002).

When collection produces episodes but none of them terminate 'normal' (all
truncated), the training runner must NOT reach the tensor cache, which would
raise the raw English DataError
("validation/test cache requires normal training episodes for statistics").
Instead the runner raises :class:`NoNormalEpisodesError`, whose message is a
clear, localized string surfaced to the GUI.

These tests run in the restricted sandbox: ``bang_rl.exceptions`` is numpy /
torch / pydantic free, so it imports directly here; the i18n key presence is
checked by inspecting ``bang_rl/gui/i18n.py`` SOURCE (importing the GUI module
would pull Qt, which is not installable in-sandbox), mirroring the
source-inspection pattern used by ``test_training_runner_wiring.py`` test C.
"""

from __future__ import annotations

import ast
from pathlib import Path


# The raw English DataError text the cache would otherwise raise.  The whole
# point of the guard is that the user never sees THIS string.
_RAW_DATA_ERROR_TEXT = (
    "validation/test cache requires normal training episodes for statistics"
)


def _i18n_module_path() -> Path:
    return Path(__file__).parents[1] / "gui" / "i18n.py"


# --- Exception message / payload (pure, no numpy) ----------------------- #


def test_no_normal_episodes_error_message_is_localized_not_raw_data_error() -> None:
    from bang_rl.exceptions import (
        NO_NORMAL_EPISODES_MESSAGE_EN,
        NO_NORMAL_EPISODES_MESSAGE_KO,
        NoNormalEpisodesError,
    )

    # Both localized strings exist and are non-empty.
    assert NO_NORMAL_EPISODES_MESSAGE_KO.strip()
    assert NO_NORMAL_EPISODES_MESSAGE_EN.strip()
    # They are NOT the raw English cache DataError text.
    assert NO_NORMAL_EPISODES_MESSAGE_KO != _RAW_DATA_ERROR_TEXT
    assert NO_NORMAL_EPISODES_MESSAGE_EN != _RAW_DATA_ERROR_TEXT
    assert _RAW_DATA_ERROR_TEXT not in NO_NORMAL_EPISODES_MESSAGE_KO
    assert _RAW_DATA_ERROR_TEXT not in NO_NORMAL_EPISODES_MESSAGE_EN
    # ko is the primary language carried by str(exc) / error['message'].
    assert any(ord(ch) > 0x7F for ch in NO_NORMAL_EPISODES_MESSAGE_KO)

    error = NoNormalEpisodesError(episodes_completed=1, truncated_episodes=1)
    assert str(error) == NO_NORMAL_EPISODES_MESSAGE_KO
    assert error.message == NO_NORMAL_EPISODES_MESSAGE_KO
    assert _RAW_DATA_ERROR_TEXT not in str(error)


def test_no_normal_episodes_error_to_dict_carries_code_and_both_languages() -> None:
    from bang_rl.exceptions import (
        NO_NORMAL_EPISODES_MESSAGE_EN,
        NO_NORMAL_EPISODES_MESSAGE_KO,
        BangRLError,
        NoNormalEpisodesError,
    )

    error = NoNormalEpisodesError(episodes_completed=2, truncated_episodes=2)
    assert isinstance(error, BangRLError)

    payload = error.to_dict()
    # The services layer _error_dict puts exc.to_dict() into run.error for a
    # BangRLError, so the GUI reads these fields.
    assert payload["error_type"] == "NoNormalEpisodesError"
    assert payload["code"] == "no_normal_episodes"
    # error['message'] is the localized ko text, never the raw DataError text.
    assert payload["message"] == NO_NORMAL_EPISODES_MESSAGE_KO
    assert payload["message"] != _RAW_DATA_ERROR_TEXT
    # Both languages are available in context so the GUI can prefer the active
    # language.
    context = payload["context"]
    assert context["message_ko"] == NO_NORMAL_EPISODES_MESSAGE_KO
    assert context["message_en"] == NO_NORMAL_EPISODES_MESSAGE_EN
    assert context["episodes_completed"] == 2
    assert context["truncated_episodes"] == 2


# --- i18n key presence (source inspection, Qt-free) --------------------- #


def _i18n_translations() -> dict:
    """Parse the _STRINGS dict literal from gui/i18n.py without importing Qt."""

    source = _i18n_module_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            else:
                names = [node.target.id] if isinstance(node.target, ast.Name) else []
            if "_STRINGS" in names and isinstance(node.value, ast.Dict):
                return ast.literal_eval(node.value)
    raise AssertionError("_STRINGS dict not found in gui/i18n.py")


def test_i18n_defines_no_normal_episodes_key_for_ko_and_en() -> None:
    from bang_rl.exceptions import (
        NO_NORMAL_EPISODES_MESSAGE_EN,
        NO_NORMAL_EPISODES_MESSAGE_KO,
    )

    translations = _i18n_translations()
    assert "training.no_normal_episodes" in translations
    entry = translations["training.no_normal_episodes"]

    assert entry.get("ko", "").strip()
    assert entry.get("en", "").strip()
    # The i18n values must not be the raw English cache DataError text.
    assert entry["ko"] != _RAW_DATA_ERROR_TEXT
    assert entry["en"] != _RAW_DATA_ERROR_TEXT
    assert _RAW_DATA_ERROR_TEXT not in entry["ko"]
    assert _RAW_DATA_ERROR_TEXT not in entry["en"]
    # The i18n strings mirror the exception module constants (kept in sync).
    assert entry["ko"] == NO_NORMAL_EPISODES_MESSAGE_KO
    assert entry["en"] == NO_NORMAL_EPISODES_MESSAGE_EN
    # No em dashes allowed in localized strings.
    assert "\u2014" not in entry["ko"]
    assert "\u2014" not in entry["en"]


# --- Runner guard wiring (source inspection, Qt / numpy free) ----------- #


def test_runner_raises_guard_before_tensor_cache_writer() -> None:
    """The runner raises NoNormalEpisodesError before calling write_all.

    Inspect the runner source (importing it would pull torch) and confirm the
    zero-normal-episodes guard appears before the NPZTensorCacheWriter call and
    that the runner stays Qt-free.
    """

    source = (
        Path(__file__).parents[1] / "training" / "pipeline_runner.py"
    ).read_text(encoding="utf-8")

    assert "NoNormalEpisodesError" in source
    assert "normal_episodes == 0" in source

    guard_pos = source.index("normal_episodes == 0")
    writer_pos = source.index("NPZTensorCacheWriter(")
    assert guard_pos < writer_pos, "the guard must precede the tensor-cache writer"

    lowered = source.lower()
    assert "pyqt" not in lowered
    assert "pyqtgraph" not in lowered
    # The runner must not import the GUI i18n layer.
    assert "bang_rl.gui" not in source
