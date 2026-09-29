"""Project-local paths used by the GUI training workflow."""
from __future__ import annotations

from pathlib import Path

# ``bang_rl`` is installed editable in this project, so this remains the
# project root instead of depending on the process current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# In-project replay storage. Mirrors the models/logs/datasets convention
# (bang_rl/replays/* gitignored except a .gitkeep placeholder). Every
# human-played GUI game is auto-saved here so it can be browsed and replayed.
REPLAYS_DIR = PROJECT_ROOT / "bang_rl" / "replays"


def ensure_training_directories() -> None:
    """Create the project-local training roots if they are missing."""

    for directory in (REPORTS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR, REPLAYS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_replays_directory() -> Path:
    """Create and return the in-project replay-storage directory."""

    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    return REPLAYS_DIR


def default_training_paths() -> tuple[Path, Path, Path]:
    """Return report, output, and artifact roots after ensuring they exist."""

    ensure_training_directories()
    return REPORTS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR


def resolve_training_file(
    value: str | Path | None,
    *,
    root: Path,
    filename: str,
    suffix: str,
) -> Path:
    """Turn a configured root/directory into a unique per-job file path."""

    candidate = root if value is None else Path(value).expanduser()
    if candidate.exists() and candidate.is_dir():
        destination = candidate / filename
    elif candidate.suffix.lower() != suffix.lower():
        # A suffix-less value is treated as a directory so the GUI can expose
        # the simple project root (for example ``artifacts``).
        candidate.mkdir(parents=True, exist_ok=True)
        destination = candidate / filename
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        destination = candidate
    return destination


def latest_artifact(value: str | Path | None) -> Path | None:
    """Resolve a model file or the newest ``.bangmodel`` in a model root."""

    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        return None
    models = sorted(
        (path for path in candidate.glob("*.bangmodel") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return models[0] if models else None


__all__ = [
    "ARTIFACTS_DIR",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "REPLAYS_DIR",
    "REPORTS_DIR",
    "default_training_paths",
    "ensure_replays_directory",
    "ensure_training_directories",
    "latest_artifact",
    "resolve_training_file",
]
