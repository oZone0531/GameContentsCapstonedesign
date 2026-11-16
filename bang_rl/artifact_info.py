"""Self-describing metadata and naming for every generated artifact (headless, no Qt).

Every generated file or folder (collected datasets, tensor caches, split
manifests, inference modules, main-AI models, reports, baselines, replays, ...)
carries an :class:`ArtifactInfo` block: what it is, when and how it was made,
the initial conditions, a short result summary, where it came from and a free
text note (``notes``, "비고") that the user can edit.

Where the info lives (the *source of truth* per format):

* JSON files (reports, replays, baselines, summaries): first key ``"info"``.
* ``.phase2model`` (torch pickle): payload key ``"info"`` plus ``<file>.about.md``.
* ``.bangmodel`` (ZIP): first key ``"info"`` of ``metadata.json`` (the manifest
  ``metadata_sha256`` is recomputed) plus ``<file>.about.md``.
* Folders (datasets, tensor caches): ``ABOUT.md`` inside the folder. Hash
  verified content (``*.npz``, ``cache-*.manifest.json``, ``split_manifest.json``,
  parquet shards) is never modified, so no content hash can change.

``<file>.about.md`` / ``ABOUT.md`` are human readable (Korean) and end with a
fenced JSON block (machine readable). The note text sits between
``<!-- notes:begin -->`` and ``<!-- notes:end -->`` and may be edited by hand.

The file *name* is never needed to find or pair anything: the stable ``id``
inside the info is. Renaming files in the file explorer is therefore safe.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

INFO_FORMAT_VERSION = 1
INFO_KEY = "info"
FOLDER_SIDECAR = "ABOUT.md"
FILE_SIDECAR_SUFFIX = ".about.md"
MAX_NAME_LENGTH = 90
MAX_USER_NAME_LENGTH = 120

KIND_DATASET = "dataset"
KIND_TENSOR_CACHE = "tensor_cache"
KIND_SPLIT = "split"
KIND_INTENT = "intent_module"
KIND_EMOTION = "emotion_module"
KIND_MAIN_AI = "main_ai"
KIND_REPORT = "report"
KIND_BASELINE = "baseline"
KIND_EVALUATION = "evaluation"
KIND_REPLAY = "replay"

# kind -> (Korean label, English label, ASCII token used in file names)
KINDS: dict[str, tuple[str, str, str]] = {
    KIND_DATASET: ("학습 데이터", "Training dataset", "dataset"),
    KIND_TENSOR_CACHE: ("텐서 캐시", "Tensor cache", "cache"),
    KIND_SPLIT: ("데이터 분할 목록", "Split manifest", "split"),
    KIND_INTENT: ("의도 추론 모듈", "Intent module", "intent"),
    KIND_EMOTION: ("감정 추론 모듈", "Emotion module", "emotion"),
    KIND_MAIN_AI: ("메인 AI", "Main AI", "mainai"),
    KIND_REPORT: ("학습 보고서", "Training report", "report"),
    KIND_BASELINE: ("기준선 캐시", "Baseline cache", "baseline"),
    KIND_EVALUATION: ("평가 결과", "Evaluation output", "eval"),
    KIND_REPLAY: ("리플레이", "Replay", "replay"),
}

_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_NOTES_BEGIN = "<!-- notes:begin -->"
_NOTES_END = "<!-- notes:end -->"
_INFO_BEGIN = "<!-- info:begin -->"
_INFO_END = "<!-- info:end -->"


def now_iso() -> str:
    """Local time with UTC offset, second resolution."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def software_versions() -> dict[str, Any]:
    try:
        from importlib.metadata import version

        package = version("bang_rl")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        package = "unknown"
    return {"bang_rl": package, "info_format_version": INFO_FORMAT_VERSION}


@dataclass(slots=True)
class ArtifactInfo:
    """Metadata block stored in or next to every generated artifact."""

    kind: str
    id: str
    name: str = ""
    created_at: str = field(default_factory=now_iso)
    created_by: str = ""
    notes: str = ""
    kind_label: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    software: dict[str, Any] = field(default_factory=software_versions)
    format_version: int = INFO_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.kind_label:
            self.kind_label = KINDS.get(self.kind, (self.kind, self.kind, self.kind))[0]
        if not self.name:
            self.name = default_artifact_name(self)

    def to_dict(self) -> dict[str, Any]:
        # Key order is the reading order of the file: identity and note first.
        return {
            "kind": self.kind,
            "kind_label": self.kind_label,
            "name": self.name,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "notes": self.notes,
            "id": self.id,
            "conditions": _json_safe(self.conditions),
            "summary": _json_safe(self.summary),
            "sources": _json_safe(self.sources),
            "software": _json_safe(self.software),
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactInfo":
        kind = str(data.get("kind") or "")
        if not kind:
            raise ValueError("artifact info has no kind")
        sources = data.get("sources")
        info = cls(
            kind=kind,
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            created_at=str(data.get("created_at") or ""),
            created_by=str(data.get("created_by") or ""),
            notes=str(data.get("notes") or ""),
            kind_label=str(data.get("kind_label") or ""),
            conditions=dict(data["conditions"]) if isinstance(data.get("conditions"), Mapping) else {},
            summary=dict(data["summary"]) if isinstance(data.get("summary"), Mapping) else {},
            sources=[dict(item) for item in sources if isinstance(item, Mapping)] if isinstance(sources, (list, tuple)) else [],
            software=dict(data["software"]) if isinstance(data.get("software"), Mapping) else {},
            format_version=int(data.get("format_version") or INFO_FORMAT_VERSION),
        )
        return info

    @property
    def short_id(self) -> str:
        return short_id(self.id)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


# --------------------------------------------------------------------- naming


def short_id(identifier: str) -> str:
    """First 6 hex characters of an id (``job-`` prefixes and dashes ignored)."""

    text = re.sub(r"[^0-9a-zA-Z]", "", str(identifier).replace("job-", "", 1))
    return text[:6].lower() or "000000"


def ascii_token(value: Any) -> str:
    """Lowercase ``[a-z0-9.-]`` token (everything else collapses to ``-``)."""

    text = re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower())
    return text.strip("-.")


def _stamp(created_at: str) -> str:
    try:
        return datetime.fromisoformat(created_at).strftime("%Y%m%d-%H%M")
    except (TypeError, ValueError):
        return "00000000-0000"


def _count_token(games: Any) -> str:
    try:
        return f"{int(games)}g"
    except (TypeError, ValueError):
        return ""


def _condition_tokens(info: ArtifactInfo) -> list[str]:
    cond = info.conditions
    kind = info.kind
    tokens: list[str] = []
    if kind == KIND_DATASET:
        tokens = [ascii_token(cond.get("players", "")), _count_token(cond.get("games"))]
    elif kind in (KIND_INTENT, KIND_EMOTION):
        epochs = cond.get("total_epochs", cond.get("epochs"))
        tokens = [f"e{int(epochs)}" if isinstance(epochs, (int, float)) and not isinstance(epochs, bool) else ""]
        datasets = [item for item in info.sources if item.get("kind") == KIND_DATASET and item.get("id")]
        if datasets:
            extra = f"+{len(datasets) - 1}" if len(datasets) > 1 else ""
            tokens.append(f"from-{short_id(datasets[0]['id'])}{extra}")
    elif kind == KIND_MAIN_AI:
        # A model with refined inference modules says so instead of a game count.
        tokens = ["refined"] if info.created_by == "inference_refinement" else [_count_token(cond.get("training_games"))]
        module = next((item for item in info.sources if item.get("kind") in (KIND_INTENT, KIND_EMOTION)), None)
        if module and module.get("id"):
            tokens.append(f"mod-{short_id(module['id'])}")
    elif kind == KIND_REPORT:
        tokens = [ascii_token(str(info.created_by).replace("_", "-"))]
    elif kind == KIND_REPLAY:
        tokens = [ascii_token(cond.get("players", cond.get("mode", "")))]
    elif kind in (KIND_BASELINE, KIND_EVALUATION):
        tokens = [_count_token(cond.get("games")), ascii_token(cond.get("label", ""))]
    elif kind in (KIND_TENSOR_CACHE, KIND_SPLIT):
        source = info.sources[0] if info.sources else {}
        if source.get("id"):
            tokens = [f"from-{short_id(source['id'])}"]
    return [token for token in tokens if token]


def default_artifact_name(info: ArtifactInfo) -> str:
    """ASCII, sortable name ``YYYYMMDD-HHMM_<kind>_<conditions>_<shortid>``.

    Always at most :data:`MAX_NAME_LENGTH` characters; the short id (the part
    that tells two same-minute artifacts apart) is never cut.
    """

    token = KINDS.get(info.kind, (info.kind, info.kind, ascii_token(info.kind)))[2]
    head = [_stamp(info.created_at), token]
    tail = short_id(info.id)
    middle = _condition_tokens(info)
    parts = head + middle + [tail]
    name = "_".join(parts)
    while len(name) > MAX_NAME_LENGTH and middle:
        middle.pop()
        name = "_".join(head + middle + [tail])
    return name[:MAX_NAME_LENGTH].rstrip("_-.")


def sanitize_user_name(name: str) -> str:
    """Make a user typed name safe as a Windows file/folder name.

    Non-ASCII (Korean) text is kept. Only forbidden characters, trailing dots
    and spaces, and reserved device names are handled. Raises ``ValueError``
    when nothing usable is left.
    """

    text = _WINDOWS_FORBIDDEN.sub("_", str(name))
    text = re.sub(r"\s+", " ", text).strip().rstrip(". ")
    text = text[:MAX_USER_NAME_LENGTH].rstrip(". ")
    if not text:
        raise ValueError("name must not be empty")
    if text.split(".")[0].lower() in _WINDOWS_RESERVED:
        text = f"_{text}"
    return text


# ------------------------------------------------------------------- file I/O


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_dir():
        return target / FOLDER_SIDECAR
    return target.with_name(target.name + FILE_SIDECAR_SUFFIX)


def _is_json_file(path: Path) -> bool:
    return path.suffix.lower() == ".json"


def _is_phase2(path: Path) -> bool:
    return path.suffix.lower() == ".phase2model"


def _is_bangmodel(path: Path) -> bool:
    return path.suffix.lower() == ".bangmodel"


def render_sidecar(info: ArtifactInfo) -> str:
    """Human readable (Korean) text followed by the machine readable JSON."""

    def line(label: str, value: Any) -> str:
        return f"- **{label}**: {value}"

    english = KINDS.get(info.kind, (info.kind, info.kind, info.kind))[1]
    out = [
        f"# {info.kind_label} ({english})",
        "",
        line("이름", info.name),
        line("종류", f"{info.kind_label} / `{info.kind}`"),
        line("생성 시각", info.created_at),
        line("생성 방법", info.created_by or "-"),
        line("고유 ID", f"`{info.id}`"),
        "",
        "## 비고 (NOTES)",
        "",
        "아래 구역 안에 자유롭게 적으세요. 프로그램의 '비고' 편집과 같은 내용입니다.",
        "",
        _NOTES_BEGIN,
        info.notes,
        _NOTES_END,
        "",
        "## 초기 조건",
        "",
        *_bullets(info.conditions),
        "",
        "## 결과 요약",
        "",
        *_bullets(info.summary),
        "",
        "## 원본 (이 파일을 만들 때 사용한 입력)",
        "",
        *(
            [f"- {item.get('kind_label') or item.get('kind')}: {item.get('name') or '-'} (`{item.get('id')}`)" for item in info.sources]
            or ["- 없음"]
        ),
        "",
        "## 기계용 정보",
        "",
        "파일 이름을 바꿔도 아래의 id로 짝과 출처를 찾습니다. 이 블록은 직접 고치지 마세요.",
        "",
        _INFO_BEGIN,
        "```json",
        json.dumps(info.to_dict(), ensure_ascii=False, indent=2),
        "```",
        _INFO_END,
        "",
    ]
    return "\n".join(out)


def _bullets(values: Mapping[str, Any]) -> list[str]:
    if not values:
        return ["- 없음"]
    rows: list[str] = []
    for key, value in values.items():
        if isinstance(value, (Mapping, list, tuple)):
            text = json.dumps(_json_safe(value), ensure_ascii=False)
            if len(text) > 300:
                text = text[:297] + "..."
        else:
            text = str(value)
        rows.append(f"- {key}: {text}")
    return rows


def parse_sidecar(text: str) -> ArtifactInfo | None:
    start = text.find(_INFO_BEGIN)
    end = text.find(_INFO_END)
    if start < 0 or end < start:
        return None
    block = text[start + len(_INFO_BEGIN) : end]
    match = re.search(r"```json\s*(.*?)```", block, re.S)
    if not match:
        return None
    try:
        info = ArtifactInfo.from_dict(json.loads(match.group(1)))
    except (ValueError, TypeError):
        return None
    notes = _sidecar_notes(text)
    if notes is not None:
        info.notes = notes
    return info


def _sidecar_notes(text: str) -> str | None:
    begin = text.find(_NOTES_BEGIN)
    end = text.find(_NOTES_END)
    if begin < 0 or end < begin:
        return None
    return text[begin + len(_NOTES_BEGIN) : end].strip("\r\n").strip()


def _write_sidecar(path: Path, info: ArtifactInfo) -> Path:
    target = sidecar_path(path)
    _atomic_write_bytes(target, render_sidecar(info).encode("utf-8"))
    return target


write_sidecar = _write_sidecar


def _read_sidecar(path: Path) -> ArtifactInfo | None:
    target = sidecar_path(path)
    if not target.is_file():
        return None
    try:
        return parse_sidecar(target.read_text(encoding="utf-8"))
    except OSError:
        return None


# -- JSON files

_HEAD_BYTES = 64 * 1024


def _json_info_from_text(text: str) -> ArtifactInfo | None:
    """Parse the leading ``"info"`` object of a JSON document without loading the rest."""

    match = re.match(r'\s*\{\s*"info"\s*:\s*', text)
    if not match:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, match.end())
    except ValueError:
        return None
    if not isinstance(value, Mapping):
        return None
    try:
        return ArtifactInfo.from_dict(value)
    except (ValueError, TypeError):
        return None


def _read_json_info(path: Path) -> ArtifactInfo | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(_HEAD_BYTES)
            info = _json_info_from_text(head)
            if info is not None:
                return info
            if not re.match(r'\s*\{\s*"info"\s*:', head):
                return None
            # The info block is longer than the head (very long notes).
            return _json_info_from_text(head + handle.read())
    except OSError:
        return None


def dump_json_with_info(
    data: Mapping[str, Any],
    info: ArtifactInfo | None,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    """JSON text with ``"info"`` as the first key (rest keeps its order)."""

    rest = {key: value for key, value in data.items() if key != INFO_KEY}
    if sort_keys:
        rest = dict(sorted(rest.items()))
    document: dict[str, Any] = {}
    if info is not None:
        document[INFO_KEY] = info.to_dict()
    elif isinstance(data.get(INFO_KEY), Mapping):
        document[INFO_KEY] = data[INFO_KEY]
    document.update(rest)
    return json.dumps(document, ensure_ascii=False, indent=indent)


def _embed_json(path: Path, info: ArtifactInfo) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object; cannot add info")
    _atomic_write_bytes(path, (dump_json_with_info(data, info) + "\n").encode("utf-8"))


# -- .phase2model

def _read_phase2(path: Path) -> ArtifactInfo | None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - unreadable file has no info
        return None
    value = payload.get(INFO_KEY) if isinstance(payload, Mapping) else None
    if not isinstance(value, Mapping):
        return None
    try:
        return ArtifactInfo.from_dict(value)
    except (ValueError, TypeError):
        return None


def _embed_phase2(path: Path, info: ArtifactInfo) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} is not a module artifact")
    updated = {INFO_KEY: info.to_dict()}
    updated.update({key: value for key, value in payload.items() if key != INFO_KEY})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(updated, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# -- .bangmodel

def _read_bangmodel_info(path: Path) -> ArtifactInfo | None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            metadata_bytes = archive.read("metadata.json")
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    info = _json_info_from_text(metadata_bytes[:_HEAD_BYTES].decode("utf-8", errors="ignore"))
    if info is not None:
        return info
    try:
        value = json.loads(metadata_bytes).get(INFO_KEY)
        return ArtifactInfo.from_dict(value) if isinstance(value, Mapping) else None
    except (ValueError, TypeError, AttributeError):
        return None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def metadata_bytes_with_info(metadata: Mapping[str, Any], info: ArtifactInfo | None) -> bytes:
    """Canonical ``metadata.json`` bytes with ``info`` as the first key."""

    rest = {key: value for key, value in metadata.items() if key != INFO_KEY}
    body = _canonical(rest)
    if info is None:
        return body
    head = json.dumps(info.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if body == b"{}":
        return b'{"info":' + head + b"}"
    return b'{"info":' + head + b"," + body[1:]


def _embed_bangmodel(path: Path, info: ArtifactInfo) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        metadata = json.loads(archive.read("metadata.json"))
        weights_name = str(manifest.get("weights_file", "weights.safetensors"))
        weights = archive.read(weights_name)
    metadata_bytes = metadata_bytes_with_info(metadata, info)
    manifest["metadata_sha256"] = sha256(metadata_bytes).hexdigest()
    members = {"manifest.json": _canonical(manifest), "metadata.json": metadata_bytes, weights_name: weights}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
                for name in ("manifest.json", "metadata.json", weights_name):
                    archive.writestr(name, members[name])
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------ public API

_read_cache: dict[tuple[str, int, int, int, int], ArtifactInfo | None] = {}


def _stamp_key(path: Path) -> tuple[str, int, int, int, int]:
    stat = path.stat()
    side = sidecar_path(path)
    if side.is_file():
        side_stat = side.stat()
        side_stamp, side_size = side_stat.st_mtime_ns, side_stat.st_size
    else:
        side_stamp, side_size = 0, -1
    return (str(path), stat.st_mtime_ns, stat.st_size, side_stamp, side_size)


def _forget(path: Path) -> None:
    """Drop cached reads of ``path``.

    File times can be coarse (two writes in the same clock tick keep the same
    mtime), so every write through this module invalidates explicitly instead
    of relying on the stamp alone.
    """

    target = str(path)
    for key in [key for key in _read_cache if key[0] == target]:
        del _read_cache[key]


def read_info(path: str | Path) -> ArtifactInfo | None:
    """Info of any artifact (file or folder), or ``None`` when it has none.

    The embedded info is the source of truth; for binary files the sidecar
    supplies the (user editable) notes and is the fallback when the embedded
    block cannot be read. Results are cached per file stamp.
    """

    target = Path(path)
    if not target.exists():
        return None
    try:
        key = _stamp_key(target)
    except OSError:
        return None
    if key in _read_cache:
        cached = _read_cache[key]
        return None if cached is None else ArtifactInfo.from_dict(cached.to_dict())
    info = _read_uncached(target)
    if len(_read_cache) > 2048:
        _read_cache.clear()
    _read_cache[key] = info
    return None if info is None else ArtifactInfo.from_dict(info.to_dict())


def _read_uncached(target: Path) -> ArtifactInfo | None:
    if target.is_dir():
        return _read_sidecar(target)
    if _is_json_file(target):
        # A JSON list (or any file we must not restructure) keeps its info in a sidecar.
        return info_from_embedded(target, _read_json_info(target))
    embedded: ArtifactInfo | None = None
    if _is_phase2(target):
        embedded = _read_phase2(target)
    elif _is_bangmodel(target):
        embedded = _read_bangmodel_info(target)
    return info_from_embedded(target, embedded)


def info_from_embedded(path: str | Path, embedded: ArtifactInfo | None) -> ArtifactInfo | None:
    """Combine an already read embedded info with the sidecar of ``path``.

    The sidecar supplies the user editable notes (and is the fallback when the
    embedded block is gone). Lets callers that already loaded the file avoid a
    second read.
    """

    side = _read_sidecar(Path(path))
    if embedded is None:
        return side
    if side is not None and side.id == embedded.id:
        embedded.notes = side.notes
        embedded.name = side.name or embedded.name
    return embedded


def info_from_payload(payload: Any) -> ArtifactInfo | None:
    """Info stored under ``"info"`` of an already loaded mapping."""

    value = payload.get(INFO_KEY) if isinstance(payload, Mapping) else None
    if not isinstance(value, Mapping):
        return None
    try:
        return ArtifactInfo.from_dict(value)
    except (ValueError, TypeError):
        return None


_id_index_cache: dict[str, tuple[tuple[int, int], str | None]] = {}


def find_json_by_id(directories: "list[Path] | tuple[Path, ...]", identifier: str, *, recursive: bool = True) -> Path | None:
    """First JSON artifact whose ``info.id`` is ``identifier`` (name independent).

    Only the leading ``info`` block of each file is parsed and the result is
    cached per file stamp, so scanning a reports folder stays cheap.
    """

    wanted = str(identifier)
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        iterator = root.rglob("*.json") if recursive else root.glob("*.json")
        for path in sorted(iterator):
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = _id_index_cache.get(str(path))
            if cached is None or cached[0] != stamp:
                info = _read_json_info(path)
                cached = (stamp, info.id if info else None)
                _id_index_cache[str(path)] = cached
            if cached[1] == wanted:
                return path
    return None


def has_info(path: str | Path) -> bool:
    return read_info(path) is not None


def write_info(path: str | Path, info: ArtifactInfo) -> Path | None:
    """Store ``info`` in/next to ``path`` (embedded where the format allows) and
    (re)write the human readable sidecar. Returns the sidecar path, if any."""

    target = Path(path)
    try:
        return _write_info(target, info)
    finally:
        _forget(target)


def _write_info(target: Path, info: ArtifactInfo) -> Path | None:
    if target.is_dir():
        _write_sidecar(target, info)
        return sidecar_path(target)
    if _is_json_file(target):
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _embed_json(target, info)
            return None
        # Top level is a list: restructuring it would break readers, so the
        # info goes into ``<file>.about.md`` instead.
        return _write_sidecar(target, info)
    if _is_phase2(target):
        _embed_phase2(target, info)
    elif _is_bangmodel(target):
        _embed_bangmodel(target, info)
    return _write_sidecar(target, info)


def ensure_sidecar(path: str | Path) -> Path | None:
    """Regenerate a missing ``.about.md`` / ``ABOUT.md`` from the embedded info."""

    target = Path(path)
    side = sidecar_path(target)
    if _is_json_file(target) and _read_json_info(target) is not None:
        return None  # embedded in the file itself
    if side.is_file():
        return side
    info = read_info(target)
    if info is None:
        return None
    try:
        return _write_sidecar(target, info)
    finally:
        _forget(target)


def update_notes(path: str | Path, text: str) -> ArtifactInfo:
    """Replace the note text in the embedded info and the sidecar."""

    info = read_info(path)
    if info is None:
        raise ValueError(f"{path} has no artifact info")
    info.notes = str(text)
    write_info(path, info)
    return info


def update_name(path: str | Path, name: str) -> ArtifactInfo:
    info = read_info(path)
    if info is None:
        raise ValueError(f"{path} has no artifact info")
    info.name = name
    write_info(path, info)
    return info


def display_name(path: str | Path) -> str:
    """Readable name: the info name when present, else the file name."""

    info = read_info(path)
    if info is not None and info.name:
        return info.name
    target = Path(path)
    return target.name if target.is_dir() else target.stem


def info_tooltip(info: ArtifactInfo | None) -> str:
    """Plain text summary (kind, name, time, notes) for tooltips and lists."""

    if info is None:
        return ""
    lines = [f"{info.kind_label}: {info.name}", f"{info.created_at}"]
    if info.created_by:
        lines.append(info.created_by)
    if info.notes.strip():
        lines.append("")
        lines.append(info.notes.strip())
    return "\n".join(lines)


def new_info(
    kind: str,
    *,
    id: str,
    created_by: str,
    conditions: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    sources: list[Mapping[str, Any]] | None = None,
    notes: str = "",
    created_at: str | None = None,
    kind_label: str | None = None,
    name: str | None = None,
) -> ArtifactInfo:
    """Build an info block; the name defaults to the naming rule."""

    return ArtifactInfo(
        kind=kind,
        id=str(id),
        name=name or "",
        created_at=created_at or now_iso(),
        created_by=created_by,
        notes=notes,
        kind_label=kind_label or "",
        conditions=_json_safe(dict(conditions or {})),
        summary=_json_safe(dict(summary or {})),
        sources=_json_safe([dict(item) for item in (sources or ())]),
    )


def source_ref(kind: str, identifier: str, name: str = "") -> dict[str, Any]:
    """Reference to the input an artifact was made from."""

    label = KINDS.get(kind, (kind, kind, kind))[0]
    return {"kind": kind, "kind_label": label, "id": str(identifier), "name": name}


__all__ = [
    "ArtifactInfo",
    "FILE_SIDECAR_SUFFIX",
    "FOLDER_SIDECAR",
    "INFO_KEY",
    "KINDS",
    "KIND_BASELINE",
    "KIND_DATASET",
    "KIND_EMOTION",
    "KIND_EVALUATION",
    "KIND_INTENT",
    "KIND_MAIN_AI",
    "KIND_REPLAY",
    "KIND_REPORT",
    "KIND_SPLIT",
    "KIND_TENSOR_CACHE",
    "default_artifact_name",
    "display_name",
    "dump_json_with_info",
    "ensure_sidecar",
    "find_json_by_id",
    "info_from_embedded",
    "info_from_payload",
    "has_info",
    "info_tooltip",
    "metadata_bytes_with_info",
    "new_info",
    "now_iso",
    "read_info",
    "sanitize_user_name",
    "short_id",
    "sidecar_path",
    "source_ref",
    "update_name",
    "update_notes",
    "write_info",
    "write_sidecar",
]
