from __future__ import annotations

import hashlib
import json
import math
import numbers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path.cwd()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    safe_payload = _to_json_safe(payload)
    path.write_text(json.dumps(safe_payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    safe_event = _to_json_safe(event)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe_event, ensure_ascii=False, allow_nan=False))
        f.write("\n")
