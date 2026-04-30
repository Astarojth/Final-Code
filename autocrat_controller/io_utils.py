from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {src}")
    rows: List[Dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} is not a JSON object: {src}")
            rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return dst


def read_json(path: str | Path) -> Dict[str, Any]:
    src = Path(path).expanduser().resolve()
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {src}")
    return data


def write_json(path: str | Path, obj: Dict[str, Any]) -> Path:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")
    return dst
