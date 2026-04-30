from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return loaded


def resolve_path(base_file: str | Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(base_file).resolve().parent / candidate).resolve()
