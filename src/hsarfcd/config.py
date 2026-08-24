from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty/non-mapping configuration files."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    """Resolve user paths without embedding machine-specific paths in source code."""
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base).expanduser() / path
    return path.resolve()


def require_keys(mapping: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing required configuration keys: {', '.join(missing)}")

