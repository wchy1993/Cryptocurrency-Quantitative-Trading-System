from __future__ import annotations

import os
from pathlib import Path


def read_secret(name: str, env_file: str | Path = ".env") -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()

    path = Path(env_file)
    if not path.exists():
        return None

    prefix = f"{name}="
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith(prefix):
                continue
            return line[len(prefix) :].strip().strip('"').strip("'")
    return None


def mask_secret(value: str | None, visible_prefix: int = 5, visible_suffix: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= visible_prefix + visible_suffix:
        return "***"
    return f"{value[:visible_prefix]}...{value[-visible_suffix:]}"
