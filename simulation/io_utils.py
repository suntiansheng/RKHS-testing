"""Atomic checkpoints and content hashes."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, write: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    def write(stream):
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")

    _atomic_write(path, write)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, lambda stream: frame.to_csv(stream, index=False))
