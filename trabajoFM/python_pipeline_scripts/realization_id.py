from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .utils import ensure_dir


IDS_FILE = Path(__file__).resolve().parent.parent / "config" / "provenance" / "ids.json"
LOCK_FILE = Path(str(IDS_FILE) + ".lock")


def _read_ids() -> dict:
    if IDS_FILE.exists():
        return json.loads(IDS_FILE.read_text(encoding="utf-8"))
    return {"last": 0}


def _write_ids(data: dict) -> None:
    ensure_dir(IDS_FILE.parent)
    IDS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def next_id(max_id: int = 1_000_000 - 1) -> int:
    """Allocate the next integer ID in [1, max_id].

    Not concurrency-proof across processes, but safe for typical single-user workflows.
    """
    ensure_dir(IDS_FILE.parent)
    data = _read_ids()
    last = int(data.get("last", 0))
    if last >= max_id:
        raise RuntimeError("No IDs left in the configured range")
    cur = last + 1
    data["last"] = cur
    _write_ids(data)
    return cur


def format_id(realization_id: int, width: int = 6) -> str:
    return f"{realization_id:0{width}d}"


def next_run_id(max_id: int = 1_000_000 - 1) -> int:
    """Allocate the next Monte Carlo run/batch ID.

    Backed by the same ids.json file under the key 'last_run'.
    """
    ensure_dir(IDS_FILE.parent)
    data = _read_ids()
    last = int(data.get("last_run", 0))
    if last >= max_id:
        raise RuntimeError("No run IDs left in the configured range")
    cur = last + 1
    data["last_run"] = cur
    _write_ids(data)
    return cur
