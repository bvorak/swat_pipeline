from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .utils import get_logger


def parse_std(path: Path) -> Dict[str, Any]:
    log = get_logger(__name__)
    log.info("Parsing .std file at %s (stub)", path)
    # TODO: implement real parsing
    return {"path": str(path), "parsed": True}

