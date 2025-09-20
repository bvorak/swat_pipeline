from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .utils import get_logger


def read_fig(path: Path) -> Dict[str, Any]:
    log = get_logger(__name__)
    log.info("Reading FIG from %s (stub)", path)
    # TODO: implement actual FIG parsing
    return {"path": str(path), "content": "stub"}


def insert_recyear(fig: Dict[str, Any], year: int) -> Dict[str, Any]:
    log = get_logger(__name__)
    log.info("Inserting RECYEAR=%s (stub)", year)
    fig["recyear"] = year
    return fig


def write_fig(path: Path, data: Dict[str, Any]) -> None:
    log = get_logger(__name__)
    log.info("Writing FIG to %s (stub)", path)
    # TODO: implement actual FIG writing
    Path(path).write_text("FIG STUB", encoding="utf-8")

