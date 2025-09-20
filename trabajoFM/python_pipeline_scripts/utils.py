from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Annotate base directory to resolve relative paths later
    cfg.setdefault("_base_dir", str(path.parent))
    return cfg


def resolve_path(config: Dict[str, Any], relative: str | Path) -> Path:
    base = Path(config.get("_base_dir", ".")).resolve()
    return (base / relative).resolve()


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    key = name or "root"
    if key in _LOGGER_CACHE:
        return _LOGGER_CACHE[key]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_level = logging.INFO
    log_dir = None
    log_file = None
    log_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    if config is not None:
        cfg_log = config.get("logging", {})
        level_name = str(cfg_log.get("level", "INFO")).upper()
        log_level = getattr(logging, level_name, logging.INFO)
        log_dir = cfg_log.get("log_dir", None)
        log_file = cfg_log.get("file", None)
        log_fmt = cfg_log.get("format", log_fmt)

    logger.setLevel(log_level)

    formatter = logging.Formatter(log_fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir and log_file:
        if config is not None:
            base_log_dir = resolve_path(config, log_dir)
        else:
            base_log_dir = Path(log_dir)
        ensure_dir(base_log_dir)
        fh_path = base_log_dir / str(log_file)
        fh = RotatingFileHandler(fh_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGER_CACHE[key] = logger
    return logger


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    base = Path(config.get("_base_dir", ".")).resolve()
    def r(p: str | Path) -> Path:
        return (base / p).resolve()

    return {
        "project_root": r(config.get("paths", {}).get("project_root", ".")),
        "data_root": r(config.get("paths", {}).get("data_root", "data")),
        "outputs_root": r(config.get("paths", {}).get("outputs_root", "outputs")),
        "base_txtinout": r(config.get("paths", {}).get("base_txtinout", "TxtInOut")),
        "swat_executable": r(config.get("paths", {}).get("swat_executable", "swat.exe")),
    }
