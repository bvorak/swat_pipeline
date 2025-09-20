#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_project_to_sys_path(config_path: Path) -> None:
    base = config_path.resolve().parent
    project_root = (base / '..').resolve()
    sys.path.insert(0, str(project_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trabajoFM pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="baseline",
        help="Input group to use (see config.input_groups)",
    )
    args = parser.parse_args()

    _add_project_to_sys_path(args.config)

    from python_pipeline_scripts import utils, runner, recyear_writer

    config = utils.load_config(args.config)
    log = utils.get_logger(__name__, config)
    log.info("Loaded config from %s", args.config)
    log.info("Using input group: %s", args.group)

    # Example pre-processing step that would materialize RECYEAR files, etc.
    recyear_writer.write_recyear_files(args.group, config)

    # Run the main model and report outcome
    result = runner.run_swat_model(config, group=args.group)
    if not result.success:
        log.error("SWAT run failed: %s (code=%s)", result.message, result.returncode)
        return 1

    log.info("Pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
