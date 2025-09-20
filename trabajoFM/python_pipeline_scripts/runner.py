from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .utils import get_logger, get_paths, resolve_path


@dataclasses.dataclass
class RunnerResult:
    success: bool
    returncode: int
    exe: Union[str, Path]
    txtinout: Path
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    message: str = ""
    elapsed_seconds: float = 0.0


def _resolve_txtinout(path: Union[str, Path]) -> Path:
    p = Path(path).resolve()
    if "txtinout" in p.name.lower() and p.is_dir():
        return p
    # If the passed path is a project root containing TxtInOut, use that
    candidate = p / "TxtInOut"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"A folder having the string 'txtinout' in its name was not found at this path: {p}")


def run_swat(
    textinout_dir: Optional[Union[str, Path]] = None,
    exe_path: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    expect_plus: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> RunnerResult:
    """
    Run a SWAT/SWAT+ model by invoking the executable with cwd=TxtInOut.

    - textinout_dir: path to the TxtInOut dir, or to a project root that contains TxtInOut
    - exe_path: full path to swat2012.exe or swatplus-rel.exe; if None, try to resolve from config or common locations
    - timeout: optional seconds to kill the run if it hangs
    - expect_plus: set True if you know it’s SWAT+ (changes default exe name)
    - config: optional repo config to enable file logging to config/logs per settings

    Returns RunnerResult with success flag, returncode, paths to captured stdio, and message.
    """
    log = get_logger(__name__, config)

    # Resolve TxtInOut: prefer explicit argument; fallback to config.paths.base_txtinout
    txtinout: Optional[Path] = None
    if textinout_dir is not None:
        try:
            txtinout = _resolve_txtinout(textinout_dir)
        except FileNotFoundError as e:
            log.warning("Could not resolve provided TxtInOut '%s': %s", textinout_dir, e)
            txtinout = None

    if txtinout is None and config is not None:
        try:
            cfg_txt = get_paths(config)["base_txtinout"]
            txtinout = _resolve_txtinout(cfg_txt)
            log.info("Using TxtInOut from config.paths.base_txtinout: %s", txtinout)
        except Exception as e:
            log.debug("Config base_txtinout not usable: %s", e)

    if txtinout is None:
        msg = "TxtInOut not provided and not resolvable from config.paths.base_txtinout"
        log.error(msg)
        return RunnerResult(False, 2, str(exe_path or "<unknown>"), Path(".").resolve(), message=msg)
    cio = txtinout / "file.cio"
    if not cio.exists():
        msg = f"file.cio not found in {txtinout}"
        log.error(msg)
        return RunnerResult(
            success=False,
            returncode=127,
            exe=str(exe_path or "<unknown>"),
            txtinout=txtinout,
            message=msg,
        )

    # Choose sensible default EXE name if not given
    default_exe_name = "swatplus-rel.exe" if expect_plus else "swat2012.exe"

    exe: Union[str, Path]
    if exe_path:
        exe = Path(exe_path).resolve()
        if not exe.exists():
            msg = f"SWAT executable not found: {exe}"
            log.error(msg)
            return RunnerResult(False, 127, exe, txtinout, message=msg)
    else:
        # Try resolve from config first
        if config:
            cfg_exe = resolve_path(config, get_paths(config)["swat_executable"])  # already absolute
            if cfg_exe.exists():
                exe = cfg_exe
            else:
                exe = None  # type: ignore[assignment]
        else:
            exe = None  # type: ignore[assignment]

        if not exe:
            exe_in_txt = txtinout / default_exe_name
            if exe_in_txt.exists():
                exe = exe_in_txt
            else:
                exe_in_root = txtinout.parent / default_exe_name
                if exe_in_root.exists():
                    try:
                        shutil.copy2(exe_in_root, exe_in_txt)
                        exe = exe_in_txt
                    except Exception:
                        # Fall through to PATH if copy fails
                        exe = default_exe_name
                else:
                    # Last resort: rely on PATH
                    exe = default_exe_name

    log.info("Running SWAT: exe=%s | cwd=%s", exe, txtinout)

    stdout_path = txtinout / "_swat_stdout.txt"
    stderr_path = txtinout / "_swat_stderr.txt"

    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(exe)],
            cwd=str(txtinout),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        stdout_path.write_text(e.stdout or "", encoding="utf-8", errors="ignore") if e.stdout else None
        stderr_path.write_text(e.stderr or "", encoding="utf-8", errors="ignore") if e.stderr else None
        msg = f"SWAT timed out after {timeout}s"
        log.error(msg)
        return RunnerResult(False, 124, exe, txtinout, stdout_path, stderr_path, msg, elapsed)
    except FileNotFoundError as e:
        elapsed = time.perf_counter() - t0
        msg = f"Executable not found or not runnable: {exe} ({e})"
        log.error(msg)
        return RunnerResult(False, 127, exe, txtinout, None, None, msg, elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        msg = f"Unexpected error running SWAT: {e}"
        log.exception(msg)
        return RunnerResult(False, 1, exe, txtinout, None, None, msg, elapsed)

    elapsed = time.perf_counter() - t0

    # Persist stdio for later inspection
    stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="ignore")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="ignore")

    if completed.returncode != 0:
        msg = (
            "SWAT run failed. See _swat_stdout.txt/_swat_stderr.txt and .std/.out files for details."
        )
        log.error("%s | returncode=%s", msg, completed.returncode)
        return RunnerResult(False, completed.returncode, exe, txtinout, stdout_path, stderr_path, msg, elapsed)

    log.info("SWAT completed successfully in %.2fs", elapsed)
    return RunnerResult(True, completed.returncode, exe, txtinout, stdout_path, stderr_path, "OK", elapsed)


def run_swat_model(config: Dict[str, Any], group: str = "baseline", expect_plus: bool = False) -> RunnerResult:
    """High-level convenience wrapper using repo config.

    Resolves the group folder from config.input_groups[group].folder. The folder
    may be either the project root containing TxtInOut or the TxtInOut folder itself.
    The SWAT executable path is taken from config.paths.swat_executable when present.
    """
    log = get_logger(__name__, config)

    groups = (config or {}).get("input_groups", {})
    if group not in groups:
        msg = f"Unknown input group: {group}"
        log.error(msg)
        return RunnerResult(False, 2, "<unknown>", Path("."), message=msg)

    # Resolve the provided folder relative to the config file
    group_folder = resolve_path(config, groups[group].get("folder", "."))
    log.info("Preparing to run SWAT for group='%s' in %s", group, group_folder)

    # exe path from config.paths.swat_executable if available
    exe_cfg = get_paths(config)["swat_executable"]

    return run_swat(
        textinout_dir=group_folder,
        exe_path=str(exe_cfg) if exe_cfg else None,
        expect_plus=expect_plus,
        config=config,
    )
