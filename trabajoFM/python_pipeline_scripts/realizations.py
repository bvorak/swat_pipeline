from __future__ import annotations

import dataclasses
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional, Union

from .utils import get_logger
from . import runner as _runner


@dataclasses.dataclass(frozen=True)
class RealizationSpec:
    """Specification of a single realization.

    - name: semantic name used for logs and result folders
    - folder: path to the folder that contains replacement files mirroring
      the structure under TxtInOut
    """

    name: str
    folder: Union[str, Path]


@dataclasses.dataclass
class RealizationRunResult:
    name: str
    success: bool
    message: str
    returncode: int
    outputs_dir: Optional[Path]
    runner: _runner.RunnerResult
    linked_count: int
    skipped_missing: int
    matched_count: int
    missing_in_realization: int
    extra_in_realization: int
    elapsed_seconds: float


def _compile_regexes(patterns: Iterable[str]):
    import re

    return [re.compile(p) for p in patterns]


def _iter_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _should_link(rel_posix: str, regexes) -> bool:
    return any(r.search(rel_posix) for r in regexes)


def _unlink_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        try:
            path.unlink()
        except Exception:
            # On Windows, files can be read‑only; try to make writable
            try:
                os.chmod(path, 0o666)
                path.unlink()
            except Exception:
                raise


def _replace_with_link(src_target: Path, link_path: Path, prefer: str, log, *, allow_copy_fallback: bool = True) -> str:
    """Replace link_path with a link pointing at src_target.

    prefer: 'symlink' or 'hardlink'. Falls back to the other, finally copy.
    Returns the chosen method: 'symlink' | 'hardlink' | 'copy'.
    """
    _unlink_existing(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Try symlink and hardlink depending on preference
    if prefer == "symlink":
        try:
            link_path.symlink_to(src_target)
            return "symlink"
        except Exception as e:
            log.debug("symlink_to failed for %s -> %s: %s", link_path, src_target, e)
            try:
                os.link(src_target, link_path)
                return "hardlink"
            except Exception as e2:
                log.debug("hardlink failed for %s -> %s: %s", link_path, src_target, e2)
    else:
        try:
            os.link(src_target, link_path)
            return "hardlink"
        except Exception as e:
            log.debug("hardlink failed for %s -> %s: %s", link_path, src_target, e)
            try:
                link_path.symlink_to(src_target)
                return "symlink"
            except Exception as e2:
                log.debug("symlink_to failed for %s -> %s: %s", link_path, src_target, e2)

    if allow_copy_fallback:
        # Fallback: copy
        shutil.copy2(src_target, link_path)
        return "copy"
    else:
        raise OSError(f"Link creation failed for {link_path} -> {src_target} and copying disabled")


def _normalize_glob_pattern(pat: str) -> str:
    """Sanitize user-provided glob patterns for Path.rglob.

    - Removes leading drive or root (C:\, /) to keep them relative
    - Converts '/.ext' segment endings to '/*.ext'
    - If pattern starts with '.ext', convert to '*.ext'
    """
    import re as _re

    if pat is None:
        return ""
    p = str(pat).strip()
    if not p:
        return p
    # Drop Windows drive like 'C:\' or 'D:/' if present
    p = _re.sub(r'^[A-Za-z]:[\\/]+', '', p)
    # Drop any leading slashes to keep the pattern relative
    p = p.lstrip('/\\')
    # Replace '/.ext' or '\\.ext' with '/*.ext'
    p = _re.sub(r'([\\/])\.(?=[^\\/]+$)', r'/*.', p)
    # If pattern is like '.ext', make it '*.ext'
    if p.startswith('.'):
        p = '*' + p
    return p


def _link_tree(src_root: Path, dst_root: Path, prefer_link: str, log) -> tuple[int, int]:
    """Create a mirror of src_root at dst_root using links for files.

    Returns a tuple (linked_count, copied_count). Copy fallback is disabled here to avoid large space usage.
    """
    linked = 0
    copied = 0
    for p in src_root.rglob("*"):
        rel = p.relative_to(src_root)
        out = dst_root / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                _replace_with_link(p, out, prefer_link, log, allow_copy_fallback=False)
                linked += 1
            except Exception:
                # As a last resort, copy a single file; count it to report potential space usage
                shutil.copy2(p, out)
                copied += 1
    return linked, copied


def run_realizations_batch(
    base_txtinout: Union[str, Path],
    realizations: list[RealizationSpec],
    link_file_regexes: list[str],
    outputs_to_copy: list[str],
    *,
    exe_path: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    expect_plus: bool = False,
    config: Optional[dict] = None,
    workspace_dir: Optional[Union[str, Path]] = None,
    create_workspace_copy: bool = True,
    force_recreate_workspace: bool = True,
    strict: bool = False,
    prefer_link: Optional[str] = None,
    results_parent_name: str = "_results",
    results_root: Optional[Union[str, Path]] = None,
    include_base_run: bool = False,
    clean_outputs_before_run: bool = True,
    # New: aggressively remove selected input files before linking to avoid stale usage
    preclean_input_globs: Optional[list[str]] = None,
    preclean_linked_inputs: bool = False,
) -> list[RealizationRunResult]:
    """
    Run a base SWAT/SWAT+ model for several realizations by linking select files.

    - base_txtinout: path to the base TxtInOut folder to copy once for all runs
    - realizations: list of RealizationSpec(name, folder) pointing to folders with
      replacement files mirroring the TxtInOut structure
    - link_file_regexes: regexes tested against relative POSIX paths under TxtInOut
      to determine which files should be linked to the realization folder
    - outputs_to_copy: list of relative filenames or glob patterns to copy from TxtInOut
      after each run into '<realization>/<results_parent_name>/<name>'
    - exe_path: path to swat2012.exe or swatplus-rel.exe; optional; if None the runner may
      resolve it from PATH
    - timeout: optional seconds to kill the run if it hangs
    - expect_plus: True if using SWAT+ (affects default exe resolution in runner)
    - config: optional repo config to enable logging to file via utils.get_logger
    - workspace_dir: optional directory to place the one‑time copy of base_txtinout
      (defaults to a sibling '<base>_work/TxtInOut')
    - strict: if True, missing realization files for matched paths cause that realization to fail
    - prefer_link: 'symlink' (default) or 'hardlink'. Falls back automatically; last resort is copy
    - results_parent_name: name of the subfolder created inside each realization to store outputs
    - clean_outputs_before_run: delete any previous outputs matching outputs_to_copy in the workspace
    """
    import re

    log = get_logger(__name__, config)

    # Default linking preference: Windows prefers hardlink to avoid symlink permissions
    if prefer_link is None:
        prefer_link = "hardlink" if os.name == "nt" else "symlink"

    base = Path(base_txtinout).resolve()
    if "txtinout" not in base.name.lower():
        raise ValueError(f"base_txtinout must be the TxtInOut folder, got: {base}")

    # Determine working TxtInOut
    if create_workspace_copy:
        if workspace_dir is None:
            ws_root = base.parent / (base.name + "_work")
            workspace_dir = ws_root
        else:
            workspace_dir = Path(workspace_dir)
        work_txtinout = Path(workspace_dir) / "TxtInOut"

        if work_txtinout.exists() and force_recreate_workspace:
            log.info("Removing existing workspace to recreate: %s", work_txtinout)
            shutil.rmtree(work_txtinout.parent, ignore_errors=True)

        if not work_txtinout.exists():
            log.info("Creating full workspace copy at %s", work_txtinout)
            work_txtinout.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(base, work_txtinout)
        else:
            log.info("Reusing existing workspace at %s", work_txtinout)
    else:
        # Operate in-place on the original base folder (dangerous)
        log.warning("Operating in-place on base TxtInOut (no workspace copy). The base folder will be modified.")
        work_txtinout = base

    regexes = _compile_regexes(link_file_regexes)
    all_files = _iter_files(work_txtinout)
    rel_index = {p: p.relative_to(work_txtinout).as_posix() for p in all_files}

    results: list[RealizationRunResult] = []

    # Helper to determine output destination root
    def _resolve_outputs_dir(realization_folder: Path, name: str) -> Path:
        if results_root is not None:
            root = Path(results_root)
            root.mkdir(parents=True, exist_ok=True)
            return root / name
        return realization_folder / results_parent_name / name

    # Optionally run the base model before applying any realizations
    if include_base_run:
        base_run_name = f"base_{base.parent.name}"
        log.info("=== Base realization '%s' start ===", base_run_name)

        if clean_outputs_before_run:
            removed = 0
            for pat in outputs_to_copy:
                if any(ch in pat for ch in "*?[]"):
                    for candidate in work_txtinout.glob(pat):
                        if candidate.is_file():
                            _unlink_existing(candidate)
                            removed += 1
                else:
                    candidate = work_txtinout / pat
                    if candidate.exists() and candidate.is_file():
                        _unlink_existing(candidate)
                        removed += 1
            if removed:
                log.info("[base] Removed %s stale output file(s) before run", removed)

        rr_base = _runner.run_swat(
            textinout_dir=work_txtinout,
            exe_path=exe_path,
            timeout=timeout,
            expect_plus=expect_plus,
            config=config,
        )

        base_outputs_dir = _resolve_outputs_dir(base.parent, base_run_name)
        base_outputs_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for pat in outputs_to_copy:
            if any(ch in pat for ch in "*?[]"):
                for src in work_txtinout.glob(pat):
                    if src.is_file():
                        dest = base_outputs_dir / src.name
                        if dest.exists() or dest.is_symlink():
                            _unlink_existing(dest)
                        shutil.copy2(src, dest)
                        copied += 1
            else:
                src = work_txtinout / pat
                if src.exists() and src.is_file():
                    dest = base_outputs_dir / src.name
                    if dest.exists() or dest.is_symlink():
                        _unlink_existing(dest)
                    shutil.copy2(src, dest)
                    copied += 1
        log.info("[base] Copied %s output file(s) to %s", copied, base_outputs_dir)

        results.append(
            RealizationRunResult(
                name=base_run_name,
                success=rr_base.success,
                message=("Base run OK" if rr_base.success else f"Base run failed: {rr_base.message}"),
                returncode=rr_base.returncode,
                outputs_dir=base_outputs_dir,
                runner=rr_base,
                linked_count=0,
                skipped_missing=0,
                matched_count=0,
                missing_in_realization=0,
                extra_in_realization=0,
                elapsed_seconds=rr_base.elapsed_seconds,
            )
        )

    for spec in realizations:
        start_ts = time.perf_counter()
        name = spec.name
        r_folder = Path(spec.folder).resolve()
        log.info("=== Realization '%s' start | folder=%s ===", name, r_folder)

        if not r_folder.exists():
            msg = f"Realization folder not found: {r_folder}"
            log.error(msg)
            results.append(
                RealizationRunResult(
                    name=name,
                    success=False,
                    message=msg,
                    returncode=2,
                    outputs_dir=None,
                    runner=_runner.RunnerResult(False, 2, str(exe_path or "<unknown>"), work_txtinout),
                    linked_count=0,
                    skipped_missing=0,
                    elapsed_seconds=time.perf_counter() - start_ts,
                )
            )
            continue

        # Optionally pre-clean inputs (before linking) to ensure no stale files remain
        removed_inputs = 0
        if preclean_linked_inputs:
            for path, rel in list(rel_index.items()):
                if _should_link(rel, regexes) and path.exists() and path.is_file():
                    try:
                        _unlink_existing(path)
                        removed_inputs += 1
                    except Exception as e:
                        log.warning("Failed to preclean linked input %s: %s", path, e)
        if preclean_input_globs:
            # Support glob patterns like '**/*.dat', '**/*.chm', '**/fig.fig'
            for pat in preclean_input_globs:
                norm_pat = _normalize_glob_pattern(pat)
                if not norm_pat:
                    continue
                for candidate in work_txtinout.rglob(norm_pat):
                    if candidate.is_file():
                        try:
                            _unlink_existing(candidate)
                            removed_inputs += 1
                        except Exception as e:
                            log.warning("Failed to preclean input by glob %s -> %s: %s", pat, candidate, e)
        if removed_inputs:
            log.info("Precleaned %s input file(s) before linking", removed_inputs)

        # Link selected files
        linked = 0
        skipped_missing = 0
        # Compute matched paths under working copy for statistics
        matched_paths = [rel for rel in rel_index.values() if _should_link(rel, regexes)]
        matched_count = len(matched_paths)

        for path, rel in rel_index.items():
            if _should_link(rel, regexes):
                target = r_folder / rel
                if target.exists() and target.is_file():
                    method = _replace_with_link(target, path, prefer_link, log, allow_copy_fallback=True)
                    linked += 1
                    if linked <= 5:
                        log.info("Linked (%s): %s -> %s", method, path.name, target)
                else:
                    if strict:
                        msg = f"Missing realization file for {rel}: {target}"
                        log.error(msg)
                        results.append(
                            RealizationRunResult(
                                name=name,
                                success=False,
                                message=msg,
                                returncode=2,
                                outputs_dir=None,
                                runner=_runner.RunnerResult(False, 2, str(exe_path or "<unknown>"), work_txtinout),
                                linked_count=linked,
                                skipped_missing=skipped_missing,
                                matched_count=matched_count,
                                missing_in_realization=0,
                                extra_in_realization=0,
                                elapsed_seconds=time.perf_counter() - start_ts,
                            )
                        )
                        break
                    else:
                        skipped_missing += 1
                        # Defer per-file warnings; report counts at the end

        # Compute realization-selected files for reverse-missing stats
        realization_all_files = _iter_files(r_folder)
        rel_selected = [p.relative_to(r_folder).as_posix() for p in realization_all_files if _should_link(p.relative_to(r_folder).as_posix(), regexes)]

        set_work = set(matched_paths)
        set_real = set(rel_selected)
        missing_set = set_work - set_real          # in work but not in realization
        extra_set = set_real - set_work            # in realization but not in work

        # Create links for extra files present only in the realization into the workspace
        extras_linked = 0
        for rel in extra_set:
            src = r_folder / rel
            dst = work_txtinout / rel
            try:
                method = _replace_with_link(src, dst, prefer_link, log, allow_copy_fallback=True)
                extras_linked += 1
                linked += 1
                if extras_linked <= 5:
                    log.info("Linked extra (%s): %s -> %s", method, dst.name, src)
            except Exception as e:
                log.warning("Failed to link extra file into workspace: %s -> %s (%s)", dst, src, e)

        missing_in_real = len(missing_set)
        extra_in_real = len(extra_set)

        # Summary logs
        log.info(
            "Selected %s files by patterns; linked %s; missing_in_realization=%s; extra_in_realization=%s",
            matched_count,
            linked,
            missing_in_real,
            extra_in_real,
        )
        if missing_in_real > 0:
            log.warning("%s matched files in working copy had no counterpart in realization '%s'", missing_in_real, name)
        if extra_in_real > 0:
            log.warning("%s files in realization '%s' matched patterns but had no counterpart in working copy; created links in workspace", extra_in_real, name)

        # Optionally clean prior outputs matching patterns
        if clean_outputs_before_run:
            removed = 0
            for pat in outputs_to_copy:
                if any(ch in pat for ch in "*?[]"):
                    for candidate in work_txtinout.glob(pat):
                        if candidate.is_file():
                            _unlink_existing(candidate)
                            removed += 1
                else:
                    candidate = work_txtinout / pat
                    if candidate.exists() and candidate.is_file():
                        _unlink_existing(candidate)
                        removed += 1
            if removed:
                log.info("Removed %s stale output file(s) before run", removed)

        # Run model
        rr = _runner.run_swat(
            textinout_dir=work_txtinout,
            exe_path=exe_path,
            timeout=timeout,
            expect_plus=expect_plus,
            config=config,
        )

        # Prepare outputs destination
        outputs_dir = _resolve_outputs_dir(r_folder, name)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Collect outputs regardless of success to aid debugging
        copied = 0
        for pat in outputs_to_copy:
            if any(ch in pat for ch in "*?[]"):
                for src in work_txtinout.glob(pat):
                    if src.is_file():
                        dest = outputs_dir / src.name
                        if dest.exists() or dest.is_symlink():
                            _unlink_existing(dest)
                        shutil.copy2(src, dest)
                        copied += 1
            else:
                src = work_txtinout / pat
                if src.exists() and src.is_file():
                    dest = outputs_dir / src.name
                    if dest.exists() or dest.is_symlink():
                        _unlink_existing(dest)
                    shutil.copy2(src, dest)
                    copied += 1
                else:
                    log.debug("Output not found to copy: %s", src)

        log.info(
            "Realization '%s': copied %s output file(s) to %s",
            name,
            copied,
            outputs_dir,
        )

        elapsed = time.perf_counter() - start_ts
        if rr.success:
            msg = f"Realization '{name}' completed successfully in {elapsed:.2f}s"
            log.info(msg)
            results.append(
                RealizationRunResult(
                    name=name,
                    success=True,
                    message=msg,
                    returncode=rr.returncode,
                    outputs_dir=outputs_dir,
                    runner=rr,
                    linked_count=linked,
                    skipped_missing=skipped_missing,
                    matched_count=matched_count,
                    missing_in_realization=missing_in_real,
                    extra_in_realization=extra_in_real,
                    elapsed_seconds=elapsed,
                )
            )
        else:
            msg = f"Realization '{name}' failed: {rr.message}"
            log.error(msg)
            results.append(
                RealizationRunResult(
                    name=name,
                    success=False,
                    message=msg,
                    returncode=rr.returncode,
                    outputs_dir=outputs_dir,
                    runner=rr,
                    linked_count=linked,
                    skipped_missing=skipped_missing,
                    matched_count=matched_count,
                    missing_in_realization=missing_in_real,
                    extra_in_realization=extra_in_real,
                    elapsed_seconds=elapsed,
                )
            )

        log.info("=== Realization '%s' end ===", name)

    return results
