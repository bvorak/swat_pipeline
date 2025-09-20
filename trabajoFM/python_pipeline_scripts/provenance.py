from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils import ensure_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclasses.dataclass
class FileRef:
    path: str
    size: Optional[int] = None
    mtime: Optional[float] = None
    sha256: Optional[str] = None
    kind: Optional[str] = None


@dataclasses.dataclass
class Step:
    name: str
    module: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    inputs: Optional[List[FileRef]] = None
    outputs: Optional[List[FileRef]] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class RealizationProvenance:
    """Captures provenance for a single realization and appends to a JSONL ledger."""

    def __init__(
        self,
        *,
        ledger_path: Path,
        realization_id: int,
        name: str,
        base_txtinout: Path,
        realization_folder: Path,
        results_dir: Optional[Path] = None,
        parameters: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, Any]] = None,
        run_id: Optional[int] = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        ensure_dir(self.ledger_path.parent)
        self.id = realization_id
        self.name = name
        self.created_at = _now_iso()
        self.base_txtinout = str(Path(base_txtinout).resolve())
        self.realization_folder = str(Path(realization_folder).resolve())
        self.results_dir = str(Path(results_dir).resolve()) if results_dir else None
        self.parameters = parameters or {}
        self.engine = engine or {}
        self.schema_version = "1"
        self.run_id = run_id
        self.steps: List[Step] = []
        self.inputs: List[FileRef] = []
        self.outputs_summary: Dict[str, Any] = {}
        self._finalized = False

    @staticmethod
    def probe_file(path: Path, *, compute_hash: bool = False, kind: Optional[str] = None) -> FileRef:
        import hashlib

        p = Path(path)
        size = p.stat().st_size if p.exists() else None
        mtime = p.stat().st_mtime if p.exists() else None
        sha = None
        if compute_hash and p.exists() and p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            sha = h.hexdigest()
        return FileRef(path=str(p.resolve()), size=size, mtime=mtime, sha256=sha, kind=kind)

    def record_input(self, path: Path, *, compute_hash: bool = False, kind: Optional[str] = None) -> None:
        self.inputs.append(self.probe_file(path, compute_hash=compute_hash, kind=kind))

    def record_outputs(self, paths: Iterable[Path], *, compute_hash: bool = False, kind: Optional[str] = None) -> None:
        outs = [self.probe_file(p, compute_hash=compute_hash, kind=kind) for p in paths]
        # attach to last step if present, else to outputs_summary entries
        if self.steps:
            last = self.steps[-1]
            last.outputs = (last.outputs or []) + outs
        self.outputs_summary.setdefault("files", [])
        self.outputs_summary["files"].extend(dataclasses.asdict(o) for o in outs)

    @contextlib.contextmanager
    def step(
        self,
        name: str,
        *,
        module: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        inputs: Optional[Iterable[Path]] = None,
        notes: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        compute_hash_inputs: bool = False,
    ):
        st = Step(
            name=name,
            module=module,
            args=args,
            started_at=_now_iso(),
            notes=notes,
            extra=extra,
        )
        if inputs:
            st.inputs = [self.probe_file(Path(p), compute_hash=compute_hash_inputs) for p in inputs]
        self.steps.append(st)
        try:
            yield st
        finally:
            st.ended_at = _now_iso()

    def finalize(
        self,
        *,
        success: bool,
        error: Optional[str] = None,
        outputs_summary: Optional[Dict[str, Any]] = None,
        additional_fields: Optional[Dict[str, Any]] = None,
        write_copy_to: Optional[Path] = None,
    ) -> None:
        if self._finalized:
            return
        self._finalized = True
        if outputs_summary:
            self.outputs_summary.update(outputs_summary)
        rec: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "base_txtinout": self.base_txtinout,
            "realization_folder": self.realization_folder,
            "results_dir": self.results_dir,
            "parameters": self.parameters,
            "engine": self.engine,
            "inputs": [dataclasses.asdict(i) for i in self.inputs],
            "steps": [dataclasses.asdict(s) for s in self.steps],
            "status": "success" if success else "failed",
            "error": error,
            "outputs_summary": self.outputs_summary,
        }
        if additional_fields:
            rec.update(additional_fields)
        # Append JSONL
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Optionally write a copy alongside the realization for easy replay
        try:
            target = Path(write_copy_to) if write_copy_to else (Path(self.realization_folder) / "provenance.json")
            ensure_dir(target.parent)
            with target.open("w", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, indent=2))
        except Exception:
            # Non-fatal: still keep ledger entry
            pass
