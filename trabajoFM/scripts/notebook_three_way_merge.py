"""Three-way merge helper for Jupyter notebooks.

This script performs a conservative, cell-level 3-way merge:
- If only one side changed from base, keep that side.
- If both sides changed differently, emit conflict markers inside the cell source.
- Cell matching prefers metadata.id and falls back to positional index.

Usage:
    python trabajoFM/scripts/notebook_three_way_merge.py \
        --base base.ipynb \
        --ours ours.ipynb \
        --theirs theirs.ipynb \
        --output merged.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_source_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _source_text(cell: dict | None) -> str:
    if not isinstance(cell, dict):
        return ""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def _cell_key(cell: dict, idx: int) -> str:
    cid = cell.get("metadata", {}).get("id")
    if isinstance(cid, str) and cid.strip():
        return f"id:{cid}"
    cid2 = cell.get("id")
    if isinstance(cid2, str) and cid2.strip():
        return f"id:{cid2}"
    return f"idx:{idx}"


def _cells_map(nb: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i, cell in enumerate(nb.get("cells", [])):
        out[_cell_key(cell, i)] = cell
    return out


def _ordered_keys(ours: dict, theirs: dict) -> List[str]:
    keys: List[str] = []
    seen = set()
    for i, cell in enumerate(ours.get("cells", [])):
        k = _cell_key(cell, i)
        if k not in seen:
            keys.append(k)
            seen.add(k)
    for i, cell in enumerate(theirs.get("cells", [])):
        k = _cell_key(cell, i)
        if k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


def _clone_cell(cell: dict | None) -> dict:
    if isinstance(cell, dict):
        return json.loads(json.dumps(cell))
    return {"cell_type": "code", "metadata": {"language": "python"}, "source": []}


def _conflict_source(ours_text: str, base_text: str, theirs_text: str) -> List[str]:
    marker = (
        "<<<<<<< OURS\n"
        + ours_text
        + ("\n" if ours_text and not ours_text.endswith("\n") else "")
        + "||||||| BASE\n"
        + base_text
        + ("\n" if base_text and not base_text.endswith("\n") else "")
        + "=======\n"
        + theirs_text
        + ("\n" if theirs_text and not theirs_text.endswith("\n") else "")
        + ">>>>>>> THEIRS\n"
    )
    return _to_source_lines(marker)


def merge_notebooks(base_nb: dict, ours_nb: dict, theirs_nb: dict, prefer: str) -> Tuple[dict, dict]:
    base_map = _cells_map(base_nb)
    ours_map = _cells_map(ours_nb)
    theirs_map = _cells_map(theirs_nb)

    merged = {
        "cells": [],
        "metadata": json.loads(json.dumps(ours_nb.get("metadata", {}))),
        "nbformat": ours_nb.get("nbformat", 4),
        "nbformat_minor": ours_nb.get("nbformat_minor", 5),
    }

    # Keep any metadata keys that only exist in theirs.
    for k, v in (theirs_nb.get("metadata", {}) or {}).items():
        if k not in merged["metadata"]:
            merged["metadata"][k] = v

    stats = {
        "kept_ours": 0,
        "kept_theirs": 0,
        "same": 0,
        "conflicts": 0,
        "added_from_ours": 0,
        "added_from_theirs": 0,
    }

    for key in _ordered_keys(ours_nb, theirs_nb):
        base_cell = base_map.get(key)
        ours_cell = ours_map.get(key)
        theirs_cell = theirs_map.get(key)

        base_text = _source_text(base_cell)
        ours_text = _source_text(ours_cell)
        theirs_text = _source_text(theirs_cell)

        # Added/deleted edge cases
        if base_cell is None and ours_cell is not None and theirs_cell is None:
            merged["cells"].append(_clone_cell(ours_cell))
            stats["added_from_ours"] += 1
            continue
        if base_cell is None and theirs_cell is not None and ours_cell is None:
            merged["cells"].append(_clone_cell(theirs_cell))
            stats["added_from_theirs"] += 1
            continue
        if base_cell is None and ours_cell is not None and theirs_cell is not None:
            if ours_text == theirs_text:
                merged["cells"].append(_clone_cell(ours_cell))
                stats["same"] += 1
            else:
                chosen = _clone_cell(ours_cell if prefer == "ours" else theirs_cell)
                chosen.setdefault("metadata", {})
                chosen["metadata"]["merge_conflict"] = True
                chosen["source"] = _conflict_source(ours_text, "", theirs_text)
                merged["cells"].append(chosen)
                stats["conflicts"] += 1
            continue

        # If one side deleted and the other changed/kept, keep non-empty side.
        if ours_cell is None and theirs_cell is not None:
            merged["cells"].append(_clone_cell(theirs_cell))
            stats["kept_theirs"] += 1
            continue
        if theirs_cell is None and ours_cell is not None:
            merged["cells"].append(_clone_cell(ours_cell))
            stats["kept_ours"] += 1
            continue

        # Standard 3-way source merge
        if ours_text == theirs_text:
            merged["cells"].append(_clone_cell(ours_cell))
            stats["same"] += 1
            continue
        if ours_text == base_text:
            merged["cells"].append(_clone_cell(theirs_cell))
            stats["kept_theirs"] += 1
            continue
        if theirs_text == base_text:
            merged["cells"].append(_clone_cell(ours_cell))
            stats["kept_ours"] += 1
            continue

        chosen = _clone_cell(ours_cell if prefer == "ours" else theirs_cell)
        chosen.setdefault("metadata", {})
        chosen["metadata"]["merge_conflict"] = True
        chosen["source"] = _conflict_source(ours_text, base_text, theirs_text)
        merged["cells"].append(chosen)
        stats["conflicts"] += 1

    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way merge for .ipynb files with cell-level conflict markers.")
    parser.add_argument("--base", required=True, type=Path, help="Base notebook (common ancestor).")
    parser.add_argument("--ours", required=True, type=Path, help="Our notebook version.")
    parser.add_argument("--theirs", required=True, type=Path, help="Their notebook version.")
    parser.add_argument("--output", required=True, type=Path, help="Merged notebook output path.")
    parser.add_argument("--prefer", choices=["ours", "theirs"], default="ours", help="Preferred scaffold for conflict cells.")
    parser.add_argument("--fail-on-conflict", action="store_true", help="Exit non-zero if conflicts remain.")
    args = parser.parse_args()

    base_nb = _load_notebook(args.base)
    ours_nb = _load_notebook(args.ours)
    theirs_nb = _load_notebook(args.theirs)

    merged_nb, stats = merge_notebooks(base_nb, ours_nb, theirs_nb, args.prefer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged_nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print("Notebook merge complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  output: {args.output}")

    if args.fail_on_conflict and stats["conflicts"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
