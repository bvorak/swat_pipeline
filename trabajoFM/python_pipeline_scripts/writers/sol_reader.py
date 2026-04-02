from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Generic .sol reader
# ---------------------------------------------------------------------------

def read_sol_file(path: Union[str, Path]) -> Dict[str, List[float]]:
    """Parse a SWAT ``.sol`` file and return every numeric key-value row.

    Returns a dict mapping the label (stripped) to a list of floats
    (one per soil layer, or a single-element list for scalar rows).

    Non-numeric rows (header, Soil Name, Texture, etc.) are silently skipped.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    result: Dict[str, List[float]] = {}
    for line in lines:
        if ":" not in line:
            continue
        colon_idx = line.index(":")
        label = line[:colon_idx].strip()
        after_colon = line[colon_idx + 1 :]

        tokens = after_colon.split()
        if not tokens:
            continue

        # Try to parse all tokens as floats; skip row if any token is
        # non-numeric (e.g. soil name, texture code).
        values: List[float] = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except ValueError:
                break
        else:
            # All tokens parsed successfully
            if values:
                result[label] = values

    return result


def read_sol_value(
    path: Union[str, Path],
    key: str,
    *,
    layer: int = 1,
) -> float:
    """Return a single numeric value from a ``.sol`` file.

    Parameters
    ----------
    path : path to the .sol file
    key : label to match (case-sensitive, must match the stripped label
          exactly, e.g. ``"Organic Carbon [weight %]"``).
    layer : 1-based layer index.  For scalar rows the only valid value is 1.

    Raises
    ------
    KeyError  – if *key* is not found in the file.
    IndexError – if *layer* exceeds the number of values on the row.
    """
    data = read_sol_file(path)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in {path}")
    values = data[key]
    if layer < 1 or layer > len(values):
        raise IndexError(
            f"Layer {layer} out of range (file has {len(values)} value(s) for {key!r})"
        )
    return values[layer - 1]


# ---------------------------------------------------------------------------
# Convenience: read Organic Carbon for layer 1
# ---------------------------------------------------------------------------

def read_organic_carbon(path: Union[str, Path], *, layer: int = 1) -> float:
    """Shortcut: read ``Organic Carbon [weight %]`` from a .sol file."""
    return read_sol_value(path, "Organic Carbon [weight %]", layer=layer)


# ---------------------------------------------------------------------------
# Bulk reader (mirrors apply_replacements_bulk pattern from chm_writer)
# ---------------------------------------------------------------------------

def read_sol_bulk(
    txtinout: Union[str, Path],
    keys: Sequence[str],
    *,
    hru_ids: Optional[Sequence[int]] = None,
    layer: int = 1,
) -> Dict[int, Dict[str, float]]:
    """Read selected keys from every (or selected) ``.sol`` file in a TxtInOut.

    Returns ``{hru_id: {key: value, ...}, ...}``.
    Missing files or keys are silently skipped.
    """
    txtinout = Path(txtinout)
    results: Dict[int, Dict[str, float]] = {}

    if hru_ids is not None:
        sol_paths = []
        for hru_id in hru_ids:
            for candidate in (txtinout / f"{hru_id}.sol", txtinout / f"{hru_id:09d}.sol"):
                if candidate.exists():
                    sol_paths.append((hru_id, candidate))
                    break
    else:
        sol_paths = []
        for p in sorted(txtinout.glob("*.sol")):
            try:
                num = int(p.stem)
            except ValueError:
                continue
            sol_paths.append((num, p))

    for hru_id, sol_path in sol_paths:
        data = read_sol_file(sol_path)
        row: Dict[str, float] = {}
        for key in keys:
            if key in data:
                vals = data[key]
                if 1 <= layer <= len(vals):
                    row[key] = vals[layer - 1]
        if row:
            results[hru_id] = row

    return results
