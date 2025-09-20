from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import os
import re


def modify_chm_file(
    src_file: Path,
    replacements: Mapping[str, float],
    dest_file: Path,
    *,
    pperco_val: Optional[float] = None,
    print_before: bool = False,
) -> Path:
    """Read a .chm file and overwrite Layer 1 numeric values for keys found in
    `replacements`, preserving original spacing and alignment. Optionally set
    the 'Phosphorus perc coef' value (PPERCO) as well.
    """
    src_file = Path(src_file)
    dest_file = Path(dest_file)
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    lines = src_file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    modified_lines = []
    for line in lines:
        modified = False
        strip = line.strip()
        # Replace values for provided keys
        for key, new_val in replacements.items():
            if strip.startswith(key):
                colon_index = line.index(":")
                label = line[:colon_index]
                after_colon = line[colon_index + 1 :].rstrip("\n")

                matches = list(re.finditer(r"\s*\S+", after_colon))
                if not matches:
                    break
                if print_before:
                    # original value (first token after colon)
                    original_val = matches[0].group().strip()
                    print(f"{key} (original Layer 1): {original_val}")

                new_val_str = f"{float(new_val):.2f}"
                field_end_pos = matches[0].end()
                field_start_pos = matches[0].start()

                start_pos = field_end_pos - len(new_val_str)
                padding = " " * max(0, start_pos - field_start_pos)
                replaced_field = padding + new_val_str

                new_line = label + ":" + replaced_field + after_colon[matches[0].end() :] + "\n"
                modified_lines.append(new_line)
                modified = True
                break

        # Replace PPERCO value if requested
        if not modified and pperco_val is not None and strip.startswith("Phosphorus perc coef"):
            colon_index = line.index(":")
            label = line[:colon_index]
            after_colon = line[colon_index + 1 :].rstrip("\n")

            matches = list(re.finditer(r"\s*\S+", after_colon))
            if matches:
                if print_before:
                    original_val = matches[0].group().strip()
                    print(f"Phosphorus perc coef (original): {original_val}")

                new_val_str = f"{float(pperco_val):.2f}"
                field_end_pos = matches[0].end()
                field_start_pos = matches[0].start()

                start_pos = field_end_pos - len(new_val_str)
                padding = " " * max(0, start_pos - field_start_pos)
                replaced_field = padding + new_val_str

                new_line = label + ":" + replaced_field + after_colon[matches[0].end() :] + "\n"
                modified_lines.append(new_line)
                modified = True

        if not modified:
            modified_lines.append(line)

    dest_file.write_text("".join(modified_lines), encoding="utf-8")
    return dest_file


def apply_replacements_bulk(
    base_txtinout: Path,
    dest_txtinout: Path,
    *,
    hru_replacements: Mapping[int, Mapping[str, float]],
    pperco_val: Optional[float] = None,
    overwrite: bool = True,
) -> list[Path]:
    """For each HRU id in `hru_replacements`, read `<base>/<id>.chm` (zero‑padded or not),
    modify values and write `<dest>/<id>.chm` where `<id>` is zero‑padded to width 9.
    Returns list of written files.
    """
    base_txtinout = Path(base_txtinout)
    dest_txtinout = Path(dest_txtinout)
    dest_txtinout.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # Build an index of existing CHM files in the base TxtInOut. This handles
    # zero-padded names like 000123.chm by mapping int(stem)->Path.
    name_index: dict[int, Path] = {}
    for p in base_txtinout.glob("*.chm"):
        stem = p.stem
        try:
            num = int(stem)
        except Exception:
            continue
        # prefer the first occurrence; assume unique
        name_index.setdefault(num, p)
    for hru_id, repl in hru_replacements.items():
        num_id = int(hru_id)
        src = base_txtinout / f"{num_id}.chm"
        if not src.exists():
            # Try index (handles zero-padded names)
            src = name_index.get(num_id, src)
            if not src.exists():
                # Skip missing sources
                continue
        # Always write with zero-padded 9-digit filename to match base naming
        dst_name = f"{num_id:09d}.chm"
        dst = dest_txtinout / dst_name
        if dst.exists() and not overwrite:
            continue
        written.append(
            modify_chm_file(src, repl, dst, pperco_val=pperco_val, print_before=False)
        )
    return written
