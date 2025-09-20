from __future__ import annotations

import re as _re
import pandas as pd


def normalize_point_year_columns(
    df: pd.DataFrame,
    *,
    id_col: str = "GRIDCODE",
    debug: bool | None = None,
) -> pd.DataFrame:
    """Normalize population CSV to have integer year columns and drop pixel counters.

    - Keeps only population sum columns that end with a 4-digit year (1970,1981,...)
    - Renames matching columns to int years
    - Deduplicates year columns and coerces them to numeric
    """
    patterns = [
        r"(?:popul(?:ation)?_)?sum_(\d{4})$",
        r".*_(\d{4})$",
        r"(\d{4})$",
    ]

    def match_year(col: str) -> int | None:
        s = str(col)
        for pat in patterns:
            m = _re.search(pat, s, flags=_re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    keep_cols: list[str] = []
    year_map: dict[str, int] = {}
    for c in df.columns:
        y = match_year(c)
        if y is not None:
            if "pixel" in str(c).lower():
                continue
            keep_cols.append(c)
            year_map[c] = y

    if not keep_cols:
        keep_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
        year_map = {c: int(c) for c in keep_cols}

    cols = [id_col] + keep_cols if id_col in df.columns else keep_cols
    out = df[cols].rename(columns=year_map)
    out = out.loc[:, ~out.columns.duplicated()]

    if id_col in out.columns:
        try:
            out[id_col] = out[id_col].apply(lambda x: int(str(x).split(",")[0]))
        except Exception:
            pass

    for c in out.columns:
        if isinstance(c, int):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    yrs = [c for c in out.columns if isinstance(c, int)]
    if debug:
        print(f"[point] normalized years found: {yrs[:10]}{'...' if len(yrs) > 10 else ''}")
    return out

