from __future__ import annotations

import pandas as pd


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Compare two DataFrames element-wise and summarize differences.

    Returns a dict with keys:
    - any_difference: bool
    - n_diff_columns: int
    - diff_columns: list[str]
    - diff_counts: dict[column -> count_of_differences]
    """
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape for direct comparison.")

    diff_mask = df1 != df2
    any_diff = diff_mask.any().any()
    if not any_diff:
        return {
            "any_difference": False,
            "n_diff_columns": 0,
            "diff_columns": [],
            "diff_counts": {},
        }

    diff_columns = diff_mask.any(axis=0)
    diff_counts = diff_mask.sum(axis=0)[diff_columns]
    return {
        "any_difference": True,
        "n_diff_columns": int(diff_columns.sum()),
        "diff_columns": diff_columns.index[diff_columns].tolist(),
        "diff_counts": {k: int(v) for k, v in diff_counts.to_dict().items()},
    }

