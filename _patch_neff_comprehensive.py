"""
Comprehensive patch for notebook 04_sensitivity_stats.ipynb.
Re-applies ALL previous-session edits (rho1/n_eff/Hedges' g) that were lost
from disk, plus current-session additions (ci95_neff, chart variants, column order).

Strategy: for cell 20 (the sig-table engine), collapse the triple-blank-line
spacing to single-spaced code, apply all edits, then write back clean.
"""
import json, re, sys, shutil
from pathlib import Path
from datetime import datetime

NB_PATH = Path(
    r"c:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM"
    r"\scripts\Python_Pipeline_SWAT_Pascal\swat_pipeline"
    r"\trabajoFM\notebooks\04_sensitivity_stats.ipynb"
)

# ── backup ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = NB_PATH.with_suffix(f".pre_neff_patch_{ts}.ipynb")
shutil.copy2(NB_PATH, backup)
print(f"Backup: {backup.name}")

with open(NB_PATH, "r", encoding="utf-8-sig") as f:
    nb = json.load(f)

changes = []

def collapse_triple(src_lines):
    """Join source list and collapse runs of 2+ blank lines to 1."""
    text = "".join(src_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def to_source_list(text):
    """Convert text back to a source list (one entry per line, each ending \\n)."""
    lines = text.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 20: Significance-table engine (the big one)
# ═══════════════════════════════════════════════════════════════════════════════
for ci, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    raw_src = "".join(cell["source"])
    if "_SIG_COLUMN_REGISTRY" not in raw_src or "_welch_t_test" not in raw_src:
        continue

    src = collapse_triple(cell["source"])
    print(f"Cell {ci}: collapsed from {len(cell['source'])} to ~{len(src)} chars")

    # ── 1. _SIG_COLUMN_REGISTRY: add hedges_g, n_eff_A, n_eff_B ──
    old_reg = '''    "ci95_lo":  {"header": "CI\u2089\u2085 lo",  "fmt": lambda v: f"{v:+.3g}" if np.isfinite(v) else ""},
    "ci95_hi":  {"header": "CI\u2089\u2085 hi",  "fmt": lambda v: f"{v:+.3g}" if np.isfinite(v) else ""},
}'''
    new_reg = '''    "hedges_g": {"header": "Hedges\u2019 g", "fmt": lambda v: f"{v:.2f}" if np.isfinite(v) else ""},
    "n_eff_A":  {"header": "n\u1d49\u1da0\u1da0(A)", "fmt": lambda v: f"{v:.0f}" if np.isfinite(v) else ""},
    "n_eff_B":  {"header": "n\u1d49\u1da0\u1da0(B)", "fmt": lambda v: f"{v:.0f}" if np.isfinite(v) else ""},
    "ci95_lo":  {"header": "CI\u2089\u2085 lo",  "fmt": lambda v: f"{v:+.3g}" if np.isfinite(v) else ""},
    "ci95_hi":  {"header": "CI\u2089\u2085 hi",  "fmt": lambda v: f"{v:+.3g}" if np.isfinite(v) else ""},
}'''
    if old_reg in src:
        src = src.replace(old_reg, new_reg, 1)
        changes.append("_SIG_COLUMN_REGISTRY: added hedges_g, n_eff_A, n_eff_B")
    elif "hedges_g" in src and "_SIG_COLUMN_REGISTRY" in src:
        changes.append("_SIG_COLUMN_REGISTRY: hedges_g already present (skip)")
    else:
        print("WARNING: Could not patch _SIG_COLUMN_REGISTRY")

    # ── 2. _EXTENDED_STATS_KEY_MAP: add rho1 to ALL entries + RW entries ──
    key_map_replacements = [
        # maxmin_over_median
        ('"bl_median_maxmin_over_median":          {"sd": "sd_maxmin_over_median",      "n": "n_maxmin_over_median"}',
         '"bl_median_maxmin_over_median":          {"sd": "sd_maxmin_over_median",      "n": "n_maxmin_over_median",      "rho1": "rho1_maxmin_over_median"}'),
        ('"bl_median_maxmin_over_median_event":    {"sd": "sd_maxmin_over_median_event",    "n": "n_maxmin_over_median_event"}',
         '"bl_median_maxmin_over_median_event":    {"sd": "sd_maxmin_over_median_event",    "n": "n_maxmin_over_median_event",    "rho1": "rho1_maxmin_over_median_event"}'),
        ('"bl_median_maxmin_over_median_nonevent": {"sd": "sd_maxmin_over_median_nonevent", "n": "n_maxmin_over_median_nonevent"}',
         '"bl_median_maxmin_over_median_nonevent": {"sd": "sd_maxmin_over_median_nonevent", "n": "n_maxmin_over_median_nonevent", "rho1": "rho1_maxmin_over_median_nonevent"}'),
        ('"bl_mean_maxmin_over_median":            {"sd": "sd_maxmin_over_median",      "n": "n_maxmin_over_median"}',
         '"bl_mean_maxmin_over_median":            {"sd": "sd_maxmin_over_median",      "n": "n_maxmin_over_median",      "rho1": "rho1_maxmin_over_median"}'),
        ('"bl_mean_maxmin_over_median_event":      {"sd": "sd_maxmin_over_median_event",    "n": "n_maxmin_over_median_event"}',
         '"bl_mean_maxmin_over_median_event":      {"sd": "sd_maxmin_over_median_event",    "n": "n_maxmin_over_median_event",    "rho1": "rho1_maxmin_over_median_event"}'),
        ('"bl_mean_maxmin_over_median_nonevent":   {"sd": "sd_maxmin_over_median_nonevent", "n": "n_maxmin_over_median_nonevent"}',
         '"bl_mean_maxmin_over_median_nonevent":   {"sd": "sd_maxmin_over_median_nonevent", "n": "n_maxmin_over_median_nonevent", "rho1": "rho1_maxmin_over_median_nonevent"}'),
        # absolute width
        ('"median_W":             {"sd": "sd_W",          "n": "n_W"}',
         '"median_W":             {"sd": "sd_W",          "n": "n_W",          "rho1": "rho1_W"}'),
        ('"p90_W":                {"sd": None,            "n": None}',
         '"p90_W":                {"sd": None,            "n": None,           "rho1": None}'),
        ('"bl_median_W_event":    {"sd": "sd_W_event",    "n": "n_W_event"}',
         '"bl_median_W_event":    {"sd": "sd_W_event",    "n": "n_W_event",    "rho1": "rho1_W_event"}'),
        ('"bl_median_W_nonevent": {"sd": "sd_W_nonevent", "n": "n_W_nonevent"}',
         '"bl_median_W_nonevent": {"sd": "sd_W_nonevent", "n": "n_W_nonevent", "rho1": "rho1_W_nonevent"}'),
        ('"bl_abs_W_event_ratio": {"sd": None, "n": None}',
         '"bl_abs_W_event_ratio": {"sd": None, "n": None, "rho1": None}'),
        # overlay shift stats
        ('"delta_mean":                 {"sd": "sd_delta",          "n": "n_delta"}',
         '"delta_mean":                 {"sd": "sd_delta",          "n": "n_delta",          "rho1": "rho1_delta"}'),
        ('"median_delta":               {"sd": "sd_delta",          "n": "n_delta"}',
         '"median_delta":               {"sd": "sd_delta",          "n": "n_delta",          "rho1": "rho1_delta"}'),
        ('"delta_event_pairs_mean":     {"sd": "sd_delta_event",    "n": "n_delta_event"}',
         '"delta_event_pairs_mean":     {"sd": "sd_delta_event",    "n": "n_delta_event",    "rho1": "rho1_delta_event"}'),
        ('"delta_non_event_pairs_mean": {"sd": "sd_delta_nonevent", "n": "n_delta_nonevent"}',
         '"delta_non_event_pairs_mean": {"sd": "sd_delta_nonevent", "n": "n_delta_nonevent", "rho1": "rho1_delta_nonevent"}'),
        # same_day pairwise
        ('"Bias(obs-pred)": {"sd": "sd_bias", "n": "n_bias"}',
         '"Bias(obs-pred)": {"sd": "sd_bias", "n": "n_bias", "rho1": "rho1_bias"}'),
        ('"PBIAS%":         {"sd": None,      "n": None}',
         '"PBIAS%":         {"sd": None,      "n": None,     "rho1": None}'),
        ('"coverage100":    {"sd": None,      "n": None}',
         '"coverage100":    {"sd": None,      "n": None,     "rho1": None}'),
    ]

    rho1_count = 0
    for old, new in key_map_replacements:
        if old in src:
            src = src.replace(old, new, 1)
            rho1_count += 1
    if rho1_count > 0:
        changes.append(f"_EXTENDED_STATS_KEY_MAP: added rho1 to {rho1_count} entries")

    # Add RW_med/high/low and bl_RW_mean entries
    rw_insert_marker = '    "bl_abs_W_event_ratio": {"sd": None, "n": None, "rho1": None},  # ratio - no direct SD\n    # Chart 2'
    rw_new = (
        '    "bl_abs_W_event_ratio": {"sd": None, "n": None, "rho1": None},  # ratio - no direct SD\n'
        '    # Chart 1 - baseline relative width (median & mean share same daily array SD/n/rho1)\n'
        '    "RW_med":               {"sd": "sd_relative_width",          "n": "n_relative_width",          "rho1": "rho1_relative_width"},\n'
        '    "RW_high":              {"sd": "sd_relative_width_event",    "n": "n_relative_width_event",    "rho1": "rho1_relative_width_event"},\n'
        '    "RW_low":               {"sd": "sd_relative_width_nonevent", "n": "n_relative_width_nonevent", "rho1": "rho1_relative_width_nonevent"},\n'
        '    "bl_RW_mean_all":       {"sd": "sd_relative_width",          "n": "n_relative_width",          "rho1": "rho1_relative_width"},\n'
        '    "bl_RW_mean_event":     {"sd": "sd_relative_width_event",    "n": "n_relative_width_event",    "rho1": "rho1_relative_width_event"},\n'
        '    "bl_RW_mean_nonevent":  {"sd": "sd_relative_width_nonevent", "n": "n_relative_width_nonevent", "rho1": "rho1_relative_width_nonevent"},\n'
        '    # Chart 2'
    )
    if rw_insert_marker in src:
        src = src.replace(rw_insert_marker, rw_new, 1)
        changes.append("_EXTENDED_STATS_KEY_MAP: added RW and bl_RW_mean entries")

    # ── 3. _welch_t_test: full rewrite with n_eff + Hedges' g ──
    old_welch = '''def _welch_t_test(mean_a, sd_a, n_a, mean_b, sd_b, n_b):
    """Welch\u2019s two-sample t-test from summary statistics.

    Returns (t_stat, p_value, ci95_lo, ci95_hi, cohens_d).
    All NaN when inputs are insufficient.
    """'''
    new_welch = '''def _welch_t_test(mean_a, sd_a, n_a, mean_b, sd_b, n_b,
                  rho1_a=np.nan, rho1_b=np.nan):
    """Welch\u2019s two-sample t-test from summary statistics.

    When lag-1 autocorrelation (rho1_a, rho1_b) is available, the test
    uses *effective* sample sizes to correct for temporal autocorrelation:

        n_eff = n \u00b7 (1 \u2212 \u03c1\u2081) / (1 + \u03c1\u2081)     (Zwiers & von Storch, 1995)

    This is necessary because consecutive daily values in hydrological time
    series are strongly autocorrelated (typical \u03c1\u2081 \u2248 0.9\u20130.99), which inflates
    the apparent precision of the sample mean.  Without the correction, SE is
    underestimated, t is inflated, and p-values and CIs are anti-conservative.

    The *point estimate* of Cohen\u2019s d is unaffected by autocorrelation; only
    CIs and p-values need the n_eff correction.

    Hedges\u2019 g applies a small-sample bias correction to d:

        g = d \u00b7 J,   J = 1 \u2212 3 / (4\u00b7df \u2212 1)     (Hedges, 1981)

    At large n (df > 100) the correction is negligible (J > 0.99), but we
    include it for formal correctness.

    Returns
    -------
    (t_stat, p_value, ci95_lo, ci95_hi, cohens_d, hedges_g, n_eff_a, n_eff_b)
    All NaN when inputs are insufficient.

    References
    ----------
    Zwiers, F. W. & von Storch, H. (1995). J. Climate, 8(2), 336\u2013351.
    Cohen, J. (1988). Statistical Power Analysis (2nd ed.). Erlbaum.
    Hedges, L. V. (1981). J. Educ. Stat., 6(2), 107\u2013128.
    """'''
    if old_welch in src:
        src = src.replace(old_welch, new_welch, 1)
        changes.append("_welch_t_test: signature + docstring rewritten")
    elif 'rho1_a=np.nan' in src and '_welch_t_test' in src:
        changes.append("_welch_t_test: already has rho1 (skip signature)")
    else:
        print("WARNING: Could not find _welch_t_test signature to replace")

    # Replace welch body
    old_welch_body = '''    _nan5 = (np.nan,) * 5
    try:
        n_a, n_b = float(n_a), float(n_b)
        if n_a < 2 or n_b < 2 or not np.isfinite(sd_a) or not np.isfinite(sd_b):
            return _nan5
        se_a = sd_a ** 2 / n_a
        se_b = sd_b ** 2 / n_b
        se_diff = np.sqrt(se_a + se_b)
        if se_diff < 1e-30:
            return _nan5
        t_stat = (mean_a - mean_b) / se_diff
        df_num = (se_a + se_b) ** 2
        df_den = se_a ** 2 / (n_a - 1) + se_b ** 2 / (n_b - 1)
        df = df_num / df_den if df_den > 0 else 1.0
        from scipy.stats import t as t_dist
        p_value = 2 * t_dist.sf(abs(t_stat), df)
        t_crit = t_dist.ppf(0.975, df)
        ci95_lo = (mean_a - mean_b) - t_crit * se_diff
        ci95_hi = (mean_a - mean_b) + t_crit * se_diff
        pooled_sd = np.sqrt(((n_a - 1) * sd_a ** 2 + (n_b - 1) * sd_b ** 2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 1e-30 else np.nan
        return t_stat, p_value, ci95_lo, ci95_hi, cohens_d
    except Exception:
        return _nan5'''
    new_welch_body = '''    _nan8 = (np.nan,) * 8
    try:
        n_a, n_b = float(n_a), float(n_b)
        if n_a < 2 or n_b < 2 or not np.isfinite(sd_a) or not np.isfinite(sd_b):
            return _nan8
        # Effective sample sizes (Zwiers & von Storch, 1995)
        def _neff(n, rho1):
            if np.isfinite(rho1) and 0 < rho1 < 1:
                return max(2.0, n * (1 - rho1) / (1 + rho1))
            return n
        n_eff_a = _neff(n_a, rho1_a)
        n_eff_b = _neff(n_b, rho1_b)
        se_a = sd_a ** 2 / n_eff_a
        se_b = sd_b ** 2 / n_eff_b
        se_diff = np.sqrt(se_a + se_b)
        if se_diff < 1e-30:
            return _nan8
        t_stat = (mean_a - mean_b) / se_diff
        # Welch-Satterthwaite df using n_eff
        df_num = (se_a + se_b) ** 2
        df_den = se_a ** 2 / (n_eff_a - 1) + se_b ** 2 / (n_eff_b - 1)
        df = df_num / df_den if df_den > 0 else 1.0
        from scipy.stats import t as t_dist
        p_value = 2 * t_dist.sf(abs(t_stat), df)
        t_crit = t_dist.ppf(0.975, df)
        ci95_lo = (mean_a - mean_b) - t_crit * se_diff
        ci95_hi = (mean_a - mean_b) + t_crit * se_diff
        # Cohen's d uses raw n (point estimate unaffected by autocorrelation)
        pooled_sd = np.sqrt(((n_a - 1) * sd_a ** 2 + (n_b - 1) * sd_b ** 2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 1e-30 else np.nan
        # Hedges' g (Hedges, 1981): bias correction J = 1 - 3/(4*df - 1)
        df_pool = n_a + n_b - 2
        J = 1 - 3 / (4 * df_pool - 1) if df_pool > 1 else 1.0
        hedges_g = cohens_d * J if np.isfinite(cohens_d) else np.nan
        return t_stat, p_value, ci95_lo, ci95_hi, cohens_d, hedges_g, n_eff_a, n_eff_b
    except Exception:
        return _nan8'''
    if old_welch_body in src:
        src = src.replace(old_welch_body, new_welch_body, 1)
        changes.append("_welch_t_test: body rewritten with n_eff + Hedges' g")
    elif '_nan8' in src:
        changes.append("_welch_t_test: body already updated (skip)")
    else:
        print("WARNING: Could not find _welch_t_test body to replace")

    # ── 4. _one_sample_t_test: rewrite with n_eff + Hedges' g ──
    old_onesamp = '''def _one_sample_t_test(mean_a, sd_a, n_a, mu0=0.0):
    """One-sample t-test (is mean_a significantly different from mu0?).

    Returns (t_stat, p_value, ci95_lo, ci95_hi, cohens_d).
    """'''
    new_onesamp = '''def _one_sample_t_test(mean_a, sd_a, n_a, mu0=0.0, rho1_a=np.nan):
    """One-sample t-test (is mean_a significantly different from mu0?).

    Uses n_eff when rho1_a is available (see _welch_t_test docstring).

    Returns (t_stat, p_value, ci95_lo, ci95_hi, cohens_d, hedges_g, n_eff_a).
    """'''
    if old_onesamp in src:
        src = src.replace(old_onesamp, new_onesamp, 1)
        changes.append("_one_sample_t_test: signature rewritten")

    old_onesamp_body = '''    _nan5 = (np.nan,) * 5
    try:
        n_a = float(n_a)
        if n_a < 2 or not np.isfinite(sd_a) or sd_a < 1e-30:
            return _nan5
        se = sd_a / np.sqrt(n_a)
        t_stat = (mean_a - mu0) / se
        df = n_a - 1
        from scipy.stats import t as t_dist
        p_value = 2 * t_dist.sf(abs(t_stat), df)
        t_crit = t_dist.ppf(0.975, df)
        ci95_lo = mean_a - t_crit * se
        ci95_hi = mean_a + t_crit * se
        cohens_d = (mean_a - mu0) / sd_a if sd_a > 1e-30 else np.nan
        return t_stat, p_value, ci95_lo, ci95_hi, cohens_d
    except Exception:
        return _nan5'''
    new_onesamp_body = '''    _nan7 = (np.nan,) * 7
    try:
        n_a = float(n_a)
        if n_a < 2 or not np.isfinite(sd_a) or sd_a < 1e-30:
            return _nan7
        # Effective sample size
        if np.isfinite(rho1_a) and 0 < rho1_a < 1:
            n_eff_a = max(2.0, n_a * (1 - rho1_a) / (1 + rho1_a))
        else:
            n_eff_a = n_a
        se = sd_a / np.sqrt(n_eff_a)
        t_stat = (mean_a - mu0) / se
        df = n_eff_a - 1
        from scipy.stats import t as t_dist
        p_value = 2 * t_dist.sf(abs(t_stat), df)
        t_crit = t_dist.ppf(0.975, df)
        ci95_lo = mean_a - t_crit * se
        ci95_hi = mean_a + t_crit * se
        cohens_d = (mean_a - mu0) / sd_a if sd_a > 1e-30 else np.nan
        # Hedges' g
        J = 1 - 3 / (4 * (n_a - 1) - 1) if n_a > 2 else 1.0
        hedges_g = cohens_d * J if np.isfinite(cohens_d) else np.nan
        return t_stat, p_value, ci95_lo, ci95_hi, cohens_d, hedges_g, n_eff_a
    except Exception:
        return _nan7'''
    if old_onesamp_body in src:
        src = src.replace(old_onesamp_body, new_onesamp_body, 1)
        changes.append("_one_sample_t_test: body rewritten with n_eff + Hedges' g")

    # ── 5. _try_get_extended: return (sd, n, rho1) ──
    old_try_return = '''    return sd_val, n_val'''
    # Count occurrences to make sure there's only one
    if src.count(old_try_return) == 1:
        new_try_return = '''    rho1_key = key_map.get("rho1")
    rho1_val = np.nan
    if rho1_key is not None:
        rho1_raw = _section.get(rho1_key)
        if rho1_raw is not None:
            try:
                rho1_val = float(rho1_raw)
            except (TypeError, ValueError):
                pass
    return sd_val, n_val, rho1_val'''
        src = src.replace(old_try_return, new_try_return, 1)
        changes.append("_try_get_extended: now returns (sd, n, rho1)")

    # Update _try_get_extended docstring
    old_try_doc = '    Returns (sd, n) or (NaN, NaN) on any lookup failure.'
    new_try_doc = '    Returns (sd, n, rho1) or (NaN, NaN, NaN) on any lookup failure.'
    if old_try_doc in src:
        src = src.replace(old_try_doc, new_try_doc, 1)

    # Update the except/fallback return in _try_get_extended
    old_try_except = '''        return np.nan, np.nan'''
    # This appears multiple times — both in _try_get_extended's except blocks
    src = src.replace(old_try_except, '        return np.nan, np.nan, np.nan')
    changes.append("_try_get_extended: except returns updated to 3-tuple")

    # ── 6. extract_extended_stats_for_chart: store rho1 ──
    old_extract = '        sd, n = _try_get_extended('
    new_extract = '        sd, n, rho1 = _try_get_extended('
    if old_extract in src:
        src = src.replace(old_extract, new_extract, 1)
        changes.append("extract_extended_stats_for_chart: unpack 3 values")

    old_store = '            result[(scenario, variable, tn)] = {"sd": sd, "n": n}'
    new_store = '            result[(scenario, variable, tn)] = {"sd": sd, "n": n, "rho1": rho1}'
    if old_store in src:
        src = src.replace(old_store, new_store, 1)
        changes.append("extract_extended_stats_for_chart: store rho1")

    # Update extract docstring
    old_extract_doc = '    """Read raw JSON exports and extract (sd, n) for each (scenario, variable, metric).'
    new_extract_doc = '    """Read raw JSON exports and extract (sd, n, rho1) for each (scenario, variable, metric).'
    if old_extract_doc in src:
        src = src.replace(old_extract_doc, new_extract_doc, 1)

    # ── 7. _get_ext: return 3 values ──
    old_get_ext = '''    def _get_ext(ext, sc, var, tech):
        """Return (sd, n) for a (scenario, variable, tech_name) triple."""
        entry = ext.get((sc, var, tech), {})
        return entry.get("sd", np.nan), entry.get("n", np.nan)'''
    new_get_ext = '''    def _get_ext(ext, sc, var, tech):
        """Return (sd, n, rho1) for a (scenario, variable, tech_name) triple."""
        entry = ext.get((sc, var, tech), {})
        return entry.get("sd", np.nan), entry.get("n", np.nan), entry.get("rho1", np.nan)'''
    if old_get_ext in src:
        src = src.replace(old_get_ext, new_get_ext, 1)
        changes.append("_get_ext: now returns (sd, n, rho1)")

    # ── 8. _add_row: update signature, unpacking, row dict ──
    old_add_row_sig = '''    def _add_row(metric_label, sc_a, var_a, val_a,
                 sc_b, var_b, val_b,
                 sd_a, n_a, sd_b, n_b, comparison_type):'''
    new_add_row_sig = '''    def _add_row(metric_label, sc_a, var_a, val_a,
                 sc_b, var_b, val_b,
                 sd_a, n_a, sd_b, n_b, comparison_type,
                 rho1_a=np.nan, rho1_b=np.nan):'''
    if old_add_row_sig in src:
        src = src.replace(old_add_row_sig, new_add_row_sig, 1)
        changes.append("_add_row: signature updated with rho1")

    old_add_unpack = '        t_stat, p_val, ci_lo, ci_hi, d = _welch_t_test(val_a, sd_a, n_a, val_b, sd_b, n_b)'
    new_add_unpack = '        t_stat, p_val, ci_lo, ci_hi, d, g, nea, neb = _welch_t_test(\n            val_a, sd_a, n_a, val_b, sd_b, n_b,\n            rho1_a=rho1_a, rho1_b=rho1_b)'
    if old_add_unpack in src:
        src = src.replace(old_add_unpack, new_add_unpack, 1)
        changes.append("_add_row: welch call updated to 8-tuple with rho1")

    old_add_row_dict = '''            "t_stat": t_stat, "p_value": p_val, "cohens_d": d,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        }'''
    new_add_row_dict = '''            "t_stat": t_stat, "p_value": p_val, "cohens_d": d,
            "hedges_g": g, "n_eff_A": nea, "n_eff_B": neb,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        }'''
    # This pattern appears in both _add_row and _regime_row
    # Replace the first occurrence (_add_row)
    if src.count(old_add_row_dict) >= 1:
        src = src.replace(old_add_row_dict, new_add_row_dict, 1)
        changes.append("_add_row: row dict updated with hedges_g, n_eff_A, n_eff_B")

    # ── 9. Three comparison mode call sites: unpack 3 from _get_ext, pass rho1 ──
    # Pattern: sd_a, n_a = _get_ext(...) → sd_a, n_a, rho1_a = _get_ext(...)
    # And: _add_row(... sd_a, n_a, sd_b, n_b, ...) → _add_row(... rho1_a=rho1_a, rho1_b=rho1_b)
    
    # Scenario comparison
    old_sc_get = '                sd_a, n_a = _get_ext(ext_stats, sc_a, var, tech)\n                sd_b, n_b = _get_ext(ext_stats, sc_b, var, tech)'
    new_sc_get = '                sd_a, n_a, rho1_a = _get_ext(ext_stats, sc_a, var, tech)\n                sd_b, n_b, rho1_b = _get_ext(ext_stats, sc_b, var, tech)'
    if old_sc_get in src:
        src = src.replace(old_sc_get, new_sc_get, 1)
        changes.append("scenario comparison: _get_ext unpacks 3 values")

    # Variable comparison
    old_var_get = '                sd_a, n_a = _get_ext(ext_stats, sc, var_a, tech)\n                sd_b, n_b = _get_ext(ext_stats, sc, var_b, tech)'
    new_var_get = '                sd_a, n_a, rho1_a = _get_ext(ext_stats, sc, var_a, tech)\n                sd_b, n_b, rho1_b = _get_ext(ext_stats, sc, var_b, tech)'
    if old_var_get in src:
        src = src.replace(old_var_get, new_var_get, 1)
        changes.append("variable comparison: _get_ext unpacks 3 values")

    # Metric comparison
    old_met_get = '                sd_a, n_a = _get_ext(ext_stats, sc, var, tech_a)\n                sd_b, n_b = _get_ext(ext_stats, sc, var, tech_b)'
    new_met_get = '                sd_a, n_a, rho1_a = _get_ext(ext_stats, sc, var, tech_a)\n                sd_b, n_b, rho1_b = _get_ext(ext_stats, sc, var, tech_b)'
    if old_met_get in src:
        src = src.replace(old_met_get, new_met_get, 1)
        changes.append("metric comparison: _get_ext unpacks 3 values")

    # Now update the _add_row calls to pass rho1
    old_add_call = '''                _add_row(metric_label, sc_a, var, val_a,
                         sc_b, var, val_b,
                         sd_a, n_a, sd_b, n_b, "scenarios")'''
    new_add_call = '''                _add_row(metric_label, sc_a, var, val_a,
                         sc_b, var, val_b,
                         sd_a, n_a, sd_b, n_b, "scenarios",
                         rho1_a=rho1_a, rho1_b=rho1_b)'''
    if old_add_call in src:
        src = src.replace(old_add_call, new_add_call, 1)
        changes.append("scenario _add_row call: passes rho1")

    old_add_call_var = '''                _add_row(metric_label, sc, var_a, val_a,
                         sc, var_b, val_b,
                         sd_a, n_a, sd_b, n_b, "variables")'''
    new_add_call_var = '''                _add_row(metric_label, sc, var_a, val_a,
                         sc, var_b, val_b,
                         sd_a, n_a, sd_b, n_b, "variables",
                         rho1_a=rho1_a, rho1_b=rho1_b)'''
    if old_add_call_var in src:
        src = src.replace(old_add_call_var, new_add_call_var, 1)
        changes.append("variable _add_row call: passes rho1")

    old_add_call_met = '''                _add_row(label_a, sc, var, val_a,
                         label_b if label_b == label_b else label_b, sc, val_b,'''
    # This might not match exactly - let me try a more flexible search
    # Actually the metric comparison _add_row is trickier. Let me find it.
    # Let me search for the "metrics" comparison type string
    idx = src.find('"metrics")\n')
    if idx > 0:
        # Check if there's already rho1 nearby
        nearby = src[max(0,idx-200):idx+20]
        if 'rho1_a=rho1_a' not in nearby:
            # Find the _add_row call ending with "metrics")
            # Pattern: sd_a, n_a, sd_b, n_b, "metrics")
            old_met_end = 'sd_a, n_a, sd_b, n_b, "metrics")'
            new_met_end = 'sd_a, n_a, sd_b, n_b, "metrics",\n                         rho1_a=rho1_a, rho1_b=rho1_b)'
            if old_met_end in src:
                src = src.replace(old_met_end, new_met_end, 1)
                changes.append("metric _add_row call: passes rho1")

    # ── 10. baseline_zero section ──
    old_bz_get = '            sd_a, n_a = _get_ext(ext_stats, sc, var, tech)'
    # This one might appear after the comparison section
    # Find the baseline_zero section
    bz_idx = src.find('# "-- baseline_zero')
    if bz_idx < 0:
        bz_idx = src.find('baseline_zero')
    if bz_idx > 0:
        # Only replace in the baseline_zero section
        section = src[bz_idx:]
        if 'sd_a, n_a = _get_ext(' in section:
            section = section.replace('sd_a, n_a = _get_ext(', 'sd_a, n_a, rho1_a = _get_ext(', 1)
            src = src[:bz_idx] + section
            changes.append("baseline_zero: _get_ext unpacks 3 values")

        # Update the _one_sample_t_test call
        old_bz_onesamp = '            t_stat, p_val, ci_lo, ci_hi, d = _one_sample_t_test(val_a, sd_a, n_a, mu0=0.0)'
        new_bz_onesamp = '            t_stat, p_val, ci_lo, ci_hi, d, g, nea = _one_sample_t_test(val_a, sd_a, n_a, mu0=0.0, rho1_a=rho1_a)'
        if old_bz_onesamp in src:
            src = src.replace(old_bz_onesamp, new_bz_onesamp, 1)
            changes.append("baseline_zero: _one_sample_t_test updated to 7-tuple")

        # Update baseline_zero row dict
        old_bz_dict = '''                "t_stat": t_stat, "p_value": p_val, "cohens_d": d,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,'''
        new_bz_dict = '''                "t_stat": t_stat, "p_value": p_val, "cohens_d": d,
                "hedges_g": g, "n_eff_A": nea, "n_eff_B": np.nan,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,'''
        if old_bz_dict in src:
            src = src.replace(old_bz_dict, new_bz_dict, 1)
            changes.append("baseline_zero: row dict updated")

    # ── 11. _regime_row ──
    old_regime_sig = '    def _regime_row(metric_label, sc, var, ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n):'
    new_regime_sig = '    def _regime_row(metric_label, sc, var, ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n,\n                    rho1_ev=np.nan, rho1_nev=np.nan):'
    if old_regime_sig in src:
        src = src.replace(old_regime_sig, new_regime_sig, 1)
        changes.append("_regime_row: signature updated with rho1")

    old_regime_welch = '''        t_stat, p_val, ci_lo, ci_hi, d = _welch_t_test(
            ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n)'''
    new_regime_welch = '''        t_stat, p_val, ci_lo, ci_hi, d, g, nea, neb = _welch_t_test(
            ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n,
            rho1_a=rho1_ev, rho1_b=rho1_nev)'''
    if old_regime_welch in src:
        src = src.replace(old_regime_welch, new_regime_welch, 1)
        changes.append("_regime_row: welch call updated to 8-tuple")

    # _regime_row dict — second occurrence of the pattern
    if old_add_row_dict in src:
        src = src.replace(old_add_row_dict, new_add_row_dict, 1)
        changes.append("_regime_row: row dict updated")

    # ── 12. _regime_row call sites: pass rho1 ──
    # Part A: Bias
    old_bias_call = '''            if _ok(ev_val, ev_sd, ev_n) and _ok(nev_val, nev_sd, nev_n):
                rows.append(_regime_row(
                    "Mean bias (event vs non-event days)",
                    sc, var, ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n))'''
    new_bias_call = '''            ev_rho1 = ev_same.get("rho1_bias", np.nan)
            nev_rho1 = nev_same.get("rho1_bias", np.nan)

            if _ok(ev_val, ev_sd, ev_n) and _ok(nev_val, nev_sd, nev_n):
                rows.append(_regime_row(
                    "Mean bias (event vs non-event days)",
                    sc, var, ev_val, ev_sd, ev_n, nev_val, nev_sd, nev_n,
                    rho1_ev=ev_rho1, rho1_nev=nev_rho1))'''
    if old_bias_call in src:
        src = src.replace(old_bias_call, new_bias_call, 1)
        changes.append("_regime_row bias call: passes rho1")

    # Part B: Overlay shift — add rho1 read and pass
    old_overlay_block = '''        ev_mean  = oc.get("delta_event_pairs_mean")
        ev_sd    = oc.get("sd_delta_event")
        ev_n     = oc.get("n_delta_event")
        nev_mean = oc.get("delta_non_event_pairs_mean")
        nev_sd   = oc.get("sd_delta_nonevent")
        nev_n    = oc.get("n_delta_nonevent")

        if _ok(ev_mean, ev_sd, ev_n) and _ok(nev_mean, nev_sd, nev_n):
            rows.append(_regime_row(
                "Ensemble shift (event vs non-event)",
                sc, var, ev_mean, ev_sd, ev_n, nev_mean, nev_sd, nev_n))'''
    new_overlay_block = '''        ev_mean  = oc.get("delta_event_pairs_mean")
        ev_sd    = oc.get("sd_delta_event")
        ev_n     = oc.get("n_delta_event")
        ev_rho1  = oc.get("rho1_delta_event", np.nan)
        nev_mean = oc.get("delta_non_event_pairs_mean")
        nev_sd   = oc.get("sd_delta_nonevent")
        nev_n    = oc.get("n_delta_nonevent")
        nev_rho1 = oc.get("rho1_delta_nonevent", np.nan)

        if _ok(ev_mean, ev_sd, ev_n) and _ok(nev_mean, nev_sd, nev_n):
            rows.append(_regime_row(
                "Ensemble shift (event vs non-event)",
                sc, var, ev_mean, ev_sd, ev_n, nev_mean, nev_sd, nev_n,
                rho1_ev=ev_rho1, rho1_nev=nev_rho1))'''
    if old_overlay_block in src:
        src = src.replace(old_overlay_block, new_overlay_block, 1)
        changes.append("_regime_row overlay call: passes rho1")

    # ── 13. build_error_data_from_extended_stats: add ci95_neff mode ──
    old_error_doc = '    error_mode : {"sd", "se", "ci95"}'
    new_error_doc = '    error_mode : {"sd", "se", "ci95", "se_neff", "ci95_neff"}'
    if old_error_doc in src:
        src = src.replace(old_error_doc, new_error_doc, 1)

    old_error_desc = '''        "sd" \u2019 raw standard deviation, "se" \u2019 SD/sqrt(n), "ci95" \u2019 1.96*SE.'''
    new_error_desc = (
        '        "sd" \u2019 raw standard deviation, "se" \u2019 SD/sqrt(n), "ci95" \u2019 1.96*SE.\n'
        '        "se_neff" and "ci95_neff" use n_eff = n\u00b7(1\u2212\u03c1\u2081)/(1+\u03c1\u2081) instead of raw n,\n'
        '        falling back to raw n when rho1 is unavailable (Zwiers & von Storch 1995).'
    )
    if old_error_desc in src:
        src = src.replace(old_error_desc, new_error_desc, 1)

    old_error_branch = (
        '        sd = entry.get("sd", float("nan"))\n'
        '        n = entry.get("n", float("nan"))\n'
        '        if not np.isfinite(sd):\n'
        '            continue\n'
        '        if error_mode == "se":\n'
        '            if np.isfinite(n) and n > 1:\n'
        '                error_val = sd / np.sqrt(n)\n'
        '            else:\n'
        '                continue\n'
        '        elif error_mode == "ci95":\n'
        '            if np.isfinite(n) and n > 1:\n'
        '                error_val = 1.96 * sd / np.sqrt(n)\n'
        '            else:\n'
        '                continue\n'
        '        else:  # sd\n'
        '            error_val = sd'
    )
    new_error_branch = (
        '        sd = entry.get("sd", float("nan"))\n'
        '        n = entry.get("n", float("nan"))\n'
        '        rho1 = entry.get("rho1", float("nan"))\n'
        '        if not np.isfinite(sd):\n'
        '            continue\n'
        '        # Compute n_eff when rho1 is available and valid\n'
        '        if np.isfinite(rho1) and 0 < rho1 < 1 and np.isfinite(n) and n > 1:\n'
        '            n_eff = max(2.0, n * (1 - rho1) / (1 + rho1))\n'
        '        else:\n'
        '            n_eff = n  # fallback to raw n\n'
        '        if error_mode == "se":\n'
        '            if np.isfinite(n) and n > 1:\n'
        '                error_val = sd / np.sqrt(n)\n'
        '            else:\n'
        '                continue\n'
        '        elif error_mode == "se_neff":\n'
        '            if np.isfinite(n_eff) and n_eff > 1:\n'
        '                error_val = sd / np.sqrt(n_eff)\n'
        '            else:\n'
        '                continue\n'
        '        elif error_mode == "ci95":\n'
        '            if np.isfinite(n) and n > 1:\n'
        '                error_val = 1.96 * sd / np.sqrt(n)\n'
        '            else:\n'
        '                continue\n'
        '        elif error_mode == "ci95_neff":\n'
        '            if np.isfinite(n_eff) and n_eff > 1:\n'
        '                error_val = 1.96 * sd / np.sqrt(n_eff)\n'
        '            else:\n'
        '                continue\n'
        '        else:  # sd\n'
        '            error_val = sd'
    )
    if old_error_branch in src:
        src = src.replace(old_error_branch, new_error_branch, 1)
        changes.append("build_error_data: added ci95_neff/se_neff error modes")

    # Write back
    cell["source"] = to_source_list(src)
    break

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 22 (config): SIGNIFICANCE_METRICS reorder
# ═══════════════════════════════════════════════════════════════════════════════
for ci, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "SIGNIFICANCE TABLE - COLUMN CONFIGURATION" not in src:
        continue

    old_list = '''SIGNIFICANCE_METRICS: List[str] = [
    # "-- values & difference "--
    "diff",
    "rel_diff",
    "value_A",
    "value_B",
    # "-- sample descriptors (Tier 2) "--
    "sd_A",
    "sd_B",
    "n_A",
    "n_B",
    # "-- inferential statistics (Tier 2) "--
    "t_stat",
    "p_value",       #  auto-adds a "sig" stars column next to it
    "cohens_d",
    "ci95_lo",
    "ci95_hi",
]'''
    new_list = '''SIGNIFICANCE_METRICS: List[str] = [
    # "-- effect size & autocorrelation-corrected inference (new) "--
    "hedges_g",
    "n_eff_A",
    "n_eff_B",
    "cohens_d",
    # "-- inferential statistics (Tier 2) "--
    "t_stat",
    "p_value",       #  auto-adds a "sig" stars column next to it
    "ci95_lo",
    "ci95_hi",
    # "-- values & difference "--
    "diff",
    "rel_diff",
    "value_A",
    "value_B",
    # "-- sample descriptors (Tier 2) "--
    "sd_A",
    "sd_B",
    "n_A",
    "n_B",
]'''
    if old_list in src:
        cell["source"] = (src.replace(old_list, new_list, 1)).splitlines(True)
        if cell["source"] and not cell["source"][-1].endswith('\n'):
            cell["source"][-1] += '\n'
        changes.append("SIGNIFICANCE_METRICS: reordered with new stats in first half")
    break

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 23 (chart 1): variant error_mode + new variants
# ═══════════════════════════════════════════════════════════════════════════════
for ci, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "_chart_1_variants" not in src or "chart_1_metrics_by_folder" not in src:
        continue
    if "build_error_data_from_extended_stats" not in src:
        continue

    # Use variant-level error_mode
    old_call = (
        '        _ch1_error_data = build_error_data_from_extended_stats(\n'
        '            generated_specs, _ch1_error_metric_labels,\n'
        '            error_mode="ci95",\n'
        '        )'
    )
    new_call = (
        '        _ch1_error_mode = variant.get("error_mode", "ci95")\n'
        '        _ch1_error_data = build_error_data_from_extended_stats(\n'
        '            generated_specs, _ch1_error_metric_labels,\n'
        '            error_mode=_ch1_error_mode,\n'
        '        )'
    )
    if old_call in src:
        src = src.replace(old_call, new_call, 1)
        changes.append("chart 1 loop: error_mode from variant dict")
    elif '_ch1_error_mode = variant.get' in src:
        changes.append("chart 1 loop: error_mode already from variant (skip)")

    # Use variant-level error_bar_legend_label
    old_legend = '            error_bar_legend_label="95% CI",'
    new_legend = '            error_bar_legend_label=variant.get("error_bar_legend_label", "95% CI"),'
    if old_legend in src:
        src = src.replace(old_legend, new_legend, 1)
        changes.append("chart 1 loop: legend label from variant dict")
    elif 'variant.get("error_bar_legend_label"' in src:
        changes.append("chart 1 loop: legend label already from variant (skip)")

    # Add 1c_neff and 1d_neff variants if not already present
    if "chart_1c_mean_rw_neff" not in src:
        new_variants = '''
    {
        "slug": "chart_1c_mean_rw_neff",
        "labels": ch1c_semantic_labels,
        "descriptor": "Relative sensitivity (mean) \u2013 CI\u2089\u2085 corrected for autocorrelation",
        "y_label": "factor compared to envelope mean",
        "error_mode": "ci95_neff",
        "error_bar_legend_label": "95% CI (n\u1d49\u1da0\u1da0)",
        "stat_label_map": {
            "Mean relative width (all days)": "all days\\n(mean)",
            "Mean relative width (high-flow days)": "high-flow days\\n(mean)",
            "Mean relative width (baseflow days)": "baseflow days\\n(mean)",
        },
        "figsize": (14, 8),
    },

    {
        "slug": "chart_1d_maxmin_neff",
        "labels": ch1d_semantic_labels,
        "descriptor": "Spread sensitivity (max-min)/median \u2013 CI\u2089\u2085 corrected for autocorrelation",
        "y_label": "factor compared to envelope median",
        "error_mode": "ci95_neff",
        "error_bar_legend_label": "95% CI (n\u1d49\u1da0\u1da0)",
        "stat_label_map": {
            "Median (max-min/median) (all days)": "all days\\n(median)",
            "Median (max-min/median) (high-flow days)": "high-flow days\\n(median)",
            "Median (max-min/median) (baseflow days)": "baseflow days\\n(median)",
        },
        "figsize": (20, 8),
    },
'''
        close_marker = "\n]\n\nch_1_title_metadata_parts"
        if close_marker in src:
            src = src.replace(close_marker, new_variants + "]\n\nch_1_title_metadata_parts", 1)
            changes.append("chart 1: added 1c_neff and 1d_neff variants")

    cell["source"] = src.splitlines(True)
    if cell["source"] and not cell["source"][-1].endswith('\n'):
        cell["source"][-1] += '\n'
    break

# ═══════════════════════════════════════════════════════════════════════════════
#  Write back
# ═══════════════════════════════════════════════════════════════════════════════
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print(f"\nApplied {len(changes)} patches:")
for c in changes:
    print(f"  \u2713 {c}")

# Validate syntax
import ast
nb2 = json.load(open(NB_PATH, encoding="utf-8"))
errors = []
code_count = 0
for i, cell in enumerate(nb2["cells"]):
    if cell["cell_type"] != "code":
        continue
    code_count += 1
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i, e.lineno, e.msg))
if errors:
    print(f"\nSYNTAX ERRORS:")
    for ci, ln, msg in errors:
        print(f"  Cell {ci}: line {ln}: {msg}")
else:
    print(f"\nAll {code_count} code cells parse OK")
