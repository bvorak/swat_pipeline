"""
Patch 04_sensitivity_stats.ipynb:
1. Add info_box_* parameters to plot_nested_sensitivity_bars (cell 7)
2. Add info_box drawing code to the function body (cell 7)
3. Add baseline_p50 extraction + config vars to Chart 2 (cell 21)
4. Wire info_box arguments into the plot call in Chart 2 (cell 21)
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/04_sensitivity_stats.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── PATCH 1: Add info_box parameters to helper function signature (cell 7) ──
c7_src = "".join(nb["cells"][7]["source"])

PATCH1_OLD = (
    '    annotation_formatter=None,   # optional callable: '
    '(value, stat, scenario, variable, default_text) -> str | None\n'
    '\n'
    '):'
)

PATCH1_NEW = (
    '    annotation_formatter=None,   # optional callable: '
    '(value, stat, scenario, variable, default_text) -> str | None\n'
    '\n'
    '    info_box_lines=None,          # list of strings to display in an info box on the chart\n'
    '    info_box_x=0.875,             # horizontal position in axes coords (0-1); 0.875 = centre of 4th quarter\n'
    '    info_box_fs=None,             # font size (defaults to legend_fs)\n'
    '    info_box_title=None,          # optional title line rendered in bold above the values\n'
    '):'
)

if PATCH1_OLD not in c7_src:
    if "info_box_lines" in c7_src:
        print("PATCH1: already applied (info_box_lines param present)")
    else:
        raise RuntimeError("PATCH1: cannot find signature anchor in cell 7")
else:
    c7_src = c7_src.replace(PATCH1_OLD, PATCH1_NEW, 1)
    print("PATCH1: added info_box_* params to signature")

# ── PATCH 2: Add info_box drawing code before return statement (cell 7) ──
PATCH2_OLD = '    return fig, ax'

PATCH2_NEW = (
    '    # ── optional info box ──\n'
    '    if info_box_lines:\n'
    '        _ib_fs = info_box_fs if info_box_fs is not None else legend_fs\n'
    '        _ib_parts = []\n'
    '        if info_box_title:\n'
    '            _ib_parts.append(info_box_title)\n'
    '        _ib_parts.extend(info_box_lines)\n'
    '        _ib_text = "\\n".join(_ib_parts)\n'
    '        ax.text(\n'
    '            info_box_x, 0.98, _ib_text,\n'
    '            transform=ax.transAxes,\n'
    '            fontsize=_ib_fs,\n'
    '            verticalalignment="top",\n'
    '            horizontalalignment="center",\n'
    '            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8, edgecolor="gray"),\n'
    '        )\n'
    '\n'
    '    return fig, ax'
)

if PATCH2_OLD not in c7_src:
    if "info_box_lines:" in c7_src and "ax.text" in c7_src:
        print("PATCH2: already applied (info_box drawing code present)")
    else:
        raise RuntimeError("PATCH2: cannot find 'return fig, ax' in cell 7")
else:
    # Replace only the LAST occurrence (there should be only one, but be safe)
    idx = c7_src.rfind(PATCH2_OLD)
    c7_src = c7_src[:idx] + PATCH2_NEW + c7_src[idx + len(PATCH2_OLD):]
    print("PATCH2: added info_box drawing code")

nb["cells"][7]["source"] = c7_src.splitlines(keepends=True)
# Ensure last line has trailing newline
if nb["cells"][7]["source"] and not nb["cells"][7]["source"][-1].endswith("\n"):
    nb["cells"][7]["source"][-1] += "\n"

# ── PATCH 3: Add config vars + extraction logic to Chart 2 (cell 21) ──
c21_src = "".join(nb["cells"][21]["source"])

# 3a: Insert config variables after the initial "chart_2_logs = []" block
PATCH3A_OLD = 'chart_2_logs = []\n'
PATCH3A_NEW = (
    'chart_2_logs = []\n'
    '\n'
    '# ── Info-box configuration (baseline metric shown per variable) ──\n'
    'ch2_info_box_prefix = "BASE p50"                             # text before the numeric value\n'
    'ch2_info_box_x = 0.875                                       # horizontal position (0–1 in axes coords)\n'
    'ch2_info_box_metric_path = ("overlay_comparison", "BASE", "baseline_p50")\n'
)

if PATCH3A_OLD not in c21_src:
    if "ch2_info_box_prefix" in c21_src:
        print("PATCH3A: already applied (config vars present)")
    else:
        raise RuntimeError("PATCH3A: cannot find 'chart_2_logs = []' in cell 21")
else:
    c21_src = c21_src.replace(PATCH3A_OLD, PATCH3A_NEW, 1)
    print("PATCH3A: added info-box config vars")

# 3b: Insert extraction logic after the auto_metrics block, before the compare_mode block.
# Find the closing paren of build_metrics_dict_from_stat_exports(...)
PATCH3B_OLD = (
    '        verbose=False,\n'
    '\n'
    '    )\n'
    '\n'
    '\n'
    '\n'
    '    _ch2_example_export = _load_stat_export'
)
PATCH3B_NEW = (
    '        verbose=False,\n'
    '\n'
    '    )\n'
    '\n'
    '    # Extract baseline_p50 for the info box (first JSON per variable)\n'
    '    _ch2_baseline_p50 = {}\n'
    '    for _spec in generated_specs:\n'
    '        _loaded = _load_stat_export(_spec, verbose=False)\n'
    '        _var = _loaded["variable"]\n'
    '        if _var not in _ch2_baseline_p50:\n'
    '            _val = _extract_metric_value(\n'
    '                _loaded["stats"], "baseline_p50",\n'
    '                explicit_path=ch2_info_box_metric_path,\n'
    '                verbose=False,\n'
    '            )\n'
    '            _ch2_baseline_p50[_var] = _val\n'
    '    _ch2_info_box_lines = [\n'
    '        f"{var}: {ch2_info_box_prefix} = {_ch2_baseline_p50[var]:.2f}"\n'
    '        for var in _ch2_selected_variables\n'
    '        if var in _ch2_baseline_p50\n'
    '    ]\n'
    '\n'
    '\n'
    '\n'
    '    _ch2_example_export = _load_stat_export'
)

if PATCH3B_OLD not in c21_src:
    if "_ch2_baseline_p50" in c21_src:
        print("PATCH3B: already applied (extraction logic present)")
    else:
        raise RuntimeError("PATCH3B: cannot find metrics-block anchor in cell 21")
else:
    c21_src = c21_src.replace(PATCH3B_OLD, PATCH3B_NEW, 1)
    print("PATCH3B: added baseline_p50 extraction logic")

# 3c: Wire info_box arguments into the plot_nested_sensitivity_bars call
PATCH3C_OLD = '        annotation_formatter=_ch2_annotation_formatter,\n    )'
PATCH3C_NEW = (
    '        annotation_formatter=_ch2_annotation_formatter,\n'
    '\n'
    '        info_box_lines=_ch2_info_box_lines,\n'
    '        info_box_x=ch2_info_box_x,\n'
    '    )'
)

if PATCH3C_OLD not in c21_src:
    if "info_box_lines=_ch2_info_box_lines" in c21_src:
        print("PATCH3C: already applied (info_box args in call)")
    else:
        raise RuntimeError("PATCH3C: cannot find annotation_formatter kwarg anchor in cell 21")
else:
    c21_src = c21_src.replace(PATCH3C_OLD, PATCH3C_NEW, 1)
    print("PATCH3C: wired info_box arguments into plot call")

nb["cells"][21]["source"] = c21_src.splitlines(keepends=True)
if nb["cells"][21]["source"] and not nb["cells"][21]["source"][-1].endswith("\n"):
    nb["cells"][21]["source"][-1] += "\n"

# ── Save ──
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved successfully.")

# ── Verify ──
with open(NB_PATH, encoding="utf-8") as f:
    nb2 = json.load(f)

c7v = "".join(nb2["cells"][7]["source"])
c21v = "".join(nb2["cells"][21]["source"])

checks = {
    "c7 info_box_lines param": "info_box_lines=None," in c7v,
    "c7 info_box_x param": "info_box_x=0.875," in c7v,
    "c7 info_box_fs param": "info_box_fs=None," in c7v,
    "c7 info_box_title param": "info_box_title=None," in c7v,
    "c7 info_box drawing code": "if info_box_lines:" in c7v,
    "c21 ch2_info_box_prefix config": "ch2_info_box_prefix" in c21v,
    "c21 ch2_info_box_x config": "ch2_info_box_x = 0.875" in c21v,
    "c21 _ch2_baseline_p50 extraction": "_ch2_baseline_p50" in c21v,
    "c21 info_box_lines kwarg": "info_box_lines=_ch2_info_box_lines" in c21v,
    "c21 info_box_x kwarg": "info_box_x=ch2_info_box_x" in c21v,
}

print("\n=== VERIFICATION ===")
all_ok = True
for label, ok in checks.items():
    status = "OK" if ok else "MISSING"
    print(f"  {label}: {status}")
    if not ok:
        all_ok = False

if all_ok:
    print("\nAll patches verified successfully!")
else:
    print("\nWARNING: Some patches are missing!")
