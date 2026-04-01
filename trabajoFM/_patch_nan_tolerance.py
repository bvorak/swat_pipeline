"""
Patch 04_sensitivity_stats.ipynb:
1. _extract_metric_value  → return NaN instead of KeyError for missing paths
2. plot_nested_sensitivity_bars → skip NaN bars and filter NaN from max calculation
"""
import json, pathlib, sys

NB = pathlib.Path(__file__).with_name("notebooks") / "04_sensitivity_stats.ipynb"
nb = json.loads(NB.read_text("utf-8"))
cells = nb["cells"]

# ── helpers ──────────────────────────────────────────────────────────────────
def src(cell):
    return "".join(cell["source"])

def set_src(cell, new_text):
    cell["source"] = new_text.splitlines(keepends=True)
    # ensure last element has no trailing newline artefact
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1]  # keep as-is; splitlines(True) already correct

changes = 0

# ── 1. Patch _extract_metric_value in cell 5 (0-based idx 4) ────────────────
for idx, c in enumerate(cells):
    code = src(c)
    if "def _extract_metric_value(" not in code:
        continue

    # Change 1a: explicit-path KeyError → NaN
    old_explicit = (
        '                raise KeyError(\n'
        '                    f"Explicit path {explicit_path!r} for \'{technical_name}\' does not exist in the stats export."\n'
        '                )'
    )
    new_explicit = (
        '                if verbose:\n'
        '                    print(f"    explicit path {explicit_path} for \'{technical_name}\' not found, returning NaN")\n'
        '                return float(\'nan\')'
    )
    if old_explicit in code:
        code = code.replace(old_explicit, new_explicit, 1)
        changes += 1
        print(f"  [cell {idx}] patched explicit-path KeyError → NaN")
    else:
        print(f"  [cell {idx}] WARNING: explicit-path KeyError pattern not found", file=sys.stderr)

    # Change 1b: auto-discovery KeyError → NaN
    old_auto = (
        '        raise KeyError(\n'
        '            f"Metric \'{technical_name}\' was not found anywhere in the stats export. "\n'
        '            "If it is nested under an ambiguous branch, add a \'path\' entry in semantic_label_map for it."\n'
        '        )'
    )
    new_auto = (
        '        if verbose:\n'
        '            print(f"    metric \'{technical_name}\' not found anywhere in the stats export, returning NaN")\n'
        '        return float(\'nan\')'
    )
    if old_auto in code:
        code = code.replace(old_auto, new_auto, 1)
        changes += 1
        print(f"  [cell {idx}] patched auto-discovery KeyError → NaN")
    else:
        print(f"  [cell {idx}] WARNING: auto-discovery KeyError pattern not found", file=sys.stderr)

    set_src(c, code)
    break
else:
    print("  WARNING: could not find _extract_metric_value cell", file=sys.stderr)

# ── 2. Patch plot_nested_sensitivity_bars ────────────────────────────────────
for idx, c in enumerate(cells):
    code = src(c)
    if "def plot_nested_sensitivity_bars(" not in code:
        continue

    # Change 2a: filter NaN from all_values
    old_all_values = (
        '    all_values = [\n'
        '        float(metrics[stat][scenario][var])\n'
        '        for stat in stats\n'
        '        for scenario in scenarios\n'
        '        for var in variables\n'
        '    ]'
    )
    new_all_values = (
        '    all_values = [\n'
        '        float(metrics[stat][scenario][var])\n'
        '        for stat in stats\n'
        '        for scenario in scenarios\n'
        '        for var in variables\n'
        '        if not math.isnan(float(metrics[stat][scenario][var]))\n'
        '    ]'
    )
    if old_all_values in code:
        code = code.replace(old_all_values, new_all_values, 1)
        changes += 1
        print(f"  [cell {idx}] patched all_values NaN filter")
    else:
        print(f"  [cell {idx}] WARNING: all_values pattern not found", file=sys.stderr)

    # Change 2b: skip NaN bars
    old_bar = (
        '            for k, var in enumerate(variables):\n'
        '                x = subgroup_left + k * bar_width\n'
        '                y = float(block[scenario][var])\n'
        '                ax.bar(x, y, width=bar_width, color=var_to_color[var], alpha=bar_alpha)'
    )
    new_bar = (
        '            for k, var in enumerate(variables):\n'
        '                x = subgroup_left + k * bar_width\n'
        '                y = float(block[scenario][var])\n'
        '                if math.isnan(y):\n'
        '                    continue\n'
        '                ax.bar(x, y, width=bar_width, color=var_to_color[var], alpha=bar_alpha)'
    )
    if old_bar in code:
        code = code.replace(old_bar, new_bar, 1)
        changes += 1
        print(f"  [cell {idx}] patched NaN bar skip")
    else:
        print(f"  [cell {idx}] WARNING: bar-drawing pattern not found", file=sys.stderr)

    set_src(c, code)
    break
else:
    print("  WARNING: could not find plot_nested_sensitivity_bars cell", file=sys.stderr)

# ── write back ───────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), "utf-8")
print(f"\nDone — {changes} patches applied to {NB.name}")
