"""
Patch 04_sensitivity_stats.ipynb:
- Add annotation_formatter hook to plot_nested_sensitivity_bars (cell 7)
- Wire _ch2_annotation_formatter in chart 2 cell (cell 21)
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "04_sensitivity_stats.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cells by content, not index
def find_cell(nb, marker):
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if marker in src:
            return i, cell, src
    raise ValueError(f"Could not find cell containing: {marker}")

# ── 1. Patch the config cell ──
config_idx, config_cell, config_src = find_cell(nb, "chart_abbreviations")

# Only patch if not already patched
if "title_metadata_parts" not in config_src:
    title_config_block = '''
# ── Title builder configuration ──
# Which metadata parts to include in each chart title (in order).
title_metadata_parts = [
    "folder_label",
    "reach",
    "event_threshold",
    "view",
    "n",
    "measured_nonnum_policy",
    "measured_selection",
]

# Map raw folder tokens to human-readable labels.
# Accepts a simple dict (token → label) or a list of rule dicts.
title_folder_label_map = {
    "drop_mdl": "drop-MDL",
    "half_mdl": "half-MDL",
    "non-event-days": "non-events",
    "runs_188_200_201": "soil uncertainty = P75",
    "runs_205_200_204": "soil uncertainty = |median relative error|",
    "runs_187_200_202": "soil uncertainty = median relative error",
    "runs_189_200_203": "soil uncertainty = RMSE rel. center",
    "runs_153_155_157": "original TFM",
}

'''
    # Insert before the final print block
    insert_marker = 'print(f"Configured {len(batch_folders)} batch folder(s).")'
    config_src = config_src.replace(
        insert_marker,
        title_config_block + insert_marker,
    )
    
    # Also add a print for title_metadata_parts
    config_src = config_src.replace(
        'print(f"  - {folder.name}")',
        'print(f"  - {folder.name}")\nprint(f"Title metadata parts: {title_metadata_parts}")',
    )
    
    config_cell["source"] = [line + "\n" for line in config_src.split("\n")]
    # Fix last line: remove trailing newline from last element
    if config_cell["source"][-1] == "\n":
        config_cell["source"] = config_cell["source"][:-1]
    print(f"Patched config cell (cell {config_idx})")
else:
    print("Config cell already patched")


# ── 2. Patch chart 1 cell ──
chart1_idx, chart1_cell, chart1_src = find_cell(nb, 'batch_folder.name} | Relative sensitivity metrics')

if "build_semantic_chart_title" not in chart1_src:
    # Add the title builder call before plot_nested_sensitivity_bars
    chart1_src = chart1_src.replace(
        '    fig, ax = plot_nested_sensitivity_bars(\n'
        '        ch_1_auto_metrics,\n'
        '        variables_order=_ch1_selected_variables,\n'
        '        title=f"{batch_folder.name} | Relative sensitivity metrics (min-max/median)",',
        '    chart_title = build_semantic_chart_title(\n'
        '        batch_folder, generated_specs,\n'
        '        "Relative sensitivity metrics (min-max/median)",\n'
        '        title_metadata_parts=title_metadata_parts,\n'
        '        title_folder_label_map=title_folder_label_map,\n'
        '    )\n'
        '\n'
        '    fig, ax = plot_nested_sensitivity_bars(\n'
        '        ch_1_auto_metrics,\n'
        '        variables_order=_ch1_selected_variables,\n'
        '        title=chart_title,',
    )
    chart1_cell["source"] = [line + "\n" for line in chart1_src.split("\n")]
    if chart1_cell["source"][-1] == "\n":
        chart1_cell["source"] = chart1_cell["source"][:-1]
    print(f"Patched chart 1 cell (cell {chart1_idx})")
else:
    print("Chart 1 cell already patched")

# ── 3. Patch chart 2 cell ──
chart2_idx, chart2_cell, chart2_src = find_cell(nb, 'batch_folder.name} | Shift sensitivity')

if "build_semantic_chart_title" not in chart2_src:
    chart2_src = chart2_src.replace(
        '    fig, ax = plot_nested_sensitivity_bars(\n'
        '        ch_2_auto_metrics,\n'
        '        title=f"{batch_folder.name} | Shift sensitivity metrics",',
        '    chart_title = build_semantic_chart_title(\n'
        '        batch_folder, generated_specs,\n'
        '        "Shift sensitivity metrics",\n'
        '        title_metadata_parts=title_metadata_parts,\n'
        '        title_folder_label_map=title_folder_label_map,\n'
        '    )\n'
        '\n'
        '    fig, ax = plot_nested_sensitivity_bars(\n'
        '        ch_2_auto_metrics,\n'
        '        title=chart_title,',
    )
    chart2_cell["source"] = [line + "\n" for line in chart2_src.split("\n")]
    if chart2_cell["source"][-1] == "\n":
        chart2_cell["source"] = chart2_cell["source"][:-1]
    print(f"Patched chart 2 cell (cell {chart2_idx})")
else:
    print("Chart 2 cell already patched")

# ── 4. Patch chart 3 cell ──
chart3_idx, chart3_cell, chart3_src = find_cell(nb, 'batch_folder.name} | Three metric comparison')

if "build_semantic_chart_title" not in chart3_src:
    chart3_src = chart3_src.replace(
        '        title=f"{batch_folder.name} | Three metric comparison",',
        '        title=build_semantic_chart_title(\n'
        '            batch_folder, generated_specs,\n'
        '            "Three metric comparison",\n'
        '            title_metadata_parts=title_metadata_parts,\n'
        '            title_folder_label_map=title_folder_label_map,\n'
        '        ),',
    )
    chart3_cell["source"] = [line + "\n" for line in chart3_src.split("\n")]
    if chart3_cell["source"][-1] == "\n":
        chart3_cell["source"] = chart3_cell["source"][:-1]
    print(f"Patched chart 3 cell (cell {chart3_idx})")
else:
    print("Chart 3 cell already patched")

# ── Write back ──
with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("\nDone. Notebook saved.")
