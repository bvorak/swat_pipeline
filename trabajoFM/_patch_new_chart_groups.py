"""
Patch 04_sensitivity_stats.ipynb:
1. Cell 3 (format_val + semantic_label_map): add new metric entries
2. Cell 20 (ch1_semantic_labels): add new chart groups
"""
import json, pathlib, sys

NB = pathlib.Path(__file__).with_name("notebooks") / "04_sensitivity_stats.ipynb"
nb = json.loads(NB.read_text("utf-8"))
cells = nb["cells"]

def src(cell):
    return "".join(cell["source"])

def set_src(cell, new_text):
    cell["source"] = new_text.splitlines(keepends=True)

changes = 0

# ── 1. Patch cell 3 (0-based idx 2): format_val + semantic_label_map ──────
cell3 = cells[2]
code = src(cell3)

# 1a. Add new format_val entries after existing bl_ entries
old_format_tail = '  "bl_p90_W_nonevent":       "{:.1f}",\n}'
new_format_tail = (
    '  "bl_p90_W_nonevent":       "{:.1f}",\n'
    '\n'
    '  # Absolute width event ratio\n'
    '  "bl_abs_W_event_ratio":    "{:.1f}×",\n'
    '\n'
    '  # Mean-based relative width (all / event / non-event)\n'
    '  "bl_RW_mean_all":          "{:.0%}",\n'
    '  "bl_RW_mean_event":        "{:.0%}",\n'
    '  "bl_RW_mean_nonevent":     "{:.0%}",\n'
    '}'
)
if old_format_tail in code:
    code = code.replace(old_format_tail, new_format_tail, 1)
    changes += 1
    print("  [cell 3] added format_val entries")
else:
    print("  WARNING: format_val tail not found", file=sys.stderr)

# 1b. Add new semantic_label_map entries after existing bl_ entries
old_sem_tail = (
    '  "bl_p90_W_nonevent": {"path": ("overlay_comparison", "__baseline__", "p90_W_nonevent"), '
    '"label": "Ensemble p90 W (non-event days)"},\n'
    '\n'
    '}'
)
new_sem_tail = (
    '  "bl_p90_W_nonevent": {"path": ("overlay_comparison", "__baseline__", "p90_W_nonevent"), '
    '"label": "Ensemble p90 W (non-event days)"},\n'
    '\n'
    '  # Absolute width event ratio\n'
    '  "bl_abs_W_event_ratio": {"path": ("overlay_comparison", "__baseline__", "abs_width_event_ratio"), '
    '"label": "Absolute width event ratio (ensemble)"},\n'
    '\n'
    '  # Mean-based relative width\n'
    '  "bl_RW_mean_all": {"path": ("overlay_comparison", "__baseline__", "relative_width_mean"), '
    '"label": "Mean relative width (all days)"},\n'
    '  "bl_RW_mean_event": {"path": ("overlay_comparison", "__baseline__", "relative_width_mean_event"), '
    '"label": "Mean relative width (event days)"},\n'
    '  "bl_RW_mean_nonevent": {"path": ("overlay_comparison", "__baseline__", "relative_width_mean_nonevent"), '
    '"label": "Mean relative width (non-event days)"},\n'
    '\n'
    '}'
)
if old_sem_tail in code:
    code = code.replace(old_sem_tail, new_sem_tail, 1)
    changes += 1
    print("  [cell 3] added semantic_label_map entries")
else:
    print("  WARNING: semantic_label_map tail not found", file=sys.stderr)

set_src(cell3, code)

# ── 2. Patch cell 20 (0-based idx 19): ch1_semantic_labels ────────────────
cell20 = cells[19]
code20 = src(cell20)

old_labels = (
    'ch1_semantic_labels = [\n'
    '    "Model relative width median((p95-p05)/p50)",\n'
    '    "Relative width (event days)",\n'
    '    "Relative width (non-event days)",\n'
    '    "Event ratio (ensemble spread)",\n'
    ']'
)
new_labels = (
    'ch1_semantic_labels = [\n'
    '    "Model relative width median((p95-p05)/p50)",\n'
    '    "Relative width (event days)",\n'
    '    "Relative width (non-event days)",\n'
    '    "Event ratio (ensemble spread)",\n'
    '    # Absolute width group\n'
    '    "Ensemble median W (event days)",\n'
    '    "Ensemble median W (non-event days)",\n'
    '    "Absolute width event ratio (ensemble)",\n'
    '    # Mean-based relative width group\n'
    '    "Mean relative width (all days)",\n'
    '    "Mean relative width (event days)",\n'
    '    "Mean relative width (non-event days)",\n'
    ']'
)
if old_labels in code20:
    code20 = code20.replace(old_labels, new_labels, 1)
    changes += 1
    print("  [cell 20] updated ch1_semantic_labels")
else:
    print("  WARNING: ch1_semantic_labels pattern not found", file=sys.stderr)

set_src(cell20, code20)

# ── 3. Patch cell 22 (0-based idx 21): stat_label_map in the chart call ───
cell22 = cells[21]
code22 = src(cell22)

old_label_map = (
    '        stat_label_map={\n'
    '            "Model relative width median((p95-p05)/p50)": "Rel. width",\n'
    '            "Relative width (event days)": "Rel. width during events",\n'
    '            "Relative width (non-event days)": "Rel. width during non-events",\n'
    '        },'
)
new_label_map = (
    '        stat_label_map={\n'
    '            "Model relative width median((p95-p05)/p50)": "Rel. width\\n(median)",\n'
    '            "Relative width (event days)": "Rel. width\\n(events)",\n'
    '            "Relative width (non-event days)": "Rel. width\\n(non-events)",\n'
    '            "Event ratio (ensemble spread)": "Event ratio\\n(rel. width)",\n'
    '            "Ensemble median W (event days)": "Abs. width\\n(events)",\n'
    '            "Ensemble median W (non-event days)": "Abs. width\\n(non-events)",\n'
    '            "Absolute width event ratio (ensemble)": "Event ratio\\n(abs. width)",\n'
    '            "Mean relative width (all days)": "Rel. width\\n(mean, all)",\n'
    '            "Mean relative width (event days)": "Rel. width\\n(mean, events)",\n'
    '            "Mean relative width (non-event days)": "Rel. width\\n(mean, non-ev.)",\n'
    '        },'
)
if old_label_map in code22:
    code22 = code22.replace(old_label_map, new_label_map, 1)
    changes += 1
    print("  [cell 22] updated stat_label_map")
else:
    print("  WARNING: stat_label_map pattern not found", file=sys.stderr)

set_src(cell22, code22)

# ── write back ───────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), "utf-8")
print(f"\nDone — {changes} patches applied to {NB.name}")
