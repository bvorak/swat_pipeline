"""Patch: left-aligned color-coded info box in helper + Chart 2."""
import json

NB = "notebooks/04_sensitivity_stats.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# ── PATCH 1: Replace info box drawing in helper function (cell 7) ──
c7 = "".join(nb["cells"][7]["source"])

old_infobox = (
    '    # ── optional info box ──\n'
    '    if info_box_lines:\n'
    '        _ib_fs = info_box_fs if info_box_fs is not None else legend_fs\n'
    '        _ib_parts = []\n'
    '        if info_box_title:\n'
    '            _ib_parts.append(info_box_title)\n'
    '        _ib_parts.extend(info_box_lines)\n'
    '        _ib_text = "\\n".join(_ib_parts)\n'
    '        ax.text(\n'
    '            info_box_x, info_box_y, _ib_text,\n'
    '            transform=ax.transAxes,\n'
    '            fontsize=_ib_fs,\n'
    '            verticalalignment="top",\n'
    '            horizontalalignment="center",\n'
    '            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8, edgecolor="gray"),\n'
    '        )'
)

new_infobox = (
    '    # ── optional info box ──\n'
    '    if info_box_lines:\n'
    '        from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox\n'
    '        _ib_fs = info_box_fs if info_box_fs is not None else legend_fs\n'
    '        _ib_areas = []\n'
    '        if info_box_title:\n'
    '            _ib_areas.append(TextArea(info_box_title, textprops=dict(fontsize=_ib_fs, weight="bold")))\n'
    '        for _ib_entry in info_box_lines:\n'
    '            if isinstance(_ib_entry, (list, tuple)) and len(_ib_entry) == 2:\n'
    '                _ib_txt, _ib_clr = _ib_entry\n'
    '            else:\n'
    '                _ib_txt, _ib_clr = str(_ib_entry), "black"\n'
    '            _ib_areas.append(TextArea(_ib_txt, textprops=dict(fontsize=_ib_fs, color=_ib_clr)))\n'
    '        _ib_pack = VPacker(children=_ib_areas, pad=0, sep=2, align="left")\n'
    '        _ib_anchor = AnchoredOffsetbox(\n'
    '            loc="upper left", child=_ib_pack,\n'
    '            bbox_to_anchor=(info_box_x, info_box_y),\n'
    '            bbox_transform=ax.transAxes,\n'
    '            frameon=True, pad=0.4,\n'
    '        )\n'
    '        _ib_anchor.patch.set(facecolor="wheat", alpha=0.8, edgecolor="gray",\n'
    '                             boxstyle="round,pad=0.4")\n'
    '        ax.add_artist(_ib_anchor)'
)

assert old_infobox in c7, "PATCH1: old pattern not found in helper!"
c7 = c7.replace(old_infobox, new_infobox, 1)
nb["cells"][7]["source"] = c7.splitlines(keepends=True)
if nb["cells"][7]["source"] and not nb["cells"][7]["source"][-1].endswith("\n"):
    nb["cells"][7]["source"][-1] += "\n"
print("PATCH1 applied (helper: VPacker left-aligned color boxes)")

# ── PATCH 2: Update Chart 2 to pass (text, color) tuples ──
c21 = "".join(nb["cells"][21]["source"])

old_lines = (
    '    _ch2_info_box_lines = [\n'
    '        f"{var}: {ch2_info_box_prefix} = {_ch2_baseline_p50[var]:.2f}"\n'
    '        for var in _ch2_selected_variables\n'
    '        if var in _ch2_baseline_p50\n'
    '    ]'
)

new_lines = (
    '    _ch2_var_colors = {"TP": "#1f77b4", "TN": "#ff7f0e", "TKN": "#2ca02c", "TON": "#d62728", "PP": "#9467bd"}\n'
    '    _ch2_info_box_lines = [\n'
    '        (f"{var}: {ch2_info_box_prefix} {_ch2_baseline_p50[var]:.2f}", _ch2_var_colors.get(var, "black"))\n'
    '        for var in _ch2_selected_variables\n'
    '        if var in _ch2_baseline_p50\n'
    '    ]'
)

assert old_lines in c21, "PATCH2: old pattern not found in Chart 2!"
c21 = c21.replace(old_lines, new_lines, 1)
nb["cells"][21]["source"] = c21.splitlines(keepends=True)
if nb["cells"][21]["source"] and not nb["cells"][21]["source"][-1].endswith("\n"):
    nb["cells"][21]["source"][-1] += "\n"
print("PATCH2 applied (Chart 2: color tuples)")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved.")

# Verify
with open(NB, encoding="utf-8") as f:
    nb2 = json.load(f)
c7v = "".join(nb2["cells"][7]["source"])
c21v = "".join(nb2["cells"][21]["source"])
print(f"Has VPacker: {'VPacker' in c7v}")
print(f"Has AnchoredOffsetbox: {'AnchoredOffsetbox' in c7v}")
print(f'Has align="left": {chr(39)}align="left"{chr(39)} in c7v: {"align=" + chr(34) + "left" + chr(34) in c7v}')
print(f"Has color tuples: {'_ch2_var_colors' in c21v}")
