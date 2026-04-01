"""
Patch 04_sensitivity_stats.ipynb:
1. Cell 20: split ch1_semantic_labels into ch1a/ch1b/ch1c
2. Cell 22: split single chart loop into 3 chart loops (1a, 1b, 1c)
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

# ── 1. Patch cell 20 (0-based idx 19): split ch1_semantic_labels ──────────
cell20 = cells[19]
code20 = src(cell20)

old_labels = (
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

new_labels = (
    '# ── Chart 1a: Median-based relative width (existing) ──\n'
    'ch1a_semantic_labels = [\n'
    '    "Model relative width median((p95-p05)/p50)",\n'
    '    "Relative width (event days)",\n'
    '    "Relative width (non-event days)",\n'
    '    "Event ratio (ensemble spread)",\n'
    ']\n'
    '\n'
    '# ── Chart 1b: Absolute width (event vs non-event) ──\n'
    'ch1b_semantic_labels = [\n'
    '    "Ensemble median W (event days)",\n'
    '    "Ensemble median W (non-event days)",\n'
    '    "Absolute width event ratio (ensemble)",\n'
    ']\n'
    '\n'
    '# ── Chart 1c: Mean-based relative width ──\n'
    'ch1c_semantic_labels = [\n'
    '    "Mean relative width (all days)",\n'
    '    "Mean relative width (event days)",\n'
    '    "Mean relative width (non-event days)",\n'
    ']\n'
    '\n'
    '# Combined list for backward-compatibility references\n'
    'ch1_semantic_labels = ch1a_semantic_labels + ch1b_semantic_labels + ch1c_semantic_labels'
)

if old_labels in code20:
    code20 = code20.replace(old_labels, new_labels, 1)
    changes += 1
    print("  [cell 20] split ch1_semantic_labels into ch1a/ch1b/ch1c")
else:
    print("  WARNING: ch1_semantic_labels pattern not found", file=sys.stderr)

set_src(cell20, code20)

# ── 2. Replace cell 22 (0-based idx 21): 3 chart loops ───────────────────
cell22 = cells[21]

new_chart_code = r'''chart_1_metrics_by_folder = {}
chart_1_logs = []

# ── Chart configuration for 3 sub-charts ──────────────────────────────────
_chart_1_variants = [
    {
        "slug": "chart_1a_relative_width_median",
        "labels": ch1a_semantic_labels,
        "descriptor": "Relative sensitivity — median((p95−p05)/p50)",
        "y_label": "factor compared to envelope median",
        "stat_label_map": {
            "Model relative width median((p95-p05)/p50)": "Rel. width\n(all days)",
            "Relative width (event days)": "Rel. width\n(events)",
            "Relative width (non-event days)": "Rel. width\n(non-events)",
            "Event ratio (ensemble spread)": "Event ratio\n(rel. width)",
        },
        "figsize": (16, 8),
    },
    {
        "slug": "chart_1b_absolute_width",
        "labels": ch1b_semantic_labels,
        "descriptor": "Absolute envelope width — event vs non-event",
        "y_label": "width (variable units / day)",
        "stat_label_map": {
            "Ensemble median W (event days)": "Abs. width\n(events)",
            "Ensemble median W (non-event days)": "Abs. width\n(non-events)",
            "Absolute width event ratio (ensemble)": "Event ratio\n(abs. width)",
        },
        "figsize": (14, 8),
    },
    {
        "slug": "chart_1c_relative_width_mean",
        "labels": ch1c_semantic_labels,
        "descriptor": "Relative sensitivity — mean-based relative width",
        "y_label": "factor compared to envelope mean",
        "stat_label_map": {
            "Mean relative width (all days)": "Rel. width\n(all days)",
            "Mean relative width (event days)": "Rel. width\n(events)",
            "Mean relative width (non-event days)": "Rel. width\n(non-events)",
        },
        "figsize": (14, 8),
    },
]

ch_1_title_metadata_parts = [
    "folder_label",
    "event_threshold",
]

ch_1_title_folder_label_map = {
    "runs_187_200_202":      "Soil interpolation error: relative median",
    "runs_205_200_204":      "Soil interpolation error: absolute median",
    "runs_189_200_203":      "Soil interpolation error: relative RMSE",
    "runs_188_200_201":      "Soil interpolation error: 75th percentile",
    "runs_153_155_157":      "Original TFM (relative RMSE)",
    "drop_mdl":              None,
    "half_mdl":              None,
}

EVENT_THRESHOLD_TITLE_ALIASES = {
    "p50": "event defintion: above 50th flow percentile",
    "p75": "event defintion: above 75th flow percentile",
    "p90": "event defintion: above 90th flow percentile",
    "abs": "event defintion: above absolute flow value",
}

for batch_folder in batch_folders:
    generated_specs, build_log = _capture_stdout(
        build_export_specs_from_folder,
        batch_folder,
        run_to_scenario,
        variable_map,
        verbose=True,
    )

    _ch1_actual_scenarios = list(dict.fromkeys(spec["scenario"] for spec in generated_specs))
    _ch1_actual_variables = list(dict.fromkeys(spec["variable"] for spec in generated_specs))
    _ch1_selected_scenarios = _ordered_present_values(_ch1_actual_scenarios, chart_scenario_order)
    _ch1_selected_variables = _ordered_present_values(_ch1_actual_variables, chart_variable_order)

    # Build metrics for ALL ch1 labels at once (one pass over the JSON files)
    ch_1_auto_metrics = build_metrics_dict_from_stat_exports(
        generated_specs,
        ch1_semantic_labels,
        semantic_label_map=example_semantic_label_map,
        scenario_order=_ch1_selected_scenarios,
        variable_order=_ch1_selected_variables,
        format_mode="metadata",
        verbose=False,
    )

    for variant in _chart_1_variants:
        # Filter the full metrics dict to only this variant's labels
        variant_metrics = {
            "metrics": {
                k: v for k, v in ch_1_auto_metrics["metrics"].items()
                if k in variant["labels"]
            }
        }

        chart_title, chart_subtitle = build_semantic_chart_title(
            batch_folder, generated_specs,
            variant["descriptor"],
            title_metadata_parts=ch_1_title_metadata_parts,
            title_folder_label_map=ch_1_title_folder_label_map,
        )

        fig, ax = plot_nested_sensitivity_bars(
            variant_metrics,
            variables_order=_ch1_selected_variables,
            title=chart_title,
            subtitle=chart_subtitle,
            y_label=variant["y_label"],
            figsize=variant["figsize"],
            scenarios_order=_ch1_selected_scenarios,
            stats_order=variant["labels"],
            stat_label_map=variant["stat_label_map"],
            abbreviations=chart_abbreviations,
            annotate_rotation=60,
            annot_fs=22,
            tick_fs=24,
            label_fs=22,
            title_fs=28,
            subtitle_fs=18,
            legend_fs=26,
            legend_outside=False,
            legend_loc="upper right",
            legend_bbox=(0.98, 0.98),
            tight_layout_rect=(0.0, 0.02, 0.98, 0.98),
            inside_legend_top_margin_scale=0.82,
            edge_bar_padding=0.70,
            subtitle_y=-0.0075,
            scenario_fs=20,
            scenario_label_rotation=0,
            scenario_spacing=0.06,
            bottom_margin_for_labels=0.14,
            label_offset_scale=0.03,
            label_separation_scale=0.08,
            label_x_nudge_scale=0.09,
            scenario_label_y_scale=0.02,
        )
        plt.show()

        saved_path, save_log = _capture_stdout(
            _save_chart_figure,
            fig,
            batch_folder.name,
            variant["slug"],
            save_enabled=save_individual_chart_images,
            output_root=saved_chart_root,
            dpi=saved_chart_dpi,
        )
        plt.close(fig)

        chart_results_by_folder.setdefault(batch_folder.name, {})[variant["slug"] + "_metrics"] = variant_metrics
        chart_results_by_folder[batch_folder.name][variant["slug"] + "_saved_path"] = saved_path

        if save_log.strip():
            chart_1_logs.append(save_log.rstrip())

    chart_1_metrics_by_folder[batch_folder.name] = ch_1_auto_metrics
    chart_results_by_folder.setdefault(batch_folder.name, {})["chart_1_metrics"] = ch_1_auto_metrics

    chart_1_logs.append(f"Chart 1 folder: {batch_folder.name}")
    chart_1_logs.append(f"  scenarios: {_ch1_selected_scenarios}")
    chart_1_logs.append(f"  variables: {_ch1_selected_variables}")
    if build_log.strip():
        chart_1_logs.append(build_log.rstrip())
    chart_1_logs.append("")

print("\nFinished chart 1 (a/b/c) for all configured folders.")
print("\n".join(line for line in chart_1_logs if line is not None))
'''

set_src(cell22, new_chart_code)
changes += 1
print("  [cell 22] replaced with 3-variant chart loop")

# ── write back ───────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), "utf-8")
print(f"\nDone — {changes} patches applied to {NB.name}")
