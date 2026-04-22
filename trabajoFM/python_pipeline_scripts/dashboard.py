from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union, Sequence, Tuple, Any
from datetime import datetime, date
from pathlib import Path
import html

import numpy as np
import pandas as pd
import re

import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go

from .stats import (
    build_fit_diagnostics,
    compute_stats_for_view,
    export_stats_to_json,
    format_stats_text,
)
from .dashboard_helper import (
    _ensure_dt_index,
    _make_freq_string,
    _slice_time,
    _filter_season,
    _resample_series,
    _detect_value_col,
    _normalize_meas_map_for_var,
    _measured_options_for_category,
    _chem_options_with_placeholder,
    _aggregate_measured,
    _period_day_counts,
    normalize_measured_mdl_by_name,
    UNMATCHED_ANALYTE_MDL_MG_L,
    apply_measured_half_mdl_replacements,
    build_selected_measured_nonnum_audit,
    convert_measured_mgL_to_kg_per_day,
)
DASHBOARD_VERSION = "2025-09-11-chem-ui-3"

_DEFAULT_MEASURED_STATION_BY_REACH: Dict[int, str] = {
    13: "30304",
}

_AUTO_MEASURED_DEFAULTS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "KJELDAHL_OUTkg": {
        1: {
            "preferred_chemicals": ["NITROGENO KJELDAHL", "Nitrógeno Kjeldahl", "Nitrógeno KJELDAHL"],
            "enabled": True,
            "preferred_station_by_reach": {13: "30304"},
        },
        2: {
            "preferred_chemicals": ["NITORGENO TOTAL", "NITROGENO TOTAL", "Nitrógeno total", "Nitrógeno Total"],
            "enabled": False,
            "preferred_station_by_reach": {13: "30304"},
        },
    },
    "TOT_Nkg": {
        1: {
            "preferred_chemicals": ["NITROGENO KJELDAHL", "Nitrógeno Kjeldahl", "Nitrógeno KJELDAHL"],
            "enabled": True,
            "preferred_station_by_reach": {13: "30304"},
        },
        2: {
            "preferred_chemicals": ["NITRATOS", "Nitratos", "NITRATO", "Nitrato"],
            "enabled": True,
            "preferred_station_by_reach": {13: "30304"},
        },
        3: {
            "preferred_chemicals": ["NITORGENO TOTAL", "NITROGENO TOTAL", "Nitrógeno total", "Nitrógeno Total"],
            "enabled": False,
            "preferred_station": None,
        },
    },
    "TOT_Pkg": {
        1: {
            "preferred_chemicals": ["FOSFORO TOTAL", "Fósforo total", "FOSFORO total"],
            "enabled": True,
            "preferred_station_by_reach": {13: "30304"},
        },
        2: {
            "preferred_chemicals": ["FOSFATOS", "Fosfatos", "FOSFATO", "Fosfato"],
            "enabled": False,
            "preferred_station": None,
        },
        3: {
            "preferred_chemicals": ["SOLIDOS EN SUSPENSION", "Sólidos en suspensión"],
            "enabled": False,
            "preferred_station": None,
        },
    },
}


def _resolve_auto_measured_preferred_station(spec: Dict[str, Any], reach: Optional[int]) -> Optional[str]:
    station_by_reach = spec.get("preferred_station_by_reach")
    if isinstance(station_by_reach, dict):
        try:
            normalized_station_by_reach = {
                int(raw_reach): str(station)
                for raw_reach, station in station_by_reach.items()
                if station is not None
            }
        except Exception:
            normalized_station_by_reach = {}
        if reach is not None and int(reach) in normalized_station_by_reach:
            return normalized_station_by_reach[int(reach)]
        return None

    preferred_station = spec.get("preferred_station")
    if preferred_station is None:
        return None

    if reach is None:
        return str(preferred_station)

    default_station = _DEFAULT_MEASURED_STATION_BY_REACH.get(int(reach))
    if default_station is None:
        return None
    return str(preferred_station) if str(preferred_station) == str(default_station) else None


def _get_auto_measured_defaults_for_variable(
    variable: Optional[str],
    *,
    reach: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    specs = _AUTO_MEASURED_DEFAULTS.get(str(variable), {})
    return {
        int(cat): {
            **spec,
            "preferred_chemicals": list(spec.get("preferred_chemicals", [])),
            "preferred_station": _resolve_auto_measured_preferred_station(spec, reach),
        }
        for cat, spec in specs.items()
    }


def _pick_preferred_measured_option(
    options: Sequence[str],
    preferred_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    if not options:
        return None
    if preferred_candidates:
        normalized = {str(option).strip().casefold(): option for option in options}
        for candidate in preferred_candidates:
            match = normalized.get(str(candidate).strip().casefold())
            if match is not None:
                return match
    return options[0]

# moved helpers to dashboard_helper.py

# -----------------------------
# Dashboard with measured overlay
def _empty_measured_nonnum_audit(
    *,
    policy: str,
    mdl_mg_L: float,
    mdl_mg_L_by_name: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "policy": str(policy),
        "default_mdl_mg_L": float(mdl_mg_L),
        "mdl_mg_L_by_name": dict(mdl_mg_L_by_name or {}) or None,
        "unmatched_name_default_mdl_mg_L": (
            float(UNMATCHED_ANALYTE_MDL_MG_L) if mdl_mg_L_by_name else float(mdl_mg_L)
        ),
        "selection_scope": "all_measured_rows",
        "replaced_rows": 0,
        "by_analyte_station": [],
    }


def _build_measured_nonnum_audit(
    *,
    measured_df: Optional[pd.DataFrame],
    measured_selection: Optional[Dict[str, Dict[str, Any]]],
    policy: str,
    mdl_mg_L: float,
    mdl_mg_L_by_name: Optional[Dict[str, float]],
    measured_name_col: str,
    measured_station_col: str,
) -> Dict[str, Any]:
    audit = _empty_measured_nonnum_audit(
        policy=policy,
        mdl_mg_L=mdl_mg_L,
        mdl_mg_L_by_name=mdl_mg_L_by_name,
    )
    if str(policy) != "half_MDL" or not isinstance(measured_df, pd.DataFrame) or measured_df.empty:
        return audit
    audit.update(
        build_selected_measured_nonnum_audit(
            measured_df,
            measured_selection=measured_selection,
            sample_name_col=measured_name_col,
            sample_station_col=measured_station_col,
        )
    )
    return audit


def _summarize_measured_nonnum_assignments(
    audit: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(audit, dict) or str(audit.get("policy")) != "half_MDL":
        return None, None

    grouped: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for entry in audit.get("by_analyte_station") or []:
        chemical = entry.get("chemical")
        if chemical is None:
            continue
        try:
            mdl_value = float(entry.get("mdl_mg_L"))
            replacement_value = float(entry.get("replacement_value_mg_L"))
        except (TypeError, ValueError):
            continue
        key = (str(chemical), mdl_value, replacement_value)
        bucket = grouped.setdefault(key, {"count": 0, "stations": set()})
        try:
            bucket["count"] += int(entry.get("count") or 0)
        except (TypeError, ValueError):
            pass
        station = entry.get("station")
        if station is not None and str(station).strip():
            bucket["stations"].add(str(station))

    lines: List[str] = []
    for (chemical, mdl_value, replacement_value), bucket in sorted(grouped.items(), key=lambda item: item[0]):
        count = int(bucket.get("count") or 0)
        row_word = "row" if count == 1 else "rows"
        stations = sorted(bucket.get("stations") or [])
        station_suffix = f"; stations: {', '.join(stations)}" if stations else ""
        lines.append(
            f"{chemical}: MDL {mdl_value:g} mg/L -> half MDL {replacement_value:g} mg/L "
            f"({count} replaced {row_word}{station_suffix})"
        )

    if not lines:
        lines.append("No half-MDL replacements were needed in the current view.")

    if audit.get("mdl_mg_L_by_name"):
        try:
            fallback_mdl = float(audit.get("unmatched_name_default_mdl_mg_L", UNMATCHED_ANALYTE_MDL_MG_L))
        except (TypeError, ValueError):
            fallback_mdl = float(UNMATCHED_ANALYTE_MDL_MG_L)
        lines.append(f"Names without an analyte-specific override use fallback MDL {fallback_mdl:g} mg/L.")

    text = "Measured half-MDL assignments:\n" + "\n".join(f"- {line}" for line in lines)
    html_lines = "<br>".join(html.escape(line) for line in lines)
    html_text = (
        "<div style='margin-top:6px'>"
        "<b>Assigned half-MDL values</b><br>"
        f"{html_lines}"
        "</div>"
    )
    return text, html_text


def _print_measured_nonnum_assignments(
    audit: Optional[Dict[str, Any]],
    *,
    previous_text: Optional[str] = None,
    prefix: str = "[half_MDL]",
) -> Optional[str]:
    text, _ = _summarize_measured_nonnum_assignments(audit)
    if text and text != previous_text:
        for line in text.splitlines():
            print(f"{prefix} {line}")
    return text


def fan_compare_simulations_dashboard(
    sim_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    reach: Optional[int] = None,
    freq_options: Iterable[str] = ("D", "W", "M", "A"),
    max_bin_size: int = 12,
    start: Optional[Union[str, datetime, date]] = None,
    end: Optional[Union[str, datetime, date]] = None,
    how_map_defaults: Optional[Dict[str, str]] = None,
    reach_col: str = "RCH",
    date_col: str = "date",
    flow_col: str = "FLOW_OUTcms",
    template: str = "plotly_white",
    figure_width: Optional[int] = 1200,
    figure_height: int = 650,
    # Optional nested figure layout overrides for the main chart and the
    # lower duration / "vs flow percentile" chart.
    dashboard_layout: Optional[Dict[str, Any]] = None,
    # Optional independent overlays (each plotted as its own line)
    extra_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    # Measured overlay (optional)
    measured_df: Optional[pd.DataFrame] = None,
    measured_var_map: Optional[Dict[str, object]] = None,
    measured_date_col: str = "F_MUESTREO",
    measured_station_col: str = "est_estaci",
    measured_name_col: str = "NOMBRE",
    measured_value_col: Optional[str] = None,
    measured_kg_col_name: str = "kg_per_day",
    # Water flow overlay from independent dataframe (optional)
    water_flow_df: Optional[pd.DataFrame] = None,
    water_flow_date_col: str = "date",
    water_flow_value_col: Optional[str] = None,
    # Diversion overlay from CSV or dataframe (optional; plotted below zero on flow axis)
    diversion_df: Optional[Union[pd.DataFrame, str, Path]] = None,
    diversion_date_col: Optional[str] = None,
    diversion_value_col: Optional[str] = None,
    # Measured cleaning policies (defaults; also controllable via UI dropdowns)
    measured_nonnum_policy_default: str = "as_na",  # "as_na" | "drop"
    measured_negative_policy_default: str = "zero",  # "keep" | "drop" | "zero"
    # Method Detection Limit (mg/L) – halved when "half_MDL" policy is selected
    mdl_mg_L: float = 0.2,
    mdl_mg_L_by_name: Optional[Dict[str, float]] = None,
    # Optional styling for observations whose values were assigned via the
    # half-MDL policy. Styling is opt-in to preserve current visuals.
    style_half_mdl_observations: bool = False,
    half_mdl_observation_style: Optional[Dict[str, Any]] = None,
    # Optional UI default selections/toggles
    ui_defaults: Optional[Dict[str, Any]] = None,
    # Optional erosion overlay toggle default
    erosion_on_default: Optional[bool] = None,
    # Event-day background illustration (purely visual, no stats impact)
    show_event_bg: Optional[bool] = None,
    event_bg_color: str = "#fdd0a2",
    nonevent_bg_color: str = "#c6dbef",
    # Trace z-order: list of group names bottom-to-top
    trace_order: Optional[List[str]] = None,
    # Band color and opacity (hex color, base alpha 0-1)
    band_color: str = "#1f77b4",
    band_alpha: float = 0.28,
    # Optional export directory for JSON stats bundles
    stats_export_dir: Optional[Union[str, Path]] = None,
    # Debug: print pipeline info for filtering/resampling
    debug: bool = False,
):
    """
    Enhanced dashboard: adds selection and overlay of measured chemical points.

    Also supports `extra_dfs`: a dict of name -> DataFrame to plot as
    independent lines (not included in the fan quantiles). Each DataFrame is
    expected to share the same schema as `sim_dfs` (at least `reach_col`,
    `date_col`, and target variable columns). These traces follow the same
    frequency/method/filters as the fan chart.

    # (flow_strat debug line removed)
    measured_var_map expected formats per SWAT variable key:
        - dict with keys 1/2/3 (or '1'/'2'/'3') -> list of NOMBRE strings
        - list/tuple of up to three lists of NOMBRE strings

    Layout overrides:
        - pass `dashboard_layout={...}` directly, or
        - pass the same structure via `ui_defaults["dashboard_layout"]`

    Expected layout sections:
        - main_chart: top time-series figure size and margins
        - duration_chart: lower duration / vs-flow-percentile figure size and margins
    """
    normalized_mdl_mg_L_by_name = normalize_measured_mdl_by_name(
        ui_defaults.get("mdl_mg_L_by_name") if isinstance(ui_defaults, dict) else None
    )
    normalized_mdl_mg_L_by_name.update(normalize_measured_mdl_by_name(mdl_mg_L_by_name))
    half_mdl_label = f"Non-numeric handling: set to half MDL ({mdl_mg_L * 0.5:g})"
    if normalized_mdl_mg_L_by_name:
        half_mdl_label = (
            "Non-numeric handling: set to half MDL "
            "(analyte-specific overrides active; unmatched fallback MDL "
            f"{UNMATCHED_ANALYTE_MDL_MG_L:g} mg/L -> {UNMATCHED_ANALYTE_MDL_MG_L * 0.5:g} mg/L)"
        )

    # Structured figure layout overrides for the main time-series chart and the
    # lower duration / "vs flow percentile" chart. Users can pass the nested
    # dict either through the dedicated `dashboard_layout` argument or via
    # `ui_defaults["dashboard_layout"]`.
    def _merge_nested_dicts(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(overrides, dict):
            return base
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _merge_nested_dicts(base[key], value)
            else:
                base[key] = value
        return base

    def _coerce_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    def _coerce_optional_float(value: Any, default: Optional[float]) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _coerce_float(value: Any, default: float) -> float:
        coerced = _coerce_optional_float(value, default)
        return default if coerced is None else float(coerced)

    def _normalize_margin(value: Any) -> Dict[str, int]:
        normalized: Dict[str, int] = {}
        if not isinstance(value, dict):
            return normalized
        key_map = {
            "left": "l",
            "right": "r",
            "top": "t",
            "bottom": "b",
            "l": "l",
            "r": "r",
            "t": "t",
            "b": "b",
        }
        for raw_key, raw_value in value.items():
            key = key_map.get(str(raw_key).lower())
            if key is None or raw_value is None:
                continue
            try:
                normalized[key] = int(raw_value)
            except Exception:
                continue
        return normalized

    def _normalize_font(
        value: Any,
        *,
        default_size: Optional[int] = None,
        default_color: Optional[str] = None,
        default_family: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        return {
            "size": _coerce_optional_int(cfg.get("size", cfg.get("font_size")), default_size),
            "color": cfg.get("color", default_color),
            "family": cfg.get("family", default_family),
        }

    def _normalize_title_style(
        value: Any,
        *,
        default_x: Optional[float] = None,
        default_xanchor: Optional[str] = None,
        default_font_size: Optional[int] = None,
        default_font_color: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        title_x = _coerce_optional_float(cfg.get("x"), default_x)
        if title_x is not None and not (0.0 <= title_x <= 1.0):
            title_x = default_x
        return {
            "x": title_x,
            "xanchor": cfg.get("xanchor", default_xanchor),
            "font": _normalize_font(cfg.get("font"), default_size=default_font_size, default_color=default_font_color),
        }

    def _normalize_axis_style(
        value: Any,
        *,
        default_title_font_size: Optional[int] = None,
        default_tick_font_size: Optional[int] = None,
        default_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        return {
            "type": cfg.get("type", default_type),
            "title_font": _normalize_font(cfg.get("title_font"), default_size=_coerce_optional_int(cfg.get("title_font_size"), default_title_font_size)),
            "tick_font": _normalize_font(cfg.get("tick_font"), default_size=_coerce_optional_int(cfg.get("tick_font_size"), default_tick_font_size)),
            "tickangle": _coerce_optional_int(cfg.get("tickangle"), None),
            "automargin": cfg.get("automargin"),
            "showgrid": cfg.get("showgrid"),
            "zeroline": cfg.get("zeroline"),
            "title_standoff": _coerce_optional_int(cfg.get("title_standoff"), None),
        }

    def _normalize_hoverlabel_style(
        value: Any,
        *,
        default_namelength: Optional[int] = None,
        default_align: Optional[str] = None,
        default_font_size: Optional[int] = None,
        default_bgcolor: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        return {
            "namelength": _coerce_optional_int(cfg.get("namelength"), default_namelength),
            "align": cfg.get("align", default_align),
            "font_size": _coerce_optional_int(cfg.get("font_size"), default_font_size),
            "bgcolor": cfg.get("bgcolor", default_bgcolor),
        }

    def _normalize_legend_style(
        value: Any,
        *,
        default_orientation: Optional[str] = None,
        default_x: Optional[float] = None,
        default_y: Optional[float] = None,
    ) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        return {
            "orientation": cfg.get("orientation", default_orientation),
            "x": _coerce_optional_float(cfg.get("x"), default_x),
            "y": _coerce_optional_float(cfg.get("y"), default_y),
            "font": _normalize_font(cfg.get("font"), default_size=_coerce_optional_int(cfg.get("font_size"), None)),
        }

    def _normalize_rangeslider_style(value: Any) -> Dict[str, Any]:
        cfg = value if isinstance(value, dict) else {}
        return {
            "thickness": _coerce_optional_float(cfg.get("thickness"), 0.08),
            "bgcolor": cfg.get("bgcolor", "#f6f6f6"),
            "bordercolor": cfg.get("bordercolor", "#ddd"),
            "borderwidth": _coerce_optional_int(cfg.get("borderwidth"), 1),
        }

    def _compact_dict(value: Any) -> Any:
        if isinstance(value, dict):
            compacted: Dict[str, Any] = {}
            for key, raw_val in value.items():
                new_val = _compact_dict(raw_val)
                if new_val is None:
                    continue
                if isinstance(new_val, dict) and not new_val:
                    continue
                if isinstance(new_val, list) and not new_val:
                    continue
                compacted[key] = new_val
            return compacted
        if isinstance(value, list):
            return [item for item in (_compact_dict(v) for v in value) if item is not None]
        return value

    def _build_font_layout(font_style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        style = font_style if isinstance(font_style, dict) else {}
        return _compact_dict(
            {
                "size": style.get("size"),
                "color": style.get("color"),
                "family": style.get("family"),
            }
        )

    def _build_title_layout(title_style: Optional[Dict[str, Any]], *, text: str) -> Dict[str, Any]:
        style = title_style if isinstance(title_style, dict) else {}
        title_cfg: Dict[str, Any] = {"text": text}
        if style.get("x") is not None:
            title_cfg["x"] = style.get("x")
        if style.get("xanchor") is not None:
            title_cfg["xanchor"] = style.get("xanchor")
        font_cfg = _build_font_layout(style.get("font"))
        if font_cfg:
            title_cfg["font"] = font_cfg
        return title_cfg

    def _build_axis_layout(
        axis_style: Optional[Dict[str, Any]],
        *,
        title_text: Optional[str] = None,
        base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        style = axis_style if isinstance(axis_style, dict) else {}
        axis_cfg: Dict[str, Any] = dict(base or {})
        if style.get("type") is not None and axis_cfg.get("type") is None:
            axis_cfg["type"] = style.get("type")
        if title_text is not None:
            title_cfg: Dict[str, Any] = {"text": title_text}
            title_font = _build_font_layout(style.get("title_font"))
            if title_font:
                title_cfg["font"] = title_font
            if style.get("title_standoff") is not None:
                title_cfg["standoff"] = style.get("title_standoff")
            axis_cfg["title"] = title_cfg
        tick_font = _build_font_layout(style.get("tick_font"))
        if tick_font:
            axis_cfg["tickfont"] = tick_font
        for key in ("tickangle", "automargin", "showgrid", "zeroline"):
            if style.get(key) is not None:
                axis_cfg[key] = style.get(key)
        return axis_cfg

    def _build_hoverlabel_layout(style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = style if isinstance(style, dict) else {}
        return _compact_dict(
            {
                "namelength": cfg.get("namelength"),
                "align": cfg.get("align"),
                "font_size": cfg.get("font_size"),
                "bgcolor": cfg.get("bgcolor"),
            }
        )

    def _build_legend_layout(style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = style if isinstance(style, dict) else {}
        legend_cfg = _compact_dict(
            {
                "orientation": cfg.get("orientation"),
                "x": cfg.get("x"),
                "y": cfg.get("y"),
                "font": _build_font_layout(cfg.get("font")),
            }
        )
        return legend_cfg if isinstance(legend_cfg, dict) else {}

    def _build_rangeslider_layout(style: Optional[Dict[str, Any]], *, visible: bool) -> Dict[str, Any]:
        cfg = style if isinstance(style, dict) else {}
        return _compact_dict(
            {
                "visible": visible,
                "thickness": cfg.get("thickness"),
                "bgcolor": cfg.get("bgcolor"),
                "bordercolor": cfg.get("bordercolor"),
                "borderwidth": cfg.get("borderwidth"),
            }
        )

    def _apply_figure_size(fig_obj: Union[go.Figure, go.FigureWidget], chart_layout: Dict[str, Any]) -> None:
        update_kwargs: Dict[str, Any] = {}
        if chart_layout.get("width") is not None:
            update_kwargs["width"] = int(chart_layout["width"])
        if chart_layout.get("height") is not None:
            update_kwargs["height"] = int(chart_layout["height"])
        if update_kwargs:
            fig_obj.update_layout(**update_kwargs)

    def _resolve_dashboard_layout() -> Dict[str, Any]:
        resolved: Dict[str, Any] = {
            # Top time-series figure shown in the dashboard body.
            "main_chart": {
                "width": figure_width,
                "height": figure_height,
                "margin": {"l": 60, "r": 20, "t": 50},
                "bottom_margin": {
                    "with_range_slider": 150,
                    "without_range_slider": 110,
                },
                "title_annotation": {
                    "x": 0.5,
                    "y_with_range_slider": -0.22,
                    "y_without_range_slider": -0.16,
                    "xref": "paper",
                    "yref": "paper",
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"size": 22, "color": "black", "family": None},
                },
                "hovermode": "x unified",
                "hoverlabel": {
                    "namelength": -1,
                    "align": "left",
                    "font_size": 12,
                    "bgcolor": "white",
                },
                "legend": {
                    "orientation": "h",
                    "x": 0.0,
                    "y": 1.02,
                    "font": {"size": None, "color": None, "family": None},
                },
                "xaxis": {
                    "type": "date",
                    "title_font": {"size": None, "color": None, "family": None},
                    "tick_font": {"size": None, "color": None, "family": None},
                    "tickangle": None,
                    "automargin": None,
                    "showgrid": None,
                    "zeroline": None,
                    "title_standoff": None,
                    "rangeslider": {
                        "thickness": 0.08,
                        "bgcolor": "#f6f6f6",
                        "bordercolor": "#ddd",
                        "borderwidth": 1,
                    },
                },
                "yaxis": {
                    "title_font": {"size": None, "color": None, "family": None},
                    "tick_font": {"size": None, "color": None, "family": None},
                    "tickangle": None,
                    "automargin": None,
                    "showgrid": None,
                    "zeroline": None,
                    "title_standoff": None,
                },
            },
            # Lower duration / "vs flow percentile" figures shown beneath the main chart.
            "duration_chart": {
                "width": None,
                "height": None,
                "hovermode": None,
                "hoverlabel": {
                    "namelength": None,
                    "align": None,
                    "font_size": None,
                    "bgcolor": None,
                },
                "legend": {
                    "orientation": None,
                    "x": None,
                    "y": None,
                    "font": {"size": None, "color": None, "family": None},
                },
                "load_duration": {
                    "margin": {"r": 40},
                    "title": {
                        "x": 0.35,
                        "xanchor": "center",
                        "font": {"size": 22, "color": None, "family": None},
                    },
                    "xaxis": {
                        "title_font": {"size": 19, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                    "yaxis": {
                        "title_font": {"size": 21, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                },
                "flow_duration": {
                    "margin": {},
                    "title": {
                        "x": None,
                        "xanchor": None,
                        "font": {"size": None, "color": None, "family": None},
                    },
                    "xaxis": {
                        "title_font": {"size": None, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                    "yaxis": {
                        "title_font": {"size": None, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                },
                "flow_stratified": {
                    "margin": {},
                    "title": {
                        "x": None,
                        "xanchor": None,
                        "font": {"size": None, "color": None, "family": None},
                    },
                    "xaxis": {
                        "title_font": {"size": None, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                    "yaxis": {
                        "title_font": {"size": None, "color": None, "family": None},
                        "tick_font": {"size": None, "color": None, "family": None},
                        "tickangle": None,
                        "automargin": None,
                        "showgrid": None,
                        "zeroline": None,
                        "title_standoff": None,
                    },
                },
            },
        }

        if isinstance(ui_defaults, dict) and "ldc_title_x" in ui_defaults:
            resolved["duration_chart"]["load_duration"]["title"]["x"] = ui_defaults.get("ldc_title_x")
        if isinstance(ui_defaults, dict):
            _merge_nested_dicts(resolved, ui_defaults.get("dashboard_layout"))
            _merge_nested_dicts(resolved, ui_defaults.get("layout_config"))
        _merge_nested_dicts(resolved, dashboard_layout)

        main_chart = resolved.get("main_chart", {}) if isinstance(resolved.get("main_chart"), dict) else {}
        duration_chart = resolved.get("duration_chart", {}) if isinstance(resolved.get("duration_chart"), dict) else {}

        main_margin = {"l": 60, "r": 20, "t": 50}
        raw_main_margin = _normalize_margin(main_chart.get("margin") or main_chart.get("margins"))
        legacy_main_bottom = raw_main_margin.pop("b", None)
        main_margin.update(raw_main_margin)

        bottom_block = main_chart.get("bottom_margin", {}) if isinstance(main_chart.get("bottom_margin"), dict) else {}
        title_block = main_chart.get("title_annotation", {}) if isinstance(main_chart.get("title_annotation"), dict) else {}
        main_xaxis_block = main_chart.get("xaxis", {}) if isinstance(main_chart.get("xaxis"), dict) else {}
        main_yaxis_block = main_chart.get("yaxis", {}) if isinstance(main_chart.get("yaxis"), dict) else {}

        resolved["main_chart"] = {
            "width": _coerce_optional_int(main_chart.get("width", main_chart.get("figure_width")), figure_width),
            "height": _coerce_optional_int(main_chart.get("height", main_chart.get("figure_height")), int(figure_height)),
            "margin": main_margin,
            "bottom_margin": {
                "with_range_slider": _coerce_optional_int(
                    bottom_block.get("with_range_slider", bottom_block.get("with_rangeslider", legacy_main_bottom)),
                    150,
                ),
                "without_range_slider": _coerce_optional_int(
                    bottom_block.get("without_range_slider", bottom_block.get("without_rangeslider", legacy_main_bottom)),
                    110,
                ),
            },
            "title_annotation": {
                "x": _coerce_float(title_block.get("x"), 0.5),
                "y_with_range_slider": _coerce_float(
                    title_block.get("y_with_range_slider", title_block.get("with_range_slider")),
                    -0.22,
                ),
                "y_without_range_slider": _coerce_float(
                    title_block.get("y_without_range_slider", title_block.get("without_range_slider")),
                    -0.16,
                ),
                "xref": title_block.get("xref", "paper"),
                "yref": title_block.get("yref", "paper"),
                "xanchor": title_block.get("xanchor", "center"),
                "yanchor": title_block.get("yanchor", "top"),
                "font": _normalize_font(title_block.get("font"), default_size=_coerce_optional_int(title_block.get("font_size"), 22), default_color=title_block.get("font_color", "black")),
            },
            "hovermode": str(main_chart.get("hovermode", "x unified")),
            "hoverlabel": _normalize_hoverlabel_style(
                main_chart.get("hoverlabel"),
                default_namelength=-1,
                default_align="left",
                default_font_size=12,
                default_bgcolor="white",
            ),
            "legend": _normalize_legend_style(
                main_chart.get("legend"),
                default_orientation="h",
                default_x=0.0,
                default_y=1.02,
            ),
            "xaxis": {
                **_normalize_axis_style(main_xaxis_block, default_type="date"),
                "rangeslider": _normalize_rangeslider_style(main_xaxis_block.get("rangeslider")),
            },
            "yaxis": _normalize_axis_style(main_yaxis_block),
        }

        legacy_duration_title_x = duration_chart.get("title_x")
        load_duration_block = duration_chart.get("load_duration", {}) if isinstance(duration_chart.get("load_duration"), dict) else {}
        flow_duration_block = duration_chart.get("flow_duration", {}) if isinstance(duration_chart.get("flow_duration"), dict) else {}
        flow_stratified_block = duration_chart.get("flow_stratified", {}) if isinstance(duration_chart.get("flow_stratified"), dict) else {}

        def _resolve_variant_margin(default_margin: Dict[str, int], parent_block: Dict[str, Any], variant_block: Dict[str, Any]) -> Dict[str, int]:
            merged = dict(default_margin)
            merged.update(_normalize_margin(parent_block.get("margin") or parent_block.get("margins")))
            merged.update(_normalize_margin(variant_block.get("margin") or variant_block.get("margins")))
            return merged

        load_duration_default_x = _coerce_optional_float(legacy_duration_title_x, 0.35)
        if load_duration_default_x is None:
            load_duration_default_x = 0.35

        resolved["duration_chart"] = {
            "width": _coerce_optional_int(duration_chart.get("width", duration_chart.get("figure_width")), None),
            "height": _coerce_optional_int(duration_chart.get("height", duration_chart.get("figure_height")), None),
            "hovermode": duration_chart.get("hovermode"),
            "hoverlabel": _normalize_hoverlabel_style(duration_chart.get("hoverlabel")),
            "legend": _normalize_legend_style(duration_chart.get("legend")),
            "load_duration": {
                "margin": _resolve_variant_margin({"r": 40}, duration_chart, load_duration_block),
                "title": _normalize_title_style(
                    load_duration_block.get("title"),
                    default_x=load_duration_default_x,
                    default_xanchor="center",
                    default_font_size=22,
                ),
                "xaxis": _normalize_axis_style(load_duration_block.get("xaxis"), default_title_font_size=19),
                "yaxis": _normalize_axis_style(load_duration_block.get("yaxis"), default_title_font_size=21),
            },
            "flow_duration": {
                "margin": _resolve_variant_margin({}, {}, flow_duration_block),
                "title": _normalize_title_style(flow_duration_block.get("title")),
                "xaxis": _normalize_axis_style(flow_duration_block.get("xaxis")),
                "yaxis": _normalize_axis_style(flow_duration_block.get("yaxis")),
            },
            "flow_stratified": {
                "margin": _resolve_variant_margin({}, {}, flow_stratified_block),
                "title": _normalize_title_style(flow_stratified_block.get("title")),
                "xaxis": _normalize_axis_style(flow_stratified_block.get("xaxis")),
                "yaxis": _normalize_axis_style(flow_stratified_block.get("yaxis")),
            },
        }
        return resolved

    resolved_dashboard_layout = _resolve_dashboard_layout()
    main_chart_layout = resolved_dashboard_layout["main_chart"]
    duration_chart_layout = resolved_dashboard_layout["duration_chart"]

    def _build_duration_chart_layout_update(
        variant_key: str,
        *,
        title_text: str,
        xaxis_title_text: str,
        yaxis_title_text: str,
        base_xaxis: Optional[Dict[str, Any]] = None,
        base_yaxis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        variant_style = duration_chart_layout.get(variant_key, {}) if isinstance(duration_chart_layout.get(variant_key), dict) else {}
        layout_update: Dict[str, Any] = {
            "title": _build_title_layout(variant_style.get("title"), text=title_text),
            "xaxis": _build_axis_layout(variant_style.get("xaxis"), title_text=xaxis_title_text, base=base_xaxis),
            "yaxis": _build_axis_layout(variant_style.get("yaxis"), title_text=yaxis_title_text, base=base_yaxis),
            "margin": dict(variant_style.get("margin", {})),
        }
        if duration_chart_layout.get("hovermode") is not None:
            layout_update["hovermode"] = duration_chart_layout.get("hovermode")
        hoverlabel_layout = _build_hoverlabel_layout(duration_chart_layout.get("hoverlabel"))
        if hoverlabel_layout:
            layout_update["hoverlabel"] = hoverlabel_layout
        legend_layout = _build_legend_layout(duration_chart_layout.get("legend"))
        if legend_layout:
            layout_update["legend"] = legend_layout
        return layout_update

    def _resolve_half_mdl_observation_style() -> Dict[str, Any]:
        resolved: Dict[str, Any] = {
            "enabled": bool(style_half_mdl_observations),
            "main_chart": {
                "name_suffix": "half-MDL",
                "marker": {
                    "symbol": "diamond-open",
                    "size": 13,
                    "color": None,
                    "opacity": 1.0,
                    "line": {"color": "#111", "width": 1.6},
                },
            },
            "duration_chart": {
                "affected": {
                    "name": "Measured (half-MDL)",
                    "marker": {
                        "symbol": "circle-open",
                        "size": 9,
                        "color": None,
                        "opacity": 1.0,
                        "line": {"color": "#111", "width": 1.4},
                    },
                },
                "unaffected": {
                    "name_paired": "Measured",
                    "name_order_stats": "Measured (all)",
                },
            },
        }
        if isinstance(ui_defaults, dict):
            if "style_half_mdl_observations" in ui_defaults:
                resolved["enabled"] = bool(ui_defaults.get("style_half_mdl_observations"))
            _merge_nested_dicts(resolved, ui_defaults.get("half_mdl_observation_style"))
        _merge_nested_dicts(resolved, half_mdl_observation_style)

        def _normalize_marker_style(
            raw: Any,
            *,
            default_symbol: str,
            default_size: int,
            default_line_color: str,
            default_line_width: float,
        ) -> Dict[str, Any]:
            cfg = raw if isinstance(raw, dict) else {}
            line_cfg = cfg.get("line") if isinstance(cfg.get("line"), dict) else {}
            return {
                "symbol": cfg.get("symbol", default_symbol),
                "size": _coerce_optional_int(cfg.get("size"), default_size),
                "color": cfg.get("color"),
                "opacity": _coerce_optional_float(cfg.get("opacity"), 1.0),
                "line": {
                    "color": line_cfg.get("color", default_line_color),
                    "width": _coerce_optional_float(line_cfg.get("width"), default_line_width),
                },
            }

        resolved["enabled"] = bool(resolved.get("enabled"))
        main_cfg = resolved.get("main_chart") if isinstance(resolved.get("main_chart"), dict) else {}
        duration_cfg = resolved.get("duration_chart") if isinstance(resolved.get("duration_chart"), dict) else {}
        affected_cfg = duration_cfg.get("affected") if isinstance(duration_cfg.get("affected"), dict) else {}
        unaffected_cfg = duration_cfg.get("unaffected") if isinstance(duration_cfg.get("unaffected"), dict) else {}
        resolved["main_chart"] = {
            "name_suffix": str(main_cfg.get("name_suffix") or "half-MDL"),
            "marker": _normalize_marker_style(
                main_cfg.get("marker"),
                default_symbol="diamond-open",
                default_size=13,
                default_line_color="#111",
                default_line_width=1.6,
            ),
        }
        resolved["duration_chart"] = {
            "affected": {
                "name": str(affected_cfg.get("name") or "Measured (half-MDL)"),
                "marker": _normalize_marker_style(
                    affected_cfg.get("marker"),
                    default_symbol="circle-open",
                    default_size=9,
                    default_line_color="#111",
                    default_line_width=1.4,
                ),
            },
            "unaffected": {
                "name_paired": str(unaffected_cfg.get("name_paired") or "Measured"),
                "name_order_stats": str(unaffected_cfg.get("name_order_stats") or "Measured (all)"),
            },
        }
        return resolved

    resolved_half_mdl_observation_style = _resolve_half_mdl_observation_style()

    def _build_marker_with_overrides(base_marker: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        marker = dict(base_marker or {})
        style_cfg = overrides if isinstance(overrides, dict) else {}
        if style_cfg.get("symbol") is not None:
            marker["symbol"] = style_cfg.get("symbol")
        if style_cfg.get("size") is not None:
            marker["size"] = style_cfg.get("size")
        if style_cfg.get("color") is not None:
            marker["color"] = style_cfg.get("color")
        if style_cfg.get("opacity") is not None:
            marker["opacity"] = style_cfg.get("opacity")
        merged_line = dict(marker.get("line") or {})
        line_cfg = style_cfg.get("line") if isinstance(style_cfg.get("line"), dict) else {}
        if line_cfg.get("color") is not None:
            merged_line["color"] = line_cfg.get("color")
        if line_cfg.get("width") is not None:
            merged_line["width"] = line_cfg.get("width")
        if merged_line:
            marker["line"] = merged_line
        return marker

    def _coerce_external_overlay_df(
        data: Optional[Union[pd.DataFrame, str, Path]],
        *,
        label: str,
    ) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"{label} path not found: {path}")
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path, encoding="utf-8")
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path)
            raise ValueError(f"{label} must be a DataFrame, CSV, or Excel file. Got: {path}")
        raise TypeError(f"{label} must be a DataFrame, CSV path, or Excel path.")

    def _pick_best_date_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if explicit and explicit in df.columns:
            return explicit
        patterns = ["fecha", "date", "día", "dia", "day"]
        cols = list(df.columns)
        for col in cols:
            name = str(col).lower()
            if any(pat in name for pat in patterns):
                return col
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return cols[0] if cols else None

    def _pick_best_diversion_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if explicit and explicit in df.columns:
            return explicit
        cols = list(df.columns)
        candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
        if not candidates:
            return None

        def _score(col: str) -> tuple[int, int]:
            name = str(col).lower()
            score = 0
            if "diversion" in name:
                score += 100
            if "m³/day" in name or "m3/day" in name or "m3_d" in name:
                score += 10
            return (score, int(df[col].notna().sum()))

        return max(candidates, key=_score)

    def _aligned_overlay_axis_range(
        values: Union[pd.Series, np.ndarray, Sequence[float]],
        *,
        primary_range: Optional[Sequence[float]] = None,
        default_range: Tuple[float, float] = (0.0, 1.0),
        pad_scale: float = 1.05,
    ) -> List[float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            raw_min = float(np.nanmin(arr))
            raw_max = float(np.nanmax(arr))
            if raw_min == raw_max:
                raw_max = raw_min + 1.0
        else:
            raw_min, raw_max = float(default_range[0]), float(default_range[1])

        zero_frac = 0.5
        try:
            if primary_range is not None and len(primary_range) == 2:
                y0, y1 = float(primary_range[0]), float(primary_range[1])
                if y1 != y0:
                    if y0 < 0.0 < y1:
                        zero_frac = (0.0 - y0) / (y1 - y0)
                    elif 0.0 <= y0:
                        zero_frac = 0.0
                    elif 0.0 >= y1:
                        zero_frac = 1.0
        except Exception:
            pass

        pos_max = max(0.0, raw_max)
        neg_min = min(0.0, raw_min)
        scale_req_pos = (pos_max / (1.0 - zero_frac)) if (1.0 - zero_frac) > 1e-9 else (np.inf if pos_max > 0 else 0.0)
        scale_req_neg = ((-neg_min) / zero_frac) if zero_frac > 1e-9 else (np.inf if neg_min < 0 else 0.0)
        scale = max(scale_req_pos, scale_req_neg)
        if not np.isfinite(scale) or scale == 0.0:
            scale = max(abs(raw_min), abs(raw_max)) or 1.0
        scale *= pad_scale
        return [-zero_frac * scale, (1.0 - zero_frac) * scale]

    diversion_source_df = _coerce_external_overlay_df(diversion_df, label="diversion_df")
    if how_map_defaults is None:
        how_map_defaults = {}

    # Discover reaches and count runs
    all_reaches = set()
    number_of_simulations = 0
    for df in sim_dfs.values():
        if reach_col in df.columns:
            all_reaches.update(df[reach_col].dropna().unique().tolist())
        number_of_simulations += 1
    reach_choices = sorted(int(r) for r in all_reaches if pd.notna(r))
    if not reach_choices:
        raise ValueError("No reaches found.")
    # Prefer reach 13 by default if present; otherwise fall back to first available
    preferred_default_reach = 13
    if reach is None:
        reach = preferred_default_reach if preferred_default_reach in reach_choices else reach_choices[0]
    else:
        try:
            reach = int(reach)
        except Exception:
            reach = preferred_default_reach if preferred_default_reach in reach_choices else reach_choices[0]
        if reach not in reach_choices:
            reach = preferred_default_reach if preferred_default_reach in reach_choices else reach_choices[0]
    # Extract run number from first sim key (e.g., run000091_real000364_1 -> run 91)
    _first_key = next(iter(sim_dfs.keys()), None)
    _run_label: Optional[str] = None
    if _first_key is not None:
        try:
            m = re.search(r"run(\d+)", str(_first_key), flags=re.IGNORECASE)
            if m:
                _run_label = f"run {int(m.group(1))}"
        except Exception:
            _run_label = None

    # Ensure season_months is defined before any debug/log usage; it can be overridden by ui_defaults below.
    season_months: Optional[Iterable[int]] = None

    # Debug helper
    def _dbg(*args, **kwargs):
        if debug or (isinstance(ui_defaults, dict) and bool(ui_defaults.get("debug", False))):
            try:
                print("[dash]", *args, **kwargs)
            except Exception:
                pass

    def _dbg_df_info(df: Optional[pd.DataFrame], label: str, *, date_col: Optional[str] = None):
        if not (debug or (isinstance(ui_defaults, dict) and bool(ui_defaults.get("debug", False)))):
            return
        try:
            if df is None:
                print(f"[dash] {label}: None")
                return
            if isinstance(df, pd.Series):
                idx = df.index
                n = int(df.shape[0])
                print(f"[dash] {label}: n={n}, idx=[{idx.min()}..{idx.max()}]")
                return
            n = int(df.shape[0])
            rng = None
            if date_col and (date_col in df.columns):
                dt = pd.to_datetime(df[date_col])
                rng = (dt.min(), dt.max())
            print(f"[dash] {label}: n={n}" + (f", {date_col}=[{rng[0]}..{rng[1]}]" if rng else ""))
        except Exception:
            pass

    _dbg("Init dashboard", dict(reach=reach, start=start, end=end, season_months=season_months))

    # Allow overriding time window from ui_defaults
    if isinstance(ui_defaults, dict):
        if ui_defaults.get("start") is not None:
            start = ui_defaults.get("start")
        if ui_defaults.get("end") is not None:
            end = ui_defaults.get("end")
        if ui_defaults.get("season_months") is not None:
            season_months = ui_defaults.get("season_months")

    if stats_export_dir is None:
        stats_export_dir = Path(__file__).resolve().parent.parent / "config" / "outputs" / "dashboard_stats"
    else:
        stats_export_dir = Path(stats_export_dir)

    # Widgets for simulations
    num_sim = widgets.HTML(value=(
        f"Number of initializations: {number_of_simulations} - {_run_label}"
        if _run_label else f"Number of initializations: {number_of_simulations}"
    ))
    # Derived variable: KJELDAHL (ORGN_OUTkg + NH4_OUTkg)
    SYN_VAR = "KJELDAHL_OUTkg"
    DERIVED_COMPONENTS = ("ORGN_OUTkg", "NH4_OUTkg")
    def _has_components(df: pd.DataFrame) -> bool:
        try:
            return all(c in df.columns for c in DERIVED_COMPONENTS)
        except Exception:
            return False
    derived_available = any(_has_components(df) for df in sim_dfs.values())
    variables_with_combo = list(variables)
    if derived_available and (SYN_VAR not in variables_with_combo):
        variables_with_combo.append(SYN_VAR)
    _dbg("variables", dict(n=len(variables_with_combo), has_KJELDAHL=(SYN_VAR in variables_with_combo)))
    dd_var = widgets.Dropdown(options=variables_with_combo, value=variables_with_combo[0], description="Variable:", layout=widgets.Layout(width="360px"))
    dd_reach = widgets.Dropdown(options=reach_choices, value=reach, description="Reach:", layout=widgets.Layout(width="180px"))
    dd_freq = widgets.Dropdown(options=list(freq_options), value="D", description="Freq:", layout=widgets.Layout(width="140px"))
    sl_bin = widgets.IntSlider(value=1, min=1, max=max_bin_size, step=1, description="Bin:", continuous_update=False, layout=widgets.Layout(width="300px"))
    dd_method = widgets.Dropdown(options=["sum", "mean", "flow_weighted_mean"], value="mean", description="Method:", layout=widgets.Layout(width="280px"))
    # New: choose which flow source to use for unit conversions (external vs SWAT avg)
    flow_source_options = [("Use external flow", "external"), ("Use SWAT avg flow", "swat_avg")]
    dd_flow_source = widgets.Dropdown(options=flow_source_options, value=("swat_avg" if (isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty) else "external"), description="Units via:", layout=widgets.Layout(width="280px"))
    # Info label for conversion statement
    lbl_units = widgets.HTML(value="")
    # Units/mode toggle: Load (kg/day) vs Concentration (mg/L)
    tg_units = widgets.ToggleButtons(
        options=[("Load (kg/day)", "load"), ("Concentration (mg/L)", "conc")],
        value="load",
        description="Compare:",
        layout=widgets.Layout(width="360px"),
        style=dict(button_width="160px"),
    )
    cb_autoscale_y_live = widgets.Checkbox(value=True, description="Auto-scale Y on zoom")
    cb_range_slider = widgets.Checkbox(value=True, description="Show range slider")
    cb_show_names_in_tooltip = widgets.Checkbox(value=False, description="Names in tooltip")
    cb_show_ensemble = widgets.Checkbox(value=True, description="Show input scenarios")
    # Event-day background illustration (purely visual, no stats impact)
    cb_show_event_bg = widgets.Checkbox(value=(show_event_bg if show_event_bg is not None else False),
                                        description="Show event background")
    cp_event_color = widgets.ColorPicker(value=event_bg_color, description="High-flow:", concise=True,
                                         layout=widgets.Layout(width="140px"))
    cp_nonevent_color = widgets.ColorPicker(value=nonevent_bg_color, description="Baseflow:", concise=True,
                                            layout=widgets.Layout(width="140px"))

    # Measured controls
    measured_present = measured_df is not None and isinstance(measured_df, pd.DataFrame) and not measured_df.empty
    # Detect potential measured columns for load and concentration
    measured_load_col: Optional[str] = None
    measured_conc_col: Optional[str] = None
    if measured_present:
        # Prefer explicit parameter if provided
        if measured_value_col and measured_value_col in measured_df.columns:
            # If provided column looks like a concentration label, keep as conc; else as load
            if str(measured_value_col).strip().lower() in {"resultado", "result", "concentracion", "concentración", "concentration", "mg/l", "mg_l"}:
                measured_conc_col = measured_value_col
            else:
                measured_load_col = measured_value_col
        # Try to auto-detect known columns
        if measured_load_col is None and "kg_per_day" in measured_df.columns:
            measured_load_col = "kg_per_day"
        # Common Spanish lab export names for concentration
        for cand in ["RESULTADO", "Resultado", "CONCENTRACION", "concentracion", "CONCENTRACIÓN", "concentración"]:
            if measured_conc_col is None and cand in measured_df.columns:
                measured_conc_col = cand
                break
        # Fallback to prior heuristic if still missing both
        if measured_load_col is None and measured_conc_col is None:
            auto = _detect_value_col(measured_df)
            measured_load_col = auto
        # If nothing at all was detected, raise to help caller
        if measured_load_col is None and measured_conc_col is None:
            raise ValueError("Unable to detect measured value column. Please pass measured_value_col.")

    # Water flow overlay config
    flow_meas_col: Optional[str] = None
    diversion_meas_col: Optional[str] = None
    diversion_source_date_col: Optional[str] = None

    # Per-category: enable, chem-name dropdown, station selector
    cat_symbols = {1: "star", 2: "circle", 3: "square"}
    cat_labels = {1: "Map 1", 2: "Map 2", 3: "Map 3"}

    cb_meas_on = widgets.Checkbox(value=measured_present, description="Show measured")
    # Default ON when a water_flow_df is provided; we will auto-pick the column later
    cb_flow_on = widgets.Checkbox(value=(isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty), description="Show Measured water flow (m3/d)")
    cb_diversion_on = widgets.Checkbox(
        value=(isinstance(diversion_source_df, pd.DataFrame) and not diversion_source_df.empty),
        description="Show diversion below 0 (m3/d)",
    )
    # SWAT avg flow availability and toggle (FLOW_OUT * 86400)
    def _has_swat_flow(_df: pd.DataFrame) -> bool:
        if not isinstance(_df, pd.DataFrame):
            return False
        try:
            cols = list(map(str, getattr(_df, 'columns', [])))
        except Exception:
            return False
        # Quick checks for common names
        if ("FLOW_OUTcmscms" in cols) or ("FLOW_OUTcms" in cols) or ("FLOW_OUT" in cols):
            return True
        low = [c.lower() for c in cols]
        return any(c.startswith("flow_out") for c in low)
    swat_flow_available = any(_has_swat_flow(df) for df in sim_dfs.values())
    cb_swat_flow_on = widgets.Checkbox(value=swat_flow_available, description="Show SWAT avg flow (m3/d)")
    # Erosion availability across simulations and toggle (supports both *tons and legacy names)
    def _has_erosion_cols(_df: pd.DataFrame) -> bool:
        try:
            cols = set(map(str, getattr(_df, 'columns', [])))
        except Exception:
            return False
        return ({"SED_INtons", "SED_OUTtons"} <= cols) or ({"SED_IN", "SED_OUT"} <= cols)
    erosion_available = any(isinstance(df, pd.DataFrame) and _has_erosion_cols(df) for df in sim_dfs.values())
    cb_erosion_on = widgets.Checkbox(value=(erosion_on_default if erosion_on_default is not None else erosion_available), description="Show erosion (SED_IN - SED_OUT)")
    # -----------------------------
    # Configurable event detection controls (replace legacy 'outliers')
    # -----------------------------
    # User chooses a flow source for event detection (can differ from unit conversion source)
    dd_event_source = widgets.Dropdown(
        options=[("Events via external flow", "external"), ("Events via SWAT avg flow", "swat_avg")],
        value=("external" if (isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty) else "swat_avg"),
        description="Events via:",
        layout=widgets.Layout(width="280px"),
    )
    # Threshold method: common percentile tokens or absolute
    dd_event_threshold = widgets.Dropdown(
        options=[
            ("p95 (>= 95th %)", "p95"),
            ("p90 (>= 90th %)", "p90"),
            ("p75 (>= 75th %)", "p75"),
            ("p60 (>= 60th %)", "p60"),
            ("p50 (>= 50th %)", "p50"),
            ("Absolute…", "abs"),
        ],
        value="p95",
        description="Threshold:",
        layout=widgets.Layout(width="220px"),
    )
    tb_event_abs = widgets.BoundedFloatText(value=np.nan, min=0.0, max=1e12, step=1.0, description="Abs value:", layout=widgets.Layout(width="220px"))
    # Minimum event length (days)
    fl_event_min_days = widgets.BoundedFloatText(value=1.0, min=0.0, max=30.0, step=0.5, description="Min days:", layout=widgets.Layout(width="180px"))
    # Buffer (days) exclude +/- around detected event days
    sl_event_buffer_days = widgets.IntSlider(value=1, min=0, max=7, step=1, description="Buffer (days)", continuous_update=False, layout=widgets.Layout(width="280px"))
    # Event view toggle
    tg_event_view = widgets.ToggleButtons(
        options=[
            ("All days", "all"),
            ("Non-event days", "non_events"),
            ("Event days", "events"),
        ],
        value="all",
        description="View:",
        layout=widgets.Layout(width="360px"),
        style={"button_width": "120px"},
    )
    # Dynamic help label
    lbl_events_help = widgets.HTML(value="<i>Events: days where flow >= threshold and lasting >= Min days. Use the view toggle to switch between event and non-event periods.</i>")
    def _toggle_abs_vis(change=None):
        show = (dd_event_threshold.value == "abs")
        tb_event_abs.layout.display = "block" if show else "none"
    _toggle_abs_vis()
    dd_event_threshold.observe(_toggle_abs_vis, names="value")
    # Apply ui_defaults overrides if provided
    try:
        if isinstance(ui_defaults, dict):
            if ui_defaults.get("event_source") in {"external", "swat_avg"}:
                dd_event_source.value = ui_defaults.get("event_source")
            if isinstance(ui_defaults.get("event_min_days"), (int, float)):
                v = float(ui_defaults.get("event_min_days"))
                if 0.0 <= v <= 30.0:
                    fl_event_min_days.value = v
            if ui_defaults.get("event_threshold") in {"p95", "p90", "p75", "p60", "p50", "abs"}:
                dd_event_threshold.value = ui_defaults.get("event_threshold")
            if isinstance(ui_defaults.get("event_abs_value"), (int, float)):
                av = float(ui_defaults.get("event_abs_value"))
                if av >= 0:
                    tb_event_abs.value = av
            if isinstance(ui_defaults.get("event_buffer_days"), (int, float)):
                b = int(ui_defaults.get("event_buffer_days"))
                if 0 <= b <= 7:
                    sl_event_buffer_days.value = b
            if ui_defaults.get("event_view") in {"all", "events", "non_events"}:
                tg_event_view.value = ui_defaults.get("event_view")
            elif isinstance(ui_defaults.get("exclude_events"), bool):
                tg_event_view.value = "non_events" if ui_defaults.get("exclude_events") else "all"
    except Exception:
        pass
    # Cleaning policy dropdowns for measured data
    dd_meas_nonnum = widgets.Dropdown(
        options=[
            ("Non-numeric handling: keep as NA (recommended)", "as_na"),
            ("Non-numeric handling: set to 0", "zero"),
            ("Non-numeric handling: drop rows", "drop"),
            (half_mdl_label, "half_MDL"),
        ],
        value=(measured_nonnum_policy_default if measured_nonnum_policy_default in {"as_na", "drop", "zero"} else "as_na"),
        description="Non-numeric:",
        layout=widgets.Layout(width="420px"),
    )
    dd_meas_negative = widgets.Dropdown(
        options=[
            ("Negative mg/L: set to 0", "zero"),
            ("Negative mg/L: drop rows", "drop"),
            ("Negative mg/L: keep as-is", "keep"),
        ],
        value=(measured_negative_policy_default if measured_negative_policy_default in {"zero", "drop", "keep"} else "zero"),
        description="Negatives:",
        layout=widgets.Layout(width="420px"),
    )
    # Deviation highlighting controls
    cb_flag_dev = widgets.Checkbox(value=True, description="Flag deviations")
    # Factor-of difference vs p90 (order of magnitude), default 10x
    sl_dev_factor = widgets.FloatSlider(value=10.0, min=2.0, max=100.0, step=0.5, description="Factor:", continuous_update=False, layout=widgets.Layout(width="360px"))
    # If independent water_flow_df is provided, prefer it over measured_df
    if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
        # determine column if not defined yet
        if not flow_meas_col:
            if water_flow_value_col and water_flow_value_col in water_flow_df.columns:
                flow_meas_col = water_flow_value_col
            else:
                for _c in water_flow_df.columns:
                    if "water_flow_m3_d" in str(_c).lower():
                        flow_meas_col = _c
                        break
                if not flow_meas_col:
                    for _c in water_flow_df.columns:
                        if pd.api.types.is_numeric_dtype(water_flow_df[_c]):
                            flow_meas_col = _c
                            break
        if flow_meas_col:
            cb_flow_on.value = True
            try:
                print("Detected water flow series in water_flow_df - will also map water flow.")
            except Exception:
                pass
    if isinstance(diversion_source_df, pd.DataFrame) and not diversion_source_df.empty:
        diversion_source_date_col = _pick_best_date_col(diversion_source_df, explicit=diversion_date_col)
        diversion_meas_col = _pick_best_diversion_col(diversion_source_df, explicit=diversion_value_col)
        if diversion_source_date_col is None or diversion_meas_col is None:
            raise ValueError("Unable to detect diversion date/value columns. Please pass diversion_date_col and diversion_value_col.")
        cb_diversion_on.value = True
        try:
            print(
                f"Detected diversion series in diversion_df: date='{diversion_source_date_col}', "
                f"value='{diversion_meas_col}' - will plot below zero."
            )
        except Exception:
            pass
    # Extra overlays toggles
    extra_present = isinstance(extra_dfs, dict) and bool(extra_dfs)
    cb_extra: Dict[str, widgets.Checkbox] = {}
    if extra_present:
        for _name in extra_dfs.keys():
            cb_extra[str(_name)] = widgets.Checkbox(value=True, description=f"Show: {str(_name)}")


    cb_cat = {
        1: widgets.Checkbox(value=True, description=f"{cat_labels[1]}"),
        2: widgets.Checkbox(value=False, description=f"{cat_labels[2]}"),
        3: widgets.Checkbox(value=False, description=f"{cat_labels[3]}"),
    }
    dd_cat_name: Dict[int, widgets.Dropdown] = {i: widgets.Dropdown(options=[], value=None, description="Chem:", layout=widgets.Layout(width="280px")) for i in (1, 2, 3)}
    ms_cat_stations: Dict[int, widgets.SelectMultiple] = {i: widgets.SelectMultiple(options=[], value=(), description="Stations:", layout=widgets.Layout(width="280px", height="120px")) for i in (1, 2, 3)}
    cat_vbox: Dict[int, widgets.VBox] = {}

    # Default method per selected var
    def _default_method_for_var(v: str) -> str:
        if v in how_map_defaults:
            return how_map_defaults[v]
        if "Conc" in v or "mg/L" in v:
            return "mean"
        if any(u in v.lower() for u in ["kg", "tons", "mg"]):
            return "sum"
        return "mean"

    dd_method.value = _default_method_for_var(dd_var.value)

    def _update_method_options_for_mode():
        # In concentration mode, allow mean and flow_weighted_mean (mean is default)
        if tg_units.value == "conc":
            allowed = ["flow_weighted_mean", "mean"]
            current = dd_method.value
            dd_method.options = allowed
            if current not in allowed:
                dd_method.value = "mean"
        else:
            allowed = ["sum", "mean", "flow_weighted_mean"]
            current = dd_method.value
            dd_method.options = allowed
            if current not in allowed:
                dd_method.value = _default_method_for_var(dd_var.value)
    # Initialize method options according to default unit mode
    _update_method_options_for_mode()

    out = widgets.Output()
    # Reload/apply changes button and stale-indicator row
    btn_reload = widgets.Button(icon='refresh', tooltip='Apply changes', button_style='warning', layout=widgets.Layout(width='120px'))
    lbl_reload = widgets.HTML("")
    reload_bar = widgets.HBox([lbl_reload, btn_reload])
    try:
        reload_bar.layout.justify_content = 'flex-end'
        reload_bar.layout.align_items = 'center'
        reload_bar.layout.display = 'none'  # hidden until there are unapplied changes
        lbl_reload.layout.margin = '0 8px 0 0'
    except Exception:
        pass
    # Stats view (HTML) and diagnostics panel; shown side-by-side
    stats_html = widgets.HTML(value="")
    btn_save_stats = widgets.Button(
        description="Save stats",
        icon="save",
        tooltip=f"Export current stats and dashboard state to JSON\nDirectory: {stats_export_dir}",
        layout=widgets.Layout(width="120px"),
        disabled=True,
    )
    lbl_save_stats = widgets.HTML("")
    stats_panel = widgets.VBox([
        widgets.HBox([btn_save_stats, lbl_save_stats]),
        stats_html,
    ])
    duration_box = widgets.HBox([], layout=widgets.Layout(width="100%", flex_wrap="wrap", justify_content="flex-start"))
    diag_box = widgets.VBox([])
    # Styling and initial layout
    try:
        stats_panel.layout.width = "40%"
        stats_panel.layout.min_width = "300px"
        stats_html.layout.padding = "8px"
        stats_html.layout.overflow = "auto"
        stats_html.layout.border = "1px solid #ddd"
        diag_box.layout.width = "60%"
        diag_box.layout.padding = "8px"
        diag_box.layout.overflow = "auto"
        diag_box.layout.border = "1px solid #ddd"
        # Hidden by default until diagnostics are requested
        diag_box.layout.display = "none"
    except Exception:
        pass
    cb_show_diags = widgets.Checkbox(value=True, description="Show diagnostics")
    # New checkbox: when enabled, add sediment net (SED_IN - SED_OUT) overlay to load duration curve diagnostics
    cb_ldc_sediment = widgets.Checkbox(value=False, description="LDC sediment overlay")
    cb_flow_strat = widgets.Checkbox(value=False, description="Flow-strat curve")
    ms_flow_regimes = widgets.SelectMultiple(
        options=["event", "non-event"],
        value=("event", "non-event"),
        description="Regimes:",
        layout=widgets.Layout(width="170px", height="60px")
    )
    cb_flow_total_band = widgets.Checkbox(value=False, description="Total band")
    cb_flow_total_only = widgets.Checkbox(value=False, description="Total only")
    cb_flow_overlay = widgets.Checkbox(value=True, description="Overlay")
    # New: total aggregation semantics dropdown (median collapse vs full envelope)
    dd_flow_total_mode = widgets.Dropdown(
        options=[
            ("Median collapse", "median"),
            ("Min/Max envelope", "extents"),  # min(min) / max(max) semantics
        ],
        value="median",
        description="Total mode:",
        layout=widgets.Layout(width="190px")
    )
    # Stats behavior controls
    dd_lag_metric = widgets.Dropdown(options=["r", "NSE"], value="r", description="Lag by:", layout=widgets.Layout(width="140px"))
    sl_max_lag = widgets.IntSlider(value=2, min=0, max=5, step=1, description="Lag±:", continuous_update=False, layout=widgets.Layout(width="220px"))
    sel_local_K = widgets.SelectMultiple(options=[1, 2], value=(1, 2), description="Local K:", layout=widgets.Layout(width="160px", height="60px"))
    cb_log_metrics = widgets.Checkbox(value=True, description="Log metrics")
    # Precompute a fast lookup for stations per chemical name to keep UI responsive
    _meas_chem2stations: Dict[str, List[str]] = {}
    if measured_present:
        try:
            df_ = measured_df[[measured_name_col, measured_station_col]].dropna()
            df_[measured_name_col] = df_[measured_name_col].astype(str)
            df_[measured_station_col] = df_[measured_station_col].astype(str)
            _meas_chem2stations = (
                df_.groupby(measured_name_col)[measured_station_col]
                   .apply(lambda s: sorted(pd.Index(s).unique().tolist()))
                   .to_dict()
            )
        except Exception:
            _meas_chem2stations = {}

    _last = {
        "aligned_df": None,
        "y_fixed": None,
        "fig": None,
        "q_df": None,
        "meas_series": [],
        "meas_series_half_mdl_flags": [],
        "flow_series": None,  # aggregated external water flow series for current settings (m3/day)
        "diversion_series": None,  # diversion overlay plotted below zero on the flow axis
        "measured_nonnum_audit": None,
        "last_measured_nonnum_summary": None,
        "swat_flow_series": None,  # mean across runs of SWAT FLOW_OUT * 86400 (m3/day)
        "flow_y_range": None, # last y2 range
        "extra_series": {},   # name -> pd.Series for extra overlays (current settings)
        "erosion_series": None,  # mean across runs of (SED_IN - SED_OUT)
        "event_context": None,
        "flow_strat_bundle": None,
        "flow_regime_visible": {"event", "non-event"},
        "flow_total_band": False,
        "flow_total_only": False,
        "flow_overlay": True,
        "flow_total_mode": "median",  # 'median' | 'extents'
        "latest_stats_export_payload": None,
        "latest_stats_export_path": None,
        "trace_order": None,
    }
    _TRACE_ORDER_DEFAULT = ("bands", "extra", "measured_flow", "swat_flow", "erosion", "central", "measured")
    _state = {"updating": False, "duration_refresher": None, "measured_defaults_var": None,
              "_gen": 0, "_gen_lock": __import__('threading').Lock(),
              "_xrange_timer": None, "_xrange_timer_lock": __import__('threading').Lock()}

    TICK_STOPS = [
        dict(dtickrange=[None, 1000 * 60 * 60 * 24], value="%Y-%m-%d\n%H:%M"),
        dict(dtickrange=[1000 * 60 * 60 * 24, 1000 * 60 * 60 * 24 * 28], value="%Y-%m-%d"),
        dict(dtickrange=[1000 * 60 * 60 * 24 * 28, 1000 * 60 * 60 * 24 * 365], value="%Y-%m"),
        dict(dtickrange=[1000 * 60 * 60 * 24 * 365, None], value="%Y"),
    ]

    # Robust selector for a water-flow column in water_flow_df
    def _pick_best_flow_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        # If caller provided a valid explicit column, honor it
        if explicit and explicit in df.columns:
            return explicit
        # Ranking of name patterns (highest priority first)
        patterns = [
            "water_flow_m3_d", "flow_m3_d", "flow", "caudal", "q_m3", "q", "m3_d", "m3/day", "cms", "m3s",
        ]
        cols = list(df.columns)
        # Prefer non-boolean numeric cols
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
        # Add non-numeric columns as fallback (we will coerce later)
        candidates = num_cols + [c for c in cols if c not in num_cols and c != 'outliers']
        # Score by name match and non-null count
        def _score(col: str) -> tuple[int, int]:
            name = str(col).lower()
            name_score = 0
            for i, pat in enumerate(patterns[::-1]):  # later entries lower weight
                if pat in name:
                    name_score = i + 1
                    break
            nonnull = int(df[col].notna().sum())
            return (name_score, nonnull)
        if not candidates:
            return None
        best = max(candidates, key=_score)
        return best

    # Robust selector for SWAT flow column in simulation DataFrames
    def _pick_best_swat_flow_col(df: pd.DataFrame) -> Optional[str]:
        try:
            cols = list(map(str, getattr(df, 'columns', [])))
        except Exception:
            return None
        priority = [
            "FLOW_OUTcmscms",  # sometimes duplicated suffix
            "FLOW_OUTcms",
            "FLOW_OUT",
        ]
        for name in priority:
            if name in cols:
                return name
        # Fallback: first column that matches pattern FLOW_OUT (case-insensitive)
        low = {c.lower(): c for c in cols}
        for key, orig in low.items():
            if key.startswith("flow_out"):
                return orig
        return None

    def _refresh_measured_controls(*_):
        if not measured_present:
            return
        if _state["updating"]:
            return
        _state["updating"] = True
        var = dd_var.value
        auto_defaults = _get_auto_measured_defaults_for_variable(var, reach=dd_reach.value)
        norm_map = _normalize_meas_map_for_var(measured_var_map or {}, var)
        for cat, spec in auto_defaults.items():
            if not norm_map.get(cat) and spec.get("preferred_chemicals"):
                norm_map[cat] = list(spec["preferred_chemicals"])
        variable_changed = (_state.get("measured_defaults_var") != var)
        for i in (1, 2, 3):
            allowed = norm_map.get(i, [])
            default_spec = auto_defaults.get(i, {})
            opts = _measured_options_for_category(measured_df, measured_name_col, allowed)
            target_chem = _pick_preferred_measured_option(opts, default_spec.get("preferred_chemicals"))
            # Update chem dropdown robustly avoiding invalid intermediate states
            old_val = dd_cat_name[i].value
            if old_val is not None and old_val not in opts:
                if opts:
                    # transitional include old value, then finalize
                    dd_cat_name[i].options = _chem_options_with_placeholder(opts, extra=old_val)
                    dd_cat_name[i].value = target_chem
                    dd_cat_name[i].options = _chem_options_with_placeholder(opts)
                else:
                    # no options -> placeholder only
                    dd_cat_name[i].options = _chem_options_with_placeholder([])
                    dd_cat_name[i].value = None
            else:
                dd_cat_name[i].options = _chem_options_with_placeholder(opts)
                if target_chem is not None and (variable_changed or dd_cat_name[i].value is None):
                    dd_cat_name[i].value = target_chem
            # Populate station options for this category (based on selected chem)
            if dd_cat_name[i].value is not None:
                sts = _meas_chem2stations.get(str(dd_cat_name[i].value), [])
            else:
                sts = []
            old_st = list(ms_cat_stations[i].value)
            if any(s not in sts for s in old_st):
                # transitional: include old selections to avoid validation error when setting options
                transitional = list(dict.fromkeys(old_st + sts))
                ms_cat_stations[i].options = transitional
                valid_sel = tuple([s for s in old_st if s in sts])
                ms_cat_stations[i].value = valid_sel
                ms_cat_stations[i].options = sts
            else:
                ms_cat_stations[i].options = sts
            # auto-select default station by map if none selected
            preferred_station = default_spec.get("preferred_station")
            if variable_changed and opts:
                if preferred_station and (preferred_station in sts):
                    ms_cat_stations[i].value = (preferred_station,)
                elif ("enabled" in default_spec) and not bool(default_spec.get("enabled")):
                    ms_cat_stations[i].value = ()
                elif not ms_cat_stations[i].value:
                    ms_cat_stations[i].value = tuple(sts)
            elif not ms_cat_stations[i].value:
                if preferred_station and (preferred_station in sts):
                    ms_cat_stations[i].value = (preferred_station,)
                else:
                    ms_cat_stations[i].value = tuple(sts)
            # Hide category box if no options
            if i in cat_vbox:
                cat_vbox[i].layout.display = None if opts else 'none'
            # Disable checkbox when no options (and uncheck)
            cb_cat[i].disabled = not bool(opts)
            if not opts:
                cb_cat[i].value = False
            elif variable_changed and ("enabled" in default_spec):
                cb_cat[i].value = bool(default_spec.get("enabled"))
        _state["measured_defaults_var"] = var
        _state["updating"] = False

    # Apply UI defaults before wiring observers for a snappier initial render
    if isinstance(ui_defaults, dict):
        # Variable first (no observers yet)
        if ui_defaults.get("variable") in variables_with_combo:
            dd_var.value = ui_defaults.get("variable")
        # Reach
        if ui_defaults.get("reach") in reach_choices:
            dd_reach.value = int(ui_defaults.get("reach"))
        # Frequency and bin
        if ui_defaults.get("freq") in list(freq_options):
            dd_freq.value = ui_defaults.get("freq")
        if isinstance(ui_defaults.get("bin"), (int, float)):
            b = int(ui_defaults.get("bin"))
            if b >= sl_bin.min and b <= sl_bin.max:
                sl_bin.value = b
        # Compare mode and method
        if ui_defaults.get("compare_mode") in ("load", "conc"):
            tg_units.value = ui_defaults.get("compare_mode")
            _update_method_options_for_mode()
        if ui_defaults.get("method") in dd_method.options:
            dd_method.value = ui_defaults.get("method")
        # Common toggles
        if isinstance(ui_defaults.get("autoscale_y_live"), bool):
            cb_autoscale_y_live.value = bool(ui_defaults.get("autoscale_y_live"))
        if isinstance(ui_defaults.get("show_names_in_tooltip"), bool):
            cb_show_names_in_tooltip.value = bool(ui_defaults.get("show_names_in_tooltip"))
        if isinstance(ui_defaults.get("range_slider"), bool):
            cb_range_slider.value = bool(ui_defaults.get("range_slider"))
        if isinstance(ui_defaults.get("show_ensemble"), bool):
            cb_show_ensemble.value = bool(ui_defaults.get("show_ensemble"))
        if isinstance(ui_defaults.get("show_event_bg"), bool):
            cb_show_event_bg.value = bool(ui_defaults.get("show_event_bg"))
        if isinstance(ui_defaults.get("trace_order"), (list, tuple)):
            trace_order = list(ui_defaults["trace_order"])
        if isinstance(ui_defaults.get("band_color"), str):
            band_color = ui_defaults["band_color"]
        if isinstance(ui_defaults.get("band_alpha"), (int, float)):
            band_alpha = float(ui_defaults["band_alpha"])
        if isinstance(ui_defaults.get("event_bg_color"), str):
            cp_event_color.value = ui_defaults["event_bg_color"]
        if isinstance(ui_defaults.get("nonevent_bg_color"), str):
            cp_nonevent_color.value = ui_defaults["nonevent_bg_color"]
        if isinstance(ui_defaults.get("show_diags"), bool):
            cb_show_diags.value = bool(ui_defaults.get("show_diags"))
        if isinstance(ui_defaults.get("flow_strat_curve"), bool):
            cb_flow_strat.value = bool(ui_defaults.get("flow_strat_curve"))
        # Flow strat additional presets
        if isinstance(ui_defaults.get("flow_total_band"), bool):
            cb_flow_total_band.value = bool(ui_defaults.get("flow_total_band"))
        if isinstance(ui_defaults.get("flow_total_only"), bool):
            cb_flow_total_only.value = bool(ui_defaults.get("flow_total_only"))
        if isinstance(ui_defaults.get("flow_overlay"), bool):
            cb_flow_overlay.value = bool(ui_defaults.get("flow_overlay"))
        # Total mode mapping: accept internal values or display labels
        _total_mode_map = {
            "median": "median",
            "median collapse": "median",
            "min/max envelope": "extents",
            "minmax envelope": "extents",
            "extents": "extents",
        }
        if isinstance(ui_defaults.get("flow_total_mode"), str):
            key = ui_defaults.get("flow_total_mode").strip().lower()
            if key in _total_mode_map:
                dd_flow_total_mode.value = _total_mode_map[key]
        # Regime visibility preset
        if isinstance(ui_defaults.get("flow_regimes"), (list, tuple, set)):
            regs = [r for r in ui_defaults.get("flow_regimes") if r in ("event","non-event")]
            if regs:
                try:
                    ms_flow_regimes.value = tuple(dict.fromkeys(regs))  # preserve order, unique
                except Exception:
                    pass
        # Stats controls
        if ui_defaults.get("lag_metric") in ("r", "NSE"):
            dd_lag_metric.value = ui_defaults.get("lag_metric")
        if isinstance(ui_defaults.get("max_lag"), (int, float)):
            ml = int(ui_defaults.get("max_lag"))
            sl_max_lag.value = max(sl_max_lag.min, min(sl_max_lag.max, ml))
        if isinstance(ui_defaults.get("local_Ks"), (list, tuple, set)):
            ks = tuple(sorted([int(k) for k in ui_defaults.get("local_Ks") if int(k) in (1, 2)]))
            if ks:
                sel_local_K.value = ks
        if isinstance(ui_defaults.get("log_metrics"), bool):
            cb_log_metrics.value = bool(ui_defaults.get("log_metrics"))
        # Erosion toggle via ui_defaults
        if isinstance(ui_defaults.get("erosion_on"), bool):
            cb_erosion_on.value = bool(ui_defaults.get("erosion_on"))
        # LDC sediment overlay toggle
        if isinstance(ui_defaults.get("cb_ldc_sediment"), bool):
            cb_ldc_sediment.value = bool(ui_defaults.get("cb_ldc_sediment"))
        # Measured toggles
        if isinstance(ui_defaults.get("measured_on"), bool):
            cb_meas_on.value = bool(ui_defaults.get("measured_on"))
        if isinstance(ui_defaults.get("flow_on"), bool):
            cb_flow_on.value = bool(ui_defaults.get("flow_on"))
        if isinstance(ui_defaults.get("diversion_on"), bool):
            cb_diversion_on.value = bool(ui_defaults.get("diversion_on"))
        if isinstance(ui_defaults.get("swat_flow_on"), bool):
            cb_swat_flow_on.value = bool(ui_defaults.get("swat_flow_on"))
        # Cleaning policies
        if ui_defaults.get("meas_nonnum_policy") in ("as_na", "drop", "zero", "half_MDL"):
            dd_meas_nonnum.value = ui_defaults.get("meas_nonnum_policy")
        if ui_defaults.get("meas_negative_policy") in ("keep", "drop", "zero"):
            dd_meas_negative.value = ui_defaults.get("meas_negative_policy")
        # Backward compatibility: map legacy outlier keys to new event system
        if ("event_view" not in (ui_defaults or {})) and isinstance(ui_defaults.get("exclude_flow_outliers"), bool):
            tg_event_view.value = "non_events" if ui_defaults.get("exclude_flow_outliers") else "all"
        if isinstance(ui_defaults.get("outlier_buffer_days"), (int, float)):
            v = int(ui_defaults.get("outlier_buffer_days"))
            if 0 <= v <= sl_event_buffer_days.max:
                sl_event_buffer_days.value = v
        # Deviation highlighting defaults
        if isinstance(ui_defaults.get("flag_deviations"), bool):
            cb_flag_dev.value = bool(ui_defaults.get("flag_deviations"))
        if isinstance(ui_defaults.get("deviation_factor"), (int, float)):
            sl_dev_factor.value = float(ui_defaults.get("deviation_factor"))
        # Populate measured controls based on current variable
        _refresh_measured_controls()
        # Category selections
        cats = ui_defaults.get("cats") or {}
        for ci in (1, 2, 3):
            params = cats.get(ci) or cats.get(str(ci)) or {}
            chem = params.get("chem")
            stations = params.get("stations")
            if chem and chem in [opt[1] for opt in (dd_cat_name[ci].options or []) if opt[1] is not None]:
                dd_cat_name[ci].value = chem
                # Refresh stations for this category using the cached map
                _refresh_stations_for_cat(ci)
            if stations and isinstance(stations, (list, tuple)):
                sts = tuple([str(s) for s in stations])
                # Only assign valid stations
                valid = [s for s in sts if s in (ms_cat_stations[ci].options or [])]
                ms_cat_stations[ci].value = tuple(valid)
            # Enable/disable category
            if isinstance(params.get("enabled"), bool):
                cb_cat[ci].value = bool(params.get("enabled"))
        # Extra overlay visibility
        extra_vis = ui_defaults.get("extra_visible") or {}
        for name, chk in (cb_extra or {}).items():
            if name in extra_vis:
                chk.value = bool(extra_vis[name])
        # Sync _last early with any preset values for flow strat so first render honors them
        try:
            _last["flow_regime_visible"] = set(ms_flow_regimes.value)
            _last["flow_total_band"] = bool(cb_flow_total_band.value)
            _last["flow_total_only"] = bool(cb_flow_total_only.value)
            _last["flow_overlay"] = bool(cb_flow_overlay.value)
            _last["flow_total_mode"] = dd_flow_total_mode.value
        except Exception:
            pass

    def _refresh_stations_for_cat(cat: int, *_):
        if not measured_present:
            return
        chem = dd_cat_name[cat].value
        if chem is None:
            ms_cat_stations[cat].options = []
            ms_cat_stations[cat].value = ()
            return
        sts = _meas_chem2stations.get(str(chem), [])
        old_st = list(ms_cat_stations[cat].value)
        if any(s not in sts for s in old_st):
            transitional = list(dict.fromkeys(old_st + sts))
            ms_cat_stations[cat].options = transitional
            valid_sel = tuple([s for s in old_st if s in sts])
            ms_cat_stations[cat].value = valid_sel
            ms_cat_stations[cat].options = sts
        else:
            ms_cat_stations[cat].options = sts
        if not ms_cat_stations[cat].value:
            default_spec = _get_auto_measured_defaults_for_variable(dd_var.value, reach=dd_reach.value).get(cat, {})
            preferred_station = default_spec.get("preferred_station")
            if preferred_station and (preferred_station in sts):
                ms_cat_stations[cat].value = (preferred_station,)
            else:
                ms_cat_stations[cat].value = tuple(sts)

    def _hovertemplate(show_name: bool) -> str:
        return ("%{fullData.name}: %{y:.4g}<extra></extra>" if show_name else "%{y:.4g}<extra></extra>")

    def _sanitize_filename_part(value: object, *, max_len: int = 32) -> str:
        text = "unknown" if value is None else str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if not text:
            text = "unknown"
        return text[:max_len]

    def _selected_measured_state() -> Dict[str, Dict[str, Any]]:
        selected: Dict[str, Dict[str, Any]] = {}
        if not measured_present:
            return selected
        for cat in (1, 2, 3):
            selected[str(cat)] = {
                "enabled": bool(cb_cat[cat].value),
                "chemical": dd_cat_name[cat].value,
                "stations": list(ms_cat_stations[cat].value or ()),
            }
        return selected

    def _collect_dashboard_state(view_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> Dict[str, Any]:
        x0 = None
        x1 = None
        if view_window is not None:
            x0, x1 = view_window
        return {
            "variable": dd_var.value,
            "reach": dd_reach.value,
            "frequency": dd_freq.value,
            "frequency_string": _make_freq_string(dd_freq.value, sl_bin.value),
            "bin": sl_bin.value,
            "method": dd_method.value,
            "compare_mode": tg_units.value,
            "flow_source": dd_flow_source.value,
            "event_source": dd_event_source.value,
            "event_threshold": dd_event_threshold.value,
            "event_abs_value": tb_event_abs.value,
            "event_min_days": fl_event_min_days.value,
            "event_buffer_days": sl_event_buffer_days.value,
            "event_view": tg_event_view.value,
            "autoscale_y_live": cb_autoscale_y_live.value,
            "range_slider": cb_range_slider.value,
            "show_names_in_tooltip": cb_show_names_in_tooltip.value,
            "show_ensemble": cb_show_ensemble.value,
            "show_event_bg": cb_show_event_bg.value,
            "event_bg_color": cp_event_color.value,
            "nonevent_bg_color": cp_nonevent_color.value,
            "trace_order": _last.get("trace_order", list(_TRACE_ORDER_DEFAULT)),
            "band_color": band_color,
            "band_alpha": band_alpha,
            "show_diagnostics": cb_show_diags.value,
            "show_measured": cb_meas_on.value,
            "show_water_flow": cb_flow_on.value,
            "show_diversion": cb_diversion_on.value,
            "show_swat_flow": cb_swat_flow_on.value,
            "show_erosion": cb_erosion_on.value,
            "show_flow_strat": cb_flow_strat.value,
            "flow_regimes": list(ms_flow_regimes.value),
            "flow_total_band": cb_flow_total_band.value,
            "flow_total_only": cb_flow_total_only.value,
            "flow_overlay": cb_flow_overlay.value,
            "flow_total_mode": dd_flow_total_mode.value,
            "lag_metric": dd_lag_metric.value,
            "max_lag": sl_max_lag.value,
            "local_window_ks": list(sel_local_K.value),
            "log_metrics": cb_log_metrics.value,
            "measured_nonnum_policy": dd_meas_nonnum.value,
            "measured_negative_policy": dd_meas_negative.value,
            "mdl_mg_L": mdl_mg_L,
            "mdl_mg_L_by_name": dict(normalized_mdl_mg_L_by_name) or None,
            "style_half_mdl_observations": bool(resolved_half_mdl_observation_style.get("enabled")),
            "half_mdl_observation_style": resolved_half_mdl_observation_style,
            "flag_deviations": cb_flag_dev.value,
            "deviation_factor": sl_dev_factor.value,
            "start": start,
            "end": end,
            "season_months": list(season_months) if season_months is not None else None,
            "measured_selection": _selected_measured_state(),
            "extra_overlays": {name: bool(chk.value) for name, chk in cb_extra.items()},
            "view_window": {"x0": x0, "x1": x1},
            "source_arguments": {
                "reach_col": reach_col,
                "date_col": date_col,
                "flow_col": flow_col,
                "template": template,
                "figure_width": main_chart_layout["width"],
                "figure_height": main_chart_layout["height"],
                "dashboard_layout": resolved_dashboard_layout,
                "stats_export_dir": str(stats_export_dir),
            },
        }

    def _build_stats_filename_stem(view_window: Tuple[pd.Timestamp, pd.Timestamp]) -> str:
        x0, x1 = view_window
        window_token = "all"
        if x0 is not None and x1 is not None:
            window_token = f"{pd.Timestamp(x0).strftime('%Y%m%d')}-{pd.Timestamp(x1).strftime('%Y%m%d')}"
        parts = [
            _sanitize_filename_part(_run_label or "run-unknown", max_len=24),
            f"var-{_sanitize_filename_part(dd_var.value, max_len=20)}",
            f"reach-{_sanitize_filename_part(dd_reach.value, max_len=8)}",
            f"freq-{_sanitize_filename_part(_make_freq_string(dd_freq.value, sl_bin.value), max_len=12)}",
            f"method-{_sanitize_filename_part(dd_method.value, max_len=16)}",
            f"mode-{_sanitize_filename_part(tg_units.value, max_len=8)}",
            f"view-{_sanitize_filename_part(tg_event_view.value, max_len=12)}",
            window_token,
        ]
        return "_".join(parts)

    def _update_save_stats_tooltip(file_path: Optional[Union[str, Path]] = None) -> None:
        target = None
        if file_path is not None:
            target = Path(file_path)
        else:
            payload = _last.get("latest_stats_export_payload")
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            filename_stem = metadata.get("filename_stem")
            if filename_stem:
                safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(filename_stem)).strip("_") or "stats-export"
                target = stats_export_dir / f"{safe_stem}.json"
        if target is None:
            btn_save_stats.tooltip = (
                "Export current stats and dashboard state to JSON"
                f"\nDirectory: {stats_export_dir}"
            )
            return
        btn_save_stats.tooltip = (
            "Export current stats and dashboard state to JSON"
            f"\nPath: {target}"
        )

    def _median_hovertemplate(show_name: bool, run_label: Optional[str]) -> str:
        head = "%{fullData.name}:<br>" if show_name else ""
        run_line = (f"{run_label}<br>" if run_label else "")
        # customdata columns: [p05, p25, p50, p75, p95] (raw values; no rounding)
        return (
            head
            + run_line
            + "p05: %{customdata[0]}<br>"
            + "p25: %{customdata[1]}<br>"
            + "median: %{customdata[2]}<br>"
            + "p75: %{customdata[3]}<br>"
            + "p95: %{customdata[4]}<extra></extra>"
        )

    # Flow-stratified duration curve helper (min/mean/max by flow exceedance and regime)
    def _build_flow_stratified_curve(
        load_min: Optional[pd.Series], load_mean: Optional[pd.Series], load_max: Optional[pd.Series],
        flow_external: Optional[pd.Series], flow_swat: Optional[pd.Series], *,
        threshold_spec: Union[str, float] = "p75", bin_step: float = 0.05,
        prefer: str = "auto", template_name: str = "plotly_white"
    ) -> Optional[Tuple[go.FigureWidget, Dict[str, Any]]]:
        try:
            if not (isinstance(load_min, pd.Series) and isinstance(load_mean, pd.Series) and isinstance(load_max, pd.Series)):
                return None
            df_load = pd.DataFrame({"L_min": load_min, "L_mean": load_mean, "L_max": load_max}).dropna(how="any")
            if df_load.empty:
                return None
            candidates: Dict[str, pd.Series] = {}
            if isinstance(flow_external, pd.Series) and not flow_external.empty:
                candidates["external"] = flow_external
            if isinstance(flow_swat, pd.Series) and not flow_swat.empty:
                candidates["swat"] = flow_swat
            if prefer == "external" and "external" in candidates:
                flow_src = "external"; flow = candidates["external"]
            elif prefer == "swat" and "swat" in candidates:
                flow_src = "swat"; flow = candidates["swat"]
            else:
                if len(candidates) == 1:
                    flow_src, flow = next(iter(candidates.items()))
                elif len(candidates) == 2:
                    flow_src = "swat" if "swat" in candidates else "external"; flow = candidates[flow_src]
                else:
                    return None
            df = df_load.join(flow.rename("flow"), how="inner").dropna(how="any")
            if df.empty:
                return None
            # Enforce ordering per row
            bad = (df.L_min > df.L_mean) | (df.L_mean > df.L_max) | (df.L_min > df.L_max)
            if bad.any():
                reordered = np.sort(df.loc[bad, ["L_min","L_mean","L_max"]].to_numpy(dtype=float), axis=1)
                df.loc[bad, ["L_min","L_mean","L_max"]] = reordered
            # Threshold selection
            if isinstance(threshold_spec, str) and threshold_spec.lower().startswith("p"):
                try:
                    p = float(threshold_spec.lower().lstrip("p")) / 100.0
                except Exception:
                    p = 0.75
                p = min(max(p, 0.0), 1.0)
                thr = float(np.nanquantile(df.flow.to_numpy(dtype=float), p))
            else:
                try:
                    thr = float(threshold_spec)
                except Exception:
                    thr = float(np.nanquantile(df.flow.to_numpy(dtype=float), 0.75))
            df["regime"] = np.where(df.flow >= thr, "event", "non-event")
            df_sorted = df.sort_values("flow", ascending=False).copy()
            N = len(df_sorted)
            if N < 5:
                return None
            df_sorted["exceedance"] = (np.arange(N) + 1) / (N + 1.0)
            if bin_step > 0:
                edges = np.arange(0.0, 1.0 + 1e-9, bin_step)
                if edges[-1] < 1.0: edges = np.append(edges, 1.0)
                bins = pd.IntervalIndex.from_breaks(edges, closed="right")
                df_sorted["bin"] = pd.cut(df_sorted["exceedance"], bins)
                rows = []
                for (b, reg), g in df_sorted.groupby(["bin","regime"], observed=True):
                    if g.empty: continue
                    left = b.left if hasattr(b, 'left') else 0.0
                    right = b.right if hasattr(b, 'right') else 0.0
                    x_mid = 0.5 * (left + right)
                    rows.append({"x_mid": x_mid, "regime": reg,
                                 "y_min": float(np.nanmedian(g.L_min)),
                                 "y_mean": float(np.nanmedian(g.L_mean)),
                                 "y_max": float(np.nanmedian(g.L_max))})
                agg = pd.DataFrame(rows)
            else:
                agg = df_sorted.rename(columns={"exceedance":"x_mid","L_min":"y_min","L_mean":"y_mean","L_max":"y_max"})[
                    ["x_mid","regime","y_min","y_mean","y_max"]
                ]
            if agg.empty: return None
            fig = go.Figure(layout=dict(template=template_name))
            colors = {"event":"#d62728", "non-event":"#1f77b4", "total":"#555555"}
            visible_regimes = _last.get("flow_regime_visible", {"event","non-event"})
            total_only = bool(_last.get("flow_total_only", False))
            want_total_band = bool(_last.get("flow_total_band", False) or total_only)
            # Optional total aggregation (ignores regime)
            total_agg = None
            if want_total_band:
                mode = _last.get("flow_total_mode", "median")
                if mode == "extents":
                    # min(min)/max(max) envelope across regimes per bin; y_mean still central tendency (median of means)
                    tmp = agg.groupby("x_mid", as_index=False).agg({
                        "y_min": "min",
                        "y_mean": "median",
                        "y_max": "max",
                    })
                else:  # median collapse (previous behavior)
                    tmp = agg.groupby("x_mid", as_index=False).agg({
                        "y_min": "median",
                        "y_mean": "median",
                        "y_max": "median",
                    })
                total_agg = tmp.sort_values("x_mid")
            if not total_only:
                for reg in ["non-event","event"]:
                    if reg not in visible_regimes:
                        continue
                    sub = agg[agg.regime == reg].sort_values("x_mid")
                    if sub.empty:
                        continue
                    x_pct = sub.x_mid.to_numpy()*100.0
                    y_low = sub.y_min.to_numpy(); y_up = sub.y_max.to_numpy()
                    # Upper line (max)
                    fig.add_trace(go.Scatter(x=x_pct, y=y_up, mode="lines", name=f"{reg} max", line=dict(color=colors[reg], width=0.6), showlegend=False))
                    # Fill
                    if colors[reg].startswith('#') and len(colors[reg]) == 7:
                        r = int(colors[reg][1:3],16); g = int(colors[reg][3:5],16); b = int(colors[reg][5:7],16)
                        fill_col = f"rgba({r},{g},{b},0.20)"
                    else:
                        fill_col = 'rgba(0,0,0,0.15)'
                    fig.add_trace(go.Scatter(x=x_pct, y=y_low, mode="lines", name=f"{reg} band", line=dict(color=colors[reg], width=0.6), fill="tonexty", fillcolor=fill_col, hoverinfo="skip", showlegend=False))
                    # Mean line
                    fig.add_trace(go.Scatter(x=x_pct, y=sub.y_mean.to_numpy(), mode="lines", name=f"{reg} mean", line=dict(color=colors[reg], width=2)))
            if total_agg is not None:
                sub = total_agg
                x_pct = sub.x_mid.to_numpy()*100.0
                y_low = sub.y_min.to_numpy(); y_up = sub.y_max.to_numpy()
                # Total upper
                fig.add_trace(go.Scatter(x=x_pct, y=y_up, mode="lines", name="total max", line=dict(color=colors["total"], width=0.6), showlegend=False))
                # Total fill (lighter opacity if overlaying with regimes)
                r,g,b = 85,85,85
                overlay_flag = bool(_last.get("flow_overlay", True))
                total_only = bool(_last.get("flow_total_only", False))
                alpha = 0.10 if (overlay_flag and not total_only and not (visible_regimes == {"event","non-event"} and not want_total_band)) else 0.18
                fig.add_trace(go.Scatter(x=x_pct, y=y_low, mode="lines", name="total band", line=dict(color=colors["total"], width=0.6), fill="tonexty", fillcolor=f"rgba({r},{g},{b},{alpha})", hoverinfo="skip", showlegend=False))
                # Total mean
                fig.add_trace(go.Scatter(x=x_pct, y=sub.y_mean.to_numpy(), mode="lines", name="total mean", line=dict(color=colors["total"], width=3, dash="dot")))
            # Unified title to match event-context variant
            fig.update_layout(
                **_build_duration_chart_layout_update(
                    "flow_stratified",
                    title_text="Flow-stratified (event vs non-event)",
                    xaxis_title_text="Flow exceedance (% of time exceeded)",
                    yaxis_title_text="Load (units)",
                )
            )
            _apply_figure_size(fig, duration_chart_layout)
            return go.FigureWidget(fig), {"threshold": thr, "flow_source": flow_src, "binned": agg, "raw_count": N, "total": bool(total_agg is not None), "total_only": total_only, "visible_regimes": list(visible_regimes)}
        except Exception as _e_fsc:
            _dbg("flow_strat_curve_fail", str(_e_fsc))
            return None

    def _build_sim_duration_widgets(
        q_plot: Optional[pd.DataFrame],
        flow_series: Optional[pd.Series],
        template_name: str,
        y_axis_title: str,
        measured_flow_series: Optional[pd.Series] = None,
        daily_flow_series: Optional[pd.Series] = None,
    ) -> List[widgets.Widget]:
        """
        Build and return a list of Plotly FigureWidget objects representing simulation load and flow duration curves.

        Parameters:
            q_plot (Optional[pd.DataFrame]): DataFrame containing quantile data for the simulation.
            flow_series (Optional[pd.Series]): Series containing flow data for the simulation.
            template_name (str): Name of the Plotly template to use for the figures.
            y_axis_title (str): Title for the y-axis of the duration curve plot.

        Returns:
            List[widgets.Widget]: List of Plotly FigureWidget objects for display in the dashboard.
        """
        widgets_out: List[widgets.Widget] = []
        try:
            from .stats import _duration_curve_from_series as _dcfs  # type: ignore
        except Exception:
            _dbg("duration", "_duration_curve_from_series import failed")
            return widgets_out
        levels = np.linspace(1.0, 99.0, 99)
        if isinstance(q_plot, pd.DataFrame) and not q_plot.empty:
            fig_ldc = go.Figure(layout=dict(template=template_name))
            # Ensure min/max columns exist (fallback compute from UNFILTERED aligned_df_plot if omitted earlier)
            try:
                # IMPORTANT: We intentionally do NOT use the filtered aligned_df here because
                # the duration curve quantile lines are derived from the UNFILTERED data.
                # Using filtered min/max caused visual deviation when selecting "Non-event days".
                if ("min" not in q_plot.columns or "max" not in q_plot.columns) and isinstance(_last.get("aligned_df_plot"), pd.DataFrame):
                    base_plot: pd.DataFrame = _last.get("aligned_df_plot")  # type: ignore
                    if base_plot is not None and not base_plot.empty:
                        if "min" not in q_plot.columns:
                            q_plot["min"] = base_plot.min(axis=1, skipna=True)
                        if "max" not in q_plot.columns:
                            q_plot["max"] = base_plot.max(axis=1, skipna=True)
            except Exception:
                pass
            try:
                _dbg("duration q_plot cols", list(q_plot.columns))
            except Exception:
                pass
            added = False
            # --------------------------------------------------------------
            # Band mode toggle:
            #   order_stats (default): current behavior using order statistics (independent sorting via _dcfs)
            #   paired: preserve day-level pairing of min/max (and quantile lines) by ordering days using a reference
            #           series (median by default, mean if requested). This keeps relative relationships intact and
            #           allows adding additional aligned series later.
            # ui_defaults keys:
            #   ldc_band_mode: 'order_stats' | 'paired'
            #   ldc_band_order_ref: 'median' | 'mean' (only used if ldc_band_mode == 'paired')
            # --------------------------------------------------------------
            # Prepare placeholders for paired ordering reuse (for measurement overlay alignment)
            paired_index: Optional[pd.Index] = None
            paired_x_rank: Optional[np.ndarray] = None
            try:
                _ui_mode = (ui_defaults or {}).get("ldc_band_mode", "order_stats")  # type: ignore
                BAND_MODE = str(_ui_mode).lower()
                if BAND_MODE not in ("order_stats", "paired"):
                    BAND_MODE = "order_stats"
            except Exception:
                BAND_MODE = "order_stats"
            try:
                _order_ref_key = str((ui_defaults or {}).get("ldc_band_order_ref", "median")).lower()
                if _order_ref_key not in ("median", "mean"):
                    _order_ref_key = "median"
                BAND_ORDER_REF = _order_ref_key
            except Exception:
                BAND_ORDER_REF = "median"
            # ------------------------------------------------------------------
            # Configurable quantile/min-max lines to show on the duration curve.
            # COMMENT OUT or REORDER entries below to customize quickly.
            # If 'min' and 'max' columns are present they will also be used to
            # render a shaded band (added prior to the quantile lines).
            # ------------------------------------------------------------------
            PLOT_LINES = [
                ("max", "Simulation max", dict(color="#7f0000", width=1.2)),  # optional (band top)
                #("p95", "Simulation p95", dict(color="#d62728", width=1.1, dash="solid")),
                ("p75", "Simulation p75", dict(color="#ff7f0e", width=1.0, dash="dot")),
                ("p50", "Simulation median/mean", dict(color="#000000", width=2.4)),  # bold central line
                ("p25", "Simulation p25", dict(color="#1f77b4", width=1.0, dash="dot")),
                #("p05", "Simulation p05", dict(color="#2ca02c", width=1.1, dash="solid")),
                ("min", "Simulation min", dict(color="#00441b", width=1.2)),  # optional (band bottom)
            ]

            # ------------------------------------------------------------------
            # Optional line visibility filter via ui_defaults.
            #   ldc_lines: list of column keys to show, e.g. ["p50"] or ["min","p50","max"]
            #   When omitted, all PLOT_LINES entries are shown.
            # ------------------------------------------------------------------
            try:
                _ldc_lines = (ui_defaults or {}).get("ldc_lines")
                if isinstance(_ldc_lines, (list, tuple)) and _ldc_lines:
                    _ldc_lines_set = set(str(k).lower() for k in _ldc_lines)
                    PLOT_LINES = [(c, n, s) for c, n, s in PLOT_LINES if c in _ldc_lines_set]
            except Exception:
                pass

            # ------------------------------------------------------------------
            # Rolling smoother window for paired mode (applied to every trace
            # after ordering by flow/reference). Reduces visual noise while
            # preserving the flow-regime diagnostic signal.
            #   ldc_smooth_window: int (number of days). 0 or 1 = no smoothing.
            #   Defaults to 0 (off).
            # ------------------------------------------------------------------
            try:
                LDC_SMOOTH_WINDOW = int((ui_defaults or {}).get("ldc_smooth_window", 0))
                if LDC_SMOOTH_WINDOW < 2:
                    LDC_SMOOTH_WINDOW = 0
            except Exception:
                LDC_SMOOTH_WINDOW = 0

            # Toggle the "(paired)" suffix in LDC legend labels.
            # ui_defaults key: ldc_show_paired_label (bool, default True)
            _LDC_PAIRED_TAG = "<br>(paired)" if bool((ui_defaults or {}).get("ldc_show_paired_label", True)) else ""

            # --------------------------------------------------------------
            # Log-scale toggle (base 10) for the Load Duration Curve.
            # Provide via ui_defaults={"ldc_log_scale": True} when invoking
            # fan_compare_simulations_dashboard. Falls back to False.
            # Data aren't transformed; only axis scaling changes. Non-positive
            # values are suppressed automatically by Plotly on a log axis.
            # --------------------------------------------------------------
            # Log scale toggle (alias keys supported):
            #   ui_defaults{"ldc_log_scale": True}
            #   ui_defaults{"ldc_log": True}  (new alias)
            try:
                _ui = ui_defaults or {}
                if "ldc_log_scale" in _ui:
                    LDC_LOG_SCALE = bool(_ui.get("ldc_log_scale"))  # type: ignore
                elif "ldc_log" in _ui:
                    LDC_LOG_SCALE = bool(_ui.get("ldc_log"))  # type: ignore
                else:
                    LDC_LOG_SCALE = False
            except Exception:
                LDC_LOG_SCALE = False

            # Sort-by toggle: order the x-axis by load (default) or by SWAT flow.
            # ui_defaults key: ldc_sort_by: 'load' | 'flow'
            # When 'flow', the duration curve is re-ordered by flow exceedance
            # probability, which is the standard hydrological LDC convention.
            try:
                _sort_by = str((ui_defaults or {}).get("ldc_sort_by", "load")).lower()
                LDC_SORT_BY_FLOW = (_sort_by == "flow") and isinstance(flow_series, pd.Series) and not flow_series.empty
            except Exception:
                LDC_SORT_BY_FLOW = False
            if LDC_SORT_BY_FLOW:
                BAND_MODE = "paired"  # flow sorting requires paired day-level ordering

            if BAND_MODE == "paired" and all(c in q_plot.columns for c in ("min", "max")):
                try:
                    # Determine ordering reference series
                    order_ref_series: Optional[pd.Series] = None
                    if LDC_SORT_BY_FLOW:
                        # Use SWAT flow as the ordering reference (standard hydrological LDC)
                        order_ref_series = flow_series.copy()
                    elif BAND_ORDER_REF == "mean":
                        base_plot_df = _last.get("aligned_df_plot")
                        if isinstance(base_plot_df, pd.DataFrame) and not base_plot_df.empty:
                            try:
                                order_ref_series = base_plot_df.mean(axis=1, skipna=True)
                            except Exception:
                                order_ref_series = None
                    if order_ref_series is None:
                        if "p50" in q_plot.columns:
                            order_ref_series = q_plot["p50"].copy()
                        else:
                            # fallback: mid-point of min/max
                            try:
                                order_ref_series = (q_plot["min"] + q_plot["max"]) / 2.0
                            except Exception:
                                order_ref_series = None
                    # Build ordering DataFrame
                    if order_ref_series is not None:
                        df_order = pd.DataFrame({
                            "min": q_plot["min"],
                            "max": q_plot["max"],
                            "ref": order_ref_series,
                        }).dropna(subset=["min", "max", "ref"])  # keep complete cases
                        if not df_order.empty:
                            if LDC_SORT_BY_FLOW:
                                # Ascending: low flow on left (x≈0), high flow on right (x≈100)
                                # → flow-percentile convention; p75 sits at x≈75.
                                order_idx = np.argsort(df_order["ref"].to_numpy())
                            else:
                                # Descending: high loads on left (exceedance convention)
                                order_idx = np.argsort(-df_order["ref"].to_numpy())
                            df_ordered = df_order.iloc[order_idx]
                            n_ord = df_ordered.shape[0]
                            x_rank = 100.0 * (np.arange(1, n_ord + 1) / (n_ord + 1))
                            paired_index = df_ordered.index
                            paired_x_rank = x_rank
                            # Helper: optional rolling mean smoother
                            def _ldc_smooth(arr: np.ndarray, w: int) -> np.ndarray:
                                if w < 2 or arr.size < w:
                                    return arr
                                kernel = np.ones(w) / w
                                padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode='edge')
                                return np.convolve(padded, kernel, mode='valid')
                            _sw = LDC_SMOOTH_WINDOW
                            # Add band (max then min with fill)
                            _y_max_raw = df_ordered["max"].to_numpy(dtype=float)
                            _y_min_raw = df_ordered["min"].to_numpy(dtype=float)
                            fig_ldc.add_trace(
                                go.Scatter(
                                    x=x_rank,
                                    y=_ldc_smooth(_y_max_raw, _sw),
                                    mode="lines",
                                    name=f"Max simulation{_LDC_PAIRED_TAG}",
                                    line=dict(color="rgba(127,0,0,0.85)", width=0.8),
                                    showlegend=True,
                                    legendrank=500,
                                )
                            )
                            # Invisible fill trace (no legend) just for the grey band
                            fig_ldc.add_trace(
                                go.Scatter(
                                    x=x_rank,
                                    y=_ldc_smooth(_y_min_raw, _sw),
                                    mode="lines",
                                    line=dict(color="rgba(0,0,0,0)", width=0),
                                    fill="tonexty",
                                    fillcolor="rgba(100,100,100,0.18)",
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                            # Visible min line with legend
                            fig_ldc.add_trace(
                                go.Scatter(
                                    x=x_rank,
                                    y=_ldc_smooth(_y_min_raw, _sw),
                                    mode="lines",
                                    name=f"Min simulation{_LDC_PAIRED_TAG}",
                                    line=dict(color="rgba(0,68,27,0.85)", width=0.8),
                                    showlegend=True,
                                    legendrank=300,
                                )
                            )
                            added = True
                            # Reorder quantile lines using same index if present
                            for col, name, style in PLOT_LINES:
                                if col in {"min", "max"}:
                                    continue  # already displayed
                                if col not in q_plot.columns:
                                    continue
                                try:
                                    series_q = q_plot[col].reindex(df_ordered.index).to_numpy(dtype=float)
                                except Exception:
                                    continue
                                # Assign legendrank: p50 (mean) = 300, others bracket around it
                                _lrank_map = {"p75": 450, "p50": 400, "p25": 350}
                                _lrank = _lrank_map.get(col, 300)
                                fig_ldc.add_trace(
                                    go.Scatter(
                                        x=x_rank,
                                        y=_ldc_smooth(series_q, _sw),
                                        mode="lines",
                                        name=f"{name}{_LDC_PAIRED_TAG}",
                                        line=style,
                                        legendrank=_lrank,
                                    )
                                )
                            # Annotate mode
                            # (Removed previous top-right overlay annotation to declutter.)
                except Exception as _e_paired:
                    _dbg("paired_band_error", str(_e_paired))
            else:
                # Existing order_stats behavior
                # Min-max shaded band (only if both columns exist and non-empty)
                if all(c in q_plot.columns for c in ("min", "max")):
                    try:
                        s_max = q_plot["max"].dropna()
                        s_min = q_plot["min"].dropna()
                        if not s_max.empty and not s_min.empty:
                            x_max, y_max = _dcfs(s_max, levels)
                            x_min, y_min = _dcfs(s_min, levels)
                            if np.any(np.isfinite(y_max)) and np.any(np.isfinite(y_min)):
                                try:
                                    y_max_ord = np.sort(y_max)
                                    y_min_ord = np.sort(y_min)
                                except Exception:
                                    y_max_ord, y_min_ord = y_max, y_min
                                fig_ldc.add_trace(
                                    go.Scatter(
                                        x=x_max,
                                        y=y_max_ord,
                                        mode="lines",
                                        name="Max",
                                        line=dict(color="rgba(127,0,0,0.8)", width=0.8),
                                        showlegend=False,
                                    )
                                )
                                fig_ldc.add_trace(
                                    go.Scatter(
                                        x=x_min,
                                        y=y_min_ord,
                                        mode="lines",
                                        name="Min-Max band",
                                        line=dict(color="rgba(0,68,27,0.8)", width=0.8),
                                        fill="tonexty",
                                        fillcolor="rgba(100,100,100,0.18)",
                                    )
                                )
                                added = True
                    except Exception:
                        pass

                # Plot requested quantile lines (skip min/max here if already banded)
                for col, name, style in PLOT_LINES:
                    if col in {"min", "max"} and all(c in q_plot.columns for c in ("min", "max")):
                        continue
                    if col not in q_plot.columns:
                        continue
                    series = q_plot[col].dropna()
                    if series.empty:
                        continue
                    x_vals, y_vals = _dcfs(series, levels)
                    if not np.any(np.isfinite(y_vals)):
                        continue
                    try:
                        y_ordered = np.sort(y_vals)
                    except Exception:
                        y_ordered = y_vals
                    fig_ldc.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_ordered,
                            mode="lines",
                            name=name,
                            line=style,
                        )
                    )
                    added = True
            # ------------------------------------------------------------------
            # Overlay measured series (sparse points) from current view.
            # We DO NOT interpolate; each measured value is plotted at its
            # empirical exceedance probability (percent rank). The orientation
            # has low values at the left, so we sort ascending.
            # Easily disable by setting ENABLE_MEASURED_DURATION_OVERLAY=False
            # or commenting out this whole block.
            # ------------------------------------------------------------------
            ENABLE_MEASURED_DURATION_OVERLAY = True
            if ENABLE_MEASURED_DURATION_OVERLAY:
                try:
                    meas_list = _last.get("meas_series") or []
                    meas_half_mdl_flags = _last.get("meas_series_half_mdl_flags") or []
                    half_mdl_duration_style_active = (
                        bool(resolved_half_mdl_observation_style.get("enabled"))
                        and str((_last.get("measured_nonnum_audit") or {}).get("policy")) == "half_MDL"
                    )
                    duration_half_mdl_cfg = resolved_half_mdl_observation_style.get("duration_chart", {}) if isinstance(resolved_half_mdl_observation_style.get("duration_chart"), dict) else {}
                    duration_half_mdl_affected_cfg = duration_half_mdl_cfg.get("affected", {}) if isinstance(duration_half_mdl_cfg.get("affected"), dict) else {}
                    duration_half_mdl_unaffected_cfg = duration_half_mdl_cfg.get("unaffected", {}) if isinstance(duration_half_mdl_cfg.get("unaffected"), dict) else {}

                    def _duration_half_mdl_flags_for(idx: int, series_here: pd.Series) -> pd.Series:
                        if idx < len(meas_half_mdl_flags) and isinstance(meas_half_mdl_flags[idx], pd.Series):
                            return meas_half_mdl_flags[idx].reindex(series_here.index).fillna(False).astype(bool)
                        return pd.Series(False, index=series_here.index, dtype=bool)

                    if BAND_MODE == "paired" and paired_index is not None and paired_x_rank is not None:
                        # Build a combined measured series (multiple categories) preserving dates
                        combined: Dict[pd.Timestamp, list[Tuple[float, bool]]] = {}
                        if isinstance(meas_list, (list, tuple)):
                            for idx_ms, ms in enumerate(meas_list):
                                if isinstance(ms, pd.Series) and not ms.empty:
                                    ms_valid = ms.dropna()
                                    if ms_valid.empty:
                                        continue
                                    half_mdl_flags_here = _duration_half_mdl_flags_for(idx_ms, ms_valid)
                                    for dt, val in ms_valid.items():
                                        try:
                                            if not np.isfinite(val):
                                                continue
                                        except Exception:
                                            continue
                                        combined.setdefault(pd.Timestamp(dt), []).append((float(val), bool(half_mdl_flags_here.loc[dt])))
                        # Iterate in paired order; each date gets the x position from its rank
                        if combined:
                            x_pts: List[float] = []
                            y_pts: List[float] = []
                            x_pts_half_mdl: List[float] = []
                            y_pts_half_mdl: List[float] = []
                            for i, dt in enumerate(paired_index):
                                if dt in combined:
                                    x_here = float(paired_x_rank[i])
                                    for val, was_half_mdl in combined[dt]:
                                        if half_mdl_duration_style_active and was_half_mdl:
                                            x_pts_half_mdl.append(x_here)
                                            y_pts_half_mdl.append(val)
                                        else:
                                            x_pts.append(x_here)
                                            y_pts.append(val)
                            if x_pts or x_pts_half_mdl:
                                arr_x_all = np.array(x_pts + x_pts_half_mdl, dtype=float)
                                # measured extent is along the already-ranked axis
                                try:
                                    meas_x_extent = (float(np.nanmin(arr_x_all)), float(np.nanmax(arr_x_all)))  # type: ignore
                                except Exception:
                                    meas_x_extent = None  # type: ignore
                                base_duration_marker = dict(color="#d62728", size=7, line=dict(color="#333", width=0.6), symbol="circle")
                                if x_pts:
                                    fig_ldc.add_trace(
                                        go.Scatter(
                                            x=np.array(x_pts, dtype=float),
                                            y=np.array(y_pts, dtype=float),
                                            mode="markers",
                                            name=str(duration_half_mdl_unaffected_cfg.get("name_paired") or "Measured"),
                                            marker=base_duration_marker,
                                            hovertemplate="Measured: %{y:.4g}<extra></extra>",
                                            legendrank=600,
                                        )
                                    )
                                if half_mdl_duration_style_active and x_pts_half_mdl:
                                    fig_ldc.add_trace(
                                        go.Scatter(
                                            x=np.array(x_pts_half_mdl, dtype=float),
                                            y=np.array(y_pts_half_mdl, dtype=float),
                                            mode="markers",
                                            name=str(duration_half_mdl_affected_cfg.get("name") or "Measured (half-MDL)"),
                                            marker=_build_marker_with_overrides(base_duration_marker, duration_half_mdl_affected_cfg.get("marker")),
                                            hovertemplate="Measured (half-MDL): %{y:.4g}<extra></extra>",
                                            legendrank=610,
                                        )
                                    )
                                added = True
                                _dbg("duration measured overlay (paired)", dict(n_points=int(arr_x_all.size)))
                    else:
                        # Fallback to order-statistics orientation (value sorted)
                        measured_entries: List[Tuple[float, bool]] = []
                        if isinstance(meas_list, (list, tuple)):
                            for idx_ms, ms in enumerate(meas_list):
                                if isinstance(ms, pd.Series) and not ms.empty:
                                    ms_valid = ms.dropna()
                                    if ms_valid.empty:
                                        continue
                                    half_mdl_flags_here = _duration_half_mdl_flags_for(idx_ms, ms_valid)
                                    for dt, val in ms_valid.items():
                                        try:
                                            if not np.isfinite(val):
                                                continue
                                        except Exception:
                                            continue
                                        measured_entries.append((float(val), bool(half_mdl_flags_here.loc[dt])))
                        if measured_entries:
                            measured_entries.sort(key=lambda item: item[0])
                            arr_meas = np.array([val for val, _flag in measured_entries], dtype=float)
                            arr_flags = np.array([flag for _val, flag in measured_entries], dtype=bool)
                            arr_meas = arr_meas[np.isfinite(arr_meas)]
                            if arr_meas.size:
                                n_meas = arr_meas.size
                                x_meas = 100.0 * (np.arange(1, n_meas + 1) / (n_meas + 1))
                                try:
                                    meas_x_extent = (float(np.nanmin(x_meas)), float(np.nanmax(x_meas)))  # type: ignore
                                except Exception:
                                    meas_x_extent = None  # type: ignore
                                base_duration_marker = dict(color="#d62728", size=7, line=dict(color="#333", width=0.6), symbol="circle")
                                if not half_mdl_duration_style_active:
                                    fig_ldc.add_trace(
                                        go.Scatter(
                                            x=x_meas,
                                            y=arr_meas,
                                            mode="markers",
                                            name=str(duration_half_mdl_unaffected_cfg.get("name_order_stats") or "Measured (all)"),
                                            marker=base_duration_marker,
                                            hovertemplate="Measured: %{y:.4g}<extra></extra>",
                                        )
                                    )
                                else:
                                    if (~arr_flags).any():
                                        fig_ldc.add_trace(
                                            go.Scatter(
                                                x=x_meas[~arr_flags],
                                                y=arr_meas[~arr_flags],
                                                mode="markers",
                                                name=str(duration_half_mdl_unaffected_cfg.get("name_order_stats") or "Measured (all)"),
                                                marker=base_duration_marker,
                                                hovertemplate="Measured: %{y:.4g}<extra></extra>",
                                            )
                                        )
                                    if arr_flags.any():
                                        fig_ldc.add_trace(
                                            go.Scatter(
                                                x=x_meas[arr_flags],
                                                y=arr_meas[arr_flags],
                                                mode="markers",
                                                name=str(duration_half_mdl_affected_cfg.get("name") or "Measured (half-MDL)"),
                                                marker=_build_marker_with_overrides(base_duration_marker, duration_half_mdl_affected_cfg.get("marker")),
                                                hovertemplate="Measured (half-MDL): %{y:.4g}<extra></extra>",
                                            )
                                        )
                                added = True
                                _dbg("duration measured overlay", dict(n_points=int(n_meas)))
                except Exception as _e_meas_dc:
                    _dbg("duration measured overlay failed", str(_e_meas_dc))
            if added:
                # ----- Flow-regime background shading on the LDC -----
                # The boundary ensures ALL event-classified measured points
                # (threshold + min-days + buffer) sit inside the event zone.
                # We compute the full event mask, collect measured days that
                # are events, find the MINIMUM flow among them, and place the
                # boundary at that flow's sorted position.  Fallback: if no
                # measured event days exist, use the flow threshold directly.
                if LDC_SORT_BY_FLOW:
                    try:
                        if paired_index is not None and paired_x_rank is not None:
                            # ---- resolve threshold & buffer params ----
                            _thr_token_bg = str(dd_event_threshold.value) if 'dd_event_threshold' in dir() else "p75"
                            _buf_bg = int(sl_event_buffer_days.value) if ('sl_event_buffer_days' in dir() and isinstance(sl_event_buffer_days.value, (int, float))) else 0
                            _etmin_bg = float(fl_event_min_days.value) if ('fl_event_min_days' in dir() and isinstance(fl_event_min_days.value, (int, float))) else 1.0
                            # Use DAILY flow for threshold + event mask so that
                            # weekly/monthly aggregation doesn't shift the p75
                            # (aggregated means compress the distribution).
                            _daily_bg = daily_flow_series
                            if isinstance(_daily_bg, pd.Series) and not _daily_bg.empty:
                                _flow_daily_arr = _daily_bg.to_numpy(dtype=float)
                                _flow_daily_valid = _flow_daily_arr[np.isfinite(_flow_daily_arr)]
                            else:
                                _flow_daily_arr = None
                                _flow_daily_valid = np.array([], dtype=float)
                            # Threshold from aggregated flow only when no daily available
                            _flow_unsorted_bg = flow_series.reindex(q_plot.index).to_numpy(dtype=float)
                            _flow_valid_bg = _flow_unsorted_bg[np.isfinite(_flow_unsorted_bg)]
                            if _thr_token_bg == "abs":
                                _flow_thr_bg = float(tb_event_abs.value) if ('tb_event_abs' in dir() and isinstance(tb_event_abs.value, (int, float)) and not np.isnan(tb_event_abs.value)) else None
                            elif _thr_token_bg.startswith("p"):
                                _pctile = float(_thr_token_bg[1:])
                                if _flow_daily_valid.size:
                                    _flow_thr_bg = float(np.nanpercentile(_flow_daily_valid, _pctile))
                                elif _flow_valid_bg.size:
                                    _flow_thr_bg = float(np.nanpercentile(_flow_valid_bg, _pctile))
                                else:
                                    _flow_thr_bg = None
                            else:
                                if _flow_daily_valid.size:
                                    _flow_thr_bg = float(np.nanpercentile(_flow_daily_valid, 75.0))
                                elif _flow_valid_bg.size:
                                    _flow_thr_bg = float(np.nanpercentile(_flow_valid_bg, 75.0))
                                else:
                                    _flow_thr_bg = None
                            if _flow_thr_bg is not None and (_flow_daily_valid.size or _flow_valid_bg.size):
                                # ---- event mask on DAILY flow (threshold + min-days + buffer) ----
                                if _flow_daily_arr is not None and _flow_daily_valid.size:
                                    _above_bg = (_flow_daily_arr >= _flow_thr_bg) & np.isfinite(_flow_daily_arr)
                                    _min_samples_bg = max(1, int(np.ceil(_etmin_bg)))
                                    _d_runs_bg = np.diff(np.r_[0, _above_bg.astype(np.int8), 0])
                                    _starts_bg = np.where(_d_runs_bg == 1)[0]
                                    _ends_bg = np.where(_d_runs_bg == -1)[0]
                                    _core_bg = np.zeros(len(_flow_daily_arr), dtype=bool)
                                    for _s, _e in zip(_starts_bg, _ends_bg):
                                        if (_e - _s) >= _min_samples_bg:
                                            _core_bg[_s:_e] = True
                                    if _buf_bg > 0:
                                        _ev_mask_daily = _core_bg.copy()
                                        for _kb in range(1, _buf_bg + 1):
                                            _ev_mask_daily[_kb:] |= _core_bg[:-_kb]
                                            _ev_mask_daily[:-_kb] |= _core_bg[_kb:]
                                    else:
                                        _ev_mask_daily = _core_bg
                                    # Build day -> event flag lookup
                                    _daily_idx = _daily_bg.index
                                    _ev_day_set: set = set()
                                    for _kd in range(len(_daily_idx)):
                                        if _ev_mask_daily[_kd]:
                                            _ev_day_set.add(pd.Timestamp(_daily_idx[_kd]).normalize())
                                else:
                                    # Fallback: event mask on aggregated flow
                                    _above_bg = (_flow_unsorted_bg >= _flow_thr_bg) & np.isfinite(_flow_unsorted_bg)
                                    _min_samples_bg = max(1, int(np.ceil(_etmin_bg)))
                                    _d_runs_bg = np.diff(np.r_[0, _above_bg.astype(np.int8), 0])
                                    _starts_bg = np.where(_d_runs_bg == 1)[0]
                                    _ends_bg = np.where(_d_runs_bg == -1)[0]
                                    _core_bg = np.zeros(len(_flow_unsorted_bg), dtype=bool)
                                    for _s, _e in zip(_starts_bg, _ends_bg):
                                        if (_e - _s) >= _min_samples_bg:
                                            _core_bg[_s:_e] = True
                                    if _buf_bg > 0:
                                        _ev_mask_agg = _core_bg.copy()
                                        for _kb in range(1, _buf_bg + 1):
                                            _ev_mask_agg[_kb:] |= _core_bg[:-_kb]
                                            _ev_mask_agg[:-_kb] |= _core_bg[_kb:]
                                    else:
                                        _ev_mask_agg = _core_bg
                                    _ev_day_set = None  # no daily lookup available
                                # ---- collect measured dates ----
                                _meas_list_bg = _last.get("meas_series") or []
                                _meas_dates_bg: set = set()
                                if isinstance(_meas_list_bg, (list, tuple)):
                                    for _ms_bg in _meas_list_bg:
                                        if isinstance(_ms_bg, pd.Series) and not _ms_bg.empty:
                                            _meas_dates_bg.update(pd.Timestamp(d) for d in _ms_bg.dropna().index)
                                # ---- min flow among event-classified measured days ----
                                _min_ev_flow = None
                                if _ev_day_set is not None:
                                    # Daily event set available — match measured dates
                                    for _md in _meas_dates_bg:
                                        _md_n = pd.Timestamp(_md).normalize()
                                        if _md_n in _ev_day_set:
                                            _fv = _daily_bg.get(_md_n)
                                            if _fv is not None and np.isfinite(float(_fv)):
                                                _fv = float(_fv)
                                                if _min_ev_flow is None or _fv < _min_ev_flow:
                                                    _min_ev_flow = _fv
                                else:
                                    # Fallback: use aggregated mask
                                    _qidx_bg = q_plot.index
                                    for _im in range(len(_qidx_bg)):
                                        if _qidx_bg[_im] in _meas_dates_bg and _ev_mask_agg[_im]:
                                            _fv = _flow_unsorted_bg[_im]
                                            if np.isfinite(_fv) and (_min_ev_flow is None or _fv < _min_ev_flow):
                                                _min_ev_flow = _fv
                                _boundary_flow = _min_ev_flow if _min_ev_flow is not None else _flow_thr_bg
                                # ---- find boundary position in sorted chart ----
                                _flow_sorted_bg = flow_series.reindex(paired_index).to_numpy(dtype=float)
                                _boundary_x = None
                                for _i_bg in range(len(_flow_sorted_bg)):
                                    if np.isfinite(_flow_sorted_bg[_i_bg]) and _flow_sorted_bg[_i_bg] >= _boundary_flow:
                                        _n_pts = len(paired_x_rank)
                                        _half_gap = (paired_x_rank[1] - paired_x_rank[0]) / 2.0 if _n_pts >= 2 else 0.5
                                        _boundary_x = float(paired_x_rank[_i_bg]) - _half_gap
                                        break
                                # ---- draw ----
                                if _boundary_x is not None:
                                    _ev_hex_bg = cp_event_color.value if 'cp_event_color' in dir() else "#fdd0a2"
                                    _ne_hex_bg = cp_nonevent_color.value if 'cp_nonevent_color' in dir() else "#c6dbef"
                                    def _hex_rgba_ldc(h: str, a: float) -> str:
                                        h = h.lstrip('#')
                                        if len(h) == 3:
                                            h = ''.join(c * 2 for c in h)
                                        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
                                    _bg_alpha = 0.12
                                    fig_ldc.add_shape(
                                        type="rect", xref="x", yref="paper",
                                        x0=0, x1=_boundary_x, y0=0, y1=1,
                                        fillcolor=_hex_rgba_ldc(_ne_hex_bg, _bg_alpha),
                                        line=dict(width=0), layer="below",
                                    )
                                    fig_ldc.add_shape(
                                        type="rect", xref="x", yref="paper",
                                        x0=_boundary_x, x1=100, y0=0, y1=1,
                                        fillcolor=_hex_rgba_ldc(_ev_hex_bg, _bg_alpha),
                                        line=dict(width=0), layer="below",
                                    )
                                    fig_ldc.add_shape(
                                        type="line", xref="x", yref="paper",
                                        x0=_boundary_x, x1=_boundary_x, y0=0, y1=1,
                                        line=dict(color="grey", width=1.2, dash="dot"),
                                        layer="above",
                                    )
                                    if not any(getattr(tr, 'name', '') == 'Event-day zone' for tr in fig_ldc.data):
                                        fig_ldc.add_trace(go.Scatter(
                                            x=[None], y=[None], mode='markers',
                                            marker=dict(size=12, color=_hex_rgba_ldc(_ev_hex_bg, 0.35), symbol='square'),
                                            name='Event-day zone', showlegend=True,
                                            legendrank=100,
                                        ))
                                        fig_ldc.add_trace(go.Scatter(
                                            x=[None], y=[None], mode='markers',
                                            marker=dict(size=12, color=_hex_rgba_ldc(_ne_hex_bg, 0.35), symbol='square'),
                                            name='Non-event zone', showlegend=True,
                                            legendrank=200,
                                        ))
                                    _dbg("ldc_event_bg", dict(boundary=_boundary_x, boundary_flow=_boundary_flow, threshold=_flow_thr_bg, min_ev_meas_flow=_min_ev_flow))
                    except Exception as _e_ldc_bg:
                        _dbg("ldc_event_bg_failed", str(_e_ldc_bg))
                # Apply optional log scaling before final layout so we can adjust label
                if 'LDC_LOG_SCALE' in locals() and LDC_LOG_SCALE:
                    try:
                        candidate_cols = [c for c in ["p05","p10","p25","p50","p60","p75","p90","p95","min","max"] if c in q_plot.columns]
                        vals = q_plot[candidate_cols].to_numpy(dtype=float) if candidate_cols else np.array([])
                        if vals.size and np.any(vals > 0):
                            # Determine min/max positive for dynamic tick range
                            pos_vals = vals[vals > 0]
                            vmin = float(np.nanmin(pos_vals)) if pos_vals.size else 1.0
                            vmax = float(np.nanmax(pos_vals)) if pos_vals.size else 10.0
                            # Compute integer exponent bounds
                            emin = int(np.floor(np.log10(vmin)))
                            emax = int(np.ceil(np.log10(vmax)))
                            # ui_defaults key 'ldc_log_ticks_full' -> if True keep default (2,5) minor ticks
                            full_ticks = bool((ui_defaults or {}).get("ldc_log_ticks_full", False))
                            if full_ticks:
                                fig_ldc.update_yaxes(type="log")
                            else:
                                tickvals = [10 ** e for e in range(emin, emax + 1)]
                                fig_ldc.update_yaxes(type="log", tickvals=tickvals, ticktext=[f"1e{e}" if abs(e) > 2 else f"{10**e:g}" for e in range(emin, emax + 1)])
                            pass  # log scale already applied above
                        else:
                            pass
                    except Exception:
                        pass
                # Extract unit from y_axis_title (e.g. "Concentration (mg/L)" -> "mg/L")
                import re as _re_ldc
                _unit_match = _re_ldc.search(r'\(([^)]+)\)', y_axis_title)
                y_axis_final = _unit_match.group(1) + " (log)" if _unit_match else y_axis_title

                # Ensure x-axis: flow-percentile (0=low,100=high) when flow-sorted, otherwise exceedance
                # Decide x-axis range: optionally clip to measured overlay extent if requested in ui_defaults
                clip_meas = bool((ui_defaults or {}).get("ldc_clip_meas_extent", False)) if 'ui_defaults' in locals() else False
                if LDC_SORT_BY_FLOW:
                    _x_axis_label = "Flow percentile (%) \u2014 low \u2192 high"
                else:
                    _x_axis_label = "Exceedance probability (%)"
                if clip_meas and 'meas_x_extent' in locals() and isinstance(meas_x_extent, tuple) and meas_x_extent:
                    try:
                        xmin, xmax = meas_x_extent  # type: ignore
                        if np.isfinite(xmin) and np.isfinite(xmax):
                            # Add small padding (at least 0.5%)
                            span = max(1e-6, xmax - xmin)
                            pad = max(0.5, 0.02 * span)
                            xmin_clip = max(0.0, xmin - pad)
                            xmax_clip = min(100.0, xmax + pad)
                            # Ensure reasonable minimum span
                            if (xmax_clip - xmin_clip) < 2.0:
                                need = 2.0 - (xmax_clip - xmin_clip)
                                xmin_clip = max(0.0, xmin_clip - need/2)
                                xmax_clip = min(100.0, xmax_clip + need/2)
                            xaxis_obj = dict(range=[xmin_clip, xmax_clip])
                        else:
                            xaxis_obj = dict(range=[0,100])
                    except Exception:
                        xaxis_obj = dict(range=[0,100])
                else:
                    xaxis_obj = dict(range=[0,100])

                # Optional y-axis clipping when x clipping is enabled to focus on visible measurement-driven subset
                if clip_meas:
                    try:
                        # Gather all y values already plotted (simulation traces + measured) to compute focused range
                        y_all = []
                        for tr in fig_ldc.data:
                            try:
                                if hasattr(tr, 'y') and tr.y is not None:
                                    arr_y = np.array(tr.y, dtype=float)
                                    arr_y = arr_y[np.isfinite(arr_y)]
                                    if arr_y.size:
                                        y_all.append(arr_y)
                            except Exception:
                                continue
                        if y_all:
                            ycat = np.concatenate(y_all)
                            ycat = ycat[np.isfinite(ycat)]
                            if ycat.size:
                                ymin = float(np.nanmin(ycat))
                                ymax = float(np.nanmax(ycat))
                                if ymin == ymax:
                                    ymin -= 0.5 if ymin > 0 else 1.0
                                    ymax += 0.5 if ymax > 0 else 1.0
                                # Add padding 5%
                                pad = 0.05 * (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                                y0_clip = ymin - pad
                                y1_clip = ymax + pad
                                if 'LDC_LOG_SCALE' in locals() and LDC_LOG_SCALE:
                                    # Ensure positive lower bound for log scale
                                    # Move ymin upward to smallest positive among data if needed
                                    pos_vals = ycat[ycat > 0]
                                    if pos_vals.size:
                                        ymin_pos = float(np.nanmin(pos_vals))
                                        # Keep at most one order of magnitude below min positive
                                        y0_clip = max(ymin_pos / 10.0, ymin_pos * 0.5)
                                        y1_clip = ymax + pad
                                fig_ldc.update_yaxes(range=[y0_clip, y1_clip])
                    except Exception:
                        pass

                _ldc_var_name = dd_var.value if 'dd_var' in dir() else ""
                try:
                    _is_conc = (tg_units.value == "conc")
                except Exception:
                    _is_conc = False
                if LDC_SORT_BY_FLOW:
                    _ldc_suffix = "Concentration vs Flow Percentile" if _is_conc else "Load vs Flow Percentile"
                else:
                    _ldc_suffix = "Simulation Duration Curve (" + ("Concentration" if _is_conc else "Load") + ")"
                _ldc_title_text = f"{_ldc_var_name} \u2013 {_ldc_suffix}" if _ldc_var_name else _ldc_suffix
                fig_ldc.update_layout(
                    **_build_duration_chart_layout_update(
                        "load_duration",
                        title_text=_ldc_title_text,
                        xaxis_title_text=_x_axis_label,
                        yaxis_title_text=y_axis_final,
                        base_xaxis=xaxis_obj,
                    )
                )
                _apply_figure_size(fig_ldc, duration_chart_layout)
                widgets_out.append(go.FigureWidget(fig_ldc))
        sim_flow_series = None
        if isinstance(flow_series, pd.Series) and not flow_series.empty:
            sim_flow_series = flow_series.dropna()
        # Skip legacy flow duration curve if flow-stratified curve will be displayed
        if sim_flow_series is not None and not sim_flow_series.empty and not bool(cb_flow_strat.value):
            fig_fdc = go.Figure(layout=dict(template=template_name))
            x_flow, y_flow = _dcfs(sim_flow_series, levels)
            if np.any(np.isfinite(y_flow)):
                try:
                    y_flow_ordered = np.sort(y_flow)
                except Exception:
                    y_flow_ordered = y_flow
                fig_fdc.add_trace(
                    go.Scatter(
                        x=x_flow,
                        y=y_flow_ordered,
                        mode="lines",
                        name="Simulated flow",
                        line=dict(color="#17becf", width=2),
                    )
                )
                # Overlay measured water flow duration curve when available
                if isinstance(measured_flow_series, pd.Series) and not measured_flow_series.empty:
                    try:
                        mf = measured_flow_series.dropna()
                        if not mf.empty:
                            x_mf, y_mf = _dcfs(mf, levels)
                            if np.any(np.isfinite(y_mf)):
                                try:
                                    y_mf_ordered = np.sort(y_mf)
                                except Exception:
                                    y_mf_ordered = y_mf
                                fig_fdc.add_trace(
                                    go.Scatter(
                                        x=x_mf,
                                        y=y_mf_ordered,
                                        mode="lines",
                                        name="Measured flow",
                                        line=dict(color="#1f77b4", width=2, dash="dash"),
                                    )
                                )
                    except Exception:
                        pass
                fig_fdc.update_layout(
                    **_build_duration_chart_layout_update(
                        "flow_duration",
                        title_text="Flow Duration Curve",
                        xaxis_title_text=r"% of Time where flow is Exceeded",
                        yaxis_title_text="Flow (m3/day)",
                    )
                )
                _apply_figure_size(fig_fdc, duration_chart_layout)
                widgets_out.append(go.FigureWidget(fig_fdc))
        return widgets_out

    def _compute_and_plot():
        #reload(.stats)
        if _state.get("updating"):
            return
        _state["updating"] = True
        def _release():
            try:
                _state["updating"] = False
            except Exception:
                pass
        # Clear stale overlay/UI banner if present
        try:
            reload_bar.layout.display = 'none'
        except Exception:
            pass
        try:
            if _last.get("fig") is not None and _state.get("stale_overlay", False):
                # remove overlay by resetting shapes
                try:
                    fig_old = _last.get("fig")
                    if hasattr(fig_old.layout, 'shapes'):
                        # remove all shapes we added (we only ever add one stale overlay)
                        fig_old.update_layout(shapes=[])
                except Exception:
                    pass
                _state["stale_overlay"] = False
        except Exception:
            pass
        freq_str = _make_freq_string(dd_freq.value, sl_bin.value)
        var = dd_var.value
        # ---- bump generation counter so stale background threads bail ----
        with _state["_gen_lock"]:
            _state["_gen"] += 1
            _current_gen = _state["_gen"]
        # Debug mapping for measured presets
        if measured_present:
            try:
                nm = _normalize_meas_map_for_var(measured_var_map or {}, var)
                _dbg("measured_var_map", {k: len(v) for k, v in nm.items()})
            except Exception:
                pass
        method = dd_method.value
        is_conc_mode = (tg_units.value == "conc")
        # Ensure method options match mode (may correct invalid states)
        _update_method_options_for_mode()
        _dbg("compute", dict(var=var, reach=dd_reach.value, freq=freq_str, method=method, mode=("conc" if is_conc_mode else "load")))

        try:
            duration_box.children = [widgets.HTML("<i>Loading duration curves...</i>")]
        except Exception:
            pass

        # Pre-compute daily external flow series and SWAT avg daily flow series for conversions
        s_external_flow_daily: Optional[pd.Series] = None
        if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
            try:
                use_flow_col = None
                if water_flow_value_col and water_flow_value_col in water_flow_df.columns:
                    use_flow_col = water_flow_value_col
                elif flow_meas_col and flow_meas_col in water_flow_df.columns:
                    use_flow_col = flow_meas_col
                else:
                    use_flow_col = _pick_best_flow_col(water_flow_df, explicit=None)
                if use_flow_col:
                    fdf = water_flow_df[[water_flow_date_col, use_flow_col]].copy()
                    fdf[water_flow_date_col] = pd.to_datetime(fdf[water_flow_date_col], errors='coerce').dt.floor('D')
                    fdf[use_flow_col] = pd.to_numeric(fdf[use_flow_col], errors='coerce').astype(float)
                    fdf = fdf.dropna(subset=[water_flow_date_col, use_flow_col])
                    s_external_flow_daily = fdf.groupby(water_flow_date_col)[use_flow_col].sum(min_count=1)
                    if start is not None:
                        s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index >= pd.to_datetime(start).floor('D')]
                    if end is not None:
                        s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index <= pd.to_datetime(end).floor('D')]
                    if season_months:
                        months = set(int(m) for m in season_months)
                        s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index.month.isin(months)]
                    s_external_flow_daily.index.name = None
            except Exception:
                s_external_flow_daily = None

        # SWAT average flow (m3/day) as daily series
        s_swat_avg_daily: Optional[pd.Series] = None
        try:
            per_sim_daily: Dict[str, pd.Series] = {}
            for sim_name, df in sim_dfs.items():
                fcol = _pick_best_swat_flow_col(df)
                if not fcol or reach_col not in df.columns or date_col not in df.columns or fcol not in df.columns:
                    continue
                subf = df[df[reach_col] == dd_reach.value][[date_col, fcol]].copy()
                if subf.empty:
                    continue
                subf = _ensure_dt_index(subf, date_col)
                if start or end:
                    subf = _slice_time(subf, start, end)
                if season_months:
                    subf = _filter_season(subf, season_months)
                if subf.empty:
                    continue
                try:
                    subf[fcol] = pd.to_numeric(subf[fcol], errors='coerce').astype(float)
                except Exception:
                    pass
                with np.errstate(invalid='ignore'):
                    subf["__m3day__"] = subf[fcol].astype(float) * 86400.0
                s_day = subf["__m3day__"].groupby(subf.index.floor('D')).sum(min_count=1).dropna()
                if not s_day.empty:
                    per_sim_daily[str(sim_name)] = s_day
            if per_sim_daily:
                aligned = pd.concat(per_sim_daily.values(), axis=1).sort_index()
                s_swat_avg_daily = aligned.mean(axis=1, skipna=True).dropna()
                s_swat_avg_daily.index.name = None
        except Exception:
            s_swat_avg_daily = None

        # Build event day sets using configurable detection
        event_mode = str(tg_event_view.value or "all")
        selected_days_set = None
        event_day_set = None
        buffered_event_days = None
        full_days_set = None
        try:
            candidate_modes = {"events", "non_events", "all"}
            if event_mode in candidate_modes:
                ev_source = str(dd_event_source.value)
                if ev_source == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                    s_events_flow = s_external_flow_daily.copy()
                elif ev_source == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                    s_events_flow = s_swat_avg_daily.copy()
                else:
                    s_events_flow = None
                if s_events_flow is not None and not s_events_flow.empty:
                    df_ev = pd.DataFrame({"date": pd.to_datetime(s_events_flow.index).floor('D'), "Q": s_events_flow.values})
                    token = str(dd_event_threshold.value)
                    if token == "abs":
                        thr_val = float(tb_event_abs.value) if isinstance(tb_event_abs.value, (int, float)) and not np.isnan(tb_event_abs.value) else None
                        thr_def = thr_val if thr_val is not None else None
                    else:
                        thr_def = token
                    if thr_def is not None:
                        from .dashboard_helper import add_event_flags
                        etmin = float(fl_event_min_days.value) if isinstance(fl_event_min_days.value, (int, float)) else 1.0
                        df_flags = add_event_flags(df_ev, thresholds={"main": thr_def}, intervals={"main": etmin}, time_col="date", flow_col="Q")
                        if "main_event" in df_flags.columns:
                            event_days = pd.to_datetime(df_flags.loc[df_flags["main_event"], :].index).floor('D').unique()
                            event_day_set = set(pd.to_datetime(event_days).tolist())
                            full_days_set = set(pd.to_datetime(df_ev["date"]).unique().tolist())
                            buf = int(sl_event_buffer_days.value) if isinstance(sl_event_buffer_days.value, (int, float)) else 0
                            buffered_event_days = set()
                            for d in event_day_set:
                                d0 = pd.Timestamp(d).normalize()
                                for k in range(-buf, buf + 1):
                                    buffered_event_days.add(d0 + pd.Timedelta(days=int(k)))
                            if event_mode == "events":
                                selected_days_set = set(buffered_event_days)
                            elif event_mode == "non_events" and full_days_set is not None:
                                selected_days_set = set(full_days_set) - set(buffered_event_days)
                            elif event_mode == "all":
                                # All modeled days regardless of event classification
                                selected_days_set = set(full_days_set)
                            else:  # fallback
                                selected_days_set = set(full_days_set) if full_days_set is not None else set(event_day_set)
                            _dbg("events", dict(
                                mode=event_mode,
                                events=len(event_day_set or []),
                                buffer=len(set(buffered_event_days) - set(event_day_set)),
                                keep=len(selected_days_set)
                            ))
        except Exception as e:
            _dbg("event detection failed", e)
            selected_days_set = None

        def _event_index(values):
            if values is None:
                return None
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(list(values)))
                idx = idx.floor('D')
                idx = idx.unique()
                idx = idx.sort_values()
                return idx
            except Exception:
                return None

        idx_events = _event_index(event_day_set)
        idx_buffered = _event_index(buffered_event_days)
        if idx_buffered is None:
            idx_buffered = idx_events
        idx_all_days = _event_index(full_days_set)
        idx_selected = _event_index(selected_days_set)
        if idx_all_days is not None and idx_buffered is not None:
            idx_non_events = idx_all_days.difference(idx_buffered)
            if idx_non_events.empty:
                idx_non_events = None
        else:
            idx_non_events = None
        _last["event_context"] = {
            "mode": event_mode,
            "events": idx_events,
            "buffered_events": idx_buffered,
            "non_events": idx_non_events,
            "selected": idx_selected,
            "all_days": idx_all_days,
        }
        # Preserve daily SWAT flow for LDC threshold (immune to aggregation)
        _last["swat_flow_daily"] = s_swat_avg_daily

        # Extract a single resampled series per run for the selected reach/variable
        # Maintain filtered (stats) and unfiltered (plot) collections
        per_sim: Dict[str, pd.Series] = {}
        per_sim_plot: Dict[str, pd.Series] = {}
        _raw_daily_end: Optional[pd.Timestamp] = None  # track pre-resample daily range
        for sim_name, df in sim_dfs.items():
            # Build subset depending on variable (derived vs direct)
            if var == SYN_VAR:
                if not all(c in df.columns for c in DERIVED_COMPONENTS):
                    _dbg(f"skip run {sim_name}: components for {SYN_VAR} missing")
                    continue
                sub_cols = [date_col] + list(DERIVED_COMPONENTS)
            else:
                if var not in df.columns:
                    _dbg(f"skip run {sim_name}: var '{var}' not in columns")
                    continue
                sub_cols = [date_col, var]
            # Concentration computation needs flow column
            #print(_pick_best_swat_flow_col(df))
            flow_col = _pick_best_swat_flow_col(df)
            if (is_conc_mode or method == "flow_weighted_mean") and flow_col in df.columns:
                sub_cols.append(flow_col)
            sub0 = df[df[reach_col] == dd_reach.value][sub_cols].copy()
            _dbg_df_info(sub0, f"{sim_name} raw reach={dd_reach.value}", date_col=date_col)
            sub = sub0
            if var == SYN_VAR:
                with np.errstate(invalid='ignore'):
                    sub[SYN_VAR] = sub[DERIVED_COMPONENTS[0]].astype(float) + sub[DERIVED_COMPONENTS[1]].astype(float)
            if sub.empty:
                continue
            sub = _ensure_dt_index(sub, date_col)
            if start or end:
                sub = _slice_time(sub, start, end)
                _dbg_df_info(sub, f"{sim_name} after time slice")
            if season_months:
                sub = _filter_season(sub, season_months)
                _dbg_df_info(sub, f"{sim_name} after season filter")
            # Keep a copy BEFORE event-day filtering for plotting
            sub_plot = sub.copy()
            # Track the raw daily date range (pre-resample) for incomplete-period detection
            try:
                _sp_max = pd.Timestamp(sub_plot.index.max()).normalize()
                if _raw_daily_end is None or _sp_max > _raw_daily_end:
                    _raw_daily_end = _sp_max
            except Exception:
                pass
            if selected_days_set is not None:
                try:
                    day_mask = sub.index.floor('D').isin(list(selected_days_set))
                    sub = sub.loc[day_mask]
                    _dbg_df_info(sub, f"{sim_name} after event-mode filter")
                except Exception:
                    pass
            if sub.empty:
                continue
            # If in concentration mode, derive daily concentration mg/L from kg/day and chosen flow source
            if is_conc_mode:
                base_col = (SYN_VAR if var == SYN_VAR else var)
                # Determine flow source for conversion
                flow_source = str(dd_flow_source.value)
                # Prepare per-timestamp m3/day series aligned to sub.index
                if flow_source == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                    days = sub.index.floor('D')
                    f_series = s_external_flow_daily.reindex(days)
                    fvals = f_series.to_numpy(dtype=float)
                    kgd = sub[base_col].to_numpy(dtype=float)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        conc = (kgd / fvals) * 1000.0
                    sub["__conc_mgL__"] = conc
                    sub["__flow_m3d__"] = f_series.to_numpy(dtype=float)
                elif flow_source == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                    days = sub.index.floor('D')
                    f_series = s_swat_avg_daily.reindex(days)
                    fvals = f_series.to_numpy(dtype=float)
                    kgd = sub[base_col].to_numpy(dtype=float)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        conc = (kgd / fvals) * 1000.0
                    sub["__conc_mgL__"] = conc
                    sub["__flow_m3d__"] = f_series.to_numpy(dtype=float)
                else:
                    # Fallback to per-run FLOW_OUT if available
                    if flow_col not in sub.columns:
                        if "FLOW_OUTcmscms" not in sub.columns:
                            _dbg(f"skip run {sim_name}: no flow available for conc mode conversion")
                            continue
                        else:
                            flow_col = "FLOW_OUTcmscms"
                    with np.errstate(invalid='ignore', divide='ignore'):
                        sub["__conc_mgL__"] = (sub[base_col] / (sub[flow_col] * 86400.0)) * 1000.0
                        sub["__flow_m3d__"] = (sub[flow_col].astype(float) * 86400.0)
                # Aggregate concentration using chosen method (mean or flow_weighted_mean with selected flow as weights)
                how_here = dd_method.value if dd_method.value in ("flow_weighted_mean", "mean") else "mean"
                s = _resample_series(sub, "__conc_mgL__", freq=freq_str, how=how_here, flow_col="__flow_m3d__")
            else:
                base_col = (SYN_VAR if var == SYN_VAR else var)
                s = _resample_series(sub, base_col, freq=freq_str, how=method, flow_col=flow_col if flow_col in sub.columns else None)
            if s.empty:
                continue
            s.name = sim_name
            per_sim[sim_name] = s
            _dbg_df_info(s, f"{sim_name} series after resample")

            # Build plotting series (unfiltered by event filter)
            try:
                base_col_plot = (SYN_VAR if var == SYN_VAR else var)
                if is_conc_mode:
                    flow_source_p = str(dd_flow_source.value)
                    if flow_source_p == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                        days_p = sub_plot.index.floor('D')
                        f_series_p = s_external_flow_daily.reindex(days_p)
                        fvals_p = f_series_p.to_numpy(dtype=float)
                        kgd_p = sub_plot[base_col_plot].to_numpy(dtype=float)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            conc_p = (kgd_p / fvals_p) * 1000.0
                        sub_plot["__conc_mgL__"] = conc_p
                        sub_plot["__flow_m3d__"] = f_series_p.to_numpy(dtype=float)
                    elif flow_source_p == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                        days_p = sub_plot.index.floor('D')
                        f_series_p = s_swat_avg_daily.reindex(days_p)
                        fvals_p = f_series_p.to_numpy(dtype=float)
                        kgd_p = sub_plot[base_col_plot].to_numpy(dtype=float)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            conc_p = (kgd_p / fvals_p) * 1000.0
                        sub_plot["__conc_mgL__"] = conc_p
                        sub_plot["__flow_m3d__"] = f_series_p.to_numpy(dtype=float)
                    else:
                        # fallback to per-run flow column
                        flow_col_plot = flow_col if flow_col in sub_plot.columns else ("FLOW_OUTcmscms" if "FLOW_OUTcmscms" in sub_plot.columns else None)
                        if flow_col_plot is not None:
                            with np.errstate(invalid='ignore', divide='ignore'):
                                sub_plot["__conc_mgL__"] = (sub_plot[base_col_plot] / (sub_plot[flow_col_plot] * 86400.0)) * 1000.0
                                sub_plot["__flow_m3d__"] = (sub_plot[flow_col_plot].astype(float) * 86400.0)
                    how_plot = dd_method.value if dd_method.value in ("flow_weighted_mean", "mean") else "mean"
                    s_plot = _resample_series(sub_plot, "__conc_mgL__", freq=freq_str, how=how_plot, flow_col="__flow_m3d__")
                else:
                    s_plot = _resample_series(sub_plot, base_col_plot, freq=freq_str, how=method, flow_col=flow_col if flow_col in sub_plot.columns else None)
                if not s_plot.empty:
                    s_plot.name = sim_name
                    per_sim_plot[sim_name] = s_plot
            except Exception as _e_plot:
                _dbg("build plot series failed", dict(run=sim_name, err=str(_e_plot)))

        if not per_sim:
            with out:
                clear_output(wait=True)
                print(f"No data for reach {dd_reach.value} and variable '{var}'.")
            _release()
            return

        # Align series to a common time index (union) and build 2D matrix (T x N)
        aligned_df = pd.concat(per_sim.values(), axis=1).sort_index()
        aligned_df.index = pd.to_datetime(aligned_df.index, utc=False)

        # Drop incomplete last aggregate period (non-daily only).
        # If the raw daily data doesn't cover the full last resampled period,
        # truncate so the graph stops before the incomplete bin.
        _is_daily_freq = freq_str.upper().endswith('D') and int(''.join(c for c in freq_str if c.isdigit()) or '1') == 1
        if not _is_daily_freq and _raw_daily_end is not None and len(aligned_df) > 1:
            try:
                # The resampled index label marks the period end (e.g. month-end, year-end).
                # If the raw daily data doesn't reach that date, the period is incomplete.
                _last_label = pd.Timestamp(aligned_df.index[-1]).normalize()
                if _raw_daily_end < _last_label:
                    aligned_df = aligned_df.iloc[:-1]
                    _dbg("incomplete_period_trim", dict(dropped=str(_last_label.date()), raw_end=str(_raw_daily_end.date())))
            except Exception as _e_trim:
                _dbg("incomplete_period_trim failed", str(_e_trim))

        arr = aligned_df.to_numpy(dtype=float)  # shape: (T, N)
        # Ensure x-axis values are JSON-safe strings to avoid timezone packing issues in Jupyter
        def _to_plotly_x(idx: pd.Index) -> List[str]:
            try:
                dt = pd.to_datetime(idx, errors='coerce')
                # drop tz info if any
                try:
                    dt = dt.tz_localize(None)
                except Exception:
                    pass
                arr = dt.to_pydatetime()
                out_x: List[str] = []
                for d in arr:
                    try:
                        # Use date-only where possible (daily data); include time if present
                        if isinstance(d, datetime):
                            out_x.append(d.isoformat())
                        else:
                            out_x.append(None)
                    except Exception:
                        out_x.append(None)
                return out_x
            except Exception:
                # Fallback: cast to string
                return [str(v) for v in list(idx)]
        x_dt = _to_plotly_x(aligned_df.index)
        _last["aligned_df"] = aligned_df
        _dbg("aligned", dict(T=arr.shape[0], N=arr.shape[1]))

        # Capture sediment net series for later diagnostics (LDC overlay) early, before resampling alters availability.
        try:
            sed_series_candidate = None
            # Look in original simulation frames for chosen reach
            for df_run in sim_dfs.values():
                cols = getattr(df_run, 'columns', [])
                has_precomp = 'SED_IN_MINUS_OUT' in cols
                sin = next((c for c in ['SED_INtons','SED_IN'] if c in cols), None)
                sout = next((c for c in ['SED_OUTtons','SED_OUT'] if c in cols), None)
                if has_precomp:
                    tmp = df_run[df_run[reach_col] == dd_reach.value][[date_col, 'SED_IN_MINUS_OUT']].copy()
                    if not tmp.empty:
                        tmp = _ensure_dt_index(tmp, date_col)
                        sed_series_candidate = tmp['SED_IN_MINUS_OUT'].dropna()
                        break
                elif sin and sout:
                    tmp = df_run[df_run[reach_col] == dd_reach.value][[date_col, sin, sout]].copy()
                    if not tmp.empty:
                        tmp = _ensure_dt_index(tmp, date_col)
                        with np.errstate(invalid='ignore'):
                            sed_series_candidate = (tmp[sin].astype(float) - tmp[sout].astype(float)).dropna()
                        if sed_series_candidate is not None and not sed_series_candidate.empty:
                            break
            if sed_series_candidate is not None and not sed_series_candidate.empty:
                _last['sed_series_for_ldc'] = sed_series_candidate.sort_index()
                _dbg('store sed_series_for_ldc', dict(n=len(sed_series_candidate)))
            else:
                _last['sed_series_for_ldc'] = None
                _dbg('store sed_series_for_ldc', 'none available')
        except Exception as e:
            _last['sed_series_for_ldc'] = None
            _dbg('store sed_series_for_ldc failed', str(e))

        # Compute quantiles across runs (ignore NaNs)
        percs = [5, 10, 25, 50, 60, 75, 90, 95]
        if arr.shape[1] == 0:
            with out:
                clear_output(wait=True)
                print("No aligned data after resampling.")
            _release()
            return
        qs = np.nanpercentile(arr, percs, axis=1)  # shape: (7, T)
        q = {p: qs[i, :] for i, p in enumerate(percs)}  # p -> array(T,)
        # Store quantiles in a frame for quick lookups
        # Also compute per-timestamp min and max across simulations for duration banding
        try:
            row_min = np.nanmin(arr, axis=1)
        except Exception:
            row_min = np.full(arr.shape[0], np.nan)
        try:
            row_max = np.nanmax(arr, axis=1)
        except Exception:
            row_max = np.full(arr.shape[0], np.nan)
        q_df = pd.DataFrame({
            "min": row_min,
            "p05": q[5],
            "p10": q[10],
            "p25": q[25],
            "p50": q[50],
            "p60": q[60],
            "p75": q[75],
            "p90": q[90],
            "p95": q[95],
            "max": row_max,
        }, index=aligned_df.index)
        _last["q_df"] = q_df
        if debug or (isinstance(ui_defaults, dict) and ui_defaults.get("debug")):
            all_nan = int(q_df["p50"].isna().sum())
            _dbg("quantiles", dict(all_nan_p50=all_nan))

        # Build UNFILTERED aligned DataFrame for plotting; if empty fallback to filtered
        try:
            if per_sim_plot:
                aligned_df_plot = pd.concat(per_sim_plot.values(), axis=1).sort_index()
            else:
                aligned_df_plot = aligned_df.copy()
            aligned_df_plot.index = pd.to_datetime(aligned_df_plot.index, utc=False)
            # Trim incomplete last aggregate period (same logic as aligned_df)
            if not _is_daily_freq and _raw_daily_end is not None and len(aligned_df_plot) > 1:
                try:
                    _lp = pd.Timestamp(aligned_df_plot.index[-1]).normalize()
                    if _raw_daily_end < _lp:
                        aligned_df_plot = aligned_df_plot.iloc[:-1]
                except Exception:
                    pass
            arr_plot = aligned_df_plot.to_numpy(dtype=float)
            percs_plot = [5, 10, 25, 50, 60, 75, 90, 95]
            qs_plot = np.nanpercentile(arr_plot, percs_plot, axis=1) if arr_plot.shape[1] else np.full((len(percs_plot), 0), np.nan)
            q_plot = {p: qs_plot[i, :] if arr_plot.shape[1] else np.array([]) for i, p in enumerate(percs_plot)}
            _last["aligned_df_plot"] = aligned_df_plot
            # Effective plot end (used to clip overlay resampled series so they don't extend past trimmed main data)
            _plot_end_ts = pd.Timestamp(aligned_df_plot.index[-1]) if len(aligned_df_plot) else None
            # Compute per-timestamp min/max for UNFILTERED data so duration curve band
            # matches the quantile lines (avoids mismatch under Non-event filtering)
            try:
                row_min_plot = np.nanmin(arr_plot, axis=1)
            except Exception:
                row_min_plot = np.full(arr_plot.shape[0], np.nan)
            try:
                row_max_plot = np.nanmax(arr_plot, axis=1)
            except Exception:
                row_max_plot = np.full(arr_plot.shape[0], np.nan)
            _last["q_plot_df"] = pd.DataFrame({
                "min": row_min_plot,
                "p05": q_plot[5], "p10": q_plot[10], "p25": q_plot[25], "p50": q_plot[50], "p60": q_plot[60], "p75": q_plot[75], "p90": q_plot[90], "p95": q_plot[95],
                "max": row_max_plot,
            }, index=aligned_df_plot.index)
            # y-range based on UNFILTERED data
            finite_vals_plot = arr_plot[np.isfinite(arr_plot)]
            if finite_vals_plot.size:
                y_min = float(np.nanmin(finite_vals_plot))
                y_max = float(np.nanmax(finite_vals_plot))
                if y_min == y_max:
                    y_max = y_min + 1.0
            else:
                y_min, y_max = 0.0, 1.0
            pad = (y_max - y_min) * 0.05
            _last["y_fixed"] = [y_min - pad, y_max + pad]
        except Exception as _e_unf:
            _dbg("unfiltered plot build failed", str(_e_unf))
            aligned_df_plot = aligned_df
            arr_plot = arr
            q_plot = q
            _last["y_fixed"] = _last.get("y_fixed", None)
            _plot_end_ts = pd.Timestamp(aligned_df.index[-1]) if len(aligned_df) else None

        # Per-point human-friendly hover scaling (k = thousands, M = millions)
        # Keeps numbers readable and avoids misleading labels for small values.
        def _nan_to_none(arr: np.ndarray) -> np.ndarray:
            a = np.asarray(arr)
            out = a.astype(object)
            mask = np.isfinite(a)
            out[~mask] = None
            return out
        def _format_scale_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            vals = np.asarray(values, dtype=float)
            absv = np.abs(vals)
            labels = np.where(absv >= 1e6, "M", np.where(absv >= 1e3, "k", ""))
            factors = np.where(labels == "M", 1e-6, np.where(labels == "k", 1e-3, 1.0))
            scaled = vals * factors
            return scaled, labels

        def _make_customdata(values: np.ndarray) -> np.ndarray:
            scaled, labels = _format_scale_array(values)
            # Replace NaN with None for JSON-compatibility
            cd = np.empty((scaled.shape[0], 2), dtype=object)
            finite = np.isfinite(scaled)
            cd[:, 0] = np.where(finite, scaled, None)
            cd[:, 1] = labels
            return cd

        # Build figure (plotting uses UNFILTERED data arrays)
        fig = go.FigureWidget(layout=dict(template=template))
        _apply_figure_size(fig, main_chart_layout)
        # Z-order control: traces are collected per group, then added in order
        _active_trace_order = list(trace_order) if trace_order else list(_TRACE_ORDER_DEFAULT)
        _deferred_groups: Dict[str, list] = {k: [] for k in _TRACE_ORDER_DEFAULT}
        _last["trace_order"] = _active_trace_order
        # Grey overlay for filtered-out days (aggregate any consecutive excluded days over full date span)
        try:
            if selected_days_set is not None and len(selected_days_set) > 0:
                existing_ranges = _last.get('filtered_out_overlay')
                # Work over full continuous daily span (ensures coverage even if resampled index is sparse)
                if len(aligned_df_plot.index) > 0 and pd.api.types.is_datetime64_any_dtype(aligned_df_plot.index):
                    day_start = pd.to_datetime(aligned_df_plot.index.min()).floor('D')
                    day_end = pd.to_datetime(aligned_df_plot.index.max()).floor('D')
                    full_days = pd.date_range(start=day_start, end=day_end, freq='D')
                    excluded_days = [d for d in full_days if d not in selected_days_set]
                else:
                    excluded_days = []
                if excluded_days:
                    # Group consecutive days
                    blocks: list[list[pd.Timestamp, pd.Timestamp]] = []
                    for d in excluded_days:
                        d = pd.Timestamp(d).normalize()
                        if not blocks or d - blocks[-1][1] > pd.Timedelta(days=1):
                            blocks.append([d, d])
                        else:
                            blocks[-1][1] = d
                    if blocks != existing_ranges:
                        # Remove previous overlay shapes (keep other shapes if any by filtering on fillcolor signature)
                        prev_shapes = list(getattr(fig.layout, 'shapes', []))
                        remaining = [s for s in prev_shapes if not str(getattr(s, 'fillcolor', '') or '').startswith('rgba(90,90,90')]
                        new_shapes = []
                        for a, b in blocks:
                            new_shapes.append(dict(
                                type='rect', xref='x', yref='paper',
                                x0=a.isoformat(), x1=(b + pd.Timedelta(days=1)).isoformat(),
                                y0=0, y1=1,
                                fillcolor='rgba(90,90,90,0.30)',
                                line=dict(width=0), layer='below'
                            ))
                        fig.layout.shapes = tuple(remaining + new_shapes)
                        _last['filtered_out_overlay'] = blocks
                    # Legend proxy: only show when there are actual excluded blocks
                    if blocks and not any(getattr(tr, 'name', '') == 'Filtered (excluded from stats)' for tr in fig.data):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color='rgba(90,90,90,0.30)', symbol='square'),
                            name='Filtered (excluded from stats)',
                            hoverinfo='skip', showlegend=True
                        ))
        except Exception as _e_overlay:
            _dbg('overlay build failed', str(_e_overlay))

        # Event / non-event background illustration (purely visual, no stats impact)
        try:
            if cb_show_event_bg.value:
                _ec = _last.get("event_context") or {}
                _ev_days = _ec.get("buffered_events")  # DatetimeIndex or None
                _all_days = _ec.get("all_days")         # DatetimeIndex or None
                _dbg("event_bg: ev_days=%s all_days=%s" % (
                    len(_ev_days) if _ev_days is not None else None,
                    len(_all_days) if _all_days is not None else None,
                ))
                if _ev_days is not None and _all_days is not None and len(_ev_days) > 0 and len(_all_days) > 0:
                    _noev_days = _all_days.difference(_ev_days)
                    # Helper: group sorted DatetimeIndex into consecutive-day blocks
                    def _group_consecutive(idx: pd.DatetimeIndex):
                        if idx is None or len(idx) == 0:
                            return []
                        days_sorted = idx.sort_values().normalize()
                        blks = []
                        for d in days_sorted:
                            if not blks or d - blks[-1][1] > pd.Timedelta(days=1):
                                blks.append([d, d])
                            else:
                                blks[-1][1] = d
                        return blks

                    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
                        h = hex_color.lstrip('#')
                        if len(h) == 3:
                            h = ''.join(c * 2 for c in h)
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        return f"rgba({r},{g},{b},{alpha})"

                    ev_alpha = 0.30
                    ev_rgba = _hex_to_rgba(cp_event_color.value, ev_alpha)
                    noev_rgba = _hex_to_rgba(cp_nonevent_color.value, ev_alpha)

                    ev_blocks = _group_consecutive(_ev_days)
                    noev_blocks = _group_consecutive(_noev_days)
                    _dbg("event_bg: ev_blocks=%d noev_blocks=%d alpha=%.2f" % (len(ev_blocks), len(noev_blocks), ev_alpha))

                    # Remove previous event-bg shapes (works for both dict and Shape objects)
                    prev_shapes = list(getattr(fig.layout, 'shapes', []))
                    remaining = [s for s in prev_shapes if not str(getattr(s, 'name', '') or '').startswith('_event_bg')]
                    event_bg_shapes = []
                    for a, b in ev_blocks:
                        event_bg_shapes.append(dict(
                            type='rect', xref='x', yref='paper',
                            x0=a, x1=(b + pd.Timedelta(days=1)),
                            y0=0, y1=1, fillcolor=ev_rgba,
                            line=dict(width=0), layer='below', name='_event_bg_ev',
                        ))
                    for a, b in noev_blocks:
                        event_bg_shapes.append(dict(
                            type='rect', xref='x', yref='paper',
                            x0=a, x1=(b + pd.Timedelta(days=1)),
                            y0=0, y1=1, fillcolor=noev_rgba,
                            line=dict(width=0), layer='below', name='_event_bg_noev',
                        ))
                    fig.layout.shapes = tuple(remaining + event_bg_shapes)
                    _dbg("event_bg: total shapes set = %d" % len(fig.layout.shapes))
                    # Legend proxies for event and non-event backgrounds
                    if not any(getattr(tr, 'name', '') == 'High-flow days' for tr in fig.data):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color=ev_rgba, symbol='square'),
                            name='High-flow days', hoverinfo='skip', showlegend=True,
                        ))
                    if not any(getattr(tr, 'name', '') == 'Baseflow days' for tr in fig.data):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color=noev_rgba, symbol='square'),
                            name='Baseflow days', hoverinfo='skip', showlegend=True,
                        ))
                else:
                    _dbg("event_bg: skipped – no event/all_days data available")
            else:
                # Checkbox off – remove any leftover event-bg shapes
                prev_shapes = list(getattr(fig.layout, 'shapes', []))
                cleaned = [s for s in prev_shapes if not str(getattr(s, 'name', '') or '').startswith('_event_bg')]
                if len(cleaned) != len(prev_shapes):
                    fig.layout.shapes = tuple(cleaned)
        except Exception as _e_evbg:
            _dbg('event background build failed', str(_e_evbg))

        # Fan chart vs simplified band depending on number of runs
        # Parse band color into rgba helper
        _bc = str(band_color).lstrip('#')
        if len(_bc) == 3:
            _bc = ''.join(c * 2 for c in _bc)
        _br, _bg, _bb = int(_bc[0:2], 16), int(_bc[2:4], 16), int(_bc[4:6], 16)
        _band_alpha = float(band_alpha)
        # Derived alphas: outer band ~43% of base, inner band = base, min-max line ~71% of base, fill ~64%
        _alpha_outer = round(_band_alpha * 0.43, 3)
        _alpha_inner = round(_band_alpha, 3)
        _alpha_mm_line = round(_band_alpha * 0.71, 3)
        _alpha_mm_fill = round(_band_alpha * 0.64, 3)
        rgba = lambda a: f"rgba({_br},{_bg},{_bb},{a})"
        n_runs_here = int(arr_plot.shape[1])
        min_runs_for_bands = 5
        # Median with percentile tooltip
        def _make_customdata_multi(*arrays: Iterable[np.ndarray]) -> np.ndarray:
            # Return values per column (p05, p25, p50, p75, p95) with NaNs as None
            cols = [np.asarray(arr, dtype=float) for arr in arrays]
            n = cols[0].shape[0] if cols else 0
            m = len(cols)
            cd = np.empty((n, m), dtype=object)
            for j, col in enumerate(cols):
                finite = np.isfinite(col)
                cd[:, j] = np.where(finite, col, None)
            return cd
        _deferred_central_trace = None  # median or mean trace, added after flows/erosion
        if n_runs_here >= min_runs_for_bands:
            # Use None values for invalid points to prevent triangular fill artifacts
            x_arr = np.array(x_dt, dtype=object)
            p95 = np.asarray(q_plot[95], dtype=float); p05 = np.asarray(q_plot[5], dtype=float)
            p75 = np.asarray(q_plot[75], dtype=float); p25 = np.asarray(q_plot[25], dtype=float)
            mask90 = np.isfinite(p95) & np.isfinite(p05)
            mask50 = np.isfinite(p75) & np.isfinite(p25)
            
            # Apply masks by setting invalid values to None instead of filtering arrays
            p95_masked = np.where(mask90, p95, np.nan)
            p05_masked = np.where(mask90, p05, np.nan)
            p75_masked = np.where(mask50, p75, np.nan)
            p25_masked = np.where(mask50, p25, np.nan)
            
            if cb_show_ensemble.value:
                # 90% band (p05..p95)
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(p95_masked), mode="lines",
                    line=dict(color=rgba(_alpha_outer), width=0.5),
                    name="p95", showlegend=False, hoverinfo="skip"
                ))
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(p05_masked), mode="lines",
                    line=dict(color=rgba(_alpha_outer), width=0.5),
                    fill="tonexty", fillcolor=rgba(_alpha_outer),
                    name="p05-p95", showlegend=True, hoverinfo="skip"
                ))
                # 50% band (p25..p75)
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(p75_masked), mode="lines",
                    line=dict(color=rgba(_alpha_inner), width=0.5),
                    name="p75", showlegend=False, hoverinfo="skip"
                ))
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(p25_masked), mode="lines",
                    line=dict(color=rgba(_alpha_inner), width=0.5),
                    fill="tonexty", fillcolor=rgba(_alpha_inner),
                    name="p25-p75", showlegend=True, hoverinfo="skip"
                ))
                # Median — deferred: added after flows/erosion for correct z-order
                _deferred_central_trace = go.Scatter(
                    x=x_dt, y=_nan_to_none(q_plot[50]), mode="lines", line=dict(color="black", width=2),
                    name="median",
                    customdata=_make_customdata_multi(q_plot[5], q_plot[25], q_plot[50], q_plot[75], q_plot[95]),
                    hovertemplate=_median_hovertemplate(cb_show_names_in_tooltip.value, _run_label),
                )
        else:
            # Too few runs: show min-max envelope + mean line
            # Only compute envelope where we have sufficient data (at least 50% of runs)
            min_data_threshold = max(1, n_runs_here // 2)  # At least half the runs
            data_count = np.sum(np.isfinite(arr_plot), axis=1)  # Count finite values per time point
            sufficient_data = data_count >= min_data_threshold
            
            with np.errstate(invalid='ignore'):
                vmin = np.full(arr_plot.shape[0], np.nan)
                vmax = np.full(arr_plot.shape[0], np.nan)
                vmean = np.full(arr_plot.shape[0], np.nan)
                
                # Only compute where we have sufficient data
                if np.any(sufficient_data):
                    sufficient_indices = np.where(sufficient_data)[0]
                    vmin[sufficient_indices] = np.nanmin(arr_plot[sufficient_indices, :], axis=1)
                    vmax[sufficient_indices] = np.nanmax(arr_plot[sufficient_indices, :], axis=1)
                    vmean[sufficient_indices] = np.nanmean(arr_plot[sufficient_indices, :], axis=1)
            if cb_show_ensemble.value:
                # Max then min with fill between
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(vmax), mode="lines", line=dict(color=rgba(_alpha_mm_line), width=0.5),
                    name="max", showlegend=False, hoverinfo="skip"
                ))
                _deferred_groups["bands"].append(go.Scatter(
                    x=x_dt, y=_nan_to_none(vmin), mode="lines", line=dict(color=rgba(_alpha_mm_line), width=0.5),
                    fill="tonexty", fillcolor=rgba(_alpha_mm_fill),
                    name="min-max", showlegend=True, hoverinfo="skip"
                ))
                # Create customdata with min, mean, max and their formatted units
                mean_data = _make_customdata(vmean)
                min_data = _make_customdata(vmin)  
                max_data = _make_customdata(vmax)
                
                # Combine into 6-column customdata: [min_val, min_unit, mean_val, mean_unit, max_val, max_unit]
                combined_customdata = np.column_stack([
                    min_data[:, 0], min_data[:, 1],   # min value, min unit
                    mean_data[:, 0], mean_data[:, 1], # mean value, mean unit  
                    max_data[:, 0], max_data[:, 1]    # max value, max unit
                ])
                
                # Mean — deferred: added after flows/erosion for correct z-order
                _deferred_central_trace = go.Scatter(
                    x=x_dt, y=_nan_to_none(vmean), mode="lines", line=dict(color="black", width=2),
                    name="mean",
                    customdata=combined_customdata,
                    hovertemplate=("max: %{customdata[4]:.8g}%{customdata[5]}<br>mean: %{customdata[2]:.8g}%{customdata[3]}<br>min: %{customdata[0]:.8g}%{customdata[1]}<extra></extra>"),
                )

        # Store band data for comprehensive statistics (grouped by series)
        band_groups: Dict[str, Dict[str, pd.Series]] = {}
        ensemble_band_raw: Dict[str, pd.Series] = {}
        ensemble_band: Dict[str, pd.Series] = {}
        if arr.size:
            idx_full = aligned_df.index
            with np.errstate(invalid='ignore'):
                raw_min = np.nanmin(arr, axis=1)
                raw_max = np.nanmax(arr, axis=1)
                raw_mean = np.nanmean(arr, axis=1)
            ensemble_band_raw["min"] = pd.Series(raw_min, index=idx_full, name="min")
            ensemble_band_raw["max"] = pd.Series(raw_max, index=idx_full, name="max")
            ensemble_band_raw["mean"] = pd.Series(raw_mean, index=idx_full, name="mean")
            for perc in (5, 10, 25, 50, 60, 75, 90, 95):
                if perc in q:
                    name = f"p{perc:02d}"
                    ensemble_band_raw[name] = pd.Series(q[perc], index=idx_full, name=name)
        if n_runs_here >= min_runs_for_bands:
            # Fan chart mode: store percentile series
            # Determine if event filtering is active; relax threshold in that case
            event_filter_active = selected_days_set is not None
            if event_filter_active:
                min_data_threshold = 1  # show band wherever at least one run has data
            else:
                min_data_threshold = max(1, n_runs_here // 2)  # At least half the runs normally
            data_count = np.sum(np.isfinite(arr), axis=1)  # Count finite values per time point
            sufficient_data = data_count >= min_data_threshold

            # Create percentile arrays with NaN where insufficient data
            min_vals = np.full(arr.shape[0], np.nan)
            max_vals = np.full(arr.shape[0], np.nan)
            p05_vals = np.full(arr.shape[0], np.nan)
            p25_vals = np.full(arr.shape[0], np.nan)
            p50_vals = np.full(arr.shape[0], np.nan)
            p75_vals = np.full(arr.shape[0], np.nan)
            p95_vals = np.full(arr.shape[0], np.nan)
            mean_vals = np.full(arr.shape[0], np.nan)

            # Only compute where we have sufficient data
            if np.any(sufficient_data):
                sufficient_indices = np.where(sufficient_data)[0]
                min_vals[sufficient_data] = np.nanmin(arr[sufficient_data, :], axis=1)
                max_vals[sufficient_data] = np.nanmax(arr[sufficient_data, :], axis=1)
                p05_vals[sufficient_data] = q[5][sufficient_data]
                p25_vals[sufficient_data] = q[25][sufficient_data]
                p50_vals[sufficient_data] = q[50][sufficient_data]
                p75_vals[sufficient_data] = q[75][sufficient_data]
                p95_vals[sufficient_data] = q[95][sufficient_data]
                mean_vals[sufficient_data] = np.nanmean(arr[sufficient_data, :], axis=1)

            # Create band data series only for time points with sufficient data
            if np.any(sufficient_data):
                valid_indices = aligned_df.index[sufficient_data]
                ensemble_band["min"] = pd.Series(min_vals[sufficient_data], index=valid_indices, name="min")
                ensemble_band["max"] = pd.Series(max_vals[sufficient_data], index=valid_indices, name="max")
                ensemble_band["p05"] = pd.Series(p05_vals[sufficient_data], index=valid_indices, name="p05")
                ensemble_band["p25"] = pd.Series(p25_vals[sufficient_data], index=valid_indices, name="p25")
                ensemble_band["p50"] = pd.Series(p50_vals[sufficient_data], index=valid_indices, name="p50")
                ensemble_band["p75"] = pd.Series(p75_vals[sufficient_data], index=valid_indices, name="p75")
                ensemble_band["p95"] = pd.Series(p95_vals[sufficient_data], index=valid_indices, name="p95")
                ensemble_band["mean"] = pd.Series(mean_vals[sufficient_data], index=valid_indices, name="mean")
        else:
            # Min-max envelope mode: store min/max/mean series
            event_filter_active = selected_days_set is not None
            if event_filter_active:
                min_data_threshold = 1
            else:
                min_data_threshold = max(1, n_runs_here // 2)  # At least half the runs
            data_count = np.sum(np.isfinite(arr), axis=1)  # Count finite values per time point
            sufficient_data = data_count >= min_data_threshold

            with np.errstate(invalid='ignore'):
                min_vals = np.full(arr.shape[0], np.nan)
                max_vals = np.full(arr.shape[0], np.nan)
                mean_vals = np.full(arr.shape[0], np.nan)
                p05_vals = np.full(arr.shape[0], np.nan)
                p25_vals = np.full(arr.shape[0], np.nan)
                p50_vals = np.full(arr.shape[0], np.nan)
                p75_vals = np.full(arr.shape[0], np.nan)
                p95_vals = np.full(arr.shape[0], np.nan)

                # Only compute where we have sufficient data
                if np.any(sufficient_data):
                    sufficient_indices = np.where(sufficient_data)[0]
                    min_vals[sufficient_data] = np.nanmin(arr[sufficient_data, :], axis=1)
                    max_vals[sufficient_data] = np.nanmax(arr[sufficient_data, :], axis=1)
                    mean_vals[sufficient_data] = np.nanmean(arr[sufficient_data, :], axis=1)
                    if isinstance(q, dict):
                        if 5 in q:
                            p05_vals[sufficient_data] = q[5][sufficient_data]
                        if 25 in q:
                            p25_vals[sufficient_data] = q[25][sufficient_data]
                        if 50 in q:
                            p50_vals[sufficient_data] = q[50][sufficient_data]
                        if 75 in q:
                            p75_vals[sufficient_data] = q[75][sufficient_data]
                        if 95 in q:
                            p95_vals[sufficient_data] = q[95][sufficient_data]

                # Create band data series only for time points with sufficient data
                if np.any(sufficient_data):
                    valid_indices = aligned_df.index[sufficient_data]
                    ensemble_band["min"] = pd.Series(min_vals[sufficient_data], index=valid_indices, name="min")
                    ensemble_band["max"] = pd.Series(max_vals[sufficient_data], index=valid_indices, name="max")
                    ensemble_band["mean"] = pd.Series(mean_vals[sufficient_data], index=valid_indices, name="mean")
                    ensemble_band["p05"] = pd.Series(p05_vals[sufficient_data], index=valid_indices, name="p05")
                    ensemble_band["p25"] = pd.Series(p25_vals[sufficient_data], index=valid_indices, name="p25")
                    ensemble_band["p50"] = pd.Series(p50_vals[sufficient_data], index=valid_indices, name="p50")
                    ensemble_band["p75"] = pd.Series(p75_vals[sufficient_data], index=valid_indices, name="p75")
                    ensemble_band["p95"] = pd.Series(p95_vals[sufficient_data], index=valid_indices, name="p95")
        if ensemble_band:
            band_groups["ensemble"] = ensemble_band

        def _sanitize_series_key(name: object) -> str:
            txt = str(name) if name is not None else "series"
            cleaned = re.sub(r"[^0-9A-Za-z]+", "_", txt).strip("_")
            return cleaned.lower() or "series"

        def _extend_relative_bands(target_groups: Dict[str, Dict[str, pd.Series]], new_series: Dict[str, pd.Series], *, prefix: str = "") -> None:
            if not isinstance(target_groups, dict) or not isinstance(new_series, dict) or not new_series:
                return
            base = target_groups.get("ensemble")
            if not isinstance(base, dict) or "mean" not in base:
                return
            base_mean = base.get("mean")
            if not isinstance(base_mean, pd.Series) or base_mean.empty:
                return
            offsets: Dict[str, pd.Series] = {}
            for key, series in base.items():
                if key == "mean" or not isinstance(series, pd.Series):
                    continue
                aligned_mean = base_mean.reindex(series.index)
                offsets[key] = series - aligned_mean
            for raw_name, s_center in new_series.items():
                if not isinstance(s_center, pd.Series) or s_center.empty:
                    continue
                safe_name = _sanitize_series_key(raw_name)
                if not safe_name:
                    continue
                label = f"{prefix}_{safe_name}" if prefix else safe_name
                if label == "ensemble":
                    label = f"{label}_series"
                derived: Dict[str, pd.Series] = {"mean": s_center.copy()}
                for key, offset in offsets.items():
                    aligned_offset = offset.reindex(s_center.index)
                    derived[key] = (s_center + aligned_offset).rename(key)
                target_groups[label] = derived


        # Optional independent overlays: plot each as its own line, not part of fan
        # Also retain per-overlay resampled series for stats correlations
        _last["extra_series"] = {}
        if isinstance(extra_dfs, dict) and extra_dfs:
            extra_palette = [
                "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#1f77b4", "#d62728",
            ]
            ei = 0
            for name, df_ex in extra_dfs.items():
                try:
                    if not isinstance(df_ex, pd.DataFrame) or df_ex.empty:
                        continue
                    if reach_col not in df_ex.columns or date_col not in df_ex.columns:
                        continue
                    if isinstance(cb_extra, dict) and name in cb_extra and not cb_extra[name].value:
                        continue
                    extra_flow_col = flow_col if flow_col in df_ex.columns else _dashboard_pick_best_swat_flow_col(df_ex)
                    # Prepare sub DataFrame depending on variable (derived vs direct)
                    if var == SYN_VAR:
                        # Need components present to build derived; skip if missing
                        if not all(c in df_ex.columns for c in DERIVED_COMPONENTS):
                            continue
                        cols = [date_col] + list(DERIVED_COMPONENTS)
                    else:
                        if var not in df_ex.columns:
                            continue
                        cols = [date_col, var]
                    # Need flow for concentration
                    if (is_conc_mode or dd_method.value == "flow_weighted_mean") and extra_flow_col in df_ex.columns:
                        cols.append(extra_flow_col)
                    sub = df_ex[df_ex[reach_col] == dd_reach.value][cols].copy()
                    if sub.empty:
                        continue
                    sub = _ensure_dt_index(sub, date_col)
                    if var == SYN_VAR:
                        # Build derived column
                        with np.errstate(invalid='ignore'):
                            sub[SYN_VAR] = sub[DERIVED_COMPONENTS[0]].astype(float) + sub[DERIVED_COMPONENTS[1]].astype(float)
                    if start or end:
                        sub = _slice_time(sub, start, end)
                    if season_months:
                        sub = _filter_season(sub, season_months)
                    if sub.empty:
                        continue
                    # Build a filtered copy for stats only (event filtering affects stats, not display)
                    if selected_days_set is not None:
                        try:
                            day_mask = sub.index.floor('D').isin(list(selected_days_set))
                            sub_filt = sub.loc[day_mask]
                        except Exception:
                            sub_filt = sub
                    else:
                        sub_filt = sub
                    if is_conc_mode:
                        if extra_flow_col not in sub.columns:
                            continue
                        with np.errstate(invalid='ignore', divide='ignore'):
                            base_col = (SYN_VAR if var == SYN_VAR else var)
                            sub["__conc_mgL__"] = (sub[base_col] / (sub[extra_flow_col] * 86400.0)) * 1000.0
                        how_here = dd_method.value if dd_method.value in ("flow_weighted_mean", "mean") else "flow_weighted_mean"
                        s_ex = _resample_series(sub, "__conc_mgL__", freq=_make_freq_string(dd_freq.value, sl_bin.value), how=how_here, flow_col=extra_flow_col)
                    else:
                        base_col = (SYN_VAR if var == SYN_VAR else var)
                        s_ex = _resample_series(sub, base_col, freq=_make_freq_string(dd_freq.value, sl_bin.value), how=dd_method.value,
                                                flow_col=extra_flow_col if extra_flow_col in sub.columns else None)
                    s_ex = s_ex.dropna()
                    # Clip to main graph end (drop incomplete trailing aggregate)
                    if _plot_end_ts is not None and not s_ex.empty:
                        s_ex = s_ex.loc[s_ex.index <= _plot_end_ts]
                    if s_ex.empty:
                        continue
                    # Build filtered series for stats correlations
                    if sub_filt is not sub:
                        if is_conc_mode:
                            if extra_flow_col in sub_filt.columns:
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    base_col_f = (SYN_VAR if var == SYN_VAR else var)
                                    sub_filt["__conc_mgL__"] = (sub_filt[base_col_f] / (sub_filt[extra_flow_col] * 86400.0)) * 1000.0
                                s_ex_stats = _resample_series(sub_filt, "__conc_mgL__", freq=_make_freq_string(dd_freq.value, sl_bin.value), how=how_here, flow_col=extra_flow_col)
                            else:
                                s_ex_stats = s_ex
                        else:
                            base_col_f = (SYN_VAR if var == SYN_VAR else var)
                            s_ex_stats = _resample_series(sub_filt, base_col_f, freq=_make_freq_string(dd_freq.value, sl_bin.value), how=dd_method.value,
                                                    flow_col=extra_flow_col if extra_flow_col in sub_filt.columns else None)
                        s_ex_stats = s_ex_stats.dropna()
                    else:
                        s_ex_stats = s_ex
                    # Store filtered series for stats correlations
                    try:
                        _last["extra_series"][str(name)] = s_ex_stats if not s_ex_stats.empty else s_ex
                    except Exception:
                        pass
                    color = extra_palette[ei % len(extra_palette)]; ei += 1
                    _deferred_groups["extra"].append(go.Scatter(
                        x=_to_plotly_x(s_ex.index), y=s_ex.values, mode="lines",
                        name=str(name), line=dict(color=color, width=2),
                        customdata=_make_customdata(s_ex.values),
                        hovertemplate="%{fullData.name}: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                    ))
                except Exception:
                    # Keep plotting even if one overlay fails
                    continue

        if ensemble_band_raw:
            band_groups["ensemble_raw"] = {k: v for k, v in ensemble_band_raw.items() if isinstance(v, pd.Series)}
        _extend_relative_bands(band_groups, _last.get("extra_series", {}), prefix="extra")
        _last["band_data"] = band_groups

        # Prepare measured DataFrame (apply cleaning policies and compute kg/day when possible)
        measured_use_df = measured_df if measured_present else None
        try:
            current_nonnum_policy = str(dd_meas_nonnum.value)
        except Exception:
            current_nonnum_policy = "as_na"
        measured_nonnum_audit = _empty_measured_nonnum_audit(
            policy=current_nonnum_policy,
            mdl_mg_L=mdl_mg_L,
            mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
        )
        use_measured_load_col = None
        use_measured_conc_col = None
        if measured_present:
            # Determine a concentration column candidate
            conc_col = None
            if measured_value_col and measured_value_col in measured_df.columns:
                conc_col = str(measured_value_col)
            else:
                for cand in ["RESULTADO", "Resultado", "CONCENTRACION", "concentracion", "CONCENTRACIÓN", "concentración"]:
                    if cand in measured_df.columns:
                        conc_col = cand
                        break
            # If a kg/day column already exists, prefer it (but we may overwrite it per policy below)
            if isinstance(measured_df, pd.DataFrame) and measured_kg_col_name in measured_df.columns:
                use_measured_load_col = measured_kg_col_name
            elif "kg_per_day" in (measured_df.columns if isinstance(measured_df, pd.DataFrame) else []):
                use_measured_load_col = "kg_per_day"

            # ---------------------------------------------
            # Early application of non-numeric + negative policies
            # BEFORE any further filtering or conversion so rows aren't lost implicitly.
            # ---------------------------------------------
            try:
                policy_nonnum = str(dd_meas_nonnum.value)
                policy_neg = str(dd_meas_negative.value)
            except Exception:
                policy_nonnum = "as_na"; policy_neg = "zero"
            measured_use_df = measured_df.copy()
            # Helper inline policy applier (mirrors convert_measured_mgL_to_kg_per_day logic sans join)
            def _apply_policies_local(df_loc: pd.DataFrame, value_col: str, *, is_conc: bool) -> pd.DataFrame:
                if value_col not in df_loc.columns:
                    return df_loc
                # Coerce to numeric preserving original for policy decisions
                raw = pd.to_numeric(df_loc[value_col], errors="coerce")
                nonnum_mask = ~raw.notna()
                # Start from raw numeric (NaN where non-numeric)
                df_loc[value_col] = raw.astype(float)
                if policy_nonnum == "drop":
                    df_loc = df_loc.loc[~nonnum_mask].copy()
                elif policy_nonnum == "zero":
                    df_loc.loc[nonnum_mask, value_col] = 0.0
                elif policy_nonnum == "half_MDL" and is_conc:
                    df_loc = apply_measured_half_mdl_replacements(
                        df_loc,
                        nonnum_mask=nonnum_mask,
                        sample_value_col=value_col,
                        sample_name_col=measured_name_col,
                        mdl_mg_L=mdl_mg_L,
                        mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                    )
                # else: as_na -> leave NaN
                if is_conc:
                    # Negative policy only meaningful for concentration
                    if policy_neg == "drop":
                        df_loc = df_loc.loc[(df_loc[value_col].isna()) | (df_loc[value_col] >= 0)].copy()
                    elif policy_neg == "zero":
                        df_loc.loc[df_loc[value_col] < 0, value_col] = 0.0
                return df_loc
            # Apply to concentration column if present (preferred path for later conversion)
            if conc_col is not None:
                measured_use_df = _apply_policies_local(measured_use_df, conc_col, is_conc=True)
            # If only a load column exists, still apply non-numeric handling (treat negatives like zero policy only if set)
            elif use_measured_load_col is not None:
                measured_use_df = _apply_policies_local(measured_use_df, use_measured_load_col, is_conc=False)
            # ---------------------------------------------
            # Compute kg/day for measured using selected flow source when possible
            if conc_col is not None:
                try:
                    if str(dd_flow_source.value) == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty and isinstance(water_flow_df, pd.DataFrame):
                        flow_val_col = (
                            water_flow_value_col if water_flow_value_col and water_flow_value_col in water_flow_df.columns
                            else (flow_meas_col if flow_meas_col in (water_flow_df.columns if isinstance(water_flow_df, pd.DataFrame) else []) else None)
                        )
                        if flow_val_col is None:
                            for _c in water_flow_df.columns:
                                if pd.api.types.is_numeric_dtype(water_flow_df[_c]):
                                    flow_val_col = _c
                                    break
                        measured_use_df = convert_measured_mgL_to_kg_per_day(
                            measured_use_df,
                            water_flow_df,
                            sample_date_col=measured_date_col,
                            sample_value_col=conc_col,
                            flow_date_col=water_flow_date_col,
                            flow_value_col=str(flow_val_col),
                            kg_col=measured_kg_col_name,
                            nonnum_policy=str(dd_meas_nonnum.value),
                            negative_policy=str(dd_meas_negative.value),
                            mdl_mg_L=mdl_mg_L,
                            sample_name_col=measured_name_col,
                            mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                        )
                        use_measured_load_col = measured_kg_col_name if measured_kg_col_name in measured_use_df.columns else use_measured_load_col
                    elif str(dd_flow_source.value) == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                        # Build a minimal flow DataFrame from SWAT avg daily m3/day
                        df_flow_swat = pd.DataFrame({
                            "date": pd.to_datetime(s_swat_avg_daily.index).floor('D'),
                            "__swat_avg_m3d__": s_swat_avg_daily.values,
                        })
                        measured_use_df = convert_measured_mgL_to_kg_per_day(
                            measured_use_df,
                            df_flow_swat,
                            sample_date_col=measured_date_col,
                            sample_value_col=conc_col,
                            flow_date_col="date",
                            flow_value_col="__swat_avg_m3d__",
                            kg_col=measured_kg_col_name,
                            nonnum_policy=str(dd_meas_nonnum.value),
                            negative_policy=str(dd_meas_negative.value),
                            mdl_mg_L=mdl_mg_L,
                            sample_name_col=measured_name_col,
                            mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                        )
                        use_measured_load_col = measured_kg_col_name if measured_kg_col_name in measured_use_df.columns else use_measured_load_col
                    else:
                        # Keep the early-cleaned copy (no conversion)
                        measured_use_df = measured_use_df
                except Exception as e:
                    _dbg("measured conversion failed", e)
                    # Fall back to early-cleaned copy
                    measured_use_df = measured_use_df
            use_measured_conc_col = conc_col if conc_col in (measured_use_df.columns if isinstance(measured_use_df, pd.DataFrame) else []) else None

        # Measured overlay: per category -> per station
        if measured_present and cb_meas_on.value and isinstance(measured_use_df, pd.DataFrame) and not measured_use_df.empty:
            _meas_for_stats: List[pd.Series] = []
            _meas_half_mdl_flags: List[pd.Series] = []
            _deferred_diamond_traces: list = []  # added last for z-order
            # Color map for stations across categories (consistent colors per station)
            palette = [
                "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#1f77b4",
            ]
            station_colors: Dict[str, str] = {}
            color_idx = 0
            # Store resampled series per map per station (for later intersection)
            cat_resampled: Dict[int, Dict[str, pd.Series]] = {}
            cat_resampled_half_mdl: Dict[int, Dict[str, pd.Series]] = {}
            half_mdl_replaced_col = "__half_mdl_replaced__"
            half_mdl_style_active = (
                bool(resolved_half_mdl_observation_style.get("enabled"))
                and str(current_nonnum_policy) == "half_MDL"
            )

            # Prepare period day counts for sum-mode multiplication
            # Split measured data according to the selected event view (if any)
            measured_included_df = measured_use_df
            measured_excluded_df = None
            if selected_days_set is not None:
                try:
                    md = pd.to_datetime(measured_use_df[measured_date_col], errors='coerce').dt.floor('D')
                    keep_mask = md.isin(list(selected_days_set))
                    measured_included_df = measured_use_df.loc[keep_mask].copy()
                    measured_excluded_df = measured_use_df.loc[~keep_mask].copy()
                except Exception:
                    measured_excluded_df = None
            audit_source_df = measured_included_df.copy()
            if not audit_source_df.empty:
                audit_dates = pd.to_datetime(audit_source_df[measured_date_col], errors='coerce')
                audit_mask = audit_dates.notna()
                if start is not None:
                    audit_mask &= audit_dates >= pd.to_datetime(start)
                if end is not None:
                    audit_mask &= audit_dates <= pd.to_datetime(end)
                if season_months:
                    months = set(int(month) for month in season_months)
                    audit_mask &= audit_dates.dt.month.isin(months)
                audit_source_df = audit_source_df.loc[audit_mask].copy()
            measured_nonnum_audit = _build_measured_nonnum_audit(
                measured_df=audit_source_df,
                measured_selection=_selected_measured_state(),
                policy=current_nonnum_policy,
                mdl_mg_L=mdl_mg_L,
                mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                measured_name_col=measured_name_col,
                measured_station_col=measured_station_col,
            )
            _last["last_measured_nonnum_summary"] = _print_measured_nonnum_assignments(
                measured_nonnum_audit,
                previous_text=_last.get("last_measured_nonnum_summary"),
            )
            df_dates = measured_included_df[[measured_date_col]].copy()
            df_dates[measured_date_col] = pd.to_datetime(df_dates[measured_date_col])
            if start is not None:
                df_dates = df_dates[df_dates[measured_date_col] >= pd.to_datetime(start)]
            if end is not None:
                df_dates = df_dates[df_dates[measured_date_col] <= pd.to_datetime(end)]
            if season_months:
                months = set(int(m) for m in season_months)
                df_dates = df_dates[df_dates[measured_date_col].dt.month.isin(months)]
            if not df_dates.empty:
                days_start = df_dates[measured_date_col].min().normalize()
                days_end = df_dates[measured_date_col].max().normalize()
                period_day_counts = _period_day_counts(days_start, days_end, freq=freq_str, season_months=season_months)
            else:
                period_day_counts = pd.Series(dtype=float)

            for cat in (1, 2, 3):
                if not cb_cat[cat].value:
                    continue
                chem_name = dd_cat_name[cat].value
                stations = list(ms_cat_stations[cat].value)
                if not chem_name or not stations:
                    continue
                # Choose measured value column according to mode
                mvcol = None
                if is_conc_mode and use_measured_conc_col is not None:
                    mvcol = str(use_measured_conc_col)
                elif (not is_conc_mode) and use_measured_load_col is not None:
                    mvcol = str(use_measured_load_col)
                elif use_measured_load_col is not None:
                    mvcol = str(use_measured_load_col)
                elif use_measured_conc_col is not None:
                    mvcol = str(use_measured_conc_col)
                _dbg(f"measured map{cat}", dict(chem=chem_name, mvcol=mvcol, stations=len(stations)))
                per_station_daily = _aggregate_measured(
                    measured_included_df,
                    date_col=measured_date_col,
                    station_col=measured_station_col,
                    name_col=measured_name_col,
                    value_col=str(mvcol) if mvcol is not None else str(measured_value_col),
                    selected_name=chem_name,
                    selected_stations=stations,
                    start=start,
                    end=end,
                    season_months=season_months,
                )
                # Prepare excluded measured daily series based on event exclusion (if any)
                per_station_daily_excl_union: Dict[str, pd.Series] = {}
                if isinstance(measured_excluded_df, pd.DataFrame) and not measured_excluded_df.empty:
                    try:
                        excl_union_df = measured_excluded_df.copy()
                        per_station_daily_excl_union = _aggregate_measured(
                            excl_union_df,
                            date_col=measured_date_col,
                            station_col=measured_station_col,
                            name_col=measured_name_col,
                            value_col=str(mvcol) if mvcol is not None else str(measured_value_col),
                            selected_name=chem_name,
                            selected_stations=stations,
                            start=start,
                            end=end,
                            season_months=season_months,
                        )
                    except Exception:
                        per_station_daily_excl_union = {}
                per_station_daily_half_mdl: Dict[str, pd.Series] = {}
                if half_mdl_style_active and (half_mdl_replaced_col in measured_included_df.columns):
                    try:
                        per_station_daily_half_mdl_raw = _aggregate_measured(
                            measured_included_df,
                            date_col=measured_date_col,
                            station_col=measured_station_col,
                            name_col=measured_name_col,
                            value_col=half_mdl_replaced_col,
                            selected_name=chem_name,
                            selected_stations=stations,
                            start=start,
                            end=end,
                            season_months=season_months,
                        )
                        for st_flag, s_flag_daily in per_station_daily_half_mdl_raw.items():
                            s_flag_bool = (s_flag_daily.fillna(0.0) > 0.0).astype(bool)
                            s_flag_bool.index.name = None
                            s_flag_bool.name = st_flag
                            per_station_daily_half_mdl[st_flag] = s_flag_bool
                    except Exception:
                        per_station_daily_half_mdl = {}
                # One trace per station (collected for later intersection logic)
                for st, s_daily in per_station_daily.items():
                    _dbg_df_info(s_daily, f"measured {chem_name} st={st} daily")
                    # Resample measured series to selected bin according to method
                    if (not is_conc_mode) and dd_method.value == "sum":
                        per_mean = s_daily.resample(freq_str).mean()
                        s_aggr = per_mean
                        if not period_day_counts.empty:
                            s_aggr = (per_mean * period_day_counts)
                    elif dd_method.value == "flow_weighted_mean":
                        # No flow in measured -> fallback to mean
                        s_aggr = s_daily.resample(freq_str).mean()
                    else:
                        s_aggr = s_daily.resample(freq_str).mean()
                    s_plot = s_aggr.dropna()
                    # Clip to main graph end (drop incomplete trailing aggregate)
                    if _plot_end_ts is not None and not s_plot.empty:
                        s_plot = s_plot.loc[s_plot.index <= _plot_end_ts]
                    _dbg_df_info(s_plot, f"measured {chem_name} st={st} resampled")
                    if s_plot.empty:
                        continue
                    cat_resampled.setdefault(cat, {})[st] = s_plot
                    s_half_mdl_plot = pd.Series(False, index=s_plot.index, dtype=bool)
                    if half_mdl_style_active:
                        s_half_mdl_daily = per_station_daily_half_mdl.get(st)
                        if isinstance(s_half_mdl_daily, pd.Series) and not s_half_mdl_daily.empty:
                            s_half_mdl_daily = s_half_mdl_daily.reindex(s_daily.index).fillna(False).astype(bool)
                            s_half_mdl_plot = s_half_mdl_daily.resample(freq_str).max().reindex(s_plot.index).fillna(False).astype(bool)
                    cat_resampled_half_mdl.setdefault(cat, {})[st] = s_half_mdl_plot
                    if st not in station_colors:
                        station_colors[st] = palette[color_idx % len(palette)]
                        color_idx += 1

            # Combine across active maps per station: only keep days where all active maps have data for that station, summing map values
            active_for_intersection = [c for c in (1, 2, 3) if cb_cat[c].value and c in cat_resampled]
            if active_for_intersection:
                # Intersection of stations that exist in all active maps
                station_sets = [set(cat_resampled[c].keys()) for c in active_for_intersection if cat_resampled.get(c)]
                if station_sets:
                    common_stations = set.intersection(*station_sets) if station_sets else set()
                else:
                    common_stations = set()
                for st in sorted(common_stations):
                    series_for_station = []
                    for c in active_for_intersection:
                        s = cat_resampled[c].get(st)
                        if s is not None and not s.empty:
                            series_for_station.append(s)
                    if len(series_for_station) != len(active_for_intersection):
                        continue  # require presence in all active maps
                    # Intersect dates where all maps have values
                    intersection_idx: Optional[pd.Index] = None
                    for s in series_for_station:
                        idx = s.index
                        intersection_idx = idx if intersection_idx is None else intersection_idx.intersection(idx)
                    if intersection_idx is None or len(intersection_idx) == 0:
                        continue
                    aligned = [s.reindex(intersection_idx) for s in series_for_station]
                    df_sum = pd.concat(aligned, axis=1)
                    combined_series = df_sum.sum(axis=1, min_count=len(aligned)).dropna().sort_index()
                    if combined_series.empty:
                        continue
                    combined_half_mdl_flags = pd.Series(False, index=combined_series.index, dtype=bool)
                    if half_mdl_style_active:
                        aligned_half_mdl_flags: List[pd.Series] = []
                        for c in active_for_intersection:
                            s_flag = cat_resampled_half_mdl.get(c, {}).get(st)
                            if not isinstance(s_flag, pd.Series):
                                s_flag = pd.Series(False, index=intersection_idx, dtype=bool)
                            aligned_half_mdl_flags.append(s_flag.reindex(intersection_idx).fillna(False).astype(bool))
                        if aligned_half_mdl_flags:
                            combined_half_mdl_flags = pd.concat(aligned_half_mdl_flags, axis=1).any(axis=1).reindex(combined_series.index).fillna(False).astype(bool)
                    _meas_for_stats.append(combined_series)
                    _meas_half_mdl_flags.append(combined_half_mdl_flags)
                    label_parts = []
                    for c in active_for_intersection:
                        chem_val = dd_cat_name[c].value
                        cat_label = cat_labels.get(c, f"Map {c}")
                        if chem_val:
                            label_parts.append(f"{cat_label} [{chem_val}]")
                        else:
                            label_parts.append(cat_label)
                    base_label = " + ".join(label_parts) + f" @ {st} (intersection sum)"
                    def _append_measured_points(sel_index, color, name_suffix):
                        if sel_index is None or len(sel_index) == 0:
                            return
                        ss = combined_series.loc[sel_index]
                        if ss.empty:
                            return
                        base_marker = dict(symbol="diamond", size=11, color=color, line=dict(width=0.6, color="#333"))
                        base_name = f"{base_label} ({name_suffix})"

                        def _append_trace(series_here: pd.Series, marker_here: Dict[str, Any], trace_name: str) -> None:
                            if series_here.empty:
                                return
                            _deferred_diamond_traces.append(go.Scatter(
                                x=_to_plotly_x(series_here.index), y=series_here.values, mode="markers",
                                name=trace_name,
                                marker=marker_here,
                                customdata=_make_customdata(series_here.values),
                                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                                showlegend=True,
                            ))

                        if not half_mdl_style_active:
                            _append_trace(ss, base_marker, base_name)
                            return

                        half_mdl_mask_here = combined_half_mdl_flags.reindex(ss.index).fillna(False).astype(bool)
                        _append_trace(ss.loc[~half_mdl_mask_here], base_marker, base_name)
                        ss_half_mdl = ss.loc[half_mdl_mask_here]
                        if not ss_half_mdl.empty:
                            half_mdl_suffix = str(resolved_half_mdl_observation_style["main_chart"].get("name_suffix") or "half-MDL")
                            _append_trace(
                                ss_half_mdl,
                                _build_marker_with_overrides(base_marker, resolved_half_mdl_observation_style["main_chart"].get("marker")),
                                f"{base_label} ({name_suffix}, {half_mdl_suffix})",
                            )
                    # Color-code measured points similar to prior logic
                    try:
                        idx_days = pd.to_datetime(combined_series.index, errors='coerce').floor('D')
                        red_mask = pd.Series(False, index=combined_series.index)
                        dev_mask = pd.Series(False, index=combined_series.index)
                        if _last.get("q_df") is not None and bool(cb_flag_dev.value):
                            base = _last.get("q_df")["p90"].reindex(combined_series.index)
                            if base.isna().all() and ("p50" in _last.get("q_df").columns):
                                base = _last.get("q_df")["p50"].reindex(combined_series.index)
                            if base.isna().all() and (_last.get("aligned_df") is not None):
                                base = _last.get("aligned_df").mean(axis=1, skipna=True).reindex(combined_series.index)
                            factor = float(sl_dev_factor.value)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mvals = combined_series.to_numpy(dtype=float)
                                bvals = base.to_numpy(dtype=float)
                                denom = np.abs(bvals)
                                denom[~np.isfinite(denom) | (denom == 0.0)] = np.nan
                                ratio = np.abs(mvals) / denom
                            arr_mask = np.isfinite(ratio) & ((ratio >= factor) | (ratio <= (1.0 / factor)))
                            dev_mask = pd.Series(arr_mask, index=combined_series.index)
                        red_idx = red_mask[red_mask].index
                        orange_idx = dev_mask[~dev_mask.index.isin(red_idx) & dev_mask].index
                        green_idx = combined_series.index.difference(red_idx.union(orange_idx))
                        color_use = station_colors.get(st, "#2ca02c")
                        _append_measured_points(red_idx, "#d62728", "flow-outlier")
                        _append_measured_points(orange_idx, "#ff7f0e", "deviation")
                        _append_measured_points(green_idx, color_use, "kept")
                    except Exception:
                        _append_measured_points(combined_series.index, station_colors.get(st, "#2ca02c"), "kept")

        # Water flow overlay (from independent water_flow_df if present)
        _last["flow_series"] = None
        if cb_flow_on.value and isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
            try:
                # Ensure a valid flow column; re-detect if needed and coerce to numeric
                use_flow_col = flow_meas_col if (flow_meas_col in water_flow_df.columns) else None
                if use_flow_col is None:
                    explicit = water_flow_value_col if (water_flow_value_col and water_flow_value_col in water_flow_df.columns) else None
                    use_flow_col = _pick_best_flow_col(water_flow_df, explicit=explicit)
                if use_flow_col is None:
                    raise ValueError('No usable flow column found in water_flow_df')
                flow_df = water_flow_df[[water_flow_date_col, use_flow_col]].copy()
                flow_df[water_flow_date_col] = pd.to_datetime(flow_df[water_flow_date_col], errors='coerce')
                flow_df[use_flow_col] = pd.to_numeric(flow_df[use_flow_col], errors='coerce').astype(float)
                flow_df = flow_df.dropna(subset=[water_flow_date_col, use_flow_col])
                # Daily aggregate (sum across duplicates)
                flow_df["_date"] = flow_df[water_flow_date_col].dt.floor('D')
                s_daily = flow_df.groupby("_date")[use_flow_col].sum(min_count=1)
                s_daily.index.name = None
                # Apply time window, season, and event view filters as needed
                if start is not None:
                    s_daily = s_daily.loc[s_daily.index >= pd.to_datetime(start).floor('D')]
                if end is not None:
                    s_daily = s_daily.loc[s_daily.index <= pd.to_datetime(end).floor('D')]
                if season_months:
                    months = set(int(m) for m in season_months)
                    s_daily = s_daily.loc[s_daily.index.month.isin(months)]
                # Resample UNFILTERED daily series for display (event filtering must not affect the plot line)
                if dd_method.value == "sum":
                    s_flow = s_daily.resample(freq_str).sum(min_count=1)
                else:
                    s_flow = s_daily.resample(freq_str).mean()
                s_flow = s_flow.dropna()
                # Clip to main graph end (drop incomplete trailing period already trimmed from simulation data)
                if _plot_end_ts is not None and not s_flow.empty:
                    s_flow = s_flow.loc[s_flow.index <= _plot_end_ts]
                if not s_flow.empty:
                    _last["flow_series"] = s_flow
                    # y2 axis range
                    fmin = float(np.nanmin(s_flow.values)); fmax = float(np.nanmax(s_flow.values))
                    if fmin == fmax:
                        fmax = fmin + 1.0
                    fpad = (fmax - fmin) * 0.05
                    y2_range = [fmin - fpad, fmax + fpad]
                    _last["flow_y_range"] = y2_range
                    fig.update_layout(yaxis2=dict(
                        title="Measured water flow (m3/d)", overlaying='y', side='right', showgrid=False,
                        autorange=False, range=y2_range, title_standoff=20, automargin=True,
                        title_font_color="#1f77b4"
                    ))
                    # Single dotted blue line, with legend label requested
                    _deferred_groups["measured_flow"].append(go.Scatter(
                        x=_to_plotly_x(s_flow.index), y=s_flow.values, mode="lines",
                        name="Measured Water flow (m3/d, SAIH_corrected)", yaxis='y2',
                        line=dict(color="#1f77b4", width=1.2, dash="dot"),
                        customdata=_make_customdata(s_flow.values),
                        hovertemplate="Water flow: %{customdata[0]:.4g}%{customdata[1]} m3/d<extra></extra>",
                        visible=True,
                    ))
            except Exception:
                _last["flow_series"] = None

        # Diversion overlay (from CSV/DataFrame; plotted below zero on the flow axis)
        _last["diversion_series"] = None
        if cb_diversion_on.value and isinstance(diversion_source_df, pd.DataFrame) and not diversion_source_df.empty:
            try:
                use_diversion_col = diversion_meas_col if (diversion_meas_col in diversion_source_df.columns) else None
                if use_diversion_col is None:
                    explicit = diversion_value_col if (diversion_value_col and diversion_value_col in diversion_source_df.columns) else None
                    use_diversion_col = _pick_best_diversion_col(diversion_source_df, explicit=explicit)
                use_diversion_date_col = (
                    diversion_source_date_col if (diversion_source_date_col in diversion_source_df.columns)
                    else _pick_best_date_col(diversion_source_df, explicit=diversion_date_col)
                )
                if use_diversion_col is None or use_diversion_date_col is None:
                    raise ValueError("No usable diversion date/value columns found in diversion_df")
                diversion_plot_df = diversion_source_df[[use_diversion_date_col, use_diversion_col]].copy()
                diversion_plot_df[use_diversion_date_col] = pd.to_datetime(diversion_plot_df[use_diversion_date_col], errors='coerce')
                diversion_plot_df[use_diversion_col] = pd.to_numeric(diversion_plot_df[use_diversion_col], errors='coerce').astype(float)
                diversion_plot_df = diversion_plot_df.dropna(subset=[use_diversion_date_col, use_diversion_col])
                diversion_plot_df["_date"] = diversion_plot_df[use_diversion_date_col].dt.floor('D')
                s_daily_diversion = diversion_plot_df.groupby("_date")[use_diversion_col].sum(min_count=1)
                s_daily_diversion.index.name = None
                if start is not None:
                    s_daily_diversion = s_daily_diversion.loc[s_daily_diversion.index >= pd.to_datetime(start).floor('D')]
                if end is not None:
                    s_daily_diversion = s_daily_diversion.loc[s_daily_diversion.index <= pd.to_datetime(end).floor('D')]
                if season_months:
                    months = set(int(m) for m in season_months)
                    s_daily_diversion = s_daily_diversion.loc[s_daily_diversion.index.month.isin(months)]
                if dd_method.value == "sum":
                    s_diversion = s_daily_diversion.resample(freq_str).sum(min_count=1)
                else:
                    s_diversion = s_daily_diversion.resample(freq_str).mean()
                s_diversion = s_diversion.dropna()
                if _plot_end_ts is not None and not s_diversion.empty:
                    s_diversion = s_diversion.loc[s_diversion.index <= _plot_end_ts]
                if not s_diversion.empty:
                    s_diversion_plot = -s_diversion.abs()
                    _last["diversion_series"] = s_diversion_plot
                    _deferred_groups["measured_flow"].append(go.Scatter(
                        x=_to_plotly_x(s_diversion_plot.index), y=s_diversion_plot.values, mode="lines",
                        name="Diversion (m3/d, delta btw. SAIH corrected and uncorrected)", yaxis='y2',
                        line=dict(color="#d62728", width=1.3, dash="dash"),
                        customdata=_make_customdata(np.abs(s_diversion.values)),
                        hovertemplate="Diversion magnitude: %{customdata[0]:.4g}%{customdata[1]} m3/d (plotted below zero)<extra></extra>",
                        visible=True,
                    ))
            except Exception:
                _last["diversion_series"] = None

        # SWAT average flow overlay (from simulation DataFrames; FLOW_OUT * 86400)
        # Always compute when flow checkbox is on OR ldc_sort_by == "flow"
        _last["swat_flow_series"] = None
        _need_swat_flow = bool(cb_swat_flow_on.value) or str((ui_defaults or {}).get("ldc_sort_by", "")).lower() == "flow"
        try:
            if _need_swat_flow:
                per_sim_flow: Dict[str, pd.Series] = {}
                for sim_name, df in sim_dfs.items():
                    try:
                        fcol = _pick_best_swat_flow_col(df)
                        if not fcol:
                            continue
                        if (reach_col not in df.columns) or (date_col not in df.columns) or (fcol not in df.columns):
                            continue
                        sub = df[df[reach_col] == dd_reach.value][[date_col, fcol]].copy()
                        if sub.empty:
                            continue
                        sub = _ensure_dt_index(sub, date_col)
                        if start or end:
                            sub = _slice_time(sub, start, end)
                        if season_months:
                            sub = _filter_season(sub, season_months)
                        if sub.empty:
                            continue
                        # Convert flow to numeric and compute m3/day
                        try:
                            sub[fcol] = pd.to_numeric(sub[fcol], errors='coerce').astype(float)
                        except Exception:
                            pass
                        with np.errstate(invalid='ignore'):
                            sub["__m3day__"] = sub[fcol].astype(float) * 86400.0
                        # Daily aggregate (sum duplicates)
                        s_daily = sub["__m3day__"].groupby(sub.index.floor('D')).sum(min_count=1)
                        # Resample UNFILTERED for display
                        if dd_method.value == "sum":
                            s_res = s_daily.resample(freq_str).sum(min_count=1)
                        else:
                            s_res = s_daily.resample(freq_str).mean()
                        s_res = s_res.dropna()
                        if not s_res.empty:
                            per_sim_flow[str(sim_name)] = s_res
                    except Exception:
                        continue
                s_swat_mean = None
                if per_sim_flow:
                    try:
                        aligned = pd.concat(per_sim_flow.values(), axis=1).sort_index()
                        s_swat_mean = aligned.mean(axis=1, skipna=True)
                        s_swat_mean.name = "swat_flow_mean"
                        s_swat_mean = s_swat_mean.dropna()
                    except Exception:
                        s_swat_mean = None
                if s_swat_mean is not None and not s_swat_mean.empty:
                    # Clip to main graph end (drop incomplete trailing period)
                    if _plot_end_ts is not None:
                        s_swat_mean = s_swat_mean.loc[s_swat_mean.index <= _plot_end_ts]
                if s_swat_mean is not None and not s_swat_mean.empty:
                    _last["swat_flow_series"] = s_swat_mean
                    # Only add the trace to the main chart when the display checkbox is on
                    if bool(cb_swat_flow_on.value):
                        # Ensure y2 axis exists and add SWAT flow trace
                        fig.update_layout(yaxis2=dict(
                            title="Water flow (m3/d, SAIH_corrected)", overlaying='y', side='right', showgrid=False,
                            autorange=False, title_standoff=20, automargin=True,
                            title_font_color="#1f77b4"
                        ))
                        _deferred_groups["swat_flow"].append(go.Scatter(
                            x=_to_plotly_x(s_swat_mean.index), y=s_swat_mean.values, mode="lines",
                            name="Simulated water flow (m3/d)", yaxis='y2',
                            line=dict(color="#17becf", width=1.6, dash="solid"),
                            customdata=_make_customdata(s_swat_mean.values),
                            hovertemplate="SWAT flow: %{customdata[0]:.4g}%{customdata[1]} m3/d<extra></extra>",
                            visible=True,
                        ))
        except Exception:
            _last["swat_flow_series"] = None

        # Finalize water flow axis range to include all flow overlays present
        try:
            y2_values = []
            if isinstance(_last.get("flow_series"), pd.Series) and not _last["flow_series"].empty:
                y2_values.append(_last["flow_series"].to_numpy(dtype=float))
            if isinstance(_last.get("diversion_series"), pd.Series) and not _last["diversion_series"].empty:
                y2_values.append(_last["diversion_series"].to_numpy(dtype=float))
            if isinstance(_last.get("swat_flow_series"), pd.Series) and not _last["swat_flow_series"].empty:
                y2_values.append(_last["swat_flow_series"].to_numpy(dtype=float))
            if y2_values:
                fv = np.concatenate([v[np.isfinite(v)] for v in y2_values if v.size > 0])
                try:
                    main_rng = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                except Exception:
                    main_rng = list(_last.get('y_fixed') or [])
                y2_range = _aligned_overlay_axis_range(fv, primary_range=main_rng, default_range=(0.0, 1.0))
                _last["flow_y_range"] = y2_range
                flow_axis_title = (
                    "Water flow / diversion (m3/d)"
                    if isinstance(_last.get("diversion_series"), pd.Series) and not _last["diversion_series"].empty
                    else "Water flow (m3/d)"
                )
                fig.update_layout(yaxis2=dict(
                    title=flow_axis_title, overlaying='y', side='right', showgrid=False,
                    autorange=False, range=y2_range, title_standoff=20, automargin=True,
                    title_font_color="#1f77b4"
                ))
        except Exception:
            pass

        # Erosion overlay (mean across runs of SED_IN - SED_OUT), own scale (y3)
        try:
            _last["erosion_series"] = None
            if not bool(cb_erosion_on.value):
                raise Exception("erosion toggle off")
            # Helper to find sediment columns in a DataFrame
            def _find_sed_cols(_df: pd.DataFrame) -> Optional[Tuple[str, str]]:
                try:
                    cols = list(map(str, getattr(_df, 'columns', [])))
                except Exception:
                    return None
                if ("SED_INtons" in cols) and ("SED_OUTtons" in cols):
                    return "SED_INtons", "SED_OUTtons"
                if ("SED_IN" in cols) and ("SED_OUT" in cols):
                    return "SED_IN", "SED_OUT"
                low = {c.lower(): c for c in cols}
                if ("sed_intons" in low) and ("sed_outtons" in low):
                    return low["sed_intons"], low["sed_outtons"]
                if ("sed_in" in low) and ("sed_out" in low):
                    return low["sed_in"], low["sed_out"]
                in_cand = next((c for c in cols if ("sed_in" in c.lower() and "tons" in c.lower()) or c.lower()=="sed_in"), None)
                out_cand = next((c for c in cols if ("sed_out" in c.lower() and "tons" in c.lower()) or c.lower()=="sed_out"), None)
                if in_cand and out_cand:
                    return in_cand, out_cand
                return None
            per_sim_ero: Dict[str, pd.Series] = {}
            for sim_name, df in sim_dfs.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                pair = _find_sed_cols(df)
                if not pair:
                    continue
                in_col, out_col = pair
                sub = df[df[reach_col] == dd_reach.value][[date_col, in_col, out_col]].copy()
                if sub.empty:
                    continue
                sub = _ensure_dt_index(sub, date_col)
                if start or end:
                    sub = _slice_time(sub, start, end)
                if season_months:
                    sub = _filter_season(sub, season_months)
                if sub.empty:
                    continue
                with np.errstate(invalid='ignore'):
                    sub["__erosion__"] = sub[in_col].astype(float) - sub[out_col].astype(float)
                how_ero = dd_method.value if dd_method.value in ("sum", "mean") else "mean"
                s_er = _resample_series(sub, "__erosion__", freq=freq_str, how=how_ero)
                s_er = s_er.dropna()
                if not s_er.empty:
                    per_sim_ero[str(sim_name)] = s_er
            s_ero_mean = None
            if per_sim_ero:
                er_aligned = pd.concat(per_sim_ero.values(), axis=1).sort_index()
                s_ero_mean = er_aligned.mean(axis=1, skipna=True)
                s_ero_mean.name = "erosion_mean"
                s_ero_mean = s_ero_mean.dropna()
            if s_ero_mean is not None and not s_ero_mean.empty:
                # Clip to main graph end (drop incomplete trailing period)
                if _plot_end_ts is not None:
                    s_ero_mean = s_ero_mean.loc[s_ero_mean.index <= _plot_end_ts]
            if s_ero_mean is not None and not s_ero_mean.empty:
                _last["erosion_series"] = s_ero_mean
                # Configure y3 axis range
                ev = s_ero_mean.values[np.isfinite(s_ero_mean.values)]
                if ev.size:
                    raw_min = float(np.nanmin(ev)); raw_max = float(np.nanmax(ev))
                    if raw_min == raw_max:
                        raw_max = raw_min + 1.0
                else:
                    raw_min, raw_max = -1.0, 1.0
                # Align y3 zero to same pixel as primary y-axis zero
                main_rng = None
                try:
                    main_rng = list(fig.layout.yaxis.range) if hasattr(fig.layout, 'yaxis') and fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                except Exception:
                    main_rng = list(_last.get('y_fixed') or [])
                f = 0.5
                if main_rng and len(main_rng) == 2:
                    y0, y1 = float(main_rng[0]), float(main_rng[1])
                    if y0 < 0.0 < y1:
                        f = (0.0 - y0) / (y1 - y0)
                    elif 0.0 <= y0:
                        f = 0.0  # zero below view -> bottom
                    elif 0.0 >= y1:
                        f = 1.0  # zero above view -> top
                pos_max = max(0.0, raw_max)
                neg_min = min(0.0, raw_min)
                S_req_pos = (pos_max / (1.0 - f)) if (1.0 - f) > 1e-9 else (np.inf if pos_max > 0 else 0.0)
                S_req_neg = ((-neg_min) / f) if f > 1e-9 else (np.inf if neg_min < 0 else 0.0)
                S = max(S_req_pos, S_req_neg)
                if not np.isfinite(S) or S == 0.0:
                    S = max(abs(raw_min), abs(raw_max)) or 1.0
                S *= 1.05  # small padding
                e_min = -f * S
                e_max = (1.0 - f) * S
                y3_range = [e_min, e_max]
                fig.update_layout(yaxis3=dict(
                    title="Sedimentation tons (SED_IN - SED_OUT)", overlaying='y', side='right', showgrid=False,
                    autorange=False, range=y3_range, anchor='x', title_standoff=100, automargin=True,
                    ticklabelposition='inside', ticks='inside', title_font_color="#8c564b"
                ))
                _deferred_groups["erosion"].append(go.Scatter(
                    x=_to_plotly_x(s_ero_mean.index), y=s_ero_mean.values, mode="lines",
                    name="Erosion (SED_IN - SED_OUT)", yaxis='y3',
                    line=dict(color="#8c564b", width=1.8),
                    customdata=_make_customdata(s_ero_mean.values),
                    hovertemplate="Erosion: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                    visible=True,
                ))
                # (trace reorder handled by consolidated reorder block below)
            else:
                # Requested but couldn't find valid sediment columns across simulations
                try:
                    print("[dashboard] Warning: Erosion requested but no sediment columns found. Tried SED_INtons/SED_OUTtons and SED_IN/SED_OUT.")
                except Exception:
                    pass
        except Exception as _e:
            # No erosion overlay in this render
            try:
                if str(_e) != "erosion toggle off":
                    print(f"[dashboard] Erosion overlay skipped: {str(_e)}")
            except Exception:
                pass
            _last["erosion_series"] = None

        # Fold per-variable deferred traces into the group system
        if _deferred_central_trace is not None:
            _deferred_groups["central"].append(_deferred_central_trace)
        try:
            _deferred_groups["measured"].extend(_deferred_diamond_traces)
        except NameError:
            pass  # _deferred_diamond_traces not defined (no measured overlay)

        # Add ALL traces in the user-specified z-order (first group = bottom)
        # Use Plotly's per-trace zorder to enforce rendering order across axes.
        # All traces WITHIN a group share the same zorder so that
        # fill="tonexty" pairs stay on the same SVG layer.
        _dbg("trace_order", _active_trace_order,
             {k: len(v) for k, v in _deferred_groups.items() if v})
        _z_group = 0
        for _grp in _active_trace_order:
            _grp_traces = _deferred_groups.get(_grp, [])
            for _tr in _grp_traces:
                _tr.zorder = _z_group
                fig.add_trace(_tr)
            if _grp_traces:
                _z_group += 1
        # Safety net: add any groups not listed in trace_order
        _seen_grps = set(_active_trace_order)
        for _grp in _TRACE_ORDER_DEFAULT:
            if _grp not in _seen_grps:
                _grp_traces = _deferred_groups.get(_grp, [])
                for _tr in _grp_traces:
                    _tr.zorder = _z_group
                    fig.add_trace(_tr)
                if _grp_traces:
                    _z_group += 1

        # Save measured series for stats box
        if measured_present and cb_meas_on.value:
            _last["meas_series"] = _meas_for_stats
            _last["meas_series_half_mdl_flags"] = _meas_half_mdl_flags
        else:
            _last["meas_series"] = []
            _last["meas_series_half_mdl_flags"] = []
        _last["measured_nonnum_audit"] = measured_nonnum_audit

        # Move title below chart to avoid collision with legend
        chem_labels = []
        if measured_present:
            picks = []
            for cat in (1, 2, 3):
                # Only include chem names for active maps
                if cb_cat[cat].value:
                    val = dd_cat_name[cat].value
                    if val is not None and str(val).strip():
                        picks.append(str(val))
            if picks:
                chem_labels.append("vs. " + "; ".join(picks))
        # Use ASCII hyphen in title to avoid encoding issues
        mode_label = ("Conc mg/L" if is_conc_mode else "Load kg/day")
        title_text = f"{var} - Reach {dd_reach.value} ({freq_str}, {method}) [{mode_label}]" + ("  " + chem_labels[0] if chem_labels else "")
        slider_visible = bool(cb_range_slider.value)
        bottom_margin = int(
            main_chart_layout["bottom_margin"]["with_range_slider"]
            if slider_visible else
            main_chart_layout["bottom_margin"]["without_range_slider"]
        )
        title_y = float(
            main_chart_layout["title_annotation"]["y_with_range_slider"]
            if slider_visible else
            main_chart_layout["title_annotation"]["y_without_range_slider"]
        )
        main_plot_margin = dict(main_chart_layout["margin"])
        main_plot_margin["b"] = bottom_margin
        main_chart_update: Dict[str, Any] = {
            "title_text": None,
            "xaxis_title": None,
            "hovermode": main_chart_layout.get("hovermode") or "x unified",
            "xaxis": _build_axis_layout(
                main_chart_layout.get("xaxis"),
                base={
                    "type": main_chart_layout.get("xaxis", {}).get("type", "date"),
                    "rangeslider": _build_rangeslider_layout(main_chart_layout.get("xaxis", {}).get("rangeslider"), visible=slider_visible),
                    "tickformatstops": TICK_STOPS,
                },
            ),
            "margin": main_plot_margin,
        }
        main_hoverlabel = _build_hoverlabel_layout(main_chart_layout.get("hoverlabel"))
        if main_hoverlabel:
            main_chart_update["hoverlabel"] = main_hoverlabel
        main_legend = _build_legend_layout(main_chart_layout.get("legend"))
        if main_legend:
            main_chart_update["legend"] = main_legend
        fig.update_layout(**main_chart_update)
        main_title_annotation = main_chart_layout.get("title_annotation", {})
        main_title_font = _build_font_layout(main_title_annotation.get("font"))
        fig.add_annotation(
            x=main_title_annotation.get("x", 0.5), y=title_y,
            xref=main_title_annotation.get("xref", "paper"), yref=main_title_annotation.get("yref", "paper"),
            text=title_text,
            showarrow=False,
            xanchor=main_title_annotation.get("xanchor", "center"),
            yanchor=main_title_annotation.get("yanchor", "top"),
            font=main_title_font or {},
        )

        # Stats computation will populate the HTML block below the figure

        # fixed Y; optional live update on zoom (apply only to primary y-axis)
        y_title = ("Concentration (mg/L)" if is_conc_mode else f"{var} (kg/day)")
        _last["y_axis_title"] = y_title
        fig.update_layout(
            yaxis=_build_axis_layout(
                main_chart_layout.get("yaxis"),
                title_text=y_title,
                base={"autorange": False, "range": _last["y_fixed"]},
            )
        )

        # Update conversion statement label
        try:
            conv_lines = []
            src = str(dd_flow_source.value)
            method_label = str(dd_method.value)
            if is_conc_mode:
                if src == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                    conv_lines.append("Simulation conversion: kg/day → mg/L using external flow [m³/day]. Formula: mg/L = (kg/day ÷ m³/day) × 1000.")
                elif src == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                    conv_lines.append("Simulation conversion: kg/day → mg/L using SWAT avg FLOW_OUT × 86400 = m³/day. Formula: mg/L = (kg/day ÷ m³/day) × 1000.")
                else:
                    conv_lines.append("Simulation conversion: kg/day → mg/L using per-run FLOW_OUT × 86400 = m³/day (fallback). Formula: mg/L = (kg/day ÷ m³/day) × 1000.")
                if method_label == "flow_weighted_mean":
                    conv_lines.append("Aggregation: flow-weighted mean in mg/L using m³/day as weights.")
                else:
                    conv_lines.append("Aggregation: simple mean in mg/L.")
                conv_lines.append("Measured: displayed directly in mg/L (no conversion applied here).")
            else:
                if src == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                    conv_lines.append("Measured conversion: mg/L → kg/day using external flow [m³/day]. Formula: kg/day = mg/L × m³/day × 0.001.")
                elif src == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                    conv_lines.append("Measured conversion: mg/L → kg/day using SWAT avg FLOW_OUT × 86400 = m³/day. Formula: kg/day = mg/L × m³/day × 0.001.")
                else:
                    conv_lines.append("Measured: kg/day used as-is (no flow-based conversion available).")
                if method_label == "sum":
                    conv_lines.append("Aggregation: period sums in kg/day (measured converted daily then summed).")
                elif method_label == "flow_weighted_mean":
                    conv_lines.append("Aggregation: flow-weighted mean (where applicable); otherwise mean.")
                else:
                    conv_lines.append("Aggregation: simple mean in kg/day.")
                conv_lines.append("Simulation: series are in kg/day (no conversion).")
            lbl_units.value = "<br>".join(conv_lines)
        except Exception:
            lbl_units.value = ""
        # After locking primary y-axis, if water-flow axis exists, recompute its range to align zero
        if hasattr(fig.layout, 'yaxis2'):
            try:
                flow_vals = []
                for key in ("flow_series", "diversion_series", "swat_flow_series"):
                    s_any = _last.get(key)
                    if isinstance(s_any, pd.Series) and not s_any.empty:
                        vv = s_any.to_numpy(dtype=float)
                        vv = vv[np.isfinite(vv)]
                        if vv.size:
                            flow_vals.append(vv)
                if flow_vals:
                    main_rng = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                    y2_range = _aligned_overlay_axis_range(np.concatenate(flow_vals), primary_range=main_rng, default_range=(0.0, 1.0))
                    _last["flow_y_range"] = y2_range
                    fig.layout.yaxis2.update(autorange=False, range=y2_range)
            except Exception:
                pass
        # After locking primary y-axis, if erosion axis exists, recompute its range to align zero
        if _last.get("erosion_series") is not None and hasattr(fig.layout, 'yaxis3'):
            try:
                s_ero_mean = _last.get("erosion_series")
                ev = s_ero_mean.values[np.isfinite(s_ero_mean.values)]
                if ev.size:
                    raw_min = float(np.nanmin(ev)); raw_max = float(np.nanmax(ev))
                    if raw_min == raw_max:
                        raw_max = raw_min + 1.0
                else:
                    raw_min, raw_max = -1.0, 1.0
                main_rng = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                f = 0.5
                if main_rng and len(main_rng) == 2:
                    y0, y1 = float(main_rng[0]), float(main_rng[1])
                    if y0 < 0.0 < y1:
                        f = (0.0 - y0) / (y1 - y0)
                    elif 0.0 <= y0:
                        f = 0.0
                    elif 0.0 >= y1:
                        f = 1.0
                pos_max = max(0.0, raw_max)
                neg_min = min(0.0, raw_min)
                S_req_pos = (pos_max / (1.0 - f)) if (1.0 - f) > 1e-9 else (np.inf if pos_max > 0 else 0.0)
                S_req_neg = ((-neg_min) / f) if f > 1e-9 else (np.inf if neg_min < 0 else 0.0)
                S = max(S_req_pos, S_req_neg)
                if not np.isfinite(S) or S == 0.0:
                    S = max(abs(raw_min), abs(raw_max)) or 1.0
                S *= 1.05
                e_min = -f * S
                e_max = (1.0 - f) * S
                fig.layout.yaxis3.update(autorange=False, range=[e_min, e_max])
            except Exception:
                pass

        # Water flow trace already added above when available (no special styling)
        _last["fig"] = fig

        # Background thread to compute and update stats (HTML below the figure)
        def _compute_stats_and_update(xrange: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None):
            try:
                fig_local = _last.get("fig")
                q_local: Optional[pd.DataFrame] = _last.get("q_df")
                meas_list: List[pd.Series] = _last.get("meas_series") or []
                extras_dict = _last.get("extra_series") or {}
                if fig_local is None or q_local is None:
                    return
                # If diagnostics are not requested, clear and hide the diag box
                if not cb_show_diags.value:
                    try:
                        diag_box.children = []
                        diag_box.layout.display = "none"
                    except Exception:
                        pass
                # Determine window
                if xrange is None:
                    xr = fig_local.layout.xaxis.range
                    if xr:
                        x0 = pd.to_datetime(xr[0]); x1 = pd.to_datetime(xr[1])
                    else:
                        x0 = q_local.index.min(); x1 = q_local.index.max()
                else:
                    x0, x1 = xrange
                if x0 is None or x1 is None:
                    return
                # Compute comprehensive stats using the external module
                try:
                    stats = compute_stats_for_view(
                        q_local,
                        meas_list,
                        window=(x0, x1),
                        extras=extras_dict,
                        compute_log=bool(cb_log_metrics.value),
                        max_global_lag=int(sl_max_lag.value),
                        local_window_ks=tuple(sorted(list(sel_local_K.value))) if sel_local_K.value else (),
                        local_strategy="nearest",
                        choose_best_lag_by=str(dd_lag_metric.value),
                        band_data=_last.get("band_data", {}),
                        event_context=_last.get("event_context"),
                    )
                except Exception as e:
                    # If stats computation fails, show error in stats panel (not in the figure output)
                    _last["latest_stats_export_payload"] = None
                    btn_save_stats.disabled = True
                    lbl_save_stats.value = "<span style='color:#b94a48;'>Stats export unavailable</span>"
                    stats_html.value = f"Stats computation error: {e}"
                    return
                html_text = format_stats_text(stats)
                try:
                    view_window = (pd.Timestamp(x0), pd.Timestamp(x1))
                    dashboard_state = _collect_dashboard_state(view_window)
                    _last["latest_stats_export_payload"] = {
                        "metadata": {
                            "generated_at": datetime.utcnow().isoformat() + "Z",
                            "dashboard_version": DASHBOARD_VERSION,
                            "export_version": "stats-export-v1",
                            "run_label": _run_label or "run-unknown",
                            "source_function": "fan_compare_simulations_dashboard",
                            "filename_stem": _build_stats_filename_stem(view_window),
                            "view_window": {"x0": view_window[0], "x1": view_window[1]},
                            "dashboard_state": dashboard_state,
                            "measured_preprocessing": _last.get("measured_nonnum_audit"),
                            "stats_function": {
                                "name": "compute_stats_for_view",
                                "arguments": {
                                    "window": view_window,
                                    "compute_log": bool(cb_log_metrics.value),
                                    "max_global_lag": int(sl_max_lag.value),
                                    "local_window_ks": tuple(sorted(list(sel_local_K.value))) if sel_local_K.value else (),
                                    "local_strategy": "nearest",
                                    "choose_best_lag_by": str(dd_lag_metric.value),
                                    "has_band_data": bool(_last.get("band_data")),
                                    "has_event_context": bool(_last.get("event_context")),
                                },
                                "internal_functions_used": [
                                    "compute_stats_for_view",
                                    "format_stats_text",
                                ],
                            },
                        },
                        "stats": stats,
                    }
                    _last["latest_stats_export_path"] = None
                    btn_save_stats.disabled = False
                    lbl_save_stats.value = ""
                    _update_save_stats_tooltip()
                except Exception as export_payload_error:
                    _last["latest_stats_export_payload"] = None
                    _last["latest_stats_export_path"] = None
                    btn_save_stats.disabled = True
                    lbl_save_stats.value = f"<span style='color:#b94a48;'>Export payload error: {export_payload_error}</span>"
                    _update_save_stats_tooltip()
                try:
                    stats_html.value = html_text
                except Exception:
                    pass  # avoid printing into the figure output widget from a background thread
                # Optionally build and render diagnostics figures below
                if cb_show_diags.value:
                    # Show loading indicator in the diagnostics box
                    try:
                        diag_box.layout.display = "flex"
                        diag_box.children = [widgets.HTML("<i>⏳ Building diagnostics…</i>")]
                    except Exception:
                        pass
                    try:
                        measured_names = []
                        stations_set = set()
                        if measured_present:
                            for cat in (1, 2, 3):
                                if cb_cat[cat].value:
                                    if dd_cat_name[cat].value is not None and str(dd_cat_name[cat].value).strip():
                                        measured_names.append(str(dd_cat_name[cat].value))
                                    stations_set.update([str(s) for s in (ms_cat_stations[cat].value or [])])
                        # Build diagnostics directly from current q_df + measured (same units as main)
                        figs = build_fit_diagnostics(
                            q_local,
                            meas_list,
                            window=(x0, x1),
                            template=template,
                            title=f"Diagnostics: {dd_var.value} (Reach {dd_reach.value})",
                            lag_hist_K=int(tuple(sorted(list(sel_local_K.value)))[0]) if sel_local_K.value else 1,
                            ldc_log_scale=(bool(LDC_LOG_SCALE) if 'LDC_LOG_SCALE' in locals() else False),
                            compare_mode=("conc" if is_conc_mode else "load"),
                        )
                        # Convert to FigureWidgets for better embedding
                        children = []
                        for key in ["obs_vs_pred", "resid_hist", "resid_vs_pred", "lag_hist", "load_duration_curve"]:
                            if key in figs:
                                try:
                                    fig_here = figs[key]
                                    # Inject sediment overlay into LDC if requested
                                    if key == "load_duration_curve" and cb_ldc_sediment.value:
                                        try:
                                            sed_series = _last.get('sed_series_for_ldc')
                                            if sed_series is not None:
                                                print('[LDC-SED][debug] using stored sed_series_for_ldc length', len(sed_series))
                                            else:
                                                # Fallback to aligned_df scan (legacy path)
                                                aligned_df_any = _last.get("aligned_df")
                                                if isinstance(aligned_df_any, pd.DataFrame):
                                                    candidate_cols = [c for c in ["SED_INtons", "SED_IN", "SED_OUTtons", "SED_OUT", "SED_IN_MINUS_OUT"] if c in aligned_df_any.columns]
                                                    print("[LDC-SED][debug] fallback candidate sediment cols:", candidate_cols)
                                                    sin = next((c for c in ["SED_INtons", "SED_IN"] if c in candidate_cols), None)
                                                    sout = next((c for c in ["SED_OUTtons", "SED_OUT"] if c in candidate_cols), None)
                                                    if "SED_IN_MINUS_OUT" in candidate_cols:
                                                        sed_series = aligned_df_any["SED_IN_MINUS_OUT"].dropna()
                                                    elif sin and sout:
                                                        sed_series = (aligned_df_any[sin] - aligned_df_any[sout]).dropna()
                                                else:
                                                    print("[LDC-SED][debug] aligned_df missing or not DataFrame")
                                            if sed_series is not None and not sed_series.empty:
            					# Build exceedance curve
                                                import numpy as _np
                                                from .stats import _duration_curve_from_series as _dcfs  # type: ignore
                                                levels = _np.linspace(1.0, 99.0, 99)
                                                x_sed, y_sed = _dcfs(sed_series, levels)
                                                print("[LDC-SED][debug] first 5 exceedance pairs:", list(zip(x_sed[:5], y_sed[:5])))
                                                fig_here.update_layout(
                                                    yaxis2=dict(title="Net sediment (tons/day)", overlaying="y", side="right", showgrid=False)
                                                )
                                                fig_here.add_trace(go.Scatter(
                                                    x=x_sed,
                                                    y=y_sed,
                                                    mode="lines",
                                                    name="Net sediment",
                                                    line=dict(color="brown", width=2, dash="dot"),
                                                    yaxis="y2",
                                                ))
                                            else:
                                                print("[LDC-SED][debug] No sediment series available for overlay")
                                        except Exception as e:
                                            print("[LDC-SED][debug] exception:", e)
                                    children.append(go.FigureWidget(fig_here))
                                except Exception:
                                    children.append(widgets.HTML(f"<pre>Unable to render {key}</pre>"))
                        if children:
                            # Add a small reproducible call snippet
                            start_s = pd.to_datetime(x0).strftime('%Y-%m-%d')
                            end_s = pd.to_datetime(x1).strftime('%Y-%m-%d')
                            call_str = (
                                "from python_pipeline_scripts.stats import build_fit_diagnostics\n"
                                "# assuming you have q_df (fan quantiles) and measured_series from the dashboard context\n"
                                f"figs = build_fit_diagnostics(q_df, measured_series, window=(pd.Timestamp('{start_s}'), pd.Timestamp('{end_s}')), template='{template}', title='Diagnostics: {dd_var.value} (Reach {dd_reach.value})', lag_hist_K={int(tuple(sorted(list(sel_local_K.value)))[0]) if sel_local_K.value else 1}, ldc_log_scale={(bool(LDC_LOG_SCALE) if 'LDC_LOG_SCALE' in locals() else False)}, compare_mode=('conc' if is_conc_mode else 'load'))"
                            )
                            children.append(widgets.HTML("<b>Reproduce diagnostics</b>"))
                            children.append(widgets.HTML(f"<pre style='white-space:pre-wrap'>{call_str}</pre>"))
                            _, assignment_html = _summarize_measured_nonnum_assignments(_last.get("measured_nonnum_audit"))
                            if assignment_html:
                                children.append(widgets.HTML(assignment_html))
                            diag_box.children = children
                        else:
                            diag_box.children = [widgets.HTML("No diagnostics to display (no measured points in view)." )]
                    except Exception as e:
                        diag_box.children = [widgets.HTML(f"Diagnostics unavailable: <pre>{e}</pre>")]
            except Exception:
                # Best effort; keep UI responsive
                try:
                    stats_html.value = "Stats unavailable"
                except Exception:
                    pass

        # After the figure is rendered, kick off async stats computation for responsiveness
        import threading

        def _debounced_stats_update(view_window, delay=0.35):
            """Schedule a stats update, cancelling any pending one (debounce)."""
            import threading as _thr
            with _state["_xrange_timer_lock"]:
                old = _state.get("_xrange_timer")
                if old is not None:
                    old.cancel()
                with _state["_gen_lock"]:
                    gen = _state["_gen"]
                def _guarded():
                    if _state["_gen"] != gen:
                        return
                    _compute_stats_and_update(view_window)
                t = _thr.Timer(delay, _guarded)
                t.daemon = True
                _state["_xrange_timer"] = t
                t.start()

        def _on_xrange_change(layout, xrange):
            if _last["aligned_df"] is None:
                return
            if not cb_autoscale_y_live.value:
                fig.layout.yaxis.update(autorange=False, range=_last["y_fixed"])
                # Keep water flow axis fixed as computed
                if _last.get("flow_y_range") is not None:
                    fig.layout.yaxis2.update(autorange=False, range=_last["flow_y_range"]) if hasattr(fig.layout, 'yaxis2') else None
                # Still update stats for the new view
                try:
                    x0 = pd.to_datetime(xrange[0]); x1 = pd.to_datetime(xrange[1])
                except Exception:
                    return
                # Show loading indicator while recomputing
                try:
                    stats_html.value = "<i>⏳ Computing stats…</i>"
                except Exception:
                    pass
                if cb_show_diags.value:
                    try:
                        diag_box.layout.display = "flex"
                        diag_box.children = [widgets.HTML("<i>⏳ Building diagnostics…</i>")]
                    except Exception:
                        pass
                _debounced_stats_update((x0, x1))
                return
            try:
                x0 = pd.to_datetime(xrange[0]); x1 = pd.to_datetime(xrange[1])
            except Exception:
                return
            win = _last["aligned_df"].loc[(_last["aligned_df"].index >= x0) & (_last["aligned_df"].index <= x1)]
            if win.empty:
                return
            vals = win.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return
            ymin = float(np.nanmin(vals)); ymax = float(np.nanmax(vals))
            if ymin == ymax: ymax = ymin + 1.0
            pad_local = (ymax - ymin) * 0.05
            fig.layout.yaxis.update(autorange=False, range=[ymin - pad_local, ymax + pad_local])
            # Update water flow axis (y2) to align with visible window using all flow overlays present
            if hasattr(fig.layout, 'yaxis2'):
                vals_list = []
                for key in ("flow_series", "diversion_series", "swat_flow_series"):
                    s_any = _last.get(key)
                    if isinstance(s_any, pd.Series):
                        sf = s_any.loc[(s_any.index >= x0) & (s_any.index <= x1)]
                        if not sf.empty:
                            vv = sf.to_numpy(dtype=float)
                            vv = vv[np.isfinite(vv)]
                            if vv.size:
                                vals_list.append(vv)
                cc = np.concatenate(vals_list) if vals_list else np.array([], dtype=float)
                try:
                    main_rng = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                except Exception:
                    main_rng = list(_last.get('y_fixed') or [])
                fig.layout.yaxis2.update(
                    autorange=False,
                    range=_aligned_overlay_axis_range(cc, primary_range=main_rng, default_range=(0.0, 1.0))
                )
            # Update erosion axis (y3) to keep zero aligned with primary y zero
            if _last.get("erosion_series") is not None and hasattr(fig.layout, 'yaxis3'):
                s_er = _last["erosion_series"]
                se = s_er.loc[(s_er.index >= x0) & (s_er.index <= x1)]
                ev = se.values[np.isfinite(se.values)] if not se.empty else np.array([])
                if ev.size:
                    raw_min = float(np.nanmin(ev)); raw_max = float(np.nanmax(ev))
                    if raw_min == raw_max:
                        raw_max = raw_min + 1.0
                else:
                    raw_min, raw_max = -1.0, 1.0
                # fraction of primary y where 0 falls
                try:
                    main_rng = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else list(_last.get('y_fixed') or [])
                except Exception:
                    main_rng = list(_last.get('y_fixed') or [])
                fz = 0.5
                if main_rng and len(main_rng) == 2:
                    y0, y1 = float(main_rng[0]), float(main_rng[1])
                    if y0 < 0.0 < y1:
                        fz = (0.0 - y0) / (y1 - y0)
                    elif 0.0 <= y0:
                        fz = 0.0
                    elif 0.0 >= y1:
                        fz = 1.0
                pos_max = max(0.0, raw_max)
                neg_min = min(0.0, raw_min)
                S_req_pos = (pos_max / (1.0 - fz)) if (1.0 - fz) > 1e-9 else (np.inf if pos_max > 0 else 0.0)
                S_req_neg = ((-neg_min) / fz) if fz > 1e-9 else (np.inf if neg_min < 0 else 0.0)
                S = max(S_req_pos, S_req_neg)
                if not np.isfinite(S) or S == 0.0:
                    S = max(abs(raw_min), abs(raw_max)) or 1.0
                S *= 1.05
                e_min = -fz * S; e_max = (1.0 - fz) * S
                fig.layout.yaxis3.update(autorange=False, range=[e_min, e_max])
            # Update stats asynchronously for the new view (with loading indicator)
            try:
                stats_html.value = "<i>⏳ Computing stats…</i>"
            except Exception:
                pass
            if cb_show_diags.value:
                try:
                    diag_box.layout.display = "flex"
                    diag_box.children = [widgets.HTML("<i>⏳ Building diagnostics…</i>")]
                except Exception:
                    pass
            _debounced_stats_update((x0, x1))

        fig.layout.xaxis.on_change(_on_xrange_change, 'range')

        # Sanitize all trace data: replace NaN/Inf with None so Jupyter's JSON
        # serializer never encounters "out of range float values".  This avoids
        # the jupyter_client UserWarning that can split output streams and cause
        # intermittent duplicate/empty chart rendering artifacts.
        for _tr in fig.data:
            for _attr in ("y", "x"):
                _vals = getattr(_tr, _attr, None)
                if _vals is not None:
                    try:
                        _a = np.asarray(_vals, dtype=float)
                        if not np.all(np.isfinite(_a)):
                            setattr(_tr, _attr, np.where(np.isfinite(_a), _a, None).tolist())
                    except (TypeError, ValueError):
                        pass  # non-numeric axis (e.g. date strings) – skip
            _cd = getattr(_tr, "customdata", None)
            if _cd is not None:
                try:
                    _ca = np.asarray(_cd, dtype=float)
                    if not np.all(np.isfinite(_ca)):
                        _ca_obj = _ca.astype(object)
                        _ca_obj[~np.isfinite(_ca)] = None
                        _tr.customdata = _ca_obj.tolist()
                except (TypeError, ValueError):
                    pass  # mixed-type customdata – leave as-is

        with out:
            clear_output(wait=True)
            display(fig)

        def _async_update_duration_curves():
            # Bail early if a newer update already started
            if _state["_gen"] != _current_gen:
                return
            try:
                q_plot_local = _last.get("q_plot_df")
                # Explicitly choose flow series (prefer swat) without boolean or on Series
                flow_series_local = _last.get("swat_flow_series")
                if not (isinstance(flow_series_local, pd.Series) and not flow_series_local.empty):
                    flow_series_local = _last.get("flow_series")
                _dbg("duration context", dict(
                    has_q_plot=isinstance(q_plot_local, pd.DataFrame),
                    q_plot_shape=None if not isinstance(q_plot_local, pd.DataFrame) else q_plot_local.shape,
                    swat_flow_valid=isinstance(_last.get("swat_flow_series"), pd.Series) and not (_last.get("swat_flow_series") is None or _last.get("swat_flow_series").empty),
                    ext_flow_valid=isinstance(_last.get("flow_series"), pd.Series) and not (_last.get("flow_series") is None or _last.get("flow_series").empty)
                ))
                y_axis_title = _last.get("y_axis_title", "Value")
                # Pass measured water flow for the FDC overlay (only when flow toggle is on)
                meas_flow_local = _last.get("flow_series") if cb_flow_on.value else None
                daily_flow_local = _last.get("swat_flow_daily")
                widgets_list = _build_sim_duration_widgets(q_plot_local, flow_series_local, template, y_axis_title, measured_flow_series=meas_flow_local, daily_flow_series=daily_flow_local)
                # Check cancellation before expensive flow-strat computation
                if _state["_gen"] != _current_gen:
                    return
                # Optionally add flow-stratified curve (min/mean/max by flow exceedance) using current ensemble
                if cb_flow_strat.value:
                    try:
                        aligned_df_plot = _last.get("aligned_df_plot")
                        _dbg("flow_strat:aligned_df_plot_info", dict(
                            valid=isinstance(aligned_df_plot, pd.DataFrame) and not aligned_df_plot.empty,
                            shape=None if not isinstance(aligned_df_plot, pd.DataFrame) else aligned_df_plot.shape
                        ))
                        if isinstance(aligned_df_plot, pd.DataFrame) and not aligned_df_plot.empty:
                            # Derive per-day min/mean/max across runs
                            load_min = aligned_df_plot.min(axis=1, skipna=True)
                            load_mean = aligned_df_plot.mean(axis=1, skipna=True)
                            load_max = aligned_df_plot.max(axis=1, skipna=True)
                            # Flow candidates already daily aggregated
                            flow_ext = _last.get("flow_series")  # external aggregated
                            flow_swat = _last.get("swat_flow_series")
                            # Use event definition from _last["event_context"] if available to override percentile split
                            event_ctx = _last.get("event_context") or {}
                            idx_events = event_ctx.get("buffered_events")
                            if idx_events is None:
                                idx_events = event_ctx.get("events")
                            idx_non_events = event_ctx.get("non_events")
                            fig_fsc = None
                            bundle = None
                            _dbg("flow_strat:event_ctx_indices", dict(
                                has_events=isinstance(idx_events, pd.DatetimeIndex),
                                n_events=None if not isinstance(idx_events, pd.DatetimeIndex) else len(idx_events),
                                has_non_events=isinstance(idx_non_events, pd.DatetimeIndex),
                                n_non_events=None if not isinstance(idx_non_events, pd.DatetimeIndex) else len(idx_non_events)
                            ))
                            # Require both event and non-event indices present with minimum length
                            if (
                                isinstance(idx_events, pd.DatetimeIndex) and len(idx_events) >= 3 and
                                isinstance(idx_non_events, pd.DatetimeIndex) and len(idx_non_events) >= 3
                            ):
                                # Build regimes explicitly from event indices instead of percentile threshold
                                try:
                                    # Align flows and loads to daily index
                                    df_env = pd.DataFrame({
                                        "L_min": load_min, "L_mean": load_mean, "L_max": load_max
                                    })
                                    # choose flow source priority as in helper (swat first)
                                    flow_primary = None
                                    if isinstance(flow_swat, pd.Series) and not flow_swat.empty:
                                        flow_primary = flow_swat
                                    elif isinstance(flow_ext, pd.Series) and not flow_ext.empty:
                                        flow_primary = flow_ext
                                    if flow_primary is not None:
                                        df_env = df_env.join(flow_primary.rename("flow"), how="inner")
                                    df_env = df_env.dropna(how="any")
                                    if not df_env.empty:
                                        days = df_env.index.floor('D')
                                        reg = np.full(len(days), 'non-event', dtype=object)
                                        ev_set = set(pd.to_datetime(idx_events).floor('D').tolist())
                                        reg[[d in ev_set for d in days]] = 'event'
                                        df_env['regime'] = reg
                                        # Compute exceedance
                                        df_sorted = df_env.sort_values('flow', ascending=False).copy()
                                        N = len(df_sorted)
                                        if N >= 5:
                                            df_sorted['exceedance'] = (np.arange(N) + 1) / (N + 1.0)
                                            # Bin
                                            edges = np.arange(0.0, 1.0 + 1e-9, 0.05)
                                            if edges[-1] < 1.0:
                                                edges = np.append(edges, 1.0)
                                            bins = pd.IntervalIndex.from_breaks(edges, closed='right')
                                            df_sorted['bin'] = pd.cut(df_sorted['exceedance'], bins)
                                            rows = []
                                            for (b, rg), g in df_sorted.groupby(['bin','regime'], observed=True):
                                                if g.empty:
                                                    continue
                                                left = b.left if hasattr(b, 'left') else 0.0
                                                right = b.right if hasattr(b, 'right') else 0.0
                                                mid = 0.5 * (left + right)
                                                rows.append({
                                                    'x_mid': mid,
                                                    'regime': rg,
                                                    'y_min': float(np.nanmedian(g['L_min'])),
                                                    'y_mean': float(np.nanmedian(g['L_mean'])),
                                                    'y_max': float(np.nanmedian(g['L_max']))
                                                })
                                            agg = pd.DataFrame(rows)
                                            if not agg.empty:
                                                colors = {"event": "#d62728", "non-event": "#1f77b4", "total": "#555555"}
                                                visible_regimes = _last.get("flow_regime_visible", {"event","non-event"})
                                                total_only = bool(_last.get("flow_total_only", False))
                                                want_total_band = bool(_last.get("flow_total_band", False) or total_only)
                                                # Build optional total collapse
                                                total_agg = None
                                                if want_total_band:
                                                    mode = _last.get('flow_total_mode', 'median')
                                                    if mode == 'extents':
                                                        tmp = agg.groupby('x_mid', as_index=False).agg({
                                                            'y_min':'min','y_mean':'median','y_max':'max'
                                                        }).sort_values('x_mid')
                                                    else:
                                                        tmp = agg.groupby('x_mid', as_index=False).agg({
                                                            'y_min':'median','y_mean':'median','y_max':'median'
                                                        }).sort_values('x_mid')
                                                    total_agg = tmp
                                                fig2 = go.Figure(layout=dict(template=template))
                                                if not total_only:
                                                    for rg in ['non-event','event']:
                                                        if rg not in visible_regimes:
                                                            continue
                                                        sub = agg[agg['regime'] == rg].sort_values('x_mid')
                                                        if sub.empty:
                                                            continue
                                                        x_pct = sub['x_mid'].to_numpy()*100.0
                                                        y_up = sub['y_max'].to_numpy(); y_low = sub['y_min'].to_numpy()
                                                        fig2.add_trace(go.Scatter(x=x_pct, y=y_up, mode='lines', name=f"{rg} max", line=dict(color=colors[rg], width=0.6), showlegend=False))
                                                        if colors[rg].startswith('#') and len(colors[rg])==7:
                                                            r=int(colors[rg][1:3],16); g=int(colors[rg][3:5],16); b=int(colors[rg][5:7],16)
                                                            fill_col=f"rgba({r},{g},{b},0.20)"
                                                        else:
                                                            fill_col='rgba(0,0,0,0.15)'
                                                        fig2.add_trace(go.Scatter(x=x_pct, y=y_low, mode='lines', name=f"{rg} band", line=dict(color=colors[rg], width=0.6), fill='tonexty', fillcolor=fill_col, hoverinfo='skip', showlegend=False))
                                                        fig2.add_trace(go.Scatter(x=x_pct, y=sub['y_mean'].to_numpy(), mode='lines', name=f"{rg} mean", line=dict(color=colors[rg], width=2)))
                                                if total_agg is not None:
                                                    sub = total_agg
                                                    x_pct = sub['x_mid'].to_numpy()*100.0
                                                    y_up = sub['y_max'].to_numpy(); y_low = sub['y_min'].to_numpy()
                                                    fig2.add_trace(go.Scatter(x=x_pct, y=y_up, mode='lines', name='total max', line=dict(color=colors['total'], width=0.6), showlegend=False))
                                                    overlay_flag = bool(_last.get('flow_overlay', True))
                                                    total_only_flag = bool(_last.get('flow_total_only', False))
                                                    alpha = 0.10 if (overlay_flag and not total_only_flag) else 0.18
                                                    fig2.add_trace(go.Scatter(x=x_pct, y=y_low, mode='lines', name='total band', line=dict(color=colors['total'], width=0.6), fill='tonexty', fillcolor=f'rgba(85,85,85,{alpha})', hoverinfo='skip', showlegend=False))
                                                    fig2.add_trace(go.Scatter(x=x_pct, y=sub['y_mean'].to_numpy(), mode='lines', name='total mean', line=dict(color=colors['total'], width=3, dash='dot')))
                                                fig2.update_layout(
                                                    **_build_duration_chart_layout_update(
                                                        'flow_stratified',
                                                        title_text='Flow-stratified (event vs non-event)',
                                                        xaxis_title_text='Flow exceedance (% of time exceeded)',
                                                        yaxis_title_text=y_axis_title,
                                                    )
                                                )
                                                _apply_figure_size(fig2, duration_chart_layout)
                                                fig_fsc = go.FigureWidget(fig2)
                                                bundle = {
                                                    'source': 'event_context',
                                                    'events': int(len(idx_events)),
                                                    'non_events': int(len(idx_non_events)),
                                                    'binned_points': int(len(agg)),
                                                    'total': bool(total_agg is not None),
                                                    'total_only': total_only,
                                                    'visible_regimes': list(visible_regimes)
                                                }
                                except Exception as _e_reg_curve:
                                    _dbg('flow_strat_event_ctx_fail', str(_e_reg_curve))
                            # After attempting event-context approach, fallback if needed
                            if fig_fsc is None:
                                try:
                                    _dbg("flow_strat:fallback_attempt", dict(
                                        load_len=int(len(load_mean) if isinstance(load_mean, pd.Series) else -1),
                                        flow_ext_valid=isinstance(flow_ext, pd.Series) and not (flow_ext is None or flow_ext.empty),
                                        flow_swat_valid=isinstance(flow_swat, pd.Series) and not (flow_swat is None or flow_swat.empty)
                                    ))
                                    built = _build_flow_stratified_curve(load_min, load_mean, load_max, flow_ext, flow_swat, template_name=template)
                                    if built:
                                        fig_fsc, bundle = built
                                except Exception as _e_helper_fail:
                                    _dbg('flow_strat_curve_fail', str(_e_helper_fail))
                            if fig_fsc is not None:
                                widgets_list.append(fig_fsc)
                                _last['flow_strat_bundle'] = bundle
                            else:
                                _dbg("flow_strat:no_figure", dict(reason="no fig_fsc after attempts"))
                    except Exception as _e_fsc_outer:
                        _dbg('flow_strat_integration_fail', str(_e_fsc_outer))
                        _dbg("flow_strat:exception_context", dict(
                            aligned_valid=isinstance(_last.get("aligned_df_plot"), pd.DataFrame) and not (_last.get("aligned_df_plot") is None or _last.get("aligned_df_plot").empty),
                            event_ctx_keys=list((_last.get("event_context") or {}).keys()),
                            cb_flow_strat=cb_flow_strat.value
                        ))
                # If both load duration and flow-strat figures exist, place them side-by-side
                if widgets_list:
                    # Update flow strat state from widgets before deciding layout
                    try:
                        _last['flow_regime_visible'] = set(ms_flow_regimes.value)
                        _last['flow_total_band'] = bool(cb_flow_total_band.value)
                        _last['flow_total_only'] = bool(cb_flow_total_only.value)
                        _last['flow_overlay'] = bool(cb_flow_overlay.value)
                        _last['flow_total_mode'] = dd_flow_total_mode.value
                    except Exception:
                        pass
                    if cb_flow_strat.value and len(widgets_list) >= 2:
                        # Check cancellation before UI update
                        if _state["_gen"] != _current_gen:
                            return
                        overlay_layout = bool(cb_flow_overlay.value)
                        if overlay_layout:
                            try:
                                duration_box.children = [widgets.HBox(widgets_list, layout=widgets.Layout(justify_content='space-between'))]
                            except Exception:
                                duration_box.children = widgets_list
                        else:
                            # Vertical stacking
                            try:
                                duration_box.children = widgets_list
                            except Exception:
                                duration_box.children = widgets_list
                    else:
                        duration_box.children = widgets_list
                else:
                    duration_box.children = []
            except Exception as exc:
                try:
                    duration_box.children = [widgets.HTML(f"<i>Duration curves unavailable: {exc}</i>")]
                except Exception:
                    pass

        import threading as _dur_threading
        _dur_threading.Thread(target=_async_update_duration_curves, daemon=True).start()

        # Trigger stats computation after figure is on screen
        # (pass generation so the thread can bail if superseded)
        def _gen_guarded_stats(view_window, gen):
            if _state["_gen"] != gen:
                return
            _compute_stats_and_update(view_window)
        try:
            stats_html.value = "<i>⏳ Computing stats…</i>"
        except Exception:
            pass
        import threading
        threading.Thread(target=_gen_guarded_stats, args=(None, _current_gen), daemon=True).start()
        _release()

    # observers
    def _on_var_change(change):
        # Ensure method selection remains valid when variable changes
        new_var = change.get("new")
        default = _default_method_for_var(new_var)
        # First update options for current compare mode, then set a compatible value
        _update_method_options_for_mode()
        allowed = list(dd_method.options)
        if default not in allowed:
            # Pick a sensible fallback within allowed options
            fallback = "flow_weighted_mean" if (tg_units.value == "conc" and "flow_weighted_mean" in allowed) else allowed[0]
            dd_method.value = fallback
        else:
            dd_method.value = default
        _dbg("on_var_change", new_var)
        _dbg("on_var_change", change.get("new"))
        # Reset measured selections on variable change so per-variable defaults are re-applied.
        if measured_present:
            _state["updating"] = True
            for i in (1, 2, 3):
                dd_cat_name[i].value = None
            cb_cat[1].value = True
            cb_cat[2].value = False
            cb_cat[3].value = False
            _state["measured_defaults_var"] = None
            _state["updating"] = False
            _refresh_measured_controls()
        _mark_stale()
    dd_var.observe(_on_var_change, names="value")

    def _on_tooltip_toggle(change):
        if _last["fig"] is None:
            return
        # Update median tooltip to include percentiles and run label
        for tr in _last["fig"].data:
            if getattr(tr, 'name', '') == "median":
                tr.hovertemplate = _median_hovertemplate(change["new"], _run_label)
    cb_show_names_in_tooltip.observe(_on_tooltip_toggle, names="value")
    # Ensure median tooltip always shows percentiles according to toggle
    def _on_tooltip_toggle_fix(change):
        if _last["fig"] is None:
            return
        for tr in _last["fig"].data:
            if getattr(tr, 'name', '') == "median":
                tr.hovertemplate = _median_hovertemplate(change["new"], _run_label)
    cb_show_names_in_tooltip.observe(_on_tooltip_toggle_fix, names="value")

    # Mark stale on any change (no live recompute)
    sim_controls = [dd_var, dd_reach, dd_freq, sl_bin, dd_method, cb_autoscale_y_live, tg_units, dd_flow_source]
    def _mark_stale(*_):
        if _state.get("updating"):
            return
        # Show banner and overlay the existing figure
        try:
            lbl_reload.value = "<b>Settings changed</b>"
            reload_bar.layout.display = 'flex'
        except Exception:
            pass
        # Add grey overlay shape on current figure
        try:
            if _last.get("fig") is not None and not _state.get("stale_overlay", False):
                fig_old = _last.get("fig")
                shapes = list(fig_old.layout.shapes) if hasattr(fig_old.layout, 'shapes') and fig_old.layout.shapes else []
                shapes.append(dict(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1, fillcolor='rgba(200,200,200,0.85)', line=dict(width=0), layer='above'))
                fig_old.update_layout(shapes=shapes)
                _state["stale_overlay"] = True
        except Exception:
            pass
    for w in sim_controls:
        w.observe(_mark_stale, names="value")

    if measured_present:
        cb_meas_on.observe(_mark_stale, names="value")
        # Make water flow toggle apply immediately so the line appears/disappears on click
        def _on_flow_toggle(_):
            try:
                _compute_and_plot()
            except Exception:
                pass
        cb_flow_on.observe(_on_flow_toggle, names="value")
        cb_swat_flow_on.observe(_on_flow_toggle, names="value")
        dd_flow_source.observe(_on_flow_toggle, names="value")
        # Immediate recompute for erosion toggle as well
        def _on_erosion_toggle(_):
            try:
                _compute_and_plot()
            except Exception:
                pass
        cb_erosion_on.observe(_on_erosion_toggle, names="value")
        # Make log-metrics toggle recompute stats immediately
        def _on_log_toggle(_):
            try:
                _compute_and_plot()
            except Exception:
                pass
        cb_log_metrics.observe(_on_log_toggle, names="value")
        dd_meas_nonnum.observe(_mark_stale, names="value")
        dd_meas_negative.observe(_mark_stale, names="value")
        # Event control observers (mark stale)
        tg_event_view.observe(_mark_stale, names="value")
        sl_event_buffer_days.observe(_mark_stale, names="value")
        dd_event_threshold.observe(_mark_stale, names="value")
        tb_event_abs.observe(_mark_stale, names="value")
        fl_event_min_days.observe(_mark_stale, names="value")
        dd_event_source.observe(_mark_stale, names="value")
    else:
        # Even without measured overlay, respond to flow toggles immediately
        def _on_flow_toggle_simple(_):
            try:
                _compute_and_plot()
            except Exception:
                pass
        cb_flow_on.observe(_on_flow_toggle_simple, names="value")
        cb_swat_flow_on.observe(_on_flow_toggle_simple, names="value")
        dd_flow_source.observe(_on_flow_toggle_simple, names="value")
        if erosion_available:
            cb_erosion_on.observe(_on_flow_toggle_simple, names="value")
        # Event control observers (no measured overlay case)
        tg_event_view.observe(_mark_stale, names="value")
        sl_event_buffer_days.observe(_mark_stale, names="value")
        dd_event_threshold.observe(_mark_stale, names="value")
        tb_event_abs.observe(_mark_stale, names="value")
        fl_event_min_days.observe(_mark_stale, names="value")
        dd_event_source.observe(_mark_stale, names="value")
        cb_flag_dev.observe(_mark_stale, names="value")
        sl_dev_factor.observe(_mark_stale, names="value")
        for i in (1, 2, 3):
            cb_cat[i].observe(_mark_stale, names="value")
            # On chemical change: refresh stations for this category and redraw
            def _mk_on_name_change(ii: int):
                def _handler(_):
                    _refresh_stations_for_cat(ii)
                    _mark_stale()
                return _handler
            dd_cat_name[i].observe(_mk_on_name_change(i), names="value")
            ms_cat_stations[i].observe(_mark_stale, names="value")

    if isinstance(cb_extra, dict) and cb_extra:
        for _chk in cb_extra.values():
            _chk.observe(_mark_stale, names="value")

    def _on_range_slider_toggle(_):
        try:
            _compute_and_plot()
        except Exception:
            pass
    cb_range_slider.observe(_on_range_slider_toggle, names="value")

    def _on_ensemble_toggle(_):
        try:
            _compute_and_plot()
        except Exception:
            pass
    cb_show_ensemble.observe(_on_ensemble_toggle, names="value")

    def _on_event_bg_change(_):
        try:
            _compute_and_plot()
        except Exception:
            pass
    cb_show_event_bg.observe(_on_event_bg_change, names="value")
    cp_event_color.observe(_on_event_bg_change, names="value")
    cp_nonevent_color.observe(_on_event_bg_change, names="value")

    # Stats/dx toggles
    for _w in (dd_lag_metric, sl_max_lag, sel_local_K, cb_log_metrics, cb_show_diags, cb_ldc_sediment):
        try:
            _w.observe(lambda *_: _mark_stale(), names="value")
        except Exception:
            pass

    # Flow strat controls observers (mark stale)
    try:
        ms_flow_regimes.observe(_mark_stale, names="value")
        cb_flow_total_band.observe(_mark_stale, names="value")
        cb_flow_total_only.observe(_mark_stale, names="value")
        cb_flow_overlay.observe(_mark_stale, names="value")
        dd_flow_total_mode.observe(_mark_stale, names="value")
    except Exception:
        pass

    # Layout controls
    controls_left = widgets.VBox([num_sim, dd_var, tg_units, dd_method, dd_flow_source, lbl_units])
    base_right_children = [dd_reach, dd_freq, sl_bin, cb_autoscale_y_live, cb_show_names_in_tooltip, cb_range_slider, cb_show_ensemble,
                           cb_show_event_bg, widgets.HBox([cp_event_color, cp_nonevent_color])]

    if measured_present:
        cat_boxes = []
        for i in (1, 2, 3):
            cat_box = widgets.VBox([
                cb_cat[i],
                dd_cat_name[i],
                ms_cat_stations[i],
            ])
            cat_vbox[i] = cat_box
            cat_boxes.append(cat_box)
        # Include flow toggles if available (external and/or SWAT avg)
        flow_toggles = []
        if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
            flow_toggles.append(cb_flow_on)
        if isinstance(diversion_source_df, pd.DataFrame) and not diversion_source_df.empty:
            flow_toggles.append(cb_diversion_on)
        if swat_flow_available:
            flow_toggles.append(cb_swat_flow_on)
        flow_row = widgets.HBox(flow_toggles) if flow_toggles else widgets.HBox([])
        erosion_row = widgets.HBox([cb_erosion_on]) if erosion_available else widgets.HBox([])
        # Event configuration rows (measured present)
        event_threshold_row = widgets.HBox([dd_event_threshold, tb_event_abs])
        event_source_row = widgets.HBox([dd_event_source, fl_event_min_days])
        event_view_row = widgets.HBox([tg_event_view, sl_event_buffer_days])
        outlier_row = widgets.VBox([event_source_row, event_threshold_row, event_view_row, lbl_events_help])
        policy_row = widgets.VBox([dd_meas_nonnum, dd_meas_negative])
        deviation_row = widgets.HBox([cb_flag_dev, sl_dev_factor])
        measured_box = widgets.VBox([
            widgets.HTML("<b>Measured overlay</b>"),
            cb_meas_on,
            flow_row,
            erosion_row,
            outlier_row,
            policy_row,
            deviation_row,
            widgets.HBox(cat_boxes)
        ])
        # Extra overlays section
        if isinstance(cb_extra, dict) and cb_extra:
            extra_box = widgets.VBox([widgets.HTML("<b>Extra overlays</b>")] + list(cb_extra.values()))
            controls_right = widgets.VBox(base_right_children + [measured_box, extra_box])
        else:
            controls_right = widgets.VBox(base_right_children + [measured_box])
        _refresh_measured_controls()
    else:
        # Build flow/erosion toggles even without measured overlay
        flow_toggles = []
        if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
            flow_toggles.append(cb_flow_on)
        if isinstance(diversion_source_df, pd.DataFrame) and not diversion_source_df.empty:
            flow_toggles.append(cb_diversion_on)
        if swat_flow_available:
            flow_toggles.append(cb_swat_flow_on)
        flow_row = widgets.HBox(flow_toggles) if flow_toggles else widgets.HBox([])
        erosion_row = widgets.HBox([cb_erosion_on]) if erosion_available else widgets.HBox([])
        rows = base_right_children + ([flow_row] if flow_toggles else []) + ([erosion_row] if erosion_available else [])
        if isinstance(cb_extra, dict) and cb_extra:
            extra_box = widgets.VBox([widgets.HTML("<b>Extra overlays</b>")] + list(cb_extra.values()))
            controls_right = widgets.VBox(rows + [extra_box])
        else:
            controls_right = widgets.VBox(rows)

    # Stats controls group (small toggles)
    stats_controls = widgets.HBox([
        dd_lag_metric,
        sl_max_lag,
        sel_local_K,
        cb_log_metrics,
        cb_show_diags,
        cb_ldc_sediment,
        cb_flow_strat,
        ms_flow_regimes,
        cb_flow_total_band,
        cb_flow_total_only,
        cb_flow_overlay,
        dd_flow_total_mode,
    ])
    controls = widgets.HBox([controls_left, widgets.HBox([widgets.Label(""), controls_right])])
    stats_row = widgets.HBox([stats_panel, diag_box], layout=widgets.Layout(width="100%"))
    # Wire reload button
    def _on_reload(_):
        btn_reload.disabled = True
        try:
            lbl_reload.value = "<i>⏳ Applying…</i>"
        except Exception:
            pass
        try:
            _compute_and_plot()
        finally:
            try:
                btn_reload.disabled = False
            except Exception:
                pass

    def _on_save_stats(_):
        payload = _last.get("latest_stats_export_payload")
        if not payload:
            lbl_save_stats.value = "<span style='color:#8a6d3b;'>No computed stats to export</span>"
            _update_save_stats_tooltip()
            return
        btn_save_stats.disabled = True
        try:
            file_path = export_stats_to_json(payload, stats_export_dir)
            _last["latest_stats_export_path"] = file_path
            lbl_save_stats.value = f"<span style='color:#3c763d;'>Saved: {Path(file_path).name}</span>"
            _update_save_stats_tooltip(file_path)
        except Exception as exc:
            lbl_save_stats.value = f"<span style='color:#b94a48;'>Save failed: {exc}</span>"
            _update_save_stats_tooltip()
        finally:
            btn_save_stats.disabled = False

    btn_reload.on_click(_on_reload)
    btn_save_stats.on_click(_on_save_stats)
    display(controls, reload_bar, out, stats_controls, duration_box, stats_row)

    # Initial sync in case ui_defaults not provided or partial
    try:
        _last["flow_regime_visible"] = set(ms_flow_regimes.value)
        _last["flow_total_band"] = bool(cb_flow_total_band.value)
        _last["flow_total_only"] = bool(cb_flow_total_only.value)
        _last["flow_overlay"] = bool(cb_flow_overlay.value)
        _last["flow_total_mode"] = dd_flow_total_mode.value
    except Exception:
        pass

    _compute_and_plot()


def _dashboard_sanitize_filename_part(value: object, *, max_len: int = 32) -> str:
    text = "unknown" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        text = "unknown"
    return text[:max_len]


def _dashboard_extract_run_label(sim_dfs: Dict[str, pd.DataFrame]) -> Optional[str]:
    first_key = next(iter(sim_dfs.keys()), None)
    if first_key is None:
        return None
    try:
        match = re.search(r"run(\d+)", str(first_key), flags=re.IGNORECASE)
        if match:
            return f"run {int(match.group(1))}"
    except Exception:
        return None
    return None


def _dashboard_default_method_for_var(variable: str, how_map_defaults: Optional[Dict[str, str]] = None) -> str:
    if isinstance(how_map_defaults, dict) and variable in how_map_defaults:
        return str(how_map_defaults[variable])
    if "Conc" in variable or "mg/L" in variable:
        return "mean"
    if any(unit in str(variable).lower() for unit in ["kg", "tons", "mg"]):
        return "sum"
    return "mean"


def _dashboard_pick_best_flow_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if explicit and explicit in df.columns:
        return explicit
    patterns = [
        "water_flow_m3_d", "flow_m3_d", "flow", "caudal", "q_m3", "q", "m3_d", "m3/day", "cms", "m3s",
    ]
    cols = list(df.columns)
    num_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool]
    candidates = num_cols + [col for col in cols if col not in num_cols and col != "outliers"]
    if not candidates:
        return None

    def _score(col: str) -> tuple[int, int]:
        name = str(col).lower()
        name_score = 0
        for idx, pattern in enumerate(patterns[::-1]):
            if pattern in name:
                name_score = idx + 1
                break
        nonnull = int(df[col].notna().sum())
        return (name_score, nonnull)

    return max(candidates, key=_score)


def _dashboard_pick_best_swat_flow_col(df: pd.DataFrame) -> Optional[str]:
    try:
        cols = list(map(str, getattr(df, "columns", [])))
    except Exception:
        return None
    if not cols:
        return None
    preferred = ["FLOW_OUTcms", "FLOW_OUTcmscms", "FLOW_OUT"]
    for name in preferred:
        if name in cols:
            return name
    low = {col.lower(): col for col in cols}
    for key, original in low.items():
        if key.startswith("flow_out"):
            return original
    return None


def _dashboard_build_stats_filename_stem(
    *,
    run_label: Optional[str],
    variable: str,
    reach: object,
    frequency: str,
    bin_size: int,
    method: str,
    compare_mode: str,
    event_view: str,
    view_window: Tuple[pd.Timestamp, pd.Timestamp],
) -> str:
    x0, x1 = view_window
    window_token = "all"
    if x0 is not None and x1 is not None:
        window_token = f"{pd.Timestamp(x0).strftime('%Y%m%d')}-{pd.Timestamp(x1).strftime('%Y%m%d')}"
    parts = [
        _dashboard_sanitize_filename_part(run_label or "run-unknown", max_len=24),
        f"var-{_dashboard_sanitize_filename_part(variable, max_len=20)}",
        f"reach-{_dashboard_sanitize_filename_part(reach, max_len=8)}",
        f"freq-{_dashboard_sanitize_filename_part(_make_freq_string(frequency, bin_size), max_len=12)}",
        f"method-{_dashboard_sanitize_filename_part(method, max_len=16)}",
        f"mode-{_dashboard_sanitize_filename_part(compare_mode, max_len=8)}",
        f"view-{_dashboard_sanitize_filename_part(event_view, max_len=12)}",
        window_token,
    ]
    return "_".join(parts)


def _build_dashboard_stats_export_payload(
    *,
    stats: Dict[str, Any],
    view_window: Tuple[pd.Timestamp, pd.Timestamp],
    dashboard_state: Dict[str, Any],
    measured_preprocessing: Optional[Dict[str, Any]],
    run_label: Optional[str],
    filename_stem: str,
    source_function: str,
    compute_log: bool,
    max_global_lag: int,
    local_window_ks: Sequence[int],
    choose_best_lag_by: str,
    has_band_data: bool,
    has_event_context: bool,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "dashboard_version": DASHBOARD_VERSION,
            "export_version": "stats-export-v1",
            "run_label": run_label or "run-unknown",
            "source_function": source_function,
            "filename_stem": filename_stem,
            "view_window": {"x0": view_window[0], "x1": view_window[1]},
            "dashboard_state": dashboard_state,
            "measured_preprocessing": measured_preprocessing,
            "stats_function": {
                "name": "compute_stats_for_view",
                "arguments": {
                    "window": view_window,
                    "compute_log": bool(compute_log),
                    "max_global_lag": int(max_global_lag),
                    "local_window_ks": tuple(sorted(int(value) for value in local_window_ks)) if local_window_ks else (),
                    "local_strategy": "nearest",
                    "choose_best_lag_by": str(choose_best_lag_by),
                    "has_band_data": bool(has_band_data),
                    "has_event_context": bool(has_event_context),
                },
                "internal_functions_used": [
                    "compute_stats_for_view",
                    "format_stats_text",
                ],
            },
        },
        "stats": stats,
    }


def _infer_headless_measured_selection_defaults(
    *,
    variable: str,
    reach: Optional[int],
    measured_var_map: Optional[Dict[str, object]],
    measured_df: Optional[pd.DataFrame],
    measured_name_col: str,
    measured_station_col: str,
) -> Dict[str, Dict[str, Any]]:
    defaults: Dict[str, Dict[str, Any]] = {
        str(cat): {"enabled": False, "chemical": None, "stations": []}
        for cat in (1, 2, 3)
    }
    if not isinstance(measured_df, pd.DataFrame) or measured_df.empty:
        return defaults
    if measured_name_col not in measured_df.columns or measured_station_col not in measured_df.columns:
        return defaults

    auto_defaults = _get_auto_measured_defaults_for_variable(variable, reach=reach)
    norm_map = _normalize_meas_map_for_var(measured_var_map or {}, variable)
    for cat, spec in auto_defaults.items():
        if not norm_map.get(cat) and spec.get("preferred_chemicals"):
            norm_map[cat] = list(spec["preferred_chemicals"])
    chem_to_stations: Dict[str, List[str]] = {}
    try:
        for chem_name, group in measured_df.groupby(measured_name_col, dropna=True):
            stations = (
                group[measured_station_col]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            chem_to_stations[str(chem_name)] = sorted(stations)
    except Exception:
        chem_to_stations = {}

    for cat in (1, 2, 3):
        allowed = norm_map.get(cat, [])
        default_spec = auto_defaults.get(cat, {})
        options = _measured_options_for_category(measured_df, measured_name_col, allowed)
        if not options:
            continue
        chemical = _pick_preferred_measured_option(options, default_spec.get("preferred_chemicals"))
        if chemical is None:
            continue
        stations = list(chem_to_stations.get(str(chemical), []))
        if not stations:
            continue
        preferred_station = default_spec.get("preferred_station")
        enabled = bool(default_spec.get("enabled")) if ("enabled" in default_spec) else bool(cat == 1)
        if not enabled:
            selected_stations = []
        elif preferred_station and (preferred_station in stations):
            selected_stations = [preferred_station]
        else:
            selected_stations = stations
        defaults[str(cat)] = {
            "enabled": enabled,
            "chemical": chemical,
            "stations": selected_stations,
        }
    return defaults


def _normalize_headless_dashboard_config(
    dashboard_config: Optional[Dict[str, Any]],
    *,
    variables: List[str],
    reach_choices: List[int],
    how_map_defaults: Optional[Dict[str, str]],
    water_flow_df: Optional[pd.DataFrame],
    measured_present: bool,
    measured_var_map: Optional[Dict[str, object]],
    measured_df: Optional[pd.DataFrame],
    measured_name_col: str,
    measured_station_col: str,
    extra_dfs: Optional[Dict[str, pd.DataFrame]],
    reach_override: Optional[int],
) -> Dict[str, Any]:
    raw = dict(dashboard_config or {})
    aliases = {
        "freq": "frequency",
        "measured_on": "show_measured",
        "flow_on": "show_water_flow",
        "diversion_on": "show_diversion",
        "swat_flow_on": "show_swat_flow",
        "erosion_on": "show_erosion",
        "show_diags": "show_diagnostics",
        "local_Ks": "local_window_ks",
        "meas_negative_policy": "measured_negative_policy",
        "meas_nonnum_policy": "measured_nonnum_policy",
        "extra_visible": "extra_overlays",
    }
    for old_key, new_key in aliases.items():
        if old_key in raw and new_key not in raw:
            raw[new_key] = raw[old_key]

    if "cats" in raw and "measured_selection" not in raw and isinstance(raw["cats"], dict):
        measured_selection: Dict[str, Dict[str, Any]] = {}
        for cat in (1, 2, 3):
            cat_raw = raw["cats"].get(cat) or raw["cats"].get(str(cat)) or {}
            measured_selection[str(cat)] = {
                "enabled": bool(cat_raw.get("enabled", cat == 1)),
                "chemical": cat_raw.get("chem"),
                "stations": list(cat_raw.get("stations") or []),
            }
        raw["measured_selection"] = measured_selection

    default_reach = 13 if 13 in reach_choices else (reach_choices[0] if reach_choices else None)
    effective_reach = reach_override if reach_override is not None else raw.get("reach", default_reach)
    try:
        effective_reach = int(effective_reach) if effective_reach is not None else default_reach
    except Exception:
        effective_reach = default_reach
    if effective_reach not in reach_choices:
        effective_reach = default_reach
    default_var = str(raw.get("variable") or (variables[0] if variables else ""))
    should_infer_measured_selection = measured_present and ("measured_selection" not in raw) and ("cats" not in raw)
    inferred_measured_selection = (
        _infer_headless_measured_selection_defaults(
            variable=default_var,
            reach=effective_reach,
            measured_var_map=measured_var_map,
            measured_df=measured_df,
            measured_name_col=measured_name_col,
            measured_station_col=measured_station_col,
        )
        if should_infer_measured_selection else {}
    )
    flow_source_default = "swat_avg" if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty else "external"
    event_source_default = "external" if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty else "swat_avg"
    extra_defaults = {str(name): True for name in (extra_dfs or {}).keys()}
    normalized: Dict[str, Any] = {
        "variable": default_var,
        "reach": effective_reach,
        "frequency": raw.get("frequency", "D"),
        "bin": int(raw.get("bin", 1)),
        "method": raw.get("method") or _dashboard_default_method_for_var(default_var, how_map_defaults),
        "compare_mode": raw.get("compare_mode", "load"),
        "flow_source": raw.get("flow_source", flow_source_default),
        "event_source": raw.get("event_source", event_source_default),
        "event_threshold": raw.get("event_threshold", "p95"),
        "event_abs_value": raw.get("event_abs_value", np.nan),
        "event_min_days": float(raw.get("event_min_days", 1.0)),
        "event_buffer_days": int(raw.get("event_buffer_days", 1)),
        "event_view": raw.get("event_view", "all"),
        "show_measured": bool(raw.get("show_measured", measured_present)),
        "show_water_flow": bool(raw.get("show_water_flow", isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty)),
        "show_diversion": bool(raw.get("show_diversion", False)),
        "show_swat_flow": bool(raw.get("show_swat_flow", False)),
        "show_erosion": bool(raw.get("show_erosion", False)),
        "show_diagnostics": bool(raw.get("show_diagnostics", True)),
        "show_ensemble": bool(raw.get("show_ensemble", True)),
        "show_event_bg": bool(raw.get("show_event_bg", False)),
        "event_bg_color": str(raw.get("event_bg_color", "#fdd0a2")),
        "nonevent_bg_color": str(raw.get("nonevent_bg_color", "#c6dbef")),
        "lag_metric": raw.get("lag_metric", "r"),
        "max_lag": int(raw.get("max_lag", 2)),
        "local_window_ks": list(raw.get("local_window_ks", [1, 2])),
        "log_metrics": bool(raw.get("log_metrics", True)),
        "measured_nonnum_policy": raw.get("measured_nonnum_policy", "as_na"),
        "measured_negative_policy": raw.get("measured_negative_policy", "zero"),
        "mdl_mg_L": float(raw.get("mdl_mg_L", 0.2)),
        "mdl_mg_L_by_name": normalize_measured_mdl_by_name(raw.get("mdl_mg_L_by_name")),
        "flag_deviations": bool(raw.get("flag_deviations", True)),
        "deviation_factor": float(raw.get("deviation_factor", 10.0)),
        "start": raw.get("start"),
        "end": raw.get("end"),
        "season_months": raw.get("season_months"),
        "view_window": raw.get("view_window") or {"x0": None, "x1": None},
        "measured_selection": raw.get("measured_selection") or inferred_measured_selection,
        "extra_overlays": raw.get("extra_overlays") or extra_defaults,
        "autoscale_y_live": bool(raw.get("autoscale_y_live", True)),
        "range_slider": bool(raw.get("range_slider", True)),
        "show_names_in_tooltip": bool(raw.get("show_names_in_tooltip", False)),
        "flow_regimes": list(raw.get("flow_regimes", ["event", "non-event"])),
        "flow_total_band": bool(raw.get("flow_total_band", False)),
        "flow_total_only": bool(raw.get("flow_total_only", False)),
        "flow_overlay": bool(raw.get("flow_overlay", True)),
        "flow_total_mode": raw.get("flow_total_mode", "median"),
    }

    try:
        normalized["reach"] = int(normalized["reach"]) if normalized["reach"] is not None else default_reach
    except Exception:
        normalized["reach"] = default_reach
    if normalized["reach"] not in reach_choices:
        normalized["reach"] = default_reach

    normalized["frequency"] = str(normalized["frequency"] or "D").upper()
    if normalized["frequency"] not in {"D", "W", "M", "A"}:
        normalized["frequency"] = "D"
    normalized["compare_mode"] = "conc" if str(normalized["compare_mode"]).lower() == "conc" else "load"
    method = str(normalized["method"] or "mean")
    if normalized["compare_mode"] == "conc":
        if method not in {"flow_weighted_mean", "mean"}:
            method = "mean"
    elif method not in {"sum", "mean", "flow_weighted_mean"}:
        method = _dashboard_default_method_for_var(normalized["variable"], how_map_defaults)
    normalized["method"] = method
    normalized["event_view"] = str(normalized["event_view"] or "all")
    if normalized["event_view"] == "non_events":
        normalized["event_view"] = "non_events"
    elif normalized["event_view"] not in {"all", "events"}:
        normalized["event_view"] = "all"
    normalized["event_source"] = str(normalized["event_source"] or event_source_default)
    if normalized["event_source"] not in {"external", "swat_avg"}:
        normalized["event_source"] = event_source_default
    normalized["flow_source"] = str(normalized["flow_source"] or flow_source_default)
    measured_selection = normalized.get("measured_selection") or {}
    normalized_selection: Dict[str, Dict[str, Any]] = {}
    for cat in (1, 2, 3):
        meta = measured_selection.get(str(cat)) or measured_selection.get(cat) or {}
        normalized_selection[str(cat)] = {
            "enabled": bool(meta.get("enabled", False)),
            "chemical": meta.get("chemical"),
            "stations": list(meta.get("stations") or []),
        }
    normalized["measured_selection"] = normalized_selection
    extra_map = dict(extra_defaults)
    extra_map.update({str(name): bool(value) for name, value in (normalized.get("extra_overlays") or {}).items()})
    normalized["extra_overlays"] = extra_map
    if normalized.get("season_months") is not None:
        try:
            normalized["season_months"] = [int(value) for value in normalized["season_months"]]
        except Exception:
            normalized["season_months"] = None
    return normalized


def export_dashboard_stats_from_config(
    sim_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    *,
    dashboard_config: Dict[str, Any],
    extra_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    measured_df: Optional[pd.DataFrame] = None,
    measured_var_map: Optional[Dict[str, object]] = None,
    water_flow_df: Optional[pd.DataFrame] = None,
    stats_export_dir: Optional[Union[str, Path]] = None,
    reach_col: str = "RCH",
    date_col: str = "date",
    flow_col: str = "FLOW_OUTcms",
    measured_date_col: str = "F_MUESTREO",
    measured_station_col: str = "est_estaci",
    measured_name_col: str = "NOMBRE",
    measured_value_col: Optional[str] = None,
    measured_kg_col_name: str = "kg_per_day",
    water_flow_date_col: str = "date",
    water_flow_value_col: Optional[str] = None,
    how_map_defaults: Optional[Dict[str, str]] = None,
    mdl_mg_L: float = 0.2,
    mdl_mg_L_by_name: Optional[Dict[str, float]] = None,
    debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    if not sim_dfs:
        raise ValueError("sim_dfs must not be empty")
    if not variables:
        raise ValueError("variables must not be empty")

    measured_present = measured_df is not None and isinstance(measured_df, pd.DataFrame) and not measured_df.empty
    all_reaches = set()
    for df in sim_dfs.values():
        if isinstance(df, pd.DataFrame) and reach_col in df.columns:
            all_reaches.update(df[reach_col].dropna().unique().tolist())
    reach_choices = sorted(int(value) for value in all_reaches if pd.notna(value))
    if not reach_choices:
        raise ValueError("No reaches found in sim_dfs")

    cfg = _normalize_headless_dashboard_config(
        dashboard_config,
        variables=variables,
        reach_choices=reach_choices,
        how_map_defaults=how_map_defaults,
        water_flow_df=water_flow_df,
        measured_present=measured_present,
        measured_var_map=measured_var_map,
        measured_df=measured_df,
        measured_name_col=measured_name_col,
        measured_station_col=measured_station_col,
        extra_dfs=extra_dfs,
        reach_override=dashboard_config.get("reach") if isinstance(dashboard_config, dict) and "reach" in dashboard_config else None,
    )
    reach = int(cfg["reach"])
    variable = str(cfg["variable"])
    freq_str = _make_freq_string(cfg["frequency"], int(cfg["bin"]))
    method = str(cfg["method"])
    is_conc_mode = str(cfg["compare_mode"]) == "conc"
    season_months = cfg.get("season_months")
    run_label = _dashboard_extract_run_label(sim_dfs)
    normalized_mdl_mg_L_by_name = normalize_measured_mdl_by_name(
        cfg.get("mdl_mg_L_by_name") if cfg.get("mdl_mg_L_by_name") is not None else mdl_mg_L_by_name
    )

    def _dbg(*args: Any) -> None:
        if debug:
            try:
                print("[dash-headless]", *args)
            except Exception:
                pass

    if stats_export_dir is None:
        stats_export_dir = Path(__file__).resolve().parent.parent / "config" / "outputs" / "dashboard_stats"
    else:
        stats_export_dir = Path(stats_export_dir)

    measured_load_col: Optional[str] = None
    measured_conc_col: Optional[str] = None
    if measured_present:
        if measured_value_col and measured_value_col in measured_df.columns:
            if str(measured_value_col).strip().lower() in {"resultado", "result", "concentracion", "concentración", "concentration", "mg/l", "mg_l"}:
                measured_conc_col = measured_value_col
            else:
                measured_load_col = measured_value_col
        if measured_load_col is None and "kg_per_day" in measured_df.columns:
            measured_load_col = "kg_per_day"
        for candidate in ["RESULTADO", "Resultado", "CONCENTRACION", "concentracion", "CONCENTRACIÓN", "concentración"]:
            if measured_conc_col is None and candidate in measured_df.columns:
                measured_conc_col = candidate
                break
        if measured_load_col is None and measured_conc_col is None:
            detected = _detect_value_col(measured_df)
            measured_load_col = detected
        if measured_load_col is None and measured_conc_col is None:
            raise ValueError("Unable to detect measured value column. Please pass measured_value_col.")

    flow_meas_col: Optional[str] = None
    if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
        if water_flow_value_col and water_flow_value_col in water_flow_df.columns:
            flow_meas_col = water_flow_value_col
        else:
            for candidate in water_flow_df.columns:
                if "water_flow_m3_d" in str(candidate).lower():
                    flow_meas_col = str(candidate)
                    break
            if flow_meas_col is None:
                for candidate in water_flow_df.columns:
                    if pd.api.types.is_numeric_dtype(water_flow_df[candidate]):
                        flow_meas_col = str(candidate)
                        break

    s_external_flow_daily: Optional[pd.Series] = None
    if isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty:
        try:
            use_flow_col = None
            if water_flow_value_col and water_flow_value_col in water_flow_df.columns:
                use_flow_col = water_flow_value_col
            elif flow_meas_col and flow_meas_col in water_flow_df.columns:
                use_flow_col = flow_meas_col
            else:
                use_flow_col = _dashboard_pick_best_flow_col(water_flow_df, explicit=None)
            if use_flow_col:
                fdf = water_flow_df[[water_flow_date_col, use_flow_col]].copy()
                fdf[water_flow_date_col] = pd.to_datetime(fdf[water_flow_date_col], errors="coerce").dt.floor("D")
                fdf[use_flow_col] = pd.to_numeric(fdf[use_flow_col], errors="coerce").astype(float)
                fdf = fdf.dropna(subset=[water_flow_date_col, use_flow_col])
                s_external_flow_daily = fdf.groupby(water_flow_date_col)[use_flow_col].sum(min_count=1)
                if cfg.get("start") is not None:
                    s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index >= pd.to_datetime(cfg["start"]).floor("D")]
                if cfg.get("end") is not None:
                    s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index <= pd.to_datetime(cfg["end"]).floor("D")]
                if season_months:
                    months = set(int(month) for month in season_months)
                    s_external_flow_daily = s_external_flow_daily.loc[s_external_flow_daily.index.month.isin(months)]
                s_external_flow_daily.index.name = None
        except Exception as exc:
            _dbg("external flow prep failed", exc)
            s_external_flow_daily = None

    s_swat_avg_daily: Optional[pd.Series] = None
    try:
        per_sim_daily: Dict[str, pd.Series] = {}
        for sim_name, df in sim_dfs.items():
            fcol = _dashboard_pick_best_swat_flow_col(df)
            if not fcol or reach_col not in df.columns or date_col not in df.columns or fcol not in df.columns:
                continue
            subf = df[df[reach_col] == reach][[date_col, fcol]].copy()
            if subf.empty:
                continue
            subf = _ensure_dt_index(subf, date_col)
            if cfg.get("start") or cfg.get("end"):
                subf = _slice_time(subf, cfg.get("start"), cfg.get("end"))
            if season_months:
                subf = _filter_season(subf, season_months)
            if subf.empty:
                continue
            subf[fcol] = pd.to_numeric(subf[fcol], errors="coerce").astype(float)
            with np.errstate(invalid="ignore"):
                subf["__m3day__"] = subf[fcol].astype(float) * 86400.0
            s_day = subf["__m3day__"].groupby(subf.index.floor("D")).sum(min_count=1).dropna()
            if not s_day.empty:
                per_sim_daily[str(sim_name)] = s_day
        if per_sim_daily:
            aligned = pd.concat(per_sim_daily.values(), axis=1).sort_index()
            s_swat_avg_daily = aligned.mean(axis=1, skipna=True).dropna()
            s_swat_avg_daily.index.name = None
    except Exception as exc:
        _dbg("swat flow prep failed", exc)
        s_swat_avg_daily = None

    event_mode = str(cfg.get("event_view") or "all")
    selected_days_set = None
    event_day_set = None
    buffered_event_days = None
    full_days_set = None
    try:
        if event_mode in {"events", "non_events", "all"}:
            ev_source = str(cfg.get("event_source"))
            if ev_source == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                s_events_flow = s_external_flow_daily.copy()
            elif ev_source == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                s_events_flow = s_swat_avg_daily.copy()
            else:
                s_events_flow = None
            if s_events_flow is not None and not s_events_flow.empty:
                df_ev = pd.DataFrame({"date": pd.to_datetime(s_events_flow.index).floor("D"), "Q": s_events_flow.values})
                token = str(cfg.get("event_threshold") or "p95")
                if token == "abs":
                    abs_value = cfg.get("event_abs_value")
                    thr_def = float(abs_value) if isinstance(abs_value, (int, float)) and not np.isnan(abs_value) else None
                else:
                    thr_def = token
                if thr_def is not None:
                    from .dashboard_helper import add_event_flags

                    df_flags = add_event_flags(
                        df_ev,
                        thresholds={"main": thr_def},
                        intervals={"main": float(cfg.get("event_min_days", 1.0))},
                        time_col="date",
                        flow_col="Q",
                    )
                    if "main_event" in df_flags.columns:
                        event_days = pd.to_datetime(df_flags.loc[df_flags["main_event"], :].index).floor("D").unique()
                        event_day_set = set(pd.to_datetime(event_days).tolist())
                        full_days_set = set(pd.to_datetime(df_ev["date"]).unique().tolist())
                        buf = int(cfg.get("event_buffer_days", 1))
                        buffered_event_days = set()
                        for day in event_day_set:
                            d0 = pd.Timestamp(day).normalize()
                            for offset in range(-buf, buf + 1):
                                buffered_event_days.add(d0 + pd.Timedelta(days=int(offset)))
                        if event_mode == "events":
                            selected_days_set = set(buffered_event_days)
                        elif event_mode == "non_events" and full_days_set is not None:
                            selected_days_set = set(full_days_set) - set(buffered_event_days)
                        elif event_mode == "all":
                            selected_days_set = set(full_days_set)
    except Exception as exc:
        _dbg("event detection failed", exc)
        selected_days_set = None

    def _event_index(values: Optional[Sequence[pd.Timestamp]]) -> Optional[pd.DatetimeIndex]:
        if values is None:
            return None
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(list(values))).floor("D").unique().sort_values()
            return idx
        except Exception:
            return None

    idx_events = _event_index(event_day_set)
    idx_buffered = _event_index(buffered_event_days)
    if idx_buffered is None:
        idx_buffered = idx_events
    idx_all_days = _event_index(full_days_set)
    idx_selected = _event_index(selected_days_set)
    if idx_all_days is not None and idx_buffered is not None:
        idx_non_events = idx_all_days.difference(idx_buffered)
        if idx_non_events.empty:
            idx_non_events = None
    else:
        idx_non_events = None
    event_context = {
        "mode": event_mode,
        "events": idx_events,
        "buffered_events": idx_buffered,
        "non_events": idx_non_events,
        "selected": idx_selected,
        "all_days": idx_all_days,
    }

    syn_var = "KJELDAHL_OUTkg"
    derived_components = ("ORGN_OUTkg", "NH4_OUTkg")
    per_sim: Dict[str, pd.Series] = {}
    _raw_daily_end_hl: Optional[pd.Timestamp] = None  # track pre-resample daily range
    for sim_name, df in sim_dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if variable == syn_var:
            if not all(component in df.columns for component in derived_components):
                continue
            sub_cols = [date_col] + list(derived_components)
        else:
            if variable not in df.columns:
                continue
            sub_cols = [date_col, variable]
        per_run_flow_col = _dashboard_pick_best_swat_flow_col(df)
        if (is_conc_mode or method == "flow_weighted_mean") and per_run_flow_col in df.columns:
            sub_cols.append(per_run_flow_col)
        sub = df[df[reach_col] == reach][sub_cols].copy()
        if sub.empty:
            continue
        if variable == syn_var:
            with np.errstate(invalid="ignore"):
                sub[syn_var] = sub[derived_components[0]].astype(float) + sub[derived_components[1]].astype(float)
        sub = _ensure_dt_index(sub, date_col)
        if cfg.get("start") or cfg.get("end"):
            sub = _slice_time(sub, cfg.get("start"), cfg.get("end"))
        if season_months:
            sub = _filter_season(sub, season_months)
        # Track raw daily end before event filtering for incomplete-period detection
        try:
            _sp_max_hl = pd.Timestamp(sub.index.max()).normalize()
            if _raw_daily_end_hl is None or _sp_max_hl > _raw_daily_end_hl:
                _raw_daily_end_hl = _sp_max_hl
        except Exception:
            pass
        if selected_days_set is not None:
            mask = sub.index.floor("D").isin(list(selected_days_set))
            sub = sub.loc[mask]
        if sub.empty:
            continue
        if is_conc_mode:
            base_col = syn_var if variable == syn_var else variable
            flow_source = str(cfg.get("flow_source") or "external")
            if flow_source == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty:
                days = sub.index.floor("D")
                f_series = s_external_flow_daily.reindex(days)
                fvals = f_series.to_numpy(dtype=float)
                kgd = sub[base_col].to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sub["__conc_mgL__"] = (kgd / fvals) * 1000.0
                sub["__flow_m3d__"] = fvals
            elif flow_source == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                days = sub.index.floor("D")
                f_series = s_swat_avg_daily.reindex(days)
                fvals = f_series.to_numpy(dtype=float)
                kgd = sub[base_col].to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sub["__conc_mgL__"] = (kgd / fvals) * 1000.0
                sub["__flow_m3d__"] = fvals
            else:
                if per_run_flow_col not in sub.columns:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    sub["__conc_mgL__"] = (sub[base_col] / (sub[per_run_flow_col] * 86400.0)) * 1000.0
                    sub["__flow_m3d__"] = sub[per_run_flow_col].astype(float) * 86400.0
            how_here = method if method in {"flow_weighted_mean", "mean"} else "mean"
            s = _resample_series(sub, "__conc_mgL__", freq=freq_str, how=how_here, flow_col="__flow_m3d__")
        else:
            base_col = syn_var if variable == syn_var else variable
            s = _resample_series(sub, base_col, freq=freq_str, how=method, flow_col=per_run_flow_col if per_run_flow_col in sub.columns else None)
        if s.empty:
            continue
        s.name = str(sim_name)
        per_sim[str(sim_name)] = s

    if not per_sim:
        raise ValueError(f"No data for reach {reach} and variable '{variable}'")

    aligned_df = pd.concat(per_sim.values(), axis=1).sort_index()
    aligned_df.index = pd.to_datetime(aligned_df.index, utc=False)

    # Drop incomplete last aggregate period (non-daily only), same logic as interactive dashboard
    _is_daily_freq_hl = freq_str.upper().endswith('D') and int(''.join(c for c in freq_str if c.isdigit()) or '1') == 1
    if not _is_daily_freq_hl and _raw_daily_end_hl is not None and len(aligned_df) > 1:
        try:
            _last_label_hl = pd.Timestamp(aligned_df.index[-1]).normalize()
            if _raw_daily_end_hl < _last_label_hl:
                aligned_df = aligned_df.iloc[:-1]
        except Exception:
            pass

    arr = aligned_df.to_numpy(dtype=float)
    if arr.shape[1] == 0:
        raise ValueError("No aligned data after resampling")
    percs = [5, 10, 25, 50, 60, 75, 90, 95]
    qs = np.nanpercentile(arr, percs, axis=1)
    q = {pct: qs[idx, :] for idx, pct in enumerate(percs)}
    q_df = pd.DataFrame(
        {
            "min": np.nanmin(arr, axis=1),
            "p05": q[5],
            "p10": q[10],
            "p25": q[25],
            "p50": q[50],
            "p60": q[60],
            "p75": q[75],
            "p90": q[90],
            "p95": q[95],
            "max": np.nanmax(arr, axis=1),
        },
        index=aligned_df.index,
    )

    band_groups: Dict[str, Dict[str, pd.Series]] = {}
    ensemble_band_raw: Dict[str, pd.Series] = {}
    with np.errstate(invalid="ignore"):
        ensemble_band_raw["min"] = pd.Series(np.nanmin(arr, axis=1), index=aligned_df.index, name="min")
        ensemble_band_raw["max"] = pd.Series(np.nanmax(arr, axis=1), index=aligned_df.index, name="max")
        ensemble_band_raw["mean"] = pd.Series(np.nanmean(arr, axis=1), index=aligned_df.index, name="mean")
    for pct in (5, 10, 25, 50, 60, 75, 90, 95):
        ensemble_band_raw[f"p{pct:02d}"] = pd.Series(q[pct], index=aligned_df.index, name=f"p{pct:02d}")
    band_groups["ensemble_raw"] = ensemble_band_raw

    n_runs_here = int(arr.shape[1])
    min_runs_for_bands = 5
    event_filter_active = selected_days_set is not None
    min_data_threshold = 1 if event_filter_active else max(1, n_runs_here // 2)
    data_count = np.sum(np.isfinite(arr), axis=1)
    sufficient_data = data_count >= min_data_threshold
    ensemble_band: Dict[str, pd.Series] = {}
    valid_indices = aligned_df.index[sufficient_data]
    if len(valid_indices) > 0:
        with np.errstate(invalid="ignore"):
            min_vals = np.nanmin(arr[sufficient_data, :], axis=1)
            max_vals = np.nanmax(arr[sufficient_data, :], axis=1)
            mean_vals = np.nanmean(arr[sufficient_data, :], axis=1)
        ensemble_band["min"] = pd.Series(min_vals, index=valid_indices, name="min")
        ensemble_band["max"] = pd.Series(max_vals, index=valid_indices, name="max")
        ensemble_band["mean"] = pd.Series(mean_vals, index=valid_indices, name="mean")
        for pct in (5, 25, 50, 75, 95):
            ensemble_band[f"p{pct:02d}"] = pd.Series(q[pct][sufficient_data], index=valid_indices, name=f"p{pct:02d}")
    if ensemble_band:
        band_groups["ensemble"] = ensemble_band

    def _sanitize_series_key(name: object) -> str:
        txt = str(name) if name is not None else "series"
        cleaned = re.sub(r"[^0-9A-Za-z]+", "_", txt).strip("_")
        return cleaned.lower() or "series"

    def _extend_relative_bands(target_groups: Dict[str, Dict[str, pd.Series]], new_series: Dict[str, pd.Series], *, prefix: str = "") -> None:
        base = target_groups.get("ensemble")
        if not isinstance(base, dict) or "mean" not in base:
            return
        base_mean = base.get("mean")
        if not isinstance(base_mean, pd.Series) or base_mean.empty:
            return
        offsets: Dict[str, pd.Series] = {}
        for key, series in base.items():
            if key == "mean" or not isinstance(series, pd.Series):
                continue
            offsets[key] = series - base_mean.reindex(series.index)
        for raw_name, center_series in new_series.items():
            if not isinstance(center_series, pd.Series) or center_series.empty:
                continue
            safe_name = _sanitize_series_key(raw_name)
            label = f"{prefix}_{safe_name}" if prefix else safe_name
            if label == "ensemble":
                label = f"{label}_series"
            derived: Dict[str, pd.Series] = {"mean": center_series.copy()}
            for key, offset in offsets.items():
                derived[key] = (center_series + offset.reindex(center_series.index)).rename(key)
            target_groups[label] = derived

    extra_series: Dict[str, pd.Series] = {}
    if isinstance(extra_dfs, dict) and extra_dfs:
        for name, df_ex in extra_dfs.items():
            try:
                if not isinstance(df_ex, pd.DataFrame) or df_ex.empty:
                    continue
                if reach_col not in df_ex.columns or date_col not in df_ex.columns:
                    continue
                if not bool(cfg.get("extra_overlays", {}).get(str(name), True)):
                    continue
                extra_flow_col = flow_col if flow_col in df_ex.columns else _dashboard_pick_best_swat_flow_col(df_ex)
                if variable == syn_var:
                    if not all(component in df_ex.columns for component in derived_components):
                        continue
                    cols = [date_col] + list(derived_components)
                else:
                    if variable not in df_ex.columns:
                        continue
                    cols = [date_col, variable]
                if (is_conc_mode or method == "flow_weighted_mean") and extra_flow_col in df_ex.columns:
                    cols.append(extra_flow_col)
                sub = df_ex[df_ex[reach_col] == reach][cols].copy()
                if sub.empty:
                    continue
                sub = _ensure_dt_index(sub, date_col)
                if variable == syn_var:
                    with np.errstate(invalid="ignore"):
                        sub[syn_var] = sub[derived_components[0]].astype(float) + sub[derived_components[1]].astype(float)
                if cfg.get("start") or cfg.get("end"):
                    sub = _slice_time(sub, cfg.get("start"), cfg.get("end"))
                if season_months:
                    sub = _filter_season(sub, season_months)
                if selected_days_set is not None:
                    sub = sub.loc[sub.index.floor("D").isin(list(selected_days_set))]
                if sub.empty:
                    continue
                if is_conc_mode:
                    if extra_flow_col not in sub.columns:
                        continue
                    with np.errstate(invalid="ignore", divide="ignore"):
                        base_col = syn_var if variable == syn_var else variable
                        sub["__conc_mgL__"] = (sub[base_col] / (sub[extra_flow_col] * 86400.0)) * 1000.0
                    how_here = method if method in {"flow_weighted_mean", "mean"} else "flow_weighted_mean"
                    s_ex = _resample_series(sub, "__conc_mgL__", freq=freq_str, how=how_here, flow_col=extra_flow_col)
                else:
                    base_col = syn_var if variable == syn_var else variable
                    s_ex = _resample_series(sub, base_col, freq=freq_str, how=method, flow_col=extra_flow_col if extra_flow_col in sub.columns else None)
                s_ex = s_ex.dropna()
                if not s_ex.empty:
                    extra_series[str(name)] = s_ex
            except Exception as exc:
                _dbg("extra overlay failed", name, exc)
                continue
    _extend_relative_bands(band_groups, extra_series, prefix="extra")

    measured_use_df = measured_df.copy() if measured_present else None
    use_measured_load_col = measured_load_col
    use_measured_conc_col = measured_conc_col
    measured_nonnum_audit = _empty_measured_nonnum_audit(
        policy=str(cfg.get("measured_nonnum_policy", "as_na")),
        mdl_mg_L=float(cfg.get("mdl_mg_L", mdl_mg_L)),
        mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
    )
    if measured_present and isinstance(measured_use_df, pd.DataFrame):
        conc_col = use_measured_conc_col
        policy_nonnum = str(cfg.get("measured_nonnum_policy", "as_na"))
        policy_neg = str(cfg.get("measured_negative_policy", "zero"))
        headless_mdl = float(cfg.get("mdl_mg_L", mdl_mg_L))

        def _apply_policies_local(df_loc: pd.DataFrame, value_col_name: str, *, is_conc: bool) -> pd.DataFrame:
            if value_col_name not in df_loc.columns:
                return df_loc
            raw_numeric = pd.to_numeric(df_loc[value_col_name], errors="coerce")
            nonnum_mask = ~raw_numeric.notna()
            df_loc[value_col_name] = raw_numeric.astype(float)
            if policy_nonnum == "drop":
                df_loc = df_loc.loc[~nonnum_mask].copy()
            elif policy_nonnum == "zero":
                df_loc.loc[nonnum_mask, value_col_name] = 0.0
            elif policy_nonnum == "half_MDL" and is_conc:
                df_loc = apply_measured_half_mdl_replacements(
                    df_loc,
                    nonnum_mask=nonnum_mask,
                    sample_value_col=value_col_name,
                    sample_name_col=measured_name_col,
                    mdl_mg_L=headless_mdl,
                    mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                )
            if is_conc:
                if policy_neg == "drop":
                    df_loc = df_loc.loc[(df_loc[value_col_name].isna()) | (df_loc[value_col_name] >= 0)].copy()
                elif policy_neg == "zero":
                    df_loc.loc[df_loc[value_col_name] < 0, value_col_name] = 0.0
            return df_loc

        if conc_col is not None:
            measured_use_df = _apply_policies_local(measured_use_df, conc_col, is_conc=True)
        elif use_measured_load_col is not None:
            measured_use_df = _apply_policies_local(measured_use_df, use_measured_load_col, is_conc=False)

        if conc_col is not None:
            try:
                if str(cfg.get("flow_source")) == "external" and isinstance(s_external_flow_daily, pd.Series) and not s_external_flow_daily.empty and isinstance(water_flow_df, pd.DataFrame):
                    flow_val_col = water_flow_value_col if water_flow_value_col and water_flow_value_col in water_flow_df.columns else (flow_meas_col if flow_meas_col in water_flow_df.columns else None)
                    if flow_val_col is None:
                        for candidate in water_flow_df.columns:
                            if pd.api.types.is_numeric_dtype(water_flow_df[candidate]):
                                flow_val_col = str(candidate)
                                break
                    measured_use_df = convert_measured_mgL_to_kg_per_day(
                        measured_use_df,
                        water_flow_df,
                        sample_date_col=measured_date_col,
                        sample_value_col=conc_col,
                        flow_date_col=water_flow_date_col,
                        flow_value_col=str(flow_val_col),
                        kg_col=measured_kg_col_name,
                        nonnum_policy=policy_nonnum,
                        negative_policy=policy_neg,
                        mdl_mg_L=headless_mdl,
                        sample_name_col=measured_name_col,
                        mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                    )
                    if measured_kg_col_name in measured_use_df.columns:
                        use_measured_load_col = measured_kg_col_name
                elif str(cfg.get("flow_source")) == "swat_avg" and isinstance(s_swat_avg_daily, pd.Series) and not s_swat_avg_daily.empty:
                    df_flow_swat = pd.DataFrame({
                        "date": pd.to_datetime(s_swat_avg_daily.index).floor("D"),
                        "__swat_avg_m3d__": s_swat_avg_daily.values,
                    })
                    measured_use_df = convert_measured_mgL_to_kg_per_day(
                        measured_use_df,
                        df_flow_swat,
                        sample_date_col=measured_date_col,
                        sample_value_col=conc_col,
                        flow_date_col="date",
                        flow_value_col="__swat_avg_m3d__",
                        kg_col=measured_kg_col_name,
                        nonnum_policy=policy_nonnum,
                        negative_policy=policy_neg,
                        mdl_mg_L=headless_mdl,
                        sample_name_col=measured_name_col,
                        mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
                    )
                    if measured_kg_col_name in measured_use_df.columns:
                        use_measured_load_col = measured_kg_col_name
            except Exception as exc:
                _dbg("measured conversion failed", exc)
        if isinstance(measured_use_df, pd.DataFrame):
            if conc_col in measured_use_df.columns:
                use_measured_conc_col = conc_col

    meas_for_stats: List[pd.Series] = []
    if measured_present and bool(cfg.get("show_measured")) and isinstance(measured_use_df, pd.DataFrame) and not measured_use_df.empty:
        measured_included_df = measured_use_df
        if selected_days_set is not None:
            md = pd.to_datetime(measured_use_df[measured_date_col], errors="coerce").dt.floor("D")
            measured_included_df = measured_use_df.loc[md.isin(list(selected_days_set))].copy()
        audit_source_df = measured_included_df.copy()
        if not audit_source_df.empty:
            audit_dates = pd.to_datetime(audit_source_df[measured_date_col], errors="coerce")
            audit_mask = audit_dates.notna()
            if cfg.get("start") is not None:
                audit_mask &= audit_dates >= pd.to_datetime(cfg["start"])
            if cfg.get("end") is not None:
                audit_mask &= audit_dates <= pd.to_datetime(cfg["end"])
            if season_months:
                months = set(int(month) for month in season_months)
                audit_mask &= audit_dates.dt.month.isin(months)
            audit_source_df = audit_source_df.loc[audit_mask].copy()
        measured_nonnum_audit = _build_measured_nonnum_audit(
            measured_df=audit_source_df,
            measured_selection=cfg.get("measured_selection") or {},
            policy=str(cfg.get("measured_nonnum_policy", "as_na")),
            mdl_mg_L=float(cfg.get("mdl_mg_L", mdl_mg_L)),
            mdl_mg_L_by_name=normalized_mdl_mg_L_by_name,
            measured_name_col=measured_name_col,
            measured_station_col=measured_station_col,
        )
        _print_measured_nonnum_assignments(measured_nonnum_audit, prefix="[dash-headless][half_MDL]")
        df_dates = measured_included_df[[measured_date_col]].copy()
        df_dates[measured_date_col] = pd.to_datetime(df_dates[measured_date_col])
        if cfg.get("start") is not None:
            df_dates = df_dates[df_dates[measured_date_col] >= pd.to_datetime(cfg["start"])]
        if cfg.get("end") is not None:
            df_dates = df_dates[df_dates[measured_date_col] <= pd.to_datetime(cfg["end"])]
        if season_months:
            months = set(int(month) for month in season_months)
            df_dates = df_dates[df_dates[measured_date_col].dt.month.isin(months)]
        if not df_dates.empty:
            days_start = df_dates[measured_date_col].min().normalize()
            days_end = df_dates[measured_date_col].max().normalize()
            period_day_counts = _period_day_counts(days_start, days_end, freq=freq_str, season_months=season_months)
        else:
            period_day_counts = pd.Series(dtype=float)

        cat_resampled: Dict[int, Dict[str, pd.Series]] = {}
        measured_selection = cfg.get("measured_selection") or {}
        for cat in (1, 2, 3):
            selected = measured_selection.get(str(cat), {})
            if not selected.get("enabled"):
                continue
            chem_name = selected.get("chemical")
            stations = list(selected.get("stations") or [])
            if not chem_name or not stations:
                continue
            mvcol = None
            if is_conc_mode and use_measured_conc_col is not None:
                mvcol = str(use_measured_conc_col)
            elif (not is_conc_mode) and use_measured_load_col is not None:
                mvcol = str(use_measured_load_col)
            elif use_measured_load_col is not None:
                mvcol = str(use_measured_load_col)
            elif use_measured_conc_col is not None:
                mvcol = str(use_measured_conc_col)
            if mvcol is None:
                continue
            per_station_daily = _aggregate_measured(
                measured_included_df,
                date_col=measured_date_col,
                station_col=measured_station_col,
                name_col=measured_name_col,
                value_col=mvcol,
                selected_name=chem_name,
                selected_stations=stations,
                start=cfg.get("start"),
                end=cfg.get("end"),
                season_months=season_months,
            )
            for station, daily_series in per_station_daily.items():
                if (not is_conc_mode) and method == "sum":
                    per_mean = daily_series.resample(freq_str).mean()
                    s_aggr = per_mean * period_day_counts if not period_day_counts.empty else per_mean
                elif method == "flow_weighted_mean":
                    s_aggr = daily_series.resample(freq_str).mean()
                else:
                    s_aggr = daily_series.resample(freq_str).mean()
                s_plot = s_aggr.dropna()
                if not s_plot.empty:
                    cat_resampled.setdefault(cat, {})[station] = s_plot
        active_maps = [cat for cat in (1, 2, 3) if measured_selection.get(str(cat), {}).get("enabled") and cat in cat_resampled]
        if active_maps:
            station_sets = [set(cat_resampled[cat].keys()) for cat in active_maps if cat_resampled.get(cat)]
            common_stations = set.intersection(*station_sets) if station_sets else set()
            for station in sorted(common_stations):
                series_for_station = [cat_resampled[cat][station] for cat in active_maps if station in cat_resampled.get(cat, {})]
                if len(series_for_station) != len(active_maps):
                    continue
                intersection_idx: Optional[pd.Index] = None
                for series in series_for_station:
                    intersection_idx = series.index if intersection_idx is None else intersection_idx.intersection(series.index)
                if intersection_idx is None or len(intersection_idx) == 0:
                    continue
                combined = pd.concat([series.reindex(intersection_idx) for series in series_for_station], axis=1)
                combined_series = combined.sum(axis=1, min_count=len(series_for_station)).dropna().sort_index()
                if not combined_series.empty:
                    meas_for_stats.append(combined_series)

    view_window_raw = cfg.get("view_window") or {}
    x0 = view_window_raw.get("x0") if isinstance(view_window_raw, dict) else None
    x1 = view_window_raw.get("x1") if isinstance(view_window_raw, dict) else None
    if x0 is None:
        x0 = q_df.index.min()
    else:
        x0 = pd.to_datetime(x0)
    if x1 is None:
        x1 = q_df.index.max()
    else:
        x1 = pd.to_datetime(x1)
    if x0 is None or x1 is None:
        raise ValueError("Unable to determine stats view window")
    view_window = (pd.Timestamp(x0), pd.Timestamp(x1))

    stats = compute_stats_for_view(
        q_df,
        meas_for_stats,
        window=view_window,
        extras=extra_series,
        compute_log=bool(cfg.get("log_metrics", True)),
        max_global_lag=int(cfg.get("max_lag", 2)),
        local_window_ks=tuple(sorted(int(value) for value in cfg.get("local_window_ks", []))) if cfg.get("local_window_ks") else (),
        local_strategy="nearest",
        choose_best_lag_by=str(cfg.get("lag_metric", "r")),
        band_data=band_groups,
        event_context=event_context,
        extended_stats=bool(cfg.get("extended_stats", True)),
    )

    dashboard_state = {
        "variable": variable,
        "reach": reach,
        "frequency": cfg["frequency"],
        "frequency_string": freq_str,
        "bin": int(cfg["bin"]),
        "method": method,
        "compare_mode": cfg["compare_mode"],
        "flow_source": cfg.get("flow_source"),
        "event_source": cfg.get("event_source"),
        "event_threshold": cfg.get("event_threshold"),
        "event_abs_value": cfg.get("event_abs_value"),
        "event_min_days": cfg.get("event_min_days"),
        "event_buffer_days": cfg.get("event_buffer_days"),
        "event_view": cfg.get("event_view"),
        "autoscale_y_live": cfg.get("autoscale_y_live"),
        "range_slider": cfg.get("range_slider"),
        "show_names_in_tooltip": cfg.get("show_names_in_tooltip"),
        "show_diagnostics": cfg.get("show_diagnostics"),
        "show_measured": cfg.get("show_measured"),
        "show_water_flow": cfg.get("show_water_flow"),
        "show_diversion": cfg.get("show_diversion"),
        "show_swat_flow": cfg.get("show_swat_flow"),
        "show_erosion": cfg.get("show_erosion"),
        "show_flow_strat": False,
        "flow_regimes": list(cfg.get("flow_regimes") or []),
        "flow_total_band": cfg.get("flow_total_band"),
        "flow_total_only": cfg.get("flow_total_only"),
        "flow_overlay": cfg.get("flow_overlay"),
        "flow_total_mode": cfg.get("flow_total_mode"),
        "lag_metric": cfg.get("lag_metric"),
        "max_lag": cfg.get("max_lag"),
        "local_window_ks": list(cfg.get("local_window_ks") or []),
        "log_metrics": cfg.get("log_metrics"),
        "measured_nonnum_policy": cfg.get("measured_nonnum_policy"),
        "measured_negative_policy": cfg.get("measured_negative_policy"),
        "mdl_mg_L": float(cfg.get("mdl_mg_L", mdl_mg_L)),
        "mdl_mg_L_by_name": dict(normalized_mdl_mg_L_by_name) or None,
        "flag_deviations": cfg.get("flag_deviations"),
        "deviation_factor": cfg.get("deviation_factor"),
        "start": cfg.get("start"),
        "end": cfg.get("end"),
        "season_months": cfg.get("season_months"),
        "measured_selection": cfg.get("measured_selection"),
        "extra_overlays": cfg.get("extra_overlays"),
        "view_window": {"x0": view_window[0], "x1": view_window[1]},
        "source_arguments": {
            "reach_col": reach_col,
            "date_col": date_col,
            "flow_col": flow_col,
            "template": "plotly_white",
            "figure_width": None,
            "figure_height": None,
            "stats_export_dir": str(stats_export_dir),
        },
    }
    filename_stem = _dashboard_build_stats_filename_stem(
        run_label=run_label,
        variable=variable,
        reach=reach,
        frequency=cfg["frequency"],
        bin_size=int(cfg["bin"]),
        method=method,
        compare_mode=cfg["compare_mode"],
        event_view=str(cfg.get("event_view") or "all"),
        view_window=view_window,
    )
    payload = _build_dashboard_stats_export_payload(
        stats=stats,
        view_window=view_window,
        dashboard_state=dashboard_state,
        measured_preprocessing=measured_nonnum_audit,
        run_label=run_label,
        filename_stem=filename_stem,
        source_function="export_dashboard_stats_from_config",
        compute_log=bool(cfg.get("log_metrics", True)),
        max_global_lag=int(cfg.get("max_lag", 2)),
        local_window_ks=cfg.get("local_window_ks", []),
        choose_best_lag_by=str(cfg.get("lag_metric", "r")),
        has_band_data=bool(band_groups),
        has_event_context=bool(event_context),
    )
    file_path = export_stats_to_json(payload, stats_export_dir)
    return payload, file_path


def batch_export_dashboard_stats(
    run_numbers: List[int],
    variables: List[str],
    *,
    dashboard_config: Dict[str, Any],
    per_variable_dashboard_config: Optional[Dict[str, Dict[str, Any]]] = None,
    measured_df: Optional[pd.DataFrame] = None,
    measured_var_map: Optional[Dict[str, object]] = None,
    water_flow_df: Optional[pd.DataFrame] = None,
    sim_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    extra_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    stats_export_dir: Optional[Union[str, Path]] = None,
    how_map_defaults: Optional[Dict[str, str]] = None,
    reach_col: str = "RCH",
    date_col: str = "date",
    flow_col: str = "FLOW_OUTcms",
    measured_date_col: str = "F_MUESTREO",
    measured_station_col: str = "est_estaci",
    measured_name_col: str = "NOMBRE",
    measured_value_col: Optional[str] = None,
    measured_kg_col_name: str = "kg_per_day",
    water_flow_date_col: str = "date",
    water_flow_value_col: Optional[str] = None,
    mdl_mg_L_by_name: Optional[Dict[str, float]] = None,
    include_variables_in_folder_name: bool = False,
    debug: bool = False,
) -> Dict[int, Dict[str, Tuple[Dict[str, Any], str]]]:
    """
    Batch export dashboard stats for multiple runs.
    
    Loads simulation data for specified runs, exports stats for each run-variable combination to a
    dedicated subfolder, and returns a mapping of run_number → variable →
    (payload, exported_file_path).
    
    Parameters
    ----------
    run_numbers : List[int]
        List of run numbers to export (e.g., [184, 185, 186])
    variables : List[str]
        List of SWAT output variables to export. For each run, one stats JSON is created for each
        variable in this list, while all other dashboard config values remain unchanged.
    dashboard_config : Dict[str, Any]
        Dashboard configuration dict (reach, variable, frequency, etc.) applied to all runs
    per_variable_dashboard_config : Optional[Dict[str, Dict[str, Any]]]
        Optional variable-specific overrides merged on top of dashboard_config for the matching
        exported variable. Use this when one variable needs an explicit cats/measured_selection
        block but other variables in the same batch should keep their default mapping.
    measured_df : Optional[pd.DataFrame]
        Measured chemistry data (shared across all runs)
    measured_var_map : Optional[Dict[str, object]]
        Mapping of SWAT variables to measured chemistry options
    water_flow_df : Optional[pd.DataFrame]
        Water flow measurements (shared across all runs)
    sim_dfs : Optional[Dict[str, pd.DataFrame]]
        Pre-loaded simulation DataFrames {sim_name: df}. If None, you must call load_or_build_dfs_for_runs
        and pass the result here. Otherwise raise error.
    extra_dfs : Optional[Dict[str, pd.DataFrame]]
        Extra overlay dataframes passed through to each per-run headless export.
        The dict keys become the overlay_comparison entry names in the exported JSON.
    stats_export_dir : Optional[Union[str, Path]]
        Parent directory for the batch subfolder. Defaults to trabajoFM/config/outputs/dashboard_stats/
    how_map_defaults : Optional[Dict[str, str]]
        Aggregation method defaults for each variable
    reach_col : str
        Name of reach column in simulation DataFrames
    date_col : str
        Name of date column in simulation DataFrames
    flow_col : str
        Name of flow column in simulation DataFrames
    measured_date_col : str
        Name of date column in measured data
    measured_station_col : str
        Name of station column in measured data
    measured_name_col : str
        Name of chemistry name column in measured data
    measured_value_col : Optional[str]
        Explicit value column in measured data (auto-detected if None)
    measured_kg_col_name : str
        Name of kg/day column in measured data
    water_flow_date_col : str
        Name of date column in water flow data
    water_flow_value_col : Optional[str]
        Explicit value column in water flow data (auto-detected if None)
    include_variables_in_folder_name : bool
        If True, append a sanitized variable list to the batch folder name, for example:
        runs_184_185__vars_kjeldahl_outkg_tot_nkg
    debug : bool
        Enable debug output
    
    Returns
    -------
    Dict[int, Dict[str, Tuple[Dict[str, Any], str]]]
        Mapping of run_number → variable → (payload_dict, exported_json_path)
    
    Examples
    --------
    >>> from python_pipeline_scripts.dashboard import batch_export_dashboard_stats
    >>> results = batch_export_dashboard_stats(
    ...     [184, 185, 186],
    ...     ["TOT_Nkg", "TOT_Pkg"],
    ...     dashboard_config={"reach": 13, "variable": "TOT_Nkg", "frequency": "D", ...},
    ...     per_variable_dashboard_config={
    ...         "TOT_Nkg": {
    ...             "cats": {
    ...                 1: {"enabled": True, "chem": "NITROGENO KJELDAHL", "stations": ["30304"]},
    ...                 2: {"enabled": True, "chem": "NITRATOS", "stations": ["30304"]},
    ...                 3: {"enabled": False},
    ...             }
    ...         }
    ...     },
    ...     measured_df=measured_df,
    ...     water_flow_df=flow_df,
    ...     sim_dfs=load_or_build_dfs_for_runs([184, 185, 186], force_rebuild=False),
    ...     measured_var_map=swat_to_measured,
    ... )
    >>> # Results saved to: trabajoFM/config/outputs/dashboard_stats/runs_184_185_186/
    >>> for run_id, per_var in results.items():
    ...     for variable, (payload, path) in per_var.items():
    ...         print(f"Run {run_id} / {variable}: {path}")
    """
    if not run_numbers:
        raise ValueError("run_numbers must not be empty")
    if not variables:
        raise ValueError("variables must not be empty")
    if sim_dfs is None or not sim_dfs:
        raise ValueError("sim_dfs must be provided and non-empty. Call load_or_build_dfs_for_runs and pass the result.")
    
    run_numbers_list = list(run_numbers) if not isinstance(run_numbers, list) else run_numbers
    run_numbers_list = sorted(set(int(r) for r in run_numbers_list))
    base_dashboard_config = dict(dashboard_config or {})
    per_variable_dashboard_config = {
        str(variable_name): dict(variable_config or {})
        for variable_name, variable_config in (per_variable_dashboard_config or {}).items()
    }
    variables_list = []
    for variable_name in variables:
        variable_text = str(variable_name).strip()
        if variable_text and variable_text not in variables_list:
            variables_list.append(variable_text)
    if not variables_list:
        raise ValueError("variables must contain at least one non-empty variable name")
    
    # Create batch subfolder name: runs_184_185_186
    run_list_str = "_".join(str(r) for r in run_numbers_list)
    if include_variables_in_folder_name:
        def _sanitize_folder_token(value: str) -> str:
            cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
            return cleaned or "var"

        variable_list_str = "_".join(_sanitize_folder_token(value) for value in variables_list)
        batch_folder_name = f"runs_{run_list_str}__vars_{variable_list_str}"
    else:
        batch_folder_name = f"runs_{run_list_str}"
    
    # Resolve parent stats directory
    if stats_export_dir is None:
        stats_export_dir = Path(__file__).resolve().parent.parent / "config" / "outputs" / "dashboard_stats"
    else:
        stats_export_dir = Path(stats_export_dir)
    
    # Create batch subfolder
    batch_export_dir = stats_export_dir / batch_folder_name
    batch_export_dir.mkdir(parents=True, exist_ok=True)
    
    def _dbg(*args: Any) -> None:
        if debug:
            try:
                print("[batch-export]", *args)
            except Exception:
                pass
    
    _dbg(f"Batch export initialized: {batch_folder_name}")
    _dbg(f"Parent dir: {stats_export_dir}")
    _dbg(f"Runs: {run_numbers_list}")
    _dbg(f"Variables: {variables_list}")
    _dbg(f"Loaded sim_dfs keys: {list(sim_dfs.keys())[:5]}... ({len(sim_dfs)} total)")
    
    results: Dict[int, Dict[str, Tuple[Dict[str, Any], str]]] = {}
    errors: Dict[Tuple[int, str], str] = {}
    
    for run_id in run_numbers_list:
        try:
            # Filter sim_dfs to only include sims from this run
            # Keys are like: "rch_run000184_real000396_1"
            run_pattern = f"run{run_id:06d}"  # e.g., "run000184"
            filtered_sims = {
                k: v for k, v in sim_dfs.items()
                if run_pattern in str(k).lower()
            }
            
            if not filtered_sims:
                msg = f"No simulations found for run {run_id} (pattern: {run_pattern})"
                _dbg(f"WARNING: {msg}")
                for variable in variables_list:
                    errors[(run_id, variable)] = msg
                continue
            
            _dbg(f"Run {run_id}: Found {len(filtered_sims)} simulations")
            results.setdefault(run_id, {})
            for variable in variables_list:
                try:
                    dashboard_config_for_variable = dict(base_dashboard_config)
                    dashboard_config_for_variable.update(
                        per_variable_dashboard_config.get(variable, {})
                    )
                    dashboard_config_for_variable["variable"] = variable
                    payload, export_path = export_dashboard_stats_from_config(
                        sim_dfs=filtered_sims,
                        variables=variables_list,
                        dashboard_config=dashboard_config_for_variable,
                        extra_dfs=extra_dfs,
                        measured_df=measured_df,
                        measured_var_map=measured_var_map,
                        water_flow_df=water_flow_df,
                        stats_export_dir=str(batch_export_dir),
                        reach_col=reach_col,
                        date_col=date_col,
                        flow_col=flow_col,
                        measured_date_col=measured_date_col,
                        measured_station_col=measured_station_col,
                        measured_name_col=measured_name_col,
                        measured_value_col=measured_value_col,
                        measured_kg_col_name=measured_kg_col_name,
                        water_flow_date_col=water_flow_date_col,
                        water_flow_value_col=water_flow_value_col,
                        how_map_defaults=how_map_defaults,
                        mdl_mg_L_by_name=mdl_mg_L_by_name,
                        debug=debug,
                    )
                    results[run_id][variable] = (payload, export_path)
                    _dbg(f"Run {run_id} / {variable}: Exported to {Path(export_path).name}")
                except Exception as exc:
                    msg = f"Export failed for run {run_id}, variable {variable}: {exc}"
                    _dbg(f"ERROR: {msg}")
                    errors[(run_id, variable)] = msg
            
        except Exception as exc:
            msg = f"Run setup failed for run {run_id}: {exc}"
            _dbg(f"ERROR: {msg}")
            for variable in variables_list:
                errors[(run_id, variable)] = msg
    
    # Summary
    successful = sum(len(per_var) for per_var in results.values())
    total = len(run_numbers_list) * len(variables_list)
    _dbg(f"\nBatch export complete: {successful}/{total} run-variable exports succeeded")
    if errors:
        _dbg(f"Failed run-variable exports: {list(errors.keys())}")
        for (run_id, variable), err in errors.items():
            _dbg(f"  - Run {run_id} / {variable}: {err}")
    
    _dbg(f"Exports saved to: {batch_export_dir}/")
    
    return results




