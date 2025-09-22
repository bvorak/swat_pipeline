from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union, Sequence, Tuple, Any
from datetime import datetime, date

import numpy as np
import pandas as pd
import re

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from importlib import reload

from .stats import compute_stats_for_view, format_stats_text, build_fit_diagnostics
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
    convert_measured_mgL_to_kg_per_day,
)

DASHBOARD_VERSION = "2025-09-11-chem-ui-3"

# moved helpers to dashboard_helper.py


# -----------------------------
# Dashboard with measured overlay
# -----------------------------

def fan_compare_simulations_dashboard(
    sim_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    *,
    reach: Optional[int] = None,
    freq_options: Iterable[str] = ("D", "W", "M", "A"),
    max_bin_size: int = 12,
    start: Optional[Union[str, datetime, date]] = None,
    end: Optional[Union[str, datetime, date]] = None,
    season_months: Optional[List[int]] = None,
    how_map_defaults: Optional[Dict[str, str]] = None,
    reach_col: str = "RCH",
    date_col: str = "date",
    flow_col: str = "FLOW_OUTcms",
    template: str = "plotly_white",
    figure_width: Optional[int] = 1200,
    figure_height: int = 650,
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
    # Measured cleaning policies (defaults; also controllable via UI dropdowns)
    measured_nonnum_policy_default: str = "as_na",  # "as_na" | "drop"
    measured_negative_policy_default: str = "zero",  # "keep" | "drop" | "zero"
    # Optional UI default selections/toggles
    ui_defaults: Optional[Dict[str, Any]] = None,
    # Optional erosion overlay toggle default
    erosion_on_default: Optional[bool] = None,
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

    measured_var_map expected formats per SWAT variable key:
        - dict with keys 1/2/3 (or '1'/'2'/'3') -> list of NOMBRE strings
        - list/tuple of up to three lists of NOMBRE strings
    """
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
                try:
                    dt = pd.to_datetime(df[date_col])
                    rng = (dt.min(), dt.max())
                except Exception:
                    rng = None
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
    cb_show_names_in_tooltip = widgets.Checkbox(value=False, description="Names in tooltip")

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

    # Per-category: enable, chem-name dropdown, station selector
    cat_symbols = {1: "star", 2: "circle", 3: "square"}
    cat_labels = {1: "Map 1", 2: "Map 2", 3: "Map 3"}

    cb_meas_on = widgets.Checkbox(value=measured_present, description="Show measured")
    # Default ON when a water_flow_df is provided; we will auto-pick the column later
    cb_flow_on = widgets.Checkbox(value=(isinstance(water_flow_df, pd.DataFrame) and not water_flow_df.empty), description="Show water flow (m3/d)")
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
            ("Non-numeric handling: set to half MDL (0.1)", "half_MDL"),
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
    diag_box = widgets.VBox([])
    # Styling and initial layout
    try:
        stats_html.layout.width = "40%"
        stats_html.layout.min_width = "300px"
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
        "flow_series": None,  # aggregated external water flow series for current settings (m3/day)
        "swat_flow_series": None,  # mean across runs of SWAT FLOW_OUT * 86400 (m3/day)
        "flow_y_range": None, # last y2 range
        "extra_series": {},   # name -> pd.Series for extra overlays (current settings)
        "erosion_series": None,  # mean across runs of (SED_IN - SED_OUT)
    }
    _state = {"updating": False}

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
        # Inject default measured mapping for the derived variable if not provided
        meas_map_local = dict(measured_var_map or {})
        if var == SYN_VAR and var not in meas_map_local:
            meas_map_local[var] = {
                1: ["NITROGENO KJELDAHL", "Nitrógeno Kjeldahl", "Nitrógeno KJELDAHL"],
                2: ["NITORGENO TOTAL", "NITROGENO TOTAL", "Nitrógeno total", "Nitrógeno Total"],
            }
        norm_map = _normalize_meas_map_for_var(meas_map_local, var)
        for i in (1, 2, 3):
            allowed = norm_map.get(i, [])
            opts = _measured_options_for_category(measured_df, measured_name_col, allowed)
            # Update chem dropdown robustly avoiding invalid intermediate states
            old_val = dd_cat_name[i].value
            if old_val is not None and old_val not in opts:
                if opts:
                    # transitional include old value, then finalize
                    dd_cat_name[i].options = _chem_options_with_placeholder(opts, extra=old_val)
                    dd_cat_name[i].value = opts[0]
                    dd_cat_name[i].options = _chem_options_with_placeholder(opts)
                else:
                    # no options -> placeholder only
                    dd_cat_name[i].options = _chem_options_with_placeholder([])
                    dd_cat_name[i].value = None
            else:
                dd_cat_name[i].options = _chem_options_with_placeholder(opts)
                if dd_cat_name[i].value is None and opts:
                    dd_cat_name[i].value = opts[0]
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
            if not ms_cat_stations[i].value:
                if ((i == 1) or (var == SYN_VAR and i in (1, 2))) and ("30304" in sts):
                    ms_cat_stations[i].value = ("30304",)
                else:
                    ms_cat_stations[i].value = tuple(sts)
            # Hide category box if no options
            if i in cat_vbox:
                cat_vbox[i].layout.display = None if opts else 'none'
            # Disable checkbox when no options (and uncheck)
            cb_cat[i].disabled = not bool(opts)
            if not opts:
                cb_cat[i].value = False
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
        if isinstance(ui_defaults.get("show_diags"), bool):
            cb_show_diags.value = bool(ui_defaults.get("show_diags"))
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
        # Measured toggles
        if isinstance(ui_defaults.get("measured_on"), bool):
            cb_meas_on.value = bool(ui_defaults.get("measured_on"))
        if isinstance(ui_defaults.get("flow_on"), bool):
            cb_flow_on.value = bool(ui_defaults.get("flow_on"))
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
            if cat == 1 and ("30304" in sts):
                ms_cat_stations[cat].value = ("30304",)
            else:
                ms_cat_stations[cat].value = tuple(sts)

    def _hovertemplate(show_name: bool) -> str:
        return ("%{fullData.name}: %{y:.4g}<extra></extra>" if show_name else "%{y:.4g}<extra></extra>")

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
        try:
            if event_mode in {"events", "non_events"}:
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
                            buf = int(sl_event_buffer_days.value) if isinstance(sl_event_buffer_days.value, (int, float)) else 0
                            buffered_event_days = set()
                            for d in event_day_set:
                                d0 = pd.Timestamp(d).normalize()
                                for k in range(-buf, buf + 1):
                                    buffered_event_days.add(d0 + pd.Timedelta(days=int(k)))
                            full_days_set = set(pd.to_datetime(df_ev["date"]).unique().tolist())
                            if event_mode == "events":
                                selected_days_set = buffered_event_days
                            else:
                                selected_days_set = full_days_set - (buffered_event_days or set())
                            _dbg("events", dict(mode=event_mode, events=len(event_day_set), buffer=len((buffered_event_days or set()) - (event_day_set or set())), keep=len(selected_days_set or [])))
        except Exception as e:
            _dbg("event detection failed", e)
            selected_days_set = None
        # Extract a single resampled series per run for the selected reach/variable
        # Maintain filtered (stats) and unfiltered (plot) collections
        per_sim: Dict[str, pd.Series] = {}
        per_sim_plot: Dict[str, pd.Series] = {}
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
        q_df = pd.DataFrame({
            "p05": q[5], "p10": q[10], "p25": q[25], "p50": q[50], "p60": q[60], "p75": q[75], "p90": q[90], "p95": q[95]
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
            arr_plot = aligned_df_plot.to_numpy(dtype=float)
            percs_plot = [5, 10, 25, 50, 60, 75, 90, 95]
            qs_plot = np.nanpercentile(arr_plot, percs_plot, axis=1) if arr_plot.shape[1] else np.full((len(percs_plot), 0), np.nan)
            q_plot = {p: qs_plot[i, :] if arr_plot.shape[1] else np.array([]) for i, p in enumerate(percs_plot)}
            _last["aligned_df_plot"] = aligned_df_plot
            _last["q_plot_df"] = pd.DataFrame({
                "p05": q_plot[5], "p10": q_plot[10], "p25": q_plot[25], "p50": q_plot[50], "p60": q_plot[60], "p75": q_plot[75], "p90": q_plot[90], "p95": q_plot[95]
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
        if figure_width is not None:
            fig.layout.width = int(figure_width)
        fig.layout.height = int(figure_height)
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
                        remaining = [s for s in prev_shapes if not (isinstance(s, dict) and str(s.get('fillcolor','')).startswith('rgba(90,90,90'))]
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
                    # Legend proxy (avoid duplicates)
                    if not any(getattr(tr, 'name', '') == 'Filtered (excluded from stats)' for tr in fig.data):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode='markers',
                            marker=dict(size=10, color='rgba(90,90,90,0.30)', symbol='square'),
                            name='Filtered (excluded from stats)',
                            hoverinfo='skip', showlegend=True
                        ))
        except Exception as _e_overlay:
            _dbg('overlay build failed', str(_e_overlay))

        # Fan chart vs simplified band depending on number of runs
        color = "#1f77b4"
        rgba = lambda a: f"rgba(31,119,180,{a})"
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
            
            # 90% band (p05..p95)
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(p95_masked), mode="lines",
                line=dict(color=rgba(0.12), width=0.5),
                name="p95", showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(p05_masked), mode="lines",
                line=dict(color=rgba(0.12), width=0.5),
                fill="tonexty", fillcolor=rgba(0.12),
                name="p05-p95", showlegend=True, hoverinfo="skip"
            ))
            # 50% band (p25..p75)
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(p75_masked), mode="lines",
                line=dict(color=rgba(0.28), width=0.5),
                name="p75", showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(p25_masked), mode="lines",
                line=dict(color=rgba(0.28), width=0.5),
                fill="tonexty", fillcolor=rgba(0.28),
                name="p25-p75", showlegend=True, hoverinfo="skip"
            ))
            # Median
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(q_plot[50]), mode="lines", line=dict(color="black", width=2),
                name="median",
                customdata=_make_customdata_multi(q_plot[5], q_plot[25], q_plot[50], q_plot[75], q_plot[95]),
                hovertemplate=_median_hovertemplate(cb_show_names_in_tooltip.value, _run_label),
            ))
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
            # Max then min with fill between
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(vmax), mode="lines", line=dict(color=rgba(0.20), width=0.5),
                name="max", showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(vmin), mode="lines", line=dict(color=rgba(0.20), width=0.5),
                fill="tonexty", fillcolor=rgba(0.18),
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
            
            fig.add_trace(go.Scatter(
                x=x_dt, y=_nan_to_none(vmean), mode="lines", line=dict(color="black", width=2),
                name="mean",
                customdata=combined_customdata,
                hovertemplate=("max: %{customdata[4]:.4g}%{customdata[5]}<br>mean: %{customdata[2]:.4g}%{customdata[3]}<br>min: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>"),
            ))

        # Store band data for comprehensive statistics (grouped by series)
        band_groups: Dict[str, Dict[str, pd.Series]] = {}
        ensemble_band: Dict[str, pd.Series] = {}
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
            p05_vals = np.full(arr.shape[0], np.nan)
            p25_vals = np.full(arr.shape[0], np.nan)
            p50_vals = np.full(arr.shape[0], np.nan)
            p75_vals = np.full(arr.shape[0], np.nan)
            p95_vals = np.full(arr.shape[0], np.nan)
            mean_vals = np.full(arr.shape[0], np.nan)

            # Only compute where we have sufficient data
            if np.any(sufficient_data):
                sufficient_indices = np.where(sufficient_data)[0]
                p05_vals[sufficient_data] = q[5][sufficient_data]
                p25_vals[sufficient_data] = q[25][sufficient_data]
                p50_vals[sufficient_data] = q[50][sufficient_data]
                p75_vals[sufficient_data] = q[75][sufficient_data]
                p95_vals[sufficient_data] = q[95][sufficient_data]
                mean_vals[sufficient_data] = np.nanmean(arr[sufficient_data, :], axis=1)

            # Create band data series only for time points with sufficient data
            if np.any(sufficient_data):
                valid_indices = aligned_df.index[sufficient_data]
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
                    if (is_conc_mode or dd_method.value == "flow_weighted_mean") and flow_col in df_ex.columns:
                        cols.append(flow_col)
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
                    if selected_days_set is not None:
                        try:
                            day_mask = sub.index.floor('D').isin(list(selected_days_set))
                            sub = sub.loc[day_mask]
                        except Exception:
                            pass
                    if sub.empty:
                        continue
                    if is_conc_mode:
                        if flow_col not in sub.columns:
                            continue
                        with np.errstate(invalid='ignore', divide='ignore'):
                            base_col = (SYN_VAR if var == SYN_VAR else var)
                            sub["__conc_mgL__"] = (sub[base_col] / (sub[flow_col] * 86400.0)) * 1000.0
                        how_here = dd_method.value if dd_method.value in ("flow_weighted_mean", "mean") else "flow_weighted_mean"
                        s_ex = _resample_series(sub, "__conc_mgL__", freq=_make_freq_string(dd_freq.value, sl_bin.value), how=how_here, flow_col=flow_col)
                    else:
                        base_col = (SYN_VAR if var == SYN_VAR else var)
                        s_ex = _resample_series(sub, base_col, freq=_make_freq_string(dd_freq.value, sl_bin.value), how=dd_method.value,
                                                flow_col=flow_col if flow_col in sub.columns else None)
                    s_ex = s_ex.dropna()
                    if s_ex.empty:
                        continue
                    # Store for stats correlations
                    try:
                        _last["extra_series"][str(name)] = s_ex
                    except Exception:
                        pass
                    color = extra_palette[ei % len(extra_palette)]; ei += 1
                    fig.add_trace(go.Scatter(
                        x=_to_plotly_x(s_ex.index), y=s_ex.values, mode="lines",
                        name=str(name), line=dict(color=color, width=2),
                        customdata=_make_customdata(s_ex.values),
                        hovertemplate="%{fullData.name}: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                    ))
                except Exception:
                    # Keep plotting even if one overlay fails
                    continue

        _extend_relative_bands(band_groups, _last.get("extra_series", {}), prefix="extra")
        _last["band_data"] = band_groups

        # Prepare measured DataFrame (apply cleaning policies and compute kg/day when possible)
        measured_use_df = measured_df if measured_present else None
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
                    half_mdl = 0.1 * 0.5  # match helper logic (typical MDL 0.1 mg/L)
                    df_loc.loc[nonnum_mask, value_col] = half_mdl
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
            # Color map for stations across categories (consistent colors per station)
            # Build a global color palette
            palette = [
                "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#1f77b4",
            ]
            station_colors: Dict[str, str] = {}
            color_idx = 0

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
                # One trace per station
                for st, s_daily in per_station_daily.items():
                    _dbg_df_info(s_daily, f"measured {chem_name} st={st} daily")
                    if st not in station_colors:
                        station_colors[st] = palette[color_idx % len(palette)]
                        color_idx += 1
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
                    _dbg_df_info(s_plot, f"measured {chem_name} st={st} resampled")
                    if s_plot.empty:
                        continue
                    _meas_for_stats.append(s_plot)
                    # Color-code measured points:
                    # - Red if within flow outlier period (outlier or buffer days)
                    # - Orange if deviating vs p90 by factor threshold
                    # - Green otherwise
                    try:
                        idx_days = pd.to_datetime(s_plot.index, errors='coerce').floor('D')
                        # Flow outlier period mask
                        flow_excl_set: set = set()
                        if (outlier_day_set is not None) or (buffer_only_set is not None):
                            if outlier_day_set is not None:
                                flow_excl_set |= set(outlier_day_set)
                            if buffer_only_set is not None:
                                flow_excl_set |= set(buffer_only_set)
                        red_mask = idx_days.isin(list(flow_excl_set)) if flow_excl_set else pd.Series(False, index=s_plot.index)
                        # Deviation mask (vs p90) using factor-of threshold
                        dev_mask = pd.Series(False, index=s_plot.index)
                        if _last.get("q_df") is not None and bool(cb_flag_dev.value):
                            base = _last.get("q_df")["p90"].reindex(s_plot.index)
                            if base.isna().all() and ("p50" in _last.get("q_df").columns):
                                base = _last.get("q_df")["p50"].reindex(s_plot.index)
                            if base.isna().all() and (_last.get("aligned_df") is not None):
                                base = _last.get("aligned_df").mean(axis=1, skipna=True).reindex(s_plot.index)
                            factor = float(sl_dev_factor.value)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mvals = s_plot.to_numpy(dtype=float)
                                bvals = base.to_numpy(dtype=float)
                                denom = np.abs(bvals)
                                denom[~np.isfinite(denom) | (denom == 0.0)] = np.nan
                                ratio = np.abs(mvals) / denom
                            arr_mask = np.isfinite(ratio) & ((ratio >= factor) | (ratio <= (1.0 / factor)))
                            dev_mask = pd.Series(arr_mask, index=s_plot.index)
                        # Segment into three groups with precedence: red > orange > green
                        red_idx = red_mask[red_mask].index
                        orange_idx = dev_mask[~dev_mask.index.isin(red_idx) & dev_mask].index
                        green_idx = s_plot.index.difference(red_idx.union(orange_idx))
                        def _add_pts(sel_index, color, name_suffix):
                            if sel_index is None or len(sel_index) == 0:
                                return
                            ss = s_plot.loc[sel_index]
                            fig.add_trace(go.Scatter(
                                x=_to_plotly_x(ss.index), y=ss.values, mode="markers",
                                name=f"{chem_name} - {st} ({name_suffix})",
                                marker=dict(symbol=cat_symbols[cat], size=10, color=color, line=dict(width=0.5, color="#333")),
                                customdata=_make_customdata(ss.values),
                                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                                showlegend=True,
                            ))
                        _add_pts(red_idx, "#d62728", "flow-outlier")
                        _add_pts(orange_idx, "#ff7f0e", "deviation")
                        _add_pts(green_idx, "#2ca02c", "kept")
                    except Exception:
                        # Fallback: add all as green if something goes wrong
                        fig.add_trace(go.Scatter(
                            x=_to_plotly_x(s_plot.index), y=s_plot.values, mode="markers",
                            name=f"{chem_name} - {st} (kept)",
                            marker=dict(symbol=cat_symbols[cat], size=10, color="#2ca02c", line=dict(width=0.5, color="#333")),
                            customdata=_make_customdata(s_plot.values),
                            hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                            showlegend=True,
                        ))

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
                if selected_days_set is not None:
                    try:
                        s_daily = s_daily.loc[s_daily.index.isin(list(selected_days_set))]
                    except Exception:
                        pass
                # Resample per selected method
                if dd_method.value == "sum":
                    s_flow = s_daily.resample(freq_str).sum(min_count=1)
                else:
                    s_flow = s_daily.resample(freq_str).mean()
                s_flow = s_flow.dropna()
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
                        title="m3/day corrected water flow", overlaying='y', side='right', showgrid=False,
                        autorange=False, range=y2_range, title_standoff=20, automargin=True,
                        title_font_color="#1f77b4"
                    ))
                    # Single dotted blue line, with legend label requested
                    fig.add_trace(go.Scatter(
                        x=_to_plotly_x(s_flow.index), y=s_flow.values, mode="lines",
                        name="m3/day corrected water flow", yaxis='y2',
                        line=dict(color="#1f77b4", width=1.2, dash="dot"),
                        customdata=_make_customdata(s_flow.values),
                        hovertemplate="Water flow: %{customdata[0]:.4g}%{customdata[1]} m3/d<extra></extra>",
                        visible=True,
                    ))
                    # Reorder traces so water flow draws behind others (background)
                    try:
                        if len(fig.data) >= 1:
                            fig.data = (fig.data[-1],) + fig.data[:-1]
                    except Exception:
                        pass
            except Exception:
                _last["flow_series"] = None

        # SWAT average flow overlay (from simulation DataFrames; FLOW_OUT * 86400)
        _last["swat_flow_series"] = None
        try:
            if bool(cb_swat_flow_on.value):
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
                        if selected_days_set is not None:
                            try:
                                mask = sub.index.floor('D').isin(list(selected_days_set))
                                sub = sub.loc[mask]
                            except Exception:
                                pass
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
                        # Resample per selected method
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
                    _last["swat_flow_series"] = s_swat_mean
                    # Ensure y2 axis exists and add SWAT flow trace
                    fig.update_layout(yaxis2=dict(
                        title="m3/day corrected water flow", overlaying='y', side='right', showgrid=False,
                        autorange=False, title_standoff=20, automargin=True,
                        title_font_color="#1f77b4"
                    ))
                    fig.add_trace(go.Scatter(
                        x=_to_plotly_x(s_swat_mean.index), y=s_swat_mean.values, mode="lines",
                        name="SWAT avg flow (m3/d)", yaxis='y2',
                        line=dict(color="#17becf", width=1.6, dash="solid"),
                        customdata=_make_customdata(s_swat_mean.values),
                        hovertemplate="SWAT flow: %{customdata[0]:.4g}%{customdata[1]} m3/d<extra></extra>",
                        visible=True,
                    ))
                    # Reorder traces so SWAT flow draws behind others (background)
                    try:
                        if len(fig.data) >= 1:
                            fig.data = (fig.data[-1],) + fig.data[:-1]
                    except Exception:
                        pass
        except Exception:
            _last["swat_flow_series"] = None

        # Finalize water flow axis range to include all flow overlays present
        try:
            y2_values = []
            if isinstance(_last.get("flow_series"), pd.Series) and not _last["flow_series"].empty:
                y2_values.append(_last["flow_series"].to_numpy(dtype=float))
            if isinstance(_last.get("swat_flow_series"), pd.Series) and not _last["swat_flow_series"].empty:
                y2_values.append(_last["swat_flow_series"].to_numpy(dtype=float))
            if y2_values:
                fv = np.concatenate([v[np.isfinite(v)] for v in y2_values if v.size > 0])
                if fv.size == 0:
                    fmin, fmax = 0.0, 1.0
                else:
                    fmin = float(np.nanmin(fv)); fmax = float(np.nanmax(fv))
                    if fmin == fmax:
                        fmax = fmin + 1.0
                fpad = (fmax - fmin) * 0.05
                y2_range = [fmin - fpad, fmax + fpad]
                _last["flow_y_range"] = y2_range
                fig.update_layout(yaxis2=dict(
                    title="m3/day corrected water flow", overlaying='y', side='right', showgrid=False,
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
                fig.add_trace(go.Scatter(
                    x=_to_plotly_x(s_ero_mean.index), y=s_ero_mean.values, mode="lines",
                    name="Erosion (SED_IN - SED_OUT)", yaxis='y3',
                    line=dict(color="#8c564b", width=1.8),
                    customdata=_make_customdata(s_ero_mean.values),
                    hovertemplate="Erosion: %{customdata[0]:.4g}%{customdata[1]}<extra></extra>",
                    visible=True,
                ))
                # Reorder traces so erosion draws behind others (background)
                try:
                    if len(fig.data) >= 1:
                        fig.data = (fig.data[-1],) + fig.data[:-1]
                except Exception:
                    pass
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

        # Save measured series for stats box
        if measured_present and cb_meas_on.value:
            _last["meas_series"] = _meas_for_stats
        else:
            _last["meas_series"] = []

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
        fig.update_layout(
            title_text=None,
            xaxis_title="Date", yaxis_title=var,
            hovermode="x unified",
            hoverlabel=dict(namelength=-1, align="left", font_size=12, bgcolor="white"),
            legend=dict(orientation="h", y=1.02, x=0),
            xaxis=dict(
                type="date",
                rangeslider=dict(visible=True, thickness=0.08, bgcolor="#f6f6f6", bordercolor="#ddd", borderwidth=1),
                tickformatstops=TICK_STOPS
            ),
            margin=dict(l=60, r=20, t=50, b=150)
        )
        fig.add_annotation(
            x=0.5, y=-0.22, xref='paper', yref='paper',
            text=title_text,
            showarrow=False, xanchor='center', yanchor='top',
            font=dict(size=22, color='black')
        )

        # Stats computation will populate the HTML block below the figure

        # fixed Y; optional live update on zoom (apply only to primary y-axis)
        y_title = ("Concentration (mg/L)" if is_conc_mode else f"{var} (kg/day)")
        fig.update_layout(yaxis=dict(autorange=False, range=_last["y_fixed"], title_text=y_title))

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
                    )
                except Exception as e:
                    # If stats computation fails, provide error details
                    with out:
                        print(f"Stats computation failed: {e}")
                        import traceback
                        traceback.print_exc()
                    stats_html.value = f"Stats computation error: {e}"
                    return
                html_text = format_stats_text(stats)
                try:
                    stats_html.value = html_text
                except Exception:
                    # Fallback to printing into the output area if HTML update fails
                    with out:
                        print(html_text)
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
                        )
                        # Convert to FigureWidgets for better embedding
                        children = []
                        for key in ["obs_vs_pred", "resid_hist", "resid_vs_pred", "lag_hist"]:
                            if key in figs:
                                try:
                                    children.append(go.FigureWidget(figs[key]))
                                except Exception:
                                    children.append(widgets.HTML(f"<pre>Unable to render {key}</pre>"))
                        if children:
                            # Add a small reproducible call snippet
                            start_s = pd.to_datetime(x0).strftime('%Y-%m-%d')
                            end_s = pd.to_datetime(x1).strftime('%Y-%m-%d')
                            call_str = (
                                "from python_pipeline_scripts.stats import build_fit_diagnostics\n"
                                "# assuming you have q_df (fan quantiles) and measured_series from the dashboard context\n"
                                f"figs = build_fit_diagnostics(q_df, measured_series, window=(pd.Timestamp('{start_s}'), pd.Timestamp('{end_s}')), template='{template}', title='Diagnostics: {dd_var.value} (Reach {dd_reach.value})', lag_hist_K={int(tuple(sorted(list(sel_local_K.value)))[0]) if sel_local_K.value else 1})"
                            )
                            children.append(widgets.HTML("<b>Reproduce diagnostics</b>"))
                            children.append(widgets.HTML(f"<pre style='white-space:pre-wrap'>{call_str}</pre>"))
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
                import threading
                threading.Thread(target=_compute_stats_and_update, args=((x0, x1),), daemon=True).start()
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
                for key in ("flow_series", "swat_flow_series"):
                    s_any = _last.get(key)
                    if isinstance(s_any, pd.Series):
                        sf = s_any.loc[(s_any.index >= x0) & (s_any.index <= x1)]
                        if not sf.empty:
                            vv = sf.to_numpy(dtype=float)
                            vv = vv[np.isfinite(vv)]
                            if vv.size:
                                vals_list.append(vv)
                if vals_list:
                    cc = np.concatenate(vals_list)
                    fmin = float(np.nanmin(cc)); fmax = float(np.nanmax(cc))
                    if fmin == fmax: fmax = fmin + 1.0
                else:
                    fmin, fmax = 0.0, 1.0
                fpad = (fmax - fmin) * 0.05
                fig.layout.yaxis2.update(autorange=False, range=[fmin - fpad, fmax + fpad])
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
            import threading
            threading.Thread(target=_compute_stats_and_update, args=((x0, x1),), daemon=True).start()

        fig.layout.xaxis.on_change(_on_xrange_change, 'range')

        with out:
            clear_output(wait=True)
            display(fig)
        # Trigger stats computation after figure is on screen
        try:
            stats_html.value = "<i>⏳ Computing stats…</i>"
        except Exception:
            pass
        import threading
        threading.Thread(target=_compute_stats_and_update, args=(None,), daemon=True).start()
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
        # Reset measured selections on variable change: auto-pick first chem for each map,
        # and activate only Map 1 by default.
        if measured_present:
            _state["updating"] = True
            for i in (1, 2, 3):
                dd_cat_name[i].value = None
            cb_cat[1].value = True
            cb_cat[2].value = False
            cb_cat[3].value = False
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

    # Stats/dx toggles
    for _w in (dd_lag_metric, sl_max_lag, sel_local_K, cb_log_metrics, cb_show_diags):
        try:
            _w.observe(lambda *_: _mark_stale(), names="value")
        except Exception:
            pass

    # Layout controls
    controls_left = widgets.VBox([num_sim, dd_var, tg_units, dd_method, dd_flow_source, lbl_units])
    base_right_children = [dd_reach, dd_freq, sl_bin, cb_autoscale_y_live, cb_show_names_in_tooltip]

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
    stats_controls = widgets.HBox([dd_lag_metric, sl_max_lag, sel_local_K, cb_log_metrics, cb_show_diags])
    controls = widgets.HBox([controls_left, widgets.HBox([widgets.Label(""), controls_right])])
    stats_row = widgets.HBox([stats_html, diag_box], layout=widgets.Layout(width="100%"))
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
    btn_reload.on_click(_on_reload)
    display(controls, reload_bar, out, stats_controls, stats_row)

    _compute_and_plot()




