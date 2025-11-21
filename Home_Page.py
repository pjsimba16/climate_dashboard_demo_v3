# Home_Page.py — HF snapshot-based home page (availability + global snapshot)
import os
from pathlib import Path
from typing import Optional, Dict, Set, List
from datetime import datetime
import io

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from streamlit_plotly_events import plotly_events
except Exception:  # pragma: no cover
    plotly_events = None

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Manila")
except Exception:  # pragma: no cover
    LOCAL_TZ = None

try:
    import pycountry
except Exception:  # pragma: no cover
    pycountry = None

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Database of Subnational Climate Indicators",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root { --muted:#64748b; }

.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.0rem;
}

/* Medium+ screens: give it a comfortable max width */
@media (min-width: 1200px) {
  .block-container {
    max-width: 1400px;
  }
}

/* Large screens: let it expand more with the viewport */
@media (min-width: 1600px) {
  .block-container {
    max-width: 90vw;
  }
}


h1, h2, h3 { letter-spacing:.2px; }
.subtitle { text-align:center; color:#64748b; margin-top:-.4rem; }

.panel-left{
  position: relative;
  border: 1px solid rgba(235,92,86,0.25);
  border-radius:16px;
  padding:16px 18px;
  margin-bottom:10px;
  background: transparent !important;
  overflow: hidden;
}
.panel-left::before{
  content:"";
  position:absolute;
  inset:0;
  background: linear-gradient(135deg, rgba(235,92,86,0.10), rgba(41,128,185,0.05));
  border-radius: inherit;
  z-index: 0;
}
.panel-left > *{ position: relative; z-index: 1; }

.badgerow {
    display:flex;
    flex-wrap:wrap;
    gap:0.35rem;
    margin-top:0.4rem;
}
.badge {
    display:inline-flex;
    align-items:center;
    padding:0.12rem 0.55rem;
    border-radius:999px;
    font-size:0.75rem;
    background-color:rgba(255,255,255,0.85);
    color:#444;
    border: 1px solid rgba(0,0,0,0.04);
}
.badge b { margin-left:0.15rem; }

.card {
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:12px 14px;
    background:#fafafa;
}

.footer-box {
    padding:16px;
    border-top:1px solid #e5e7eb;
    margin-top:1rem;
    color:#64748b;
    font-size:0.8rem;
}

/* Controls row alignment */
.align-with-input { height: 1.9rem; }
div.row-widget.stSelectbox > label { font-size:0.8rem; }

/* Map: full-bleed container */
.full-bleed { width: 100vw; margin-left: calc(-50vw + 50%); }

/* Plotly tweaks */
[data-testid="stPlotlyChart"] div,
[data-testid="stPlotlyChart"] canvas {
  border-radius: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers & HF snapshot readers
# -----------------------------------------------------------------------------
HF_SPACE_BASE = "https://huggingface.co/spaces/pjsimba16/adb-climate-data/resolve/main"

def _now_label() -> str:
    try:
        now = datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.now()
        return now.strftime("%b %d, %Y %H:%M %Z")
    except Exception:
        return datetime.now().strftime("%b %d, %Y %H:%M")

def _note_err(msg: str) -> None:
    st.session_state.setdefault("hf_errors", []).append(str(msg))

@st.cache_data(ttl=24*3600, show_spinner=False)
def _read_parquet_from_space(rel_path: str, columns: list[str]) -> Optional[pd.DataFrame]:
    """Generic helper to read a parquet file from the HF Space using HTTPS."""
    url = f"{HF_SPACE_BASE}/{rel_path.lstrip('/')}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:  # pragma: no cover
        _note_err(f"HF download failed for {rel_path}: {e}")
        return None
    try:
        return pd.read_parquet(io.BytesIO(resp.content))
    except Exception as e:  # pragma: no cover
        _note_err(f"Parquet read failed for {rel_path}: {e}")
        return None

def _canonical_type(t: str) -> str:
    t = (t or "").strip().lower()
    if t in {"wind", "wind speed", "windspeed", "wind speeds"}:
        return "Wind speeds"
    if t in {"temperature", "temp"}:
        return "Temperature"
    if t in {"precipitation", "rain", "prcp", "precip"}:
        return "Precipitation"
    if t in {"humidity", "humid"}:
        return "Humidity"
    return (t or "").strip().title()

# --- Availability snapshot (HF) ---
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_availability_snapshot() -> pd.DataFrame:
    cols = ["iso3", "date", "indicator_type", "value", "code"]
    df = _read_parquet_from_space("availability_snapshot.parquet", columns=cols)
    if df is None:
        return pd.DataFrame(columns=["iso3", "indicator_type", "freq", "has_data"])
    df = df.copy()
    col_map = {c.lower(): c for c in df.columns}
    iso_col = col_map.get("iso3") or col_map.get("iso")
    type_col = col_map.get("indicator_type") or col_map.get("type")
    freq_col = col_map.get("freq") or col_map.get("frequency")
    has_col = (
        col_map.get("has_data")
        or col_map.get("has_any")
        or col_map.get("available")
    )

    if not iso_col or not type_col:
        _note_err("availability_snapshot.parquet is missing iso3/type columns.")
        return pd.DataFrame(columns=["iso3", "indicator_type", "freq", "has_data"])

    df[iso_col] = df[iso_col].astype(str).str.upper().str.strip()
    df[type_col] = df[type_col].astype(str).map(_canonical_type)
    if has_col and has_col in df.columns:
        df = df[df[has_col].astype(bool)]

    out = pd.DataFrame(
        {
            "iso3": df[iso_col].values,
            "indicator_type": df[type_col].values,
            "freq": df[freq_col].values if freq_col else "Monthly",
        }
    )
    out["has_data"] = True
    return out

@st.cache_data(ttl=24*3600, show_spinner=False)
def _scan_local_availability(_: Path, mapper: Optional[pd.DataFrame]):
    """
    Backward-compatible wrapper name, but powered purely by the HF
    availability_snapshot file (no local country_data scanning).
    """
    snap = _load_availability_snapshot()
    type_to_isos: Dict[str, Set[str]] = {}
    available_types: List[str] = []

    if snap is None or snap.empty:
        return type_to_isos, available_types

    for typ, grp in snap.groupby("indicator_type"):
        ctyp = _canonical_type(typ)
        isos = set(grp["iso3"].astype(str).str.upper().unique().tolist())
        if isos:
            type_to_isos[ctyp] = isos

    available_types = sorted(type_to_isos.keys())
    return type_to_isos, available_types

# --- Global snapshot (HF) ---
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_global_snapshot() -> Optional[pd.DataFrame]:
    cols = ["iso3", "date", "indicator_type", "value", "code"]
    df = _read_parquet_from_space("global_snapshot.parquet", columns=cols)
    if df is None or df.empty:
        return None
    df = df.copy()
    col_map = {c.lower(): c for c in df.columns}
    iso_col = col_map.get("iso3") or col_map.get("iso")
    date_col = col_map.get("date") or col_map.get("time") or col_map.get("period")
    type_col = col_map.get("indicator_type") or col_map.get("type")
    value_col = col_map.get("value") or col_map.get("val")
    code_col = col_map.get("code") or col_map.get("indicator_code")

    if not iso_col or not date_col or not type_col or not value_col:
        _note_err("global_snapshot.parquet is missing required columns.")
        return None

    df[iso_col] = df[iso_col].astype(str).str.upper().str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[type_col] = df[type_col].astype(str).map(_canonical_type)
    df = df.dropna(subset=[date_col])

    rename_map = {
        iso_col: "iso3",
        date_col: "Date",
        type_col: "indicator_type",
        value_col: "value",
    }
    if code_col:
        rename_map[code_col] = "code"

    df = df.rename(columns=rename_map)
    keep_cols = ["iso3", "Date", "indicator_type", "value"] + (
        ["code"] if "code" in df.columns else []
    )
    return df[keep_cols]

REP_CODES = {
    "Temperature": "TMPA",
    "Precipitation": "PCPA",
    "Humidity": "HUMA",
    "Wind speeds": "WSPA",
}

@st.cache_data(ttl=24*3600, show_spinner=False)
def _build_global_series_for_code(code: str) -> Optional[pd.DataFrame]:
    """Slice the global snapshot by representative code or inferred indicator_type."""
    g = _load_global_snapshot()
    if g is None or g.empty:
        return None
    df = g.copy()
    sub = None

    if "code" in df.columns:
        mask = df["code"].astype(str).str.upper() == str(code).upper()
        sub = df[mask]

    if sub is None or sub.empty:
        code_to_type = {v: k for k, v in REP_CODES.items()}
        indicator_type = code_to_type.get(str(code).upper())
        if indicator_type:
            sub = df[
                df["indicator_type"].astype(str).str.lower()
                == indicator_type.lower()
            ]

    if sub is None or sub.empty:
        return None

    out = (
        sub[["iso3", "Date", "value"]]
        .dropna(subset=["Date"])
        .sort_values(["iso3", "Date"])
        .reset_index(drop=True)
    )

    c_up = str(code).upper()
    if c_up == "TMPA":
        out = out[(out["value"] > -80) & (out["value"] < 60)]
    if c_up == "PCPA":
        out = out[(out["value"] >= 0) & (out["value"] < 2000)]
    return out

def _coverage_over_time(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    d = df[["iso3", "Date"]].copy()
    d["Date"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    cov = d.groupby("Date")["iso3"].nunique().reset_index(name="countries")
    return cov.sort_values("Date")

def _latest_by_country(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    last = df.sort_values(["iso3", "Date"]).drop_duplicates(
        subset=["iso3"], keep="last"
    )
    return last["value"]

# -----------------------------------------------------------------------------
# Availability & indicator routing
# -----------------------------------------------------------------------------

def _init_home_with_progress() -> None:
    """
    Initialize all heavy home-page data (availability + global snapshots)
    while showing a big, centered progress bar. This keeps the view focused
    on the loader until everything needed for the page is ready.
    """
    global COUNTRY_DATA_ROOT, INDICATOR_MAPPER
    global type_to_isos, AVAILABLE_INDICATORS, iso_with_data
    global g_temp, g_prec, g_hum, g_wspd, data_through, badge_dt

    # Big centered loading view
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:center;height:80vh;">
              <div style="max-width:520px;width:100%;text-align:center;">
                <h2 style="margin-bottom:0.75rem;">Loading climate indicators…</h2>
                <p style="margin-bottom:1rem;font-size:0.95rem;color:#4b5563;">
                  Preparing availability snapshots and global series. This may take a few moments on first load.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress = st.progress(0)
        status = st.empty()

        total_steps = 3
        step = 0

        def update(message: str) -> None:
            nonlocal step
            step += 1
            pct = min(100, int(step * 100 / total_steps))
            progress.progress(pct)
            status.markdown(f"**{message}**")

        # 1. Availability / indicator list
        update("Loading country availability…")
        COUNTRY_DATA_ROOT = Path("country_data")  # placeholder; we don't use local files
        INDICATOR_MAPPER = None  # no local mapper needed; we use snapshot
        type_to_isos, AVAILABLE_INDICATORS = _scan_local_availability(
            COUNTRY_DATA_ROOT, INDICATOR_MAPPER
        )
        AVAILABLE_INDICATORS = sorted({_canonical_type(t) for t in AVAILABLE_INDICATORS})
        if not AVAILABLE_INDICATORS:
            AVAILABLE_INDICATORS = ["Temperature", "Precipitation"]

        iso_with_data = set().union(*type_to_isos.values()) if type_to_isos else set()
        for t, s in type_to_isos.items():
            st.session_state[f"iso_{t.lower().replace(' ', '_')}"] = s
        st.session_state["iso_with_data"] = iso_with_data

        # 2. Global snapshot series
        update("Loading global monthly snapshots…")
        g_temp = _build_global_series_for_code(REP_CODES["Temperature"])
        g_prec = _build_global_series_for_code(REP_CODES["Precipitation"])
        g_hum  = _build_global_series_for_code(REP_CODES["Humidity"])
        g_wspd = _build_global_series_for_code(REP_CODES["Wind speeds"])

        latest_candidates: List[pd.Timestamp] = []
        for df in (g_temp, g_prec, g_hum, g_wspd):
            if df is not None and not df.empty:
                latest_candidates.append(pd.to_datetime(df["Date"]).max())
        data_through = max(latest_candidates) if latest_candidates else None
        badge_dt = (
            pd.to_datetime(data_through).strftime("%b %Y")
            if data_through is not None
            else "—"
        )

        # 3. Finalizing step
        update("Finalizing dashboard…")

    # Clear loader so the rest of the page can render
    placeholder.empty()


INDICATOR_TO_PAGE = {
    "Temperature": ("pages/1_Temperature_Dashboard.py", "1_Temperature_Dashboard"),
    "Precipitation": ("pages/2_Precipitation_Dashboard.py", "2_Precipitation_Dashboard"),
    "Humidity": ("pages/3_Humidity_Dashboard.py", "3_Humidity_Dashboard"),
    "Wind speeds": ("pages/4_Windspeeds_Dashboard.py", "4_Windspeeds_Dashboard"),
}

# Initialise heavy home-page data once, with a blocking progress bar
_init_home_with_progress()


# -----------------------------------------------------------------------------
# Simple analytics / navigation helpers
# -----------------------------------------------------------------------------
def _log_event(evt: str, payload: dict) -> None:
    st.session_state.setdefault("analytics", [])
    ts = datetime.now(LOCAL_TZ).isoformat() if LOCAL_TZ else datetime.now().isoformat()
    st.session_state["analytics"].append({"ts": ts, "event": evt, **payload})

def _navigate_to_dashboard_immediate(iso3: str, indicator_type: str) -> None:
    ind = indicator_type or "Temperature"
    page_path, page_qp = INDICATOR_TO_PAGE.get(
        ind, INDICATOR_TO_PAGE.get("Temperature")
    )

    type_isos = st.session_state.get(f"iso_{ind.lower().replace(' ', '_')}", set())
    if iso3 not in type_isos:
        if iso3 in st.session_state.get("iso_temperature", set()):
            ind = "Temperature"
            page_path, page_qp = INDICATOR_TO_PAGE["Temperature"]
        elif iso3 in st.session_state.get("iso_precipitation", set()):
            ind = "Precipitation"
            page_path, page_qp = INDICATOR_TO_PAGE["Precipitation"]
        elif iso3 in st.session_state.get("iso_humidity", set()):
            ind = "Humidity"
            page_path, page_qp = INDICATOR_TO_PAGE["Humidity"]
        elif iso3 in st.session_state.get("iso_wind_speeds", set()):
            ind = "Wind speeds"
            page_path, page_qp = INDICATOR_TO_PAGE["Wind speeds"]
        else:
            for tname, isos in type_to_isos.items():
                if iso3 in isos:
                    ind = tname
                    page_path, page_qp = INDICATOR_TO_PAGE.get(
                        ind, INDICATOR_TO_PAGE.get("Temperature")
                    )
                    break

    st.session_state["nav_iso3"] = iso3
    st.query_params.update({"page": page_qp, "iso3": iso3, "city": ""})
    try:
        st.switch_page(page_path)
    except Exception:
        st.rerun()

def _perform_nav_if_pending() -> None:
    nav = st.session_state.get("_pending_nav")
    if not nav:
        return
    iso3 = nav.get("iso3")
    indicator = nav.get("indicator") or "Temperature"
    st.session_state["_pending_nav"] = None
    if iso3:
        _navigate_to_dashboard_immediate(iso3, indicator)

_perform_nav_if_pending()

# -----------------------------------------------------------------------------
# Title & hero
# -----------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center'>Global Database of Subnational Climate Indicators</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Built and Maintained by Roshen Fernando and Patrick Jaime Simba</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='text-align:center; font-size:0.8rem; color:#777;'>Last refreshed: {_now_label()}</div>",
    unsafe_allow_html=True,
)
st.divider()

hero_left, hero_right = st.columns([0.62, 0.38], gap="large")
with hero_left:
    st.markdown(
        f"""
        <div class="panel-left">
          <h2>🌍 Explore subnational climate indicators worldwide</h2>
          <p>Click a country on the map to open its dashboard, or use Quick search to jump directly.</p>
          <div class="badgerow">
            <span class="badge">Data through: <b>{badge_dt}</b></span>
            <span class="badge">Snapshot source: <b>ADM0 Monthly TMPA/PCPA/HUMA/WSPA</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.subheader("Custom Chart Builder")
    st.write(
        "Create bespoke charts across countries and ADM1s, compare indicators, facet by indicator or geography, "
        "smooth, normalize and export."
    )
    if st.button("📈 Generate a custom chart", key="hero_custom_chart"):
        try:
            st.switch_page("pages/0_Custom_Chart.py")
        except Exception:
            st.rerun()
    st.caption("Starts a flexible chart workspace with export options.")

# --- What's inside this dashboard? (moved up under hero) ---
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
with st.expander("What’s inside this dashboard?", expanded=False):
    st.markdown(
        """
- **Geography:** Countries and first-level administrative regions (ADM1); selected cities for context.
- **Indicators:** Temperature, precipitation, humidity, and wind speeds (more types supported by the data; pages added progressively).
- **Temporal frequency:** Monthly; the global snapshot here uses ADM0 Monthly TMPA/PCPA/HUMA/WSPA.
- **Latency:** Updates typically published within weeks of source release.
- **Method summary:** Area-weighted aggregation of gridded reanalysis fields to administrative boundaries.
- **Caveats:** Administrative boundary changes, data gaps, and reanalysis corrections can affect comparability over time.
        """
    )

# First-load tip
if "first_load_hint" not in st.session_state:
    st.info("Tip: first load warms the HF snapshot cache; subsequent loads should be faster.", icon="💡")
    st.session_state["first_load_hint"] = True

# -----------------------------------------------------------------------------
# Controls row
# -----------------------------------------------------------------------------
if "region_scope" not in st.session_state:
    st.session_state["region_scope"] = "World"
if "default_indicator" not in st.session_state:
    st.session_state["default_indicator"] = (
        "Temperature" if "Temperature" in AVAILABLE_INDICATORS else AVAILABLE_INDICATORS[0]
    )

# Build countries master table
if pycountry is not None:
    all_countries = pd.DataFrame(
        [
            {"iso3": c.alpha_3, "name": c.name}
            for c in pycountry.countries
            if hasattr(c, "alpha_3")
        ]
    )
else:
    all_countries = pd.DataFrame({"iso3": sorted(list(iso_with_data))})
    all_countries["name"] = all_countries["iso3"]

all_countries["iso3"] = all_countries["iso3"].astype(str).str.upper().str.strip()
_name_overrides = {
    "CHN": "People's Republic of China",
    "TWN": "Taipei, China",
    "HKG": "Hong Kong, China",
}
all_countries["name"] = all_countries.apply(
    lambda r: _name_overrides.get(r["iso3"], r.get("name", r["iso3"])), axis=1
)

for t in AVAILABLE_INDICATORS:
    key = f"has_{t.lower().replace(' ', '_')}"
    all_countries[key] = all_countries["iso3"].isin(type_to_isos.get(t, set()))
all_countries["has_data_any"] = False
for t in AVAILABLE_INDICATORS:
    all_countries["has_data_any"] |= all_countries[f"has_{t.lower().replace(' ', '_')}"]

def _badges(iso: str) -> str:
    hits = [t for t in AVAILABLE_INDICATORS if iso in type_to_isos.get(t, set())]
    return " • ".join(hits) if hits else "—"

all_countries["badges"] = all_countries["iso3"].map(_badges)

# Continent helper
@st.cache_data(show_spinner=False)
def _continent_lookup() -> Dict[str, str]:
    gm = px.data.gapminder()
    base = dict(zip(gm["iso_alpha"], gm["continent"]))
    south_america = {
        "ARG","BOL","BRA","CHL","COL","ECU","GUY","PRY","PER","SUR","URY","VEN","FLK","GUF"
    }
    north_america = {
        "USA","CAN","MEX","GTM","BLZ","HND","SLV","NIC","CRI","PAN","CUB","DOM","HTI","JAM",
        "TTO","BRB","BHS","ATG","DMA","GRD","KNA","LCA","VCT","ABW","BES","BMU","CUW","GLP",
        "GRL","MTQ","MSR","PRI","SXM","SJM","TCA","VGB","VIR"
    }
    out: Dict[str, str] = {}
    for iso, cont in base.items():
        if cont == "Americas" and iso in south_america:
            out[iso] = "South America"
        elif cont == "Americas":
            out[iso] = "North America"
        else:
            out[iso] = cont
    return out

CONTINENT_OF = _continent_lookup()

def _isos_for_region(region_name: str, all_isos: pd.Series) -> Set[str]:
    if region_name == "World":
        return set(all_isos.tolist())
    res = {
        iso for iso in all_isos if CONTINENT_OF.get(iso, None) == region_name
    }
    return res or set(all_isos.tolist())

quick_opts = ["— Type to search —"] + sorted(all_countries["name"].tolist())
c1, c2, c3, c4 = st.columns([1.0, 0.38, 0.32, 0.20])
with c1:
    chosen = st.selectbox(
        "Quick search",
        quick_opts,
        index=0,
        help="Type a country name, or click a country on the map to navigate.",
    )
with c2:
    st.selectbox(
        "Default indicator",
        AVAILABLE_INDICATORS if AVAILABLE_INDICATORS else ["Temperature", "Precipitation"],
        key="default_indicator",
        help="Which dashboard opens when you click a country or use Quick search.",
    )
with c3:
    st.selectbox(
        "View",
        ["World", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"],
        key="region_scope",
        help="Change geographic scope.",
    )
with c4:
    st.markdown("<div class='align-with-input'></div>", unsafe_allow_html=True)
    if st.button("Reset view", use_container_width=True):
        st.session_state["region_scope"] = "World"
        _log_event("reset_view", {"to": "World"})

# Quick search navigation
if chosen and chosen != "— Type to search —":
    row = all_countries[all_countries["name"] == chosen]
    if not row.empty:
        iso3_sel = str(row.iloc[0]["iso3"]).upper()
        if iso3_sel in iso_with_data:
            _log_event(
                "quick_search_open",
                {"iso3": iso3_sel, "indicator": st.session_state.get("default_indicator", "Temperature")},
            )
            st.session_state["_pending_nav"] = {
                "iso3": iso3_sel,
                "indicator": st.session_state.get("default_indicator", "Temperature"),
            }
            st.rerun()

# -----------------------------------------------------------------------------
# World map (bigger & full-bleed)
# -----------------------------------------------------------------------------
# World map (bigger & full-bleed)
# -----------------------------------------------------------------------------
st.markdown("### Country coverage overview")
st.caption(
    "Shaded countries have at least one ADM0 climate indicator available in the database."
)

# Filter to the region in the current selector
region_isos = _isos_for_region(st.session_state["region_scope"], all_countries["iso3"])
plot_df = all_countries[all_countries["iso3"].isin(region_isos)].copy()

# Optional debug line (you already used something like this)
st.write(
    f"DEBUG world map: region_scope = {st.session_state['region_scope']} "
    f"rows = {len(plot_df)}"
)

if plot_df.empty:
    st.info("No countries available for this region yet.")
else:
    plot_df["hovertext"] = plot_df.apply(
        lambda r: f"{r['name']}<br><span>Indicators: {r['badges']}</span>",
        axis=1,
    )
    plot_df["val"] = plot_df["has_data_any"].astype(float)

    map_h = 800  # vertical height
    scope_map = {
        "World": "world",
        "Africa": "africa",
        "Asia": "asia",
        "Europe": "europe",
        "North America": "north america",
        "South America": "south america",
        "Oceania": "world",
    }

    fig = go.Figure(
        go.Choropleth(
            locations=plot_df["iso3"],
            z=plot_df["val"],
            locationmode="ISO-3",
            colorscale=[[0.0, "#d4d4d8"], [1.0, "#12a39a"]],
            zmin=0.0,
            zmax=1.0,
            autocolorscale=False,
            showscale=False,
            hoverinfo="text",
            text=plot_df["hovertext"],
            customdata=plot_df[["iso3"]].to_numpy(),
            marker_line_width=1.6,
            marker_line_color="rgba(0,0,0,0.70)",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=map_h,
        geo=dict(
            scope=scope_map.get(st.session_state["region_scope"], "world"),
            projection_type="natural earth",
            showland=True,
            landcolor="#f9fafb",
            showcountries=True,
            countrycolor="#9ca3af",
            showocean=True,
            oceancolor="#eff6ff",
        ),
    )

    # World view: slightly constrain lat so it doesn't look too zoomed out
    if st.session_state["region_scope"] == "World":
        fig.update_geos(lataxis_range=[-60, 85])

    # Full-bleed wrapper (same CSS you already have at the top)
    st.markdown('<div class="full-bleed">', unsafe_allow_html=True)

    events = []
    if plotly_events is not None:
        # On Streamlit Cloud, this call can fail if the lib is missing or buggy.
        # Wrap in try/except and fall back to a normal Plotly chart.
        try:
            events = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                override_height=map_h,
                override_width="100%",
            )
        except Exception:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            events = []
    else:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
        events = []

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle click-to-open for countries (if we got events from plotly_events)
    clicked_iso3 = None
    if events:
        e = events[0]
        if isinstance(e, dict):
            idx = e.get("pointIndex", None)
            if idx is not None and 0 <= idx < len(plot_df):
                clicked_iso3 = str(plot_df.iloc[idx]["iso3"]).upper()

    if clicked_iso3 and clicked_iso3 in iso_with_data:
        _log_event(
            "map_click_open",
            {
                "iso3": clicked_iso3,
                "indicator": st.session_state.get(
                    "default_indicator", "Temperature"
                ),
            },
        )
        st.session_state["_pending_nav"] = {
            "iso3": clicked_iso3,
            "indicator": st.session_state.get(
                "default_indicator", "Temperature"
            ),
        }
        st.rerun()


# -----------------------------------------------------------------------------
# Global snapshot section
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Global snapshot (ADM0 Monthly, representative indicators)")

cov_t = _coverage_over_time(g_temp)
cov_p = _coverage_over_time(g_prec)
cov_h = _coverage_over_time(g_hum)
cov_w = _coverage_over_time(g_wspd)

if any(d is not None for d in (cov_t, cov_p, cov_h, cov_w)):
    cov_fig = go.Figure()
    if cov_t is not None:
        cov_fig.add_trace(
            go.Scatter(
                x=cov_t["Date"],
                y=cov_t["countries"],
                mode="lines",
                name="Temperature",
            )
        )
    if cov_p is not None:
        cov_fig.add_trace(
            go.Scatter(
                x=cov_p["Date"],
                y=cov_p["countries"],
                mode="lines",
                name="Precipitation",
            )
        )
    if cov_h is not None:
        cov_fig.add_trace(
            go.Scatter(
                x=cov_h["Date"],
                y=cov_h["countries"],
                mode="lines",
                name="Humidity",
            )
        )
    if cov_w is not None:
        cov_fig.add_trace(
            go.Scatter(
                x=cov_w["Date"],
                y=cov_w["countries"],
                mode="lines",
                name="Wind speeds",
            )
        )
    cov_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        xaxis_title="Month",
        yaxis_title="Countries with data",
    )
    st.plotly_chart(cov_fig, use_container_width=True)
else:
    st.info(
        "Global coverage over time (ADM0 Monthly) is unavailable yet.",
        icon="ℹ️",
    )

# Latest datapoint histograms (two rows)
hcol1, hcol2 = st.columns(2)
with hcol1:
    s = _latest_by_country(g_temp)
    if s is not None and not s.empty:
        hist_t = px.histogram(
            pd.DataFrame({"value": s}),
            x="value",
            nbins=30,
            title="Latest datapoint per country — Temperature (TMPA)",
        )
        hist_t.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=260,
            xaxis_title="Temperature (°C)",
            yaxis_title="Countries",
        )
        st.plotly_chart(hist_t, use_container_width=True)
    else:
        st.info(
            "Latest temperature distribution unavailable (ADM0 Monthly TMPA).",
            icon="ℹ️",
        )

with hcol2:
    s = _latest_by_country(g_prec)
    if s is not None and not s.empty:
        hist_p = px.histogram(
            pd.DataFrame({"value": s}),
            x="value",
            nbins=30,
            title="Latest datapoint per country — Precipitation (PCPA)",
        )
        hist_p.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=260,
            xaxis_title="Precipitation",
            yaxis_title="Countries",
        )
        st.plotly_chart(hist_p, use_container_width=True)
    else:
        st.info(
            "Latest precipitation distribution unavailable (ADM0 Monthly PCPA).",
            icon="ℹ️",
        )

hcol3, hcol4 = st.columns(2)
with hcol3:
    s = _latest_by_country(g_hum)
    if s is not None and not s.empty:
        hist_h = px.histogram(
            pd.DataFrame({"value": s}),
            x="value",
            nbins=30,
            title="Latest datapoint per country — Humidity (HUMA)",
        )
        hist_h.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=260,
            xaxis_title="Humidity (%)",
            yaxis_title="Countries",
        )
        st.plotly_chart(hist_h, use_container_width=True)
    else:
        st.info(
            "Latest humidity distribution unavailable (ADM0 Monthly HUMA).",
            icon="ℹ️",
        )

with hcol4:
    s = _latest_by_country(g_wspd)
    if s is not None and not s.empty:
        hist_w = px.histogram(
            pd.DataFrame({"value": s}),
            x="value",
            nbins=30,
            title="Latest datapoint per country — Wind speeds (WSPA)",
        )
        hist_w.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=260,
            xaxis_title="Wind speed (m/s)",
            yaxis_title="Countries",
        )
        st.plotly_chart(hist_w, use_container_width=True)
    else:
        st.info(
            "Latest wind speeds distribution unavailable (ADM0 Monthly WSPA).",
            icon="ℹ️",
        )

# ---------- Coverage & Data sources (restored) ----------
st.divider()
k1, k2 = st.columns(2)
with k1:
    st.markdown(
        f"""
    <div class="card">
      <div style="font-size:13px;color:#64748b;">Coverage</div>
      <div style="font-size:22px;margin:.15rem 0;">
        <strong>{int(all_countries['has_data_any'].sum())}</strong> countries with at least one indicator
      </div>
      <div style="font-size:13px;color:#475569;">
        Indicators shown: {(" • ".join(AVAILABLE_INDICATORS)) if AVAILABLE_INDICATORS else "Temperature • Precipitation"}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        """
    <div class="card">
      <div style="font-size:13px;color:#64748b; display:flex; align-items:center; gap:6px;">
        Data sources
        <span title="ERA5 reanalysis (ECMWF). Gridded climate fields; aggregated to country and ADM1.">
          ⓘ
        </span>
      </div>
      <div style="font-size:22px;margin:.15rem 0;"><strong>ERA5</strong></div>
      <div style="font-size:13px;color:#475569;">Additional integrations under consideration</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Optional admin analytics
# -----------------------------------------------------------------------------
if st.query_params.get("admin", ["0"])[0] == "1":
    st.divider()
    st.subheader("Admin: Session analytics")
    logs = st.session_state.get("analytics", [])
    if logs:
        df_log = pd.DataFrame(logs)
        st.dataframe(df_log)
    else:
        st.info("No events logged yet in this session.", icon="ℹ️")

st.markdown(
    """
<div class="footer-box">
  <em>Note:</em> This page time-stamps the "Last refreshed" at render time (Asia/Manila).
</div>
""",
    unsafe_allow_html=True,
)
