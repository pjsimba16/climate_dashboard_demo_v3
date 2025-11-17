# pages/0_Custom_Chart.py
# Custom Chart Builder — ADM0/ADM1 by freq + 4 main indicators + global date fallback
import os, math
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

# Try plotly-resampler (optional)
try:
    from plotly_resampler import FigureResampler
    _HAS_RESAMPLER = True
except Exception:
    _HAS_RESAMPLER = False

try:
    import plotly.io as pio
    pio.json.config.default_engine = "json"
except Exception:
    pass

# =================== PAGE CONFIG ===================
st.set_page_config(page_title="Custom Chart Builder", layout="wide", initial_sidebar_state="collapsed")

# =================== STYLES ===================
st.markdown("""
<style>
h1.custom-title{ text-align:center; font-size:2.1rem; margin:0.6rem 0 0.35rem 0; }
.topbar{ margin-top:8px; }
.helper-text{ text-align:center; color:#4a4a4a; margin-bottom:0.8rem; }

/* badges */
.badge-row{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px 0; justify-content:center; }
.badge{ background:#f1f3f5; border:1px solid #e1e4e8; border-radius:999px; padding:4px 10px; font-size:.85rem; }

/* primary buttons */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stDownloadButton"] button,
button[kind="primary"]{
  background-color:#f55551 !important; color:white !important; border:0 !important; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# =================== COLORS ===================
CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
}
TRACE_PALETTE = [CBLIND[k] for k in ["blue","verm","green","orange","navy","pink","sky","grey","yellow"]]
def _color(i:int)->str: return TRACE_PALETTE[i % len(TRACE_PALETTE)]

# =================== COUNTRY DISPLAY ===================
try:
    import pycountry
except Exception:
    pycountry = None

def iso3_to_name(iso:str)->str:
    iso=(iso or "").upper().strip()
    if pycountry:
        try:
            c=pycountry.countries.get(alpha_3=iso)
            if c and getattr(c,"name",None): return c.name
        except Exception: pass
    return iso

_CUSTOM = {"CHN":"People's Republic of China","TWN":"Taipei, China","HKG":"Hong Kong, China"}
def display_country_name(iso:str)->str: return _CUSTOM.get((iso or "").upper().strip(), iso3_to_name(iso))

# =================== BACKEND SELECTOR ===================
DATA_BACKEND_OVERRIDE = "auto"  # "local" | "hf" | "auto"

def _secret_or_env(k, default=""):
    try:
        if hasattr(st,"secrets") and k in st.secrets: return st.secrets[k]
    except Exception: pass
    return os.getenv(k, default)

HF_REPO_ID   = _secret_or_env("HF_REPO_ID","pjsimba16/adb_climate_dashboard_v1")
HF_REPO_TYPE = _secret_or_env("HF_REPO_TYPE","space")  # space per your note
HF_TOKEN     = _secret_or_env("HF_TOKEN","") or None

def _resolve_backend() -> str:
    ov = str(DATA_BACKEND_OVERRIDE or "").lower()
    if ov in ("local","hf","auto"): return ov
    try:
        qp = st.query_params.get("backend", [])
        if qp:
            v = str(qp[0]).lower()
            if v in ("local","hf","auto"): return v
    except Exception: pass
    try:
        v = str(st.secrets.get("DATA_BACKEND","auto")).lower() if hasattr(st,"secrets") else "auto"
        if v in ("local","hf","auto"): return v
    except Exception: pass
    v = os.getenv("DATA_BACKEND","auto").lower()
    return v if v in ("local","hf","auto") else "auto"

DATA_BACKEND = _resolve_backend()

# Paths (country_data one level up from /pages/)
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
COUNTRY_DATA = ROOT / "country_data"

@st.cache_data(ttl=24*3600, show_spinner=False)
def _dl(relpath:str)->str:
    # relpath like "country_data/AFG/Monthly/AFG_ADM0_data.parquet"
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=relpath, token=HF_TOKEN)

# =================== ADM1 LIST (from mapper CSV one level up) ===================
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_city_map() -> pd.DataFrame:
    for fname in ["city_mapper_with_coords_v3.csv","city_mapper_with_coords_v2.csv"]:
        p = ROOT / fname
        if p.exists():
            try: return pd.read_csv(p)
            except Exception: pass
    # fall back empty
    return pd.DataFrame(columns=["Country","City"])

CITY_MAP = _load_city_map()
ADM1_ALL = (CITY_MAP.rename(columns={"Country":"ADM0","City":"ADM1"})[["ADM0","ADM1"]]
            .dropna().astype(str))
ADM1_ALL["ADM0"] = ADM1_ALL["ADM0"].str.upper().str.strip()
ADM1_ALL["ADM1"] = ADM1_ALL["ADM1"].str.strip()
ADM1_CHOICES = sorted(ADM1_ALL.drop_duplicates().itertuples(index=False, name=None))

# =================== AVAILABILITY (scan local or HF) ===================
MAIN_CODES = {"Temperature":"TMPA","Precipitation":"PCPA","Humidity":"HUMA","Wind speeds":"WSPA"}

def _suffixes(level: str, freq: str) -> List[str]:
    # level: "ADM0" or "ADM1"
    if freq=="Annual":
        return ["_A"]
    if freq=="Monthly":
        return ["_AM","_PM","_M"] if level=="ADM0" else ["_M"]
    if freq=="Seasonal":
        return ["_AS","_PS","_S"] if level=="ADM0" else ["_S"]
    return []

@st.cache_data(ttl=24*3600, show_spinner=False)
def _list_local_isos() -> List[str]:
    if not COUNTRY_DATA.exists(): return []
    return sorted([p.name.upper() for p in COUNTRY_DATA.iterdir() if p.is_dir() and len(p.name)==3])

@st.cache_data(ttl=24*3600, show_spinner=False)
def _list_hf_isos() -> List[str]:
    try:
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
        # Expect entries like country_data/AFG/Monthly/AFG_ADM0_data.parquet
        isos=set()
        for f in files:
            parts=f.split("/")
            if len(parts)>=4 and parts[0]=="country_data" and len(parts[1])==3:
                isos.add(parts[1].upper())
        return sorted(isos)
    except Exception:
        return []

@st.cache_data(ttl=24*3600, show_spinner=False)
def available_isos() -> List[str]:
    if DATA_BACKEND in ("hf","auto"):
        h = _list_hf_isos()
        if h: return h
    return _list_local_isos()

ALL_ISOS = available_isos()
ISO_NAME_MAP = {iso:display_country_name(iso) for iso in ALL_ISOS}

# =================== BUILD DATES ===================
def _to_date_from_parts(df: pd.DataFrame, freq: str) -> pd.Series:
    if freq=="Monthly":
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        return pd.to_datetime(dict(year=y, month=m, day=1), errors="coerce")
    if freq=="Seasonal":
        if "Year" in df.columns and "Season" in df.columns:
            y = pd.to_numeric(df["Year"], errors="coerce")
            s = df["Season"].astype(str).str.upper().str.extract(r"Q(\d)", expand=False).astype(float)
            s = s.where(s.isin([1,2,3,4]), np.nan)
            month = s.map({1:1,2:4,3:7,4:10})
            return pd.to_datetime(dict(year=y, month=month, day=1), errors="coerce")
        else:
            s = df["Season"].astype(str)
            year = s.str.extract(r"(\d{4})", expand=False).astype(float)
            q = s.str.extract(r"Q([1-4])", expand=False).astype(float)
            month = q.map({1:1,2:4,3:7,4:10})
            return pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")
    y = pd.to_numeric(df["Year"], errors="coerce")
    return pd.to_datetime(dict(year=y, month=1, day=1), errors="coerce")

# =================== FILE HELPERS ===================
def _adm0_fp_local(iso: str, freq: str) -> Path:
    return COUNTRY_DATA / iso / freq / f"{iso}_ADM0_data.parquet"

def _adm1_fp_local(iso: str, adm1: str, freq: str) -> Path:
    return COUNTRY_DATA / iso / freq / f"{adm1}_ADM1_data.parquet"

def _adm0_fp_hf(iso: str, freq: str) -> str:
    return f"country_data/{iso}/{freq}/{iso}_ADM0_data.parquet"

def _adm1_fp_hf(iso: str, adm1: str, freq: str) -> str:
    return f"country_data/{iso}/{freq}/{adm1}_ADM1_data.parquet"

# =================== READER CORE ===================
def _pick_col(cols: List[str], code: str, sfx_list: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for sfx in sfx_list:
        key = f"{code}{sfx}".lower()
        if key in lower: return lower[key]
    return None

def _read_parquet_local(path: Path, columns: Optional[List[str]]=None) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

def _read_parquet_hf(relpath: str, columns: Optional[List[str]]=None) -> Optional[pd.DataFrame]:
    try:
        p = _dl(relpath)
        return pd.read_parquet(p, columns=columns)
    except Exception:
        try:
            p = _dl(relpath)
            return pd.read_parquet(p)
        except Exception:
            return None

def _read_frame(level: str, iso: str, freq: str, adm1: Optional[str]) -> Optional[pd.DataFrame]:
    if level=="ADM0":
        if DATA_BACKEND in ("hf","auto"):
            d = _read_parquet_hf(_adm0_fp_hf(iso, freq))
            if isinstance(d, pd.DataFrame): return d
        return _read_parquet_local(_adm0_fp_local(iso, freq))
    else:
        if DATA_BACKEND in ("hf","auto"):
            d = _read_parquet_hf(_adm1_fp_hf(iso, adm1, freq))
            if isinstance(d, pd.DataFrame): return d
        return _read_parquet_local(_adm1_fp_local(iso, adm1, freq))

def _single_series(level: str, iso: str, freq: str, code: str, adm1: Optional[str]) -> pd.DataFrame:
    """
    Return standardized frame with columns:
      level, iso3, adm1, date, <code_lower>
    """
    df = _read_frame(level, iso, freq, adm1)
    if df is None or df.empty:
        return pd.DataFrame(columns=["level","iso3","adm1","date",code.lower()])
    df = df.copy()
    needed_time = {"Monthly":["Year","Month"], "Seasonal":["Season"], "Annual":["Year"]}[freq]
    for c in needed_time:
        if c not in df.columns:
            if not (freq=="Seasonal" and c=="Year"):
                return pd.DataFrame(columns=["level","iso3","adm1","date",code.lower()])
    sfx = _suffixes("ADM0" if level=="ADM0" else "ADM1", freq)
    col = _pick_col(list(df.columns), code, sfx)
    if not col:
        return pd.DataFrame(columns=["level","iso3","adm1","date",code.lower()])
    dt = _to_date_from_parts(df, freq)
    out = pd.DataFrame({
        "level": level,
        "iso3": iso,
        "adm1": (adm1 or ""),
        "date": dt,
        code.lower(): pd.to_numeric(df[col], errors="coerce")
    })
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

# =================== HIGH-LEVEL BUILDERS ===================
def _build_country_series(iso: str, freq: str) -> pd.DataFrame:
    # Returns columns: level, iso3, adm1, date, tmpa, pcpa, huma, wspa
    frames=[]
    for code in ["TMPA","PCPA","HUMA","WSPA"]:
        s = _single_series("ADM0", iso, freq, code, None)
        if s.empty: continue
        frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["level","iso3","adm1","date","tmpa","pcpa","huma","wspa"])
    d = frames[0]
    for s in frames[1:]:
        d = d.merge(s, on=["level","iso3","adm1","date"], how="outer")
    return d

def _build_adm1_series_for_country(iso: str, freq: str, only_pairs: Optional[List[Tuple[str,str]]]=None) -> pd.DataFrame:
    # Load every ADM1 time-series (or filter to only_pairs), then concat
    cand = ADM1_ALL[ADM1_ALL["ADM0"]==iso]
    if cand.empty:
        return pd.DataFrame(columns=["level","iso3","adm1","date","tmpa","pcpa","huma","wspa"])
    rows=[]
    targets = set(only_pairs) if only_pairs else None
    for adm1 in cand["ADM1"].unique():
        if targets and (iso, adm1) not in targets: 
            continue
        frames=[]
        for code in ["TMPA","PCPA","HUMA","WSPA"]:
            s = _single_series("ADM1", iso, freq, code, adm1)
            if s.empty: continue
            frames.append(s)
        if not frames:
            continue
        d = frames[0]
        for s in frames[1:]:
            d = d.merge(s, on=["level","iso3","adm1","date"], how="outer")
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["level","iso3","adm1","date","tmpa","pcpa","huma","wspa"])
    return pd.concat(rows, ignore_index=True)

# =================== PRE-AGGREGATE (no-op here) ===================
def preaggregate(df:pd.DataFrame, _value_cols:List[str])->Dict[str,pd.DataFrame]:
    return {"M":df, "Q":df, "A":df}

# =================== HEADER ===================
top = st.container()
with top:
    c1,c2,c3 = st.columns([0.12,0.76,0.12], gap="small")
    with c1:
        st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
        if st.button("← Home"):
            st.query_params.clear()
            try: st.switch_page("Home_Page.py")
            except Exception: st.rerun()
    with c2:
        st.markdown("<h1 class='custom-title'>Custom Chart Builder</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='helper-text'>Build and compare custom climate charts across countries and ADM1s. "
            "Choose indicators, chart types (including anomalies), frequency, date windows, and facets; "
            "apply smoothing, normalization, and climatology overlays; then export the figure and data.</div>",
            unsafe_allow_html=True
        )

# Defaults
if "chart_type" not in st.session_state: st.session_state["chart_type"]="Line"

# =================== CHART TYPE ===================
chart_type = st.segmented_control(
    "Chart type",
    options=["Line","Area","Bar","Scatter","Anomaly"],
    selection_mode="single",
    key="chart_type",
)

# =================== GLOBAL DATE FALLBACK ===================
@st.cache_data(ttl=24*3600, show_spinner=False)
def _global_minmax(freq: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find global min/max dates across ADM0 for any of the 4 main indicators.
    """
    def _find_any_col(cols: List[str], freq: str) -> Optional[str]:
        for code in ["TMPA","PCPA","HUMA","WSPA"]:
            col = _pick_col(cols, code, _suffixes("ADM0", freq))
            if col: return col
        return None

    dmins, dmaxs = [], []
    for iso in ALL_ISOS:
        f = _adm0_fp_local(iso, freq)
        df = _read_parquet_local(f) if f.exists() else None
        if df is None and DATA_BACKEND in ("hf","auto"):
            df = _read_parquet_hf(_adm0_fp_hf(iso, freq))
        if df is None or df.empty:
            continue

        col = _find_any_col(list(df.columns), freq)
        if not col:
            continue
        try:
            df2 = df.copy()
            df2["Date"] = _to_date_from_parts(df2, freq)
            df2 = df2.dropna(subset=["Date"])
            if not df2.empty:
                dmins.append(df2["Date"].min())
                dmaxs.append(df2["Date"].max())
        except Exception:
            continue

    if dmins and dmaxs:
        return (min(dmins), max(dmaxs))
    return (pd.Timestamp("1980-01-01"), pd.Timestamp.today().normalize())

# =================== OPTIONS ===================
with st.expander("Options", expanded=True):
    with st.form("controls"):
        st.markdown("#### Selections")

        # Geographies
        iso_opts = sorted(ALL_ISOS, key=lambda x: ISO_NAME_MAP.get(x,x))
        sel_countries = st.multiselect(
            "Countries (optional)", options=iso_opts, format_func=lambda x: ISO_NAME_MAP.get(x,x),
        )
        def _adm1label(pair): return f"{display_country_name(pair[0])} — {pair[1]}"
        sel_adm1 = st.multiselect(
            "ADM1 (optional)", options=ADM1_CHOICES, format_func=_adm1label,
        )

        # Frequency — drives which parquet and date constructor we use
        freq = st.radio(
            "Frequency", ["Monthly","Seasonal","Annual"], index=0, horizontal=True,
        )

        # Date bounds (GLOBAL FALLBACK now ensures sliders always visible)
        @st.cache_data(ttl=1200, show_spinner=False)
        def _minmax_from_selection(countries: Tuple[str,...], adm1s: Tuple[Tuple[str,str],...], freq: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
            frames=[]
            for iso in countries or ():
                frames.append(_build_country_series(iso, freq))
            if adm1s:
                by_iso = {}
                for iso, name in adm1s:
                    by_iso.setdefault(iso, set()).add(name)
                for iso, names in by_iso.items():
                    frames.append(_build_adm1_series_for_country(iso, freq, only_pairs=[(iso,n) for n in names]))
            if not frames:
                return _global_minmax(freq)
            all_dates = pd.concat([f["date"] for f in frames if not f.empty], ignore_index=True)
            if all_dates.empty:
                return _global_minmax(freq)
            return (all_dates.min(), all_dates.max())

        dmin, dmax = _minmax_from_selection(tuple(sel_countries), tuple(sel_adm1), freq)

        # Quick select + slider ALWAYS shown
        quick = st.segmented_control("Quick select", options=["10y","20y","30y","50y","All"], selection_mode="single", key="quick_sel")
        q_years = {"10y":10,"20y":20,"30y":30,"50y":50}.get(quick, None)
        default_start = (dmax - pd.DateOffset(years=q_years)).date() if q_years else dmin.date()
        default_start = max(default_start, dmin.date())

        dstart, dend = st.slider(
            "Custom range",
            min_value=dmin.date(), max_value=dmax.date(),
            value=(default_start, dmax.date()),
            format="YYYY-MM",
        )

        st.markdown("---")
        facet_by = st.selectbox("Facet", ["None","Geography (Country/ADM1)","Indicator"], index=0)

        show_points = st.checkbox("Show markers", value=False)
        hide_title  = st.checkbox("Hide chart title", value=False)

        global_window = 0
        enable_per_trace = False
        normalize_ix = False
        clim_overlay = False

        # Indicators include Humidity & Wind speeds (main indicators)
        IND_OPTIONS = ["Temperature (TMPA, °C)","Precipitation (PCPA, mm)","Humidity (HUMA, %)","Wind speeds (WSPA, m/s)"]

        if chart_type in ("Line","Area","Bar"):
            indicators = st.multiselect(
                "Indicator(s)", IND_OPTIONS, default=["Temperature (TMPA, °C)"],
                help="Select one or more main indicators."
            )
            with st.expander("Advanced (time-series)"):
                global_window = st.number_input("Smoothing window (points, 0 = off)", min_value=0, value=0, step=1)
                enable_per_trace = st.checkbox("Custom smoothing per trace")
                normalize_ix = st.checkbox("Normalize to index=100 at start date (disables dual axes)")
                clim_overlay = st.checkbox("Overlay seasonal climatology (monthly mean by calendar month)")

        elif chart_type=="Scatter":
            x_series = st.selectbox("X-axis series", IND_OPTIONS, index=0)
            y_series = st.selectbox("Y-axis series", IND_OPTIONS, index=1)
            with st.expander("Advanced (scatter)"):
                global_window = st.number_input("Smoothing window for connecting lines (0 = off)", min_value=0, value=0, step=1)

        else:  # Anomaly
            an_indicator = st.selectbox("Indicator", IND_OPTIONS, index=0)

            @st.cache_data(ttl=1200, show_spinner=False)
            def _baseline_bounds(freq: str, countries: Tuple[str, ...], adm1s: Tuple[Tuple[str, str], ...]) -> Tuple[int, int]:
                frames=[]
                for iso in countries or ():
                    frames.append(_build_country_series(iso, freq))
                if adm1s:
                    by_iso={}
                    for iso,name in adm1s:
                        by_iso.setdefault(iso,set()).add(name)
                    for iso, names in by_iso.items():
                        frames.append(_build_adm1_series_for_country(iso, freq, only_pairs=[(iso,n) for n in names]))
                if not frames:
                    gmin,gmax=_global_minmax(freq)
                    return (gmin.year, gmax.year)
                all_dates = pd.concat([f["date"] for f in frames if not f.empty], ignore_index=True)
                if all_dates.empty:
                    gmin,gmax=_global_minmax(freq)
                    return (gmin.year, gmax.year)
                return (all_dates.dt.year.min().item(), all_dates.dt.year.max().item())

            bmin, bmax = _baseline_bounds(freq, tuple(sel_countries), tuple(sel_adm1))
            base_start, base_end = st.slider("Baseline years", min_value=int(bmin), max_value=int(bmax),
                                             value=(max(int(bmin),1991), min(int(bmax),2020)))
            baseline_calc = st.selectbox("Baseline calculation method", ["Mean","Median","Min","Max"])
            an_method  = st.selectbox("Anomaly calculation method",
                                      ["Absolute (value - baseline)", "Percent of baseline", "Z-score (std from baseline)"])
            with st.expander("Advanced (anomalies)"):
                clim_overlay = st.checkbox("Overlay baseline (=0) line", value=True)
                show_points   = st.checkbox("Show markers (anomaly)", value=False)

        submitted = st.form_submit_button("Generate Chart(s)", use_container_width=True)

if not submitted:
    st.stop()

# =================== PREP DATA FOR CHART ===================
def _ind_to_col_and_label(name: str) -> Tuple[str, str]:
    # Returns (column_name_in_df, pretty_label)
    if name.startswith("Temperature"): return ("tmpa", "Temperature (°C)")
    if name.startswith("Precipitation"): return ("pcpa", "Precipitation (mm)")
    if name.startswith("Humidity"): return ("huma", "Humidity (%)")
    return ("wspa", "Wind speeds (m/s)")

@st.cache_data(ttl=900, show_spinner=False)
def _assemble_data(countries: Tuple[str,...],
                   adm1s: Tuple[Tuple[str,str],...],
                   freq: str,
                   start_date: str,
                   end_date: str) -> pd.DataFrame:
    """
    Assemble and FILTER data by [start_date, end_date] explicitly so cache
    invalidates when slider moves.
    """
    frames=[]
    for iso in countries or ():
        frames.append(_build_country_series(iso, freq))
    if adm1s:
        by_iso={}
        for iso,name in adm1s:
            by_iso.setdefault(iso,set()).add(name)
        for iso, names in by_iso.items():
            frames.append(_build_adm1_series_for_country(iso, freq, only_pairs=[(iso,n) for n in names]))
    if not frames:
        return pd.DataFrame(columns=["level","iso3","adm1","date","tmpa","pcpa","huma","wspa"])
    df = pd.concat(frames, ignore_index=True)
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    mask = (df["date"]>=sd) & (df["date"]<=ed)
    return df.loc[mask].sort_values(["level","iso3","adm1","date"]).reset_index(drop=True)

with st.spinner("Preparing chart data…"):
    WIDE = _assemble_data(tuple(sel_countries),
                          tuple(sel_adm1),
                          freq,
                          pd.to_datetime(dstart).strftime("%Y-%m-%d"),
                          pd.to_datetime(dend).strftime("%Y-%m-%d"))
    if WIDE.empty:
        st.warning("No data for the current selection / date range."); st.stop()

# Smoothing and normalization
def _apply_smooth(s:pd.Series, win:int)->pd.Series:
    return s.rolling(win, min_periods=1, center=True).mean() if (win and win>0) else s

geos = WIDE[["level","iso3","adm1"]].drop_duplicates().reset_index(drop=True)
def _geo_key(level, iso3, adm1): return f"{level}|{iso3}|{adm1 if level=='adm1' else ''}"
_GEO_IDX = {_geo_key(r.level, r.iso3, r.adm1): i for i,(_,r) in enumerate(geos.iterrows())}

def _trace_base_name(row)->str:
    return f"{display_country_name(row['iso3'])} — {row['adm1']}" if row["level"]=="adm1" and row["adm1"] else display_country_name(row["iso3"])

trace_names=[]
if chart_type in ("Line","Area","Bar"):
    chosen = (locals().get("indicators") or ["Temperature (TMPA, °C)"])
    for _,g in geos.iterrows():
        base=_trace_base_name(g)
        if len(chosen)==1:
            trace_names.append(base)
        else:
            for ind in chosen:
                trace_names.append(f"{base} — {ind.split()[0]}")

per_trace_windows: Dict[str,int] = {}
if (locals().get("enable_per_trace") and trace_names):
    cfg_df = pd.DataFrame({"Trace":trace_names, "Window (0 = off)":[locals().get("global_window",0)]*len(trace_names)})
    cfg = st.data_editor(cfg_df, num_rows="fixed", use_container_width=True, key="pertrace_cfg")
    per_trace_windows = {row["Trace"]: int(row["Window (0 = off)"] or 0) for _,row in cfg.iterrows()}

# =================== FIGURE HELPERS ===================
def _make_subplots(rows=1, cols=1, titles=None):
    specs=[[{"secondary_y":True}]*cols for _ in range(rows)]
    return make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.12, specs=specs, subplot_titles=titles)

def _new_fig(rows=1, cols=1, titles=None):
    base = _make_subplots(rows, cols, titles)
    if _HAS_RESAMPLER and st.session_state.get("chart_type") in ("Line","Area"):
        try:
            return FigureResampler(base, default_n_shown_samples=2500)
        except Exception:
            return base
    return base

def _apply_layout(fig, title_text:str, x_title:str, y_title:Optional[str], show_title:bool, is_bar:bool, stacked:bool, dual_units:bool):
    if not show_title: title_text=""
    fig.update_layout(
        height=780,
        margin=dict(l=48,r=48,t=64 if show_title else 24,b=80),
        title=title_text,
        legend=dict(orientation="h", y=1.04, x=0, title=dict(text="")),
        hovermode="x unified" if not is_bar else "x",
        barmode=("stack" if stacked else "group") if is_bar else None,
        template=None,
    )
    fig.update_xaxes(title_text=x_title, title_standoff=36, automargin=True)
    if y_title:
        fig.update_yaxes(title_text=y_title, title_standoff=86, automargin=True, secondary_y=False)
    fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=True)

def _add_ts(fig, df_geo, row, col, name, ycol, secondary=False, markers=False, is_bar=False, idx=0):
    if is_bar:
        tr = go.Bar(x=df_geo["date"], y=df_geo[ycol], name=name)
        fig.add_trace(tr, row=row, col=col, secondary_y=secondary)
        fig.data[-1].update(marker_color=_color(idx))
    else:
        mode = "lines+markers" if markers else "lines"
        tr = go.Scatter(x=df_geo["date"], y=df_geo[ycol], mode=mode, name=name, line=dict(width=2))
        fig.add_trace(tr, row=row, col=col, secondary_y=secondary)
        fig.data[-1].update(line=dict(color=_color(idx), width=2), marker_color=_color(idx))

def _add_xy(fig, df_geo, row, col, name, xcol, ycol, markers=False, idx=0):
    tr = go.Scatter(x=df_geo[xcol], y=df_geo[ycol],
                    mode="markers+lines" if markers else "markers",
                    text=df_geo["date"].dt.strftime("%Y-%m"),
                    hovertemplate="%{text}<br>X=%{x:.2f}, Y=%{y:.2f}<extra></extra>",
                    name=name)
    fig.add_trace(tr, row=row, col=col, secondary_y=False)
    fig.data[-1].update(marker_color=_color(idx), line=dict(color=_color(idx)))

# =================== TITLE ===================
def _label_geo(level, iso3, adm1):
    base=display_country_name(iso3); return f"{base} — {adm1}" if level=="adm1" and adm1 else base

def _compose_title(indicators:List[str], countries:List[str], adm1_pairs:List[Tuple[str,str]])->str:
    inds=" & ".join(indicators) if indicators else ""
    parts=[]
    if countries: parts.append(", ".join([display_country_name(i) for i in countries]))
    if adm1_pairs:
        df=pd.DataFrame(adm1_pairs, columns=["iso3","adm1"])
        chunks=[f"{display_country_name(iso)} ({', '.join(g['adm1'].tolist())})" for iso,g in df.groupby("iso3")]
        parts.append("ADM1: " + "; ".join(chunks))
    bits=[b for b in [inds]+parts if b]; return " — ".join(bits)

inds_list = (
    [locals().get("x_series"), locals().get("y_series")] if chart_type=="Scatter"
    else ([locals().get("an_indicator")] if chart_type=="Anomaly" else (locals().get("indicators") or []))
)
title_txt = _compose_title(inds_list, sel_countries, sel_adm1)

# =================== OPTIONAL SMOOTH / NORMALIZE ===================
def _apply_smoothing_block(df: pd.DataFrame, indicators: List[str], gw: int, per_trace: Dict[str,int]) -> pd.DataFrame:
    if gw<=0 and not per_trace: return df
    out=[]
    for (_, g) in df.groupby(["level","iso3","adm1"], sort=False):
        gg=g.copy()
        base=_trace_base_name(gg.iloc[0])
        for ind in indicators:
            col, lab = _ind_to_col_and_label(ind)
            if col not in gg: continue
            nm = base if len(indicators)==1 else f"{base} — {ind.split()[0]}"
            win = per_trace.get(nm, gw)
            gg[col] = _apply_smooth(gg[col], win)
        out.append(gg)
    return pd.concat(out, ignore_index=True)

def _normalize_index(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    if not indicators: return df
    out=[]
    for (_, g) in df.groupby(["level","iso3","adm1"], sort=False):
        gg=g.copy()
        for ind in indicators:
            col, _ = _ind_to_col_and_label(ind)
            if col in gg and not gg[col].dropna().empty:
                base=gg[col].iloc[0]
                if pd.notna(base) and base!=0:
                    gg[col] = 100.0 * gg[col] / base
        out.append(gg)
    return pd.concat(out, ignore_index=True)

# =================== BUILD FIGURE ===================
is_bar = (chart_type=="Bar")
stacked = (is_bar and False)  # keep group by default

if chart_type=="Scatter":
    fig = _new_fig(1,1); r=c=1
    xcol,_ = _ind_to_col_and_label(x_series); ycol,_ = _ind_to_col_and_label(y_series)
    for _,g in geos.iterrows():
        df=WIDE[(WIDE["level"]==g["level"]) & (WIDE["iso3"]==g["iso3"]) & (WIDE["adm1"]==g["adm1"])].dropna(subset=[xcol,ycol])
        if df.empty: continue
        k = _GEO_IDX.get(_geo_key(g["level"],g["iso3"],g["adm1"]), 0)
        _add_xy(fig, df, r, c, _label_geo(g["level"],g["iso3"],g["adm1"]), xcol, ycol, markers=show_points, idx=k)
    _apply_layout(fig, title_txt, x_series, y_series, not hide_title, False, False, False)

elif chart_type=="Anomaly":
    def _phase_key(s: pd.Series) -> pd.Series:
        if freq=="Monthly": return s.dt.month
        if freq=="Seasonal": return s.dt.quarter
        return pd.Series(1, index=s.index)

    def _baseline(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        ph = _phase_key(df["date"])
        agg_map = {"Mean":"mean", "Median":"median", "Min":"min", "Max":"max"}
        op = agg_map.get(baseline_calc, "mean")
        return (df.assign(_ph=ph).groupby(["level","iso3","adm1","_ph"])[col]
                .agg(baseline=op, mean="mean", std="std").reset_index())

    col, lab = _ind_to_col_and_label(an_indicator)
    full = WIDE.dropna(subset=[col]).copy()
    if full.empty:
        st.warning("No data for anomaly calculation with the selected baseline.", icon="⚠️"); st.stop()
    base = full[(full["date"].dt.year>=int(base_start)) & (full["date"].dt.year<=int(base_end))].copy()
    if base.empty:
        st.warning("Baseline window contains no data. Adjust the years.", icon="⚠️"); st.stop()

    bl = _baseline(base, col, baseline_calc)
    vis = full.copy()
    vis["_ph"] = _phase_key(vis["date"])
    out = vis.merge(bl, on=["level","iso3","adm1","_ph"], how="left")
    if an_method.startswith("Absolute"):
        out["anom"] = out[col] - out["baseline"]; unit = "Δ"
        ylab = f"Anomaly ({unit}{lab.split('(')[-1].rstrip(')')})"
    elif an_method.startswith("Percent"):
        out["anom"] = np.where(out["baseline"].abs()>0, 100*(out[col]-out["baseline"])/out["baseline"], np.nan); unit="%"
        ylab = "Anomaly (% of baseline)"
    else:
        out["anom"] = (out[col] - out["mean"]) / out["std"]; unit="σ"
        ylab = "Anomaly (σ)"

    fig = _new_fig(1,1); r=c=1
    for _,g in out[["level","iso3","adm1"]].drop_duplicates().iterrows():
        df=out[(out["level"]==g["level"]) & (out["iso3"]==g["iso3"]) & (out["adm1"]==g["adm1"])].copy()
        if df.empty: continue
        k = _GEO_IDX.get(_geo_key(g["level"],g["iso3"],g["adm1"]), 0)
        tr = go.Scatter(x=df["date"], y=df["anom"], mode="lines+markers" if show_points else "lines",
                        name=_label_geo(g["level"],g["iso3"],g["adm1"]),
                        line=dict(width=2, color=_color(k)))
        fig.add_trace(tr, row=r, col=c)
    if clim_overlay: fig.add_hline(y=0, line=dict(color="#999999", dash="dot"))
    _apply_layout(fig, title_txt, "Date", ylab, not hide_title, False, False, False)

else:
    # Line/Area/Bar
    chosen = (locals().get("indicators") or ["Temperature (TMPA, °C)"])
    WPROC = _apply_smoothing_block(WIDE, chosen, int(locals().get("global_window",0)), per_trace_windows)
    if locals().get("normalize_ix"):
        WPROC = _normalize_index(WPROC, chosen)

    if facet_by=="None":
        fig = _new_fig(1,1); r=c=1
        for _,g in geos.iterrows():
            df=WPROC[(WPROC["level"]==g["level"]) & (WPROC["iso3"]==g["iso3"]) & (WPROC["adm1"]==g["adm1"])].copy()
            base=_label_geo(g["level"],g["iso3"],g["adm1"])
            k = _GEO_IDX.get(_geo_key(g["level"],g["iso3"],g["adm1"]), 0)
            for ind in chosen:
                ycol, lab = _ind_to_col_and_label(ind)
                if ycol not in df or df[ycol].dropna().empty: continue
                nm = base if len(chosen)==1 else f"{base} — {ind.split()[0]}"
                _add_ts(fig, df, r, c, nm, ycol, secondary=False, markers=show_points, is_bar=(chart_type=="Bar"), idx=k)
        ylab = ( _ind_to_col_and_label(chosen[0])[1] if len(chosen)==1 else None )
        _apply_layout(fig, title_txt, "Date", ylab, not hide_title, (chart_type=="Bar"), False, False)

    elif facet_by=="Indicator":
        cols=min(2,len(chosen)); rows=math.ceil(len(chosen)/2)
        fig = _new_fig(rows, cols, titles=[_ind_to_col_and_label(ind)[1] for ind in chosen])
        _seen=set()
        for idx,ind in enumerate(chosen):
            rr=idx//2+1; cc=idx%2+1
            ycol, lab = _ind_to_col_and_label(ind)
            for _,g in geos.iterrows():
                df=WPROC[(WPROC["level"]==g["level"]) & (WPROC["iso3"]==g["iso3"]) & (WPROC["adm1"]==g["adm1"])].copy()
                if ycol not in df or df[ycol].dropna().empty: continue
                nm=_label_geo(g["level"],g["iso3"],g["adm1"])
                k = _GEO_IDX.get(_geo_key(g["level"],g["iso3"],g["adm1"]), 0)
                _add_ts(fig, df, rr, cc, nm, ycol, secondary=False, markers=show_points, is_bar=(chart_type=="Bar"), idx=k)
                if nm in _seen: fig.data[-1].showlegend=False
                else: _seen.add(nm)
        _apply_layout(fig, title_txt, "Date", None, not hide_title, (chart_type=="Bar"), False, False)

    else:  # Geography facets
        maxf=6
        if len(geos)>maxf:
            st.warning(f"Too many geographies selected for faceting ({len(geos)}). Showing the first {maxf}.", icon="⚠️")
            geos=geos.head(maxf)
        titles=[_label_geo(r.level,r.iso3,r.adm1) for _,r in geos.iterrows()]
        cols=3 if len(geos)>=3 else len(geos); rows=math.ceil(len(geos)/cols)
        fig = _new_fig(rows, cols, titles=titles)
        for idx,(_,g) in enumerate(geos.iterrows()):
            rr=idx//cols+1; cc=idx%cols+1
            df=WPROC[(WPROC["level"]==g["level"]) & (WPROC["iso3"]==g["iso3"]) & (WPROC["adm1"]==g["adm1"])].copy()
            k = _GEO_IDX.get(_geo_key(g["level"],g["iso3"],g["adm1"]), 0)
            for ind in chosen:
                ycol, lab = _ind_to_col_and_label(ind)
                if ycol not in df or df[ycol].dropna().empty: continue
                _add_ts(fig, df, rr, cc, ind.split()[0], ycol, secondary=False, markers=show_points, is_bar=(chart_type=="Bar"), idx=k)
        _apply_layout(fig, title_txt, "Date", None, not hide_title, (chart_type=="Bar"), False, False)

# Area fill
if chart_type=="Area":
    for tr in fig.select_traces(type="scatter"):
        tr.update(fill="tozeroy", hoverinfo="x+y+name")

# Legend placement
def _count_legend_items(fig: go.Figure) -> int:
    return sum(1 for tr in fig.data if getattr(tr, "showlegend", True))

def _auto_place_legend(fig: go.Figure, faceted: bool):
    n_items = _count_legend_items(fig)
    if faceted or n_items > 4:
        fig.update_layout(
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.16, yanchor="top", font=dict(size=12), itemwidth=80),
        )
        mb = fig.layout.margin.b if fig.layout.margin and fig.layout.margin.b is not None else 80
        fig.update_layout(margin=dict(b=max(mb, 120)))
    else:
        fig.update_layout(
            legend=dict(orientation="h", x=1.0, xanchor="right", y=1.0, yanchor="top", font=dict(size=12)),
        )
        mt = fig.layout.margin.t if fig.layout.margin and fig.layout.margin.t is not None else 64
        fig.update_layout(margin=dict(t=max(mt, 80)))
    fig.update_layout(legend_tracegroupgap=8)

_is_faceted = (facet_by != "None")
_base_fig = fig if isinstance(fig, go.Figure) else getattr(fig, "figure", fig)

if facet_by == "Indicator":
    seen = set()
    for tr in _base_fig.data:
        nm = getattr(tr, "name", None)
        if not nm: continue
        tr.legendgroup = nm
        if nm in seen: tr.showlegend = False
        else: seen.add(nm)

_auto_place_legend(_base_fig, _is_faceted)
fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=False)
fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=True)
st.plotly_chart(fig, use_container_width=True,
                config={"displaylogo":False,"toImageButtonOptions":{"format":"png","scale":2},"modeBarButtonsToAdd":["toImage"]})

# =================== BADGES ===================
st.markdown("#### Selections summary")
badges=[]
badges.append(f"<span class='badge'>Type: {chart_type}</span>")
badges.append(f"<span class='badge'>Freq: {freq}</span>")
badges.append(f"<span class='badge'>Date: {pd.to_datetime(dstart).isoformat()} → {pd.to_datetime(dend).isoformat()}</span>")
if chart_type not in ("Scatter","Anomaly"):
    if locals().get("indicators"):
        badges.append(f"<span class='badge'>Indicators: {' + '.join(indicators)}</span>")
elif chart_type=="Scatter":
    badges.extend([f"<span class='badge'>X: {x_series}</span>", f"<span class='badge'>Y: {y_series}</span>"])
else:
    badges.append(f"<span class='badge'>Baseline: {int(base_start)}-{int(base_end)}</span>")
    badges.append(f"<span class='badge'>Baseline calc: {baseline_calc}</span>")
    badges.append(f"<span class='badge'>Method: {an_method.split('(')[0].strip()}</span>")
if locals().get("show_points"): badges.append("<span class='badge'>Markers: On</span>")
if locals().get("global_window",0)>0: badges.append(f"<span class='badge'>Smooth: {int(global_window)}</span>")
if locals().get("clim_overlay"): badges.append("<span class='badge'>Climatology: On</span>")
st.markdown(f"<div class='badge-row'>{''.join(badges)}</div>", unsafe_allow_html=True)

# =================== EXPORT ===================
st.markdown("#### Export")
def _clean_export(df:pd.DataFrame)->pd.DataFrame:
    cols=["level","iso3","adm1","date","tmpa","pcpa","huma","wspa"]; keep=[c for c in cols if c in df.columns]
    return df[keep].sort_values(["level","iso3","adm1","date"])
st.download_button("Download data (CSV)", data=_clean_export(WIDE).to_csv(index=False).encode("utf-8"),
                   file_name="custom_chart_data.csv", mime="text/csv")
try:
    base_export = fig if isinstance(fig, go.Figure) else getattr(fig, "figure", fig)
    st.download_button("Download chart image (PNG)",
                       data=pio.to_image(base_export, format='png', scale=2),
                       file_name="custom_chart.png", mime="image/png")
except Exception:
    pass
