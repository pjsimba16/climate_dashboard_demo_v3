# pages/1_Temperature_Dashboard.py
# Temperature Dashboard — FULL patched version (Country-only + per-chart ADM1 compare)
# - Country map on the left (unchanged)
# - RIGHT column: global chart options (Type/Frequency/Source), KPIs, and a scrollable searchable ADM1 comparison summary
# - Charts:
#     TMPA: display options = [Show average, Show ±1σ band]
#     TMXA: display options = [Show average, Show extremes]
#     TMNA: display options = [Show average, Show extremes]
#     TMDR: no display options (fills space)
#     Percentiles: single ADM1 selector shared by all percentile charts; single percentile radio
#
# Notes:
# - Page now forces "Country (all)" as the chart scope (no ADM1 dropdown on the right)
# - Each chart has its own ADM1 multiselect (max 5) to overlay lines vs Country (all)
# - Hover shows one date at top, then blocks per ADM1 with that ADM1’s values
# - Seasonal hover header shows "SEASON YEAR" (e.g., JJA 2016)


import os, re, unicodedata, io
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# For territory insets (USA, FRA, ESP, PRT, DNK, NOR, CHN)
try:
    from shapely.affinity import translate as shp_translate
except Exception:
    from shapely import affinity as _shp_affinity

    def shp_translate(geom, xoff=0.0, yoff=0.0):
        return _shp_affinity.translate(geom, xoff=xoff, yoff=yoff)


try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# --- KPI CSS / helpers (unchanged) -------------------------------------------------
st.markdown("""
<style>
.kpi-row {
  display:flex;
  flex-wrap:wrap;
  gap:0.75rem;
  margin-bottom:0.75rem;
}
.kpi {
  flex:1 1 180px;
  min-width:160px;
  background:transparent;
  border-radius:0;
  padding:0;
  border:none;
}
.kpi .label {
  font-size:1.2rem;
  color:#64748b;
  margin-bottom:0.15rem;
}
.kpi .row {
  display:flex;
  align-items:center;
  gap:0.4rem;
}
.kpi .value {
  font-size:2.50rem;
  font-weight:600;
  color:#0f172a;
}
.kpi .badge {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  font-size:1.0rem;
  font-weight:600;
  padding:0.05rem 0.35rem;
  border-radius:999px;
}
.kpi .up   { background:#dcfce7; }
.kpi .down { background:#fee2e2; }
.kpi .flat { background:#e2e8f0; }
</style>
""", unsafe_allow_html=True)


def render_kpi(label: str, value_text: str, delta: float, show_symbol: bool, unit: str = "°C"):
    """
    Precip-style KPI:
      - label on top
      - big value
      - compact arrow/dash badge right beside the value (no duplicate numbers)
    """
    if delta is None or not (isinstance(delta, (int, float)) and np.isfinite(delta)) or abs(delta) < 1e-12:
        klass, sym, tip = "flat", "—", "No change"
    else:
        if delta > 0:
            klass, sym, tip = "up", "↑", f"+{delta:.2f} {unit}"
        else:
            klass, sym, tip = "down", "↓", f"{delta:.2f} {unit}"
    badge_html = f'<span class="badge {klass}" title="{tip}">{sym}</span>' if show_symbol else ""
    st.markdown(f"""
    <div class="kpi">
      <div class="label">{label}</div>
      <div class="row">
        <div class="value">{value_text}</div>
        {badge_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------- CONSTANTS ----------
HF_REPO_ID = "pjsimba16/adb_climate_dashboard_v1"
HF_GEO_REPO_ID = "pjsimba16/adb_climate_dashboard_v1"

# ---------- HF SPACE HELPERS (temperature data via HF space, not local) ----------
HF_SPACE_BASE = "https://huggingface.co/spaces/pjsimba16/adb-climate-data/resolve/main"


def _note_err(msg: str) -> None:
    st.session_state.setdefault("hf_errors", []).append(str(msg))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _read_parquet_from_space(rel_path: str):
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _load_availability_snapshot():
    df = _read_parquet_from_space("availability_snapshot.parquet")
    if df is None or df.empty:
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _temperature_isos_from_availability():
    snap = _load_availability_snapshot()
    if snap is None or snap.empty:
        return []
    mask = snap["indicator_type"].astype(str).str.lower() == "temperature"
    return sorted(snap.loc[mask, "iso3"].astype(str).str.upper().unique().tolist())


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _load_temperature_adm1_snapshot():
    """Latest ADM1 TMPA snapshot for all countries/frequencies (for map)."""
    df = _read_parquet_from_space("indicator_snapshots/temperature_adm1_latest.parquet")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    col_map = {c.lower(): c for c in df.columns}
    iso_col = col_map.get("iso3") or col_map.get("iso")
    adm1_col = (
        col_map.get("adm1")
        or col_map.get("city")
        or col_map.get("region")
        or col_map.get("name_1")
        or col_map.get("shapename")
    )
    freq_col = col_map.get("freq") or col_map.get("frequency")
    date_col = col_map.get("date") or col_map.get("time") or col_map.get("period")
    value_col = col_map.get("value")
    if not value_col:
        for c in df.columns:
            cl = str(c).lower()
            if "tmpa" in cl or "temp" in cl:
                value_col = c
                break
    if not iso_col or not adm1_col or not date_col or not value_col:
        _note_err("temperature_adm1_latest.parquet is missing required columns.")
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "iso3": df[iso_col].astype(str).str.upper().str.strip(),
            "ADM1": df[adm1_col].astype(str).str.strip(),
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    if freq_col and freq_col in df.columns:
        out["freq"] = df[freq_col].astype(str).str.title()
    else:
        out["freq"] = "Monthly"
    out = out.dropna(subset=["iso3", "ADM1", "Date", "value"])
    return out


SEASON_TO_MONTH = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
MONTH_TO_SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
INDICATOR_LABELS = (
    "Temperature",
    "Precipitation",
    "Temperature Thresholds",
    "Heatwaves",
    "Coldwaves",
    "Dry Conditions",
    "Wet Conditions",
    "Humidity",
    "Windspeeds",
)
CBLIND = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "verm": "#D55E00",
    "pink": "#CC79A7",
    "grey": "#999999",
    "red": "#d62728",
}

# === Seasonal/Annual label + unified range control ===
MONTH_TO_SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def _season_label_from_ts(ts: pd.Timestamp) -> str:
    return f"{MONTH_TO_SEASON[int(ts.month)]} {ts.year}"


def _hover_header_array(dates: pd.Series, freq: str) -> np.ndarray:
    dates = pd.to_datetime(dates, errors="coerce")
    if freq == "Monthly":
        return dates.dt.strftime("%b %Y").to_numpy(object)
    if freq == "Seasonal":
        return dates.apply(_season_label_from_ts).to_numpy(object)
    return dates.dt.strftime("%Y").to_numpy(object)  # Annual


def _range_picker(dates: pd.Series, key: str, freq: str):
    """
    - Seasonal: select_slider with labels like 'JJA 2018'
    - Annual:   slider with format='YYYY'
    - Monthly:  slider with format='YYYY-MM'
    Returns (start_ts, end_ts) as pandas.Timestamp (or (None,None) if no data)
    """
    # Normalize to pandas Timestamps and drop NaNs
    s = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if s.empty:
        return None, None

    # For the slider widget we use plain Python datetimes, because some
    # Streamlit versions don't accept pandas.Timestamp directly.
    dmin = s.min().to_pydatetime()
    dmax = s.max().to_pydatetime()

    if freq == "Seasonal":
        # Map each timestamp to a seasonal label (e.g. "JJA 2018") and keep
        # the first occurrence of each as the anchor for that season.
        labels = s.map(_season_label_from_ts).tolist()
        seen, uniq, idx_map = set(), [], {}
        for dt, lb in zip(s, labels):
            if lb not in seen:
                seen.add(lb)
                uniq.append(lb)
                idx_map[lb] = pd.Timestamp(dt)

        start_lb, end_lb = st.select_slider(
            "Date range",
            options=uniq,
            value=(uniq[0], uniq[-1]),
            key=key,
        )
        return idx_map[start_lb], idx_map[end_lb]

    fmt = "YYYY-MM" if freq == "Monthly" else "YYYY"
    d1, d2 = st.slider(
        "Date range",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
        format=fmt,
        key=key,
    )
    return pd.Timestamp(d1), pd.Timestamp(d2)
# ---------- PATH HELPERS ----------
def _here():
    try:
        return Path(__file__).parent.resolve()
    except Exception:
        return Path.cwd()


def _probe_country_data_dir():
    here = _here()
    for p in [
        here / "country_data",
        here.parent / "country_data",
        Path.cwd() / "country_data",
        Path("/mnt/data/country_data"),
    ]:
        if p.exists() and p.is_dir():
            return p
    return here.parent / "country_data"


def _probe_mapper_file():
    here = _here()
    for p in [
        here / "indicator_code_mapper.csv",
        here.parent / "indicator_code_mapper.csv",
        Path.cwd() / "indicator_code_mapper.csv",
        Path("/mnt/data/indicator_code_mapper.csv"),
    ]:
        if p.exists() and p.is_file():
            return p
    return None
COUNTRY_DATA_DIR = _probe_country_data_dir()
MAPPER_FILE      = _probe_mapper_file()

# ---------- GENERIC HELPERS ----------
def _suffixes_for_freq(freq, adm_level):
    f = str(freq)
    adm = str(adm_level).upper()
    if adm == "ADM1":
        return ["_M"] if f == "Monthly" else (["_S"] if f == "Seasonal" else ["_A"])
    return (
        ["_AM", "_PM"]
        if f == "Monthly"
        else (["_AS", "_PS"] if f == "Seasonal" else ["_AA"])
    )


def _pick_col(df_cols, base_code, suffixes):
    if not base_code:
        return None
    want = {f"{base_code}{s}".lower() for s in suffixes}
    lower_map = {str(c).lower(): c for c in df_cols}
    for w in want:
        if w in lower_map:
            return lower_map[w]
    return None


def _build_date_column(df, freq):
    year = pd.to_numeric(df.get("Year"), errors="coerce")
    if str(freq) == "Monthly":
        mo = pd.to_numeric(df.get("Month"), errors="coerce")
        return pd.to_datetime(
            dict(year=year, month=mo, day=1), errors="coerce"
        )
    if str(freq) == "Seasonal":
        season = df.get("Season")
        if season is None:
            return pd.Series([pd.NaT] * len(df))
        s2m = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
        mo = season.map(s2m).astype("float")
        return pd.to_datetime(
            dict(year=year, month=mo, day=1), errors="coerce"
        )
    return pd.to_datetime(dict(year=year, month=1, day=1), errors="coerce")


def _norm_str(s):
    s = (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _compose_title(code, mapper, fallback):
    if (
        isinstance(mapper, pd.DataFrame)
        and not mapper.empty
        and {"Code", "Description"}.issubset(mapper.columns)
    ):
        hit = mapper.loc[
            mapper["Code"].astype(str).str.upper() == str(code).upper()
        ]
        if not hit.empty:
            d = str(hit.iloc[0]["Description"])
            return d if d else fallback
    return fallback

# ---------- INDICATOR MAPPER ----------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_indicator_mapper():
    if MAPPER_FILE is None:
        return pd.DataFrame(columns=["Code", "Description"])
    try:
        return pd.read_csv(MAPPER_FILE)
    except Exception:
        return pd.DataFrame(columns=["Code", "Description"])

def _compose_title(code, mapper, fallback):
    if isinstance(mapper, pd.DataFrame) and not mapper.empty and {"Code","Description"}.issubset(mapper.columns):
        hit = mapper.loc[mapper["Code"].astype(str).str.upper() == str(code).upper()]
        if not hit.empty:
            d = str(hit.iloc[0]["Description"])
            return d if d else fallback
    return fallback

# ---------- FILE RESOLVERS ----------
def _adm0_file(iso3, freq):
    folder = {"Monthly": "Monthly", "Seasonal": "Seasonal", "Annual": "Annual"}[
        str(freq)
    ]
    p = (
        COUNTRY_DATA_DIR
        / str(iso3).upper()
        / folder
        / f"{str(iso3).upper()}_ADM0_data.parquet"
    )
    return p if p.exists() else None


def _adm1_file(iso3, adm1_name, freq):
    folder = {"Monthly": "Monthly", "Seasonal": "Seasonal", "Annual": "Annual"}[
        str(freq)
    ]
    base = COUNTRY_DATA_DIR / str(iso3).upper() / folder
    if not base.exists():
        return None
    candidates = [
        base / f"{adm1_name}_ADM1_data.parquet",
        base / f"{adm1_name.replace(' ', '_')}_ADM1_data.parquet",
        base / f"{adm1_name.replace('/', '-')}_ADM1_data.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    want = _norm_str(adm1_name)
    for f in sorted(base.glob("*_ADM1_data.parquet")):
        stem = f.name[: -len("_ADM1_data.parquet")]
        if _norm_str(stem) == want:
            return f
    return None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def list_available_isos():
    """List ISO3 codes with Temperature data, based on HF availability_snapshot."""
    try:
        isos = _temperature_isos_from_availability()
    except Exception:
        isos = []
    return list(isos or [])

# ---------- SERIES LOADERS ----------
@st.cache_data(ttl=1800, show_spinner=False)
def load_scope_series(iso3, freq, area_label, indicator_codes):
    """Load ADM0/ADM1 series for the given ISO3 and area from the HF Space."""
    mapper = load_indicator_mapper()
    is_country = (area_label in ("", "Country (all)"))
    suffixes = _suffixes_for_freq(freq, "ADM0" if is_country else "ADM1")
    iso3u = str(iso3 or "").upper()
    folder = {
        "Monthly": "Monthly",
        "Seasonal": "Seasonal",
        "Annual": "Annual",
    }[str(freq)]
    df_raw = None
    if is_country:
        rel_path = f"country_data/{iso3u}/{folder}/{iso3u}_ADM0_data.parquet"
        df_raw = _read_parquet_from_space(rel_path)
    else:
        adm1 = str(area_label or "")
        base_rel = f"country_data/{iso3u}/{folder}"
        candidates = [
            f"{base_rel}/{adm1}_ADM1_data.parquet",
            f"{base_rel}/{adm1.replace(' ', '_')}_ADM1_data.parquet",
            f"{base_rel}/{adm1.replace('/', '-')}_ADM1_data.parquet",
        ]
        df_raw = None
        for rel in candidates:
            df_raw = _read_parquet_from_space(rel)
            if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                break
        if df_raw is None or (isinstance(df_raw, pd.DataFrame) and df_raw.empty):
            norm = _norm_str(adm1).replace(" ", "_")
            if norm:
                rel = f"{base_rel}/{norm}_ADM1_data.parquet"
                df_raw = _read_parquet_from_space(rel)
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame(), {}
    df = df_raw.copy()
    df["date"] = _build_date_column(df, freq)
    df = df.dropna(subset=["date"]).sort_values("date")
    out = pd.DataFrame({"date": df["date"]})
    pretty = {}
    for code in indicator_codes:
        col = _pick_col(df.columns, code, suffixes)
        if col:
            out[code] = pd.to_numeric(df[col], errors="coerce")
            pretty[code] = _compose_title(code, mapper, code)
    return out.reset_index(drop=True), pretty

@st.cache_data(ttl=1800, show_spinner=False)
def load_temperature_bundle(iso3, freq):
    """
    Load all temperature-relevant columns for Country (all) in *one* HF call,
    then reuse for multiple charts.
    """
    # all codes you need across all charts for Country (all)
    codes = [
        "TMPA", "TMPV",
        "TMXA", "TMXV", "TMXX", "TMXN",
        "TMNA", "TMNV", "TMNX", "TMNN",
        "TMDR"
    ]
    df, _pretty = load_scope_series(iso3, freq, "Country (all)", codes)
    return df



def _prep_single(iso3_now, adm1_now, freq, code_avg, code_var=None, bundle=None):
    """
    If bundle is provided (Country-all DataFrame), use it instead of reloading.
    """
    if bundle is not None and adm1_now in ("", "Country (all)"):
        df_codes = bundle.copy()
    else:
        codes = [code_avg] + ([code_var] if code_var else [])
        area = (
            adm1_now
            if (adm1_now and adm1_now != "Country (all)")
            else "Country (all)"
        )
        df_codes, _ = load_scope_series(iso3_now, freq, area, codes)

    if df_codes.empty or code_avg not in df_codes:
        return pd.DataFrame(columns=["date", "avg", "var"])

    out = pd.DataFrame(
        {
            "date": df_codes["date"],
            "avg": pd.to_numeric(df_codes[code_avg], errors="coerce"),
        }
    )
    if code_var and code_var in df_codes:
        out["var"] = pd.to_numeric(df_codes[code_var], errors="coerce")

    return (
        out.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

def _name_matches(name: str, candidates: list[str]) -> bool:
    n = (str(name) or "").strip().lower()
    return any(n == c.strip().lower() or n.endswith(c.strip().lower()) for c in candidates)


def _composite_adm1_layout(iso3: str, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Re-arrange far-flung territories closer to the mainland for nicer choropleth maps.
    Logic copied from the original 1_Temperature_Dashboard_v0.py.

    Applies to: USA, FRA, ESP, PRT, DNK, NOR, CHN.
    For other ISO3 codes, returns gdf unchanged.
    """
    if gdf is None or gdf.empty:
        return gdf

    iso = (iso3 or "").upper()
    INTEREST = {"USA", "FRA", "ESP", "PRT", "DNK", "NOR", "CHN"}
    if iso not in INTEREST:
        return gdf

    name_col = (
        "shapeName"
        if "shapeName" in gdf.columns
        else ("NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0])
    )

    # Work in Web Mercator (meters) for consistent translations
    g = gdf.to_crs(3857).copy()
    g["_cx"] = g.geometry.centroid.x
    g["_cy"] = g.geometry.centroid.y

    # Mainland bounding boxes per country (in Web Mercator)
    def _usa(df): return (df["_cx"].between(-1.42e7, -7.3e6)) & (df["_cy"].between(2.8e6, 6.5e6))
    def _fra(df): return (df["_cx"].between(-7.1e5, 1.2e6)) & (df["_cy"].between(5.0e6, 6.7e6))
    def _esp(df): return (df["_cx"].between(-1.3e6, -2.0e5)) & (df["_cy"].between(4.0e6, 5.1e6))
    def _prt(df): return (df["_cx"].between(-1.35e6, -9.0e5)) & (df["_cy"].between(4.2e6, 4.9e6))
    def _dnk(df): return (df["_cx"].between(7.0e5, 1.4e6)) & (df["_cy"].between(7.0e6, 7.8e6))
    def _nor(df): return (df["_cx"].between(3.0e5, 1.7e6)) & (df["_cy"].between(7.1e6, 9.2e6))
    def _chn(df): return (df["_cx"].between(7.0e5, 1.4e6)) & (df["_cy"].between(3.0e6, 6.0e6))

    RULE = {"USA": _usa, "FRA": _fra, "ESP": _esp, "PRT": _prt, "DNK": _dnk, "NOR": _nor, "CHN": _chn}

    m_main = RULE[iso](g)
    if not m_main.any():
        return gdf

    minx, miny, maxx, maxy = g.loc[m_main, "geometry"].total_bounds
    width = maxx - minx
    height = maxy - miny

    TERR = {
        "USA": {
            "Alaska": ["Alaska"],
            "Hawaii": ["Hawaii"],
            "Puerto Rico": ["Puerto Rico"],
            "Guam": ["Guam"],
            "American Samoa": ["American Samoa"],
            "Northern Mariana Islands": [
                "Northern Mariana Islands",
                "Commonwealth of the Northern Mariana Islands",
                "N. Mariana Islands",
            ],
            "U.S. Virgin Islands": [
                "United States Virgin Islands",
                "U.S. Virgin Islands",
                "Virgin Islands of the United States",
            ],
            "UM": [
                "United States Minor Outlying Islands",
                "Baker Island",
                "Howland Island",
                "Jarvis Island",
                "Johnston Atoll",
                "Kingman Reef",
                "Midway Atoll",
                "Palmyra Atoll",
                "Wake Island",
            ],
        },
        "FRA": {
            "Guyane": ["Guyane", "French Guiana"],
            "Guadeloupe": ["Guadeloupe"],
            "Martinique": ["Martinique"],
            "Mayotte": ["Mayotte"],
            "La Réunion": ["La Réunion", "Reunion", "Réunion"],
        },
        "ESP": {
            "Islas Canarias": ["Canarias", "Islas Canarias", "Canary Islands"],
            "Ceuta": ["Ceuta"],
            "Melilla": ["Melilla"],
        },
        "PRT": {
            "Açores": ["Açores", "Azores"],
            "Madeira": ["Madeira", "Região Autónoma da Madeira"],
        },
        "DNK": {
            "Greenland": ["Greenland", "Kalaallit Nunaat", "Grønland"],
            "Faroe": ["Faroe", "Faroe Islands", "Føroyar"],
        },
        "NOR": {"Svalbard": ["Svalbard"], "Jan Mayen": ["Jan Mayen"]},
        "CHN": {
            "Hong Kong": ["Hong Kong", "Hong Kong SAR"],
            "Macao": ["Macao", "Macau", "Macao SAR", "Macau SAR"],
        },
    }

    def _move(mask, x_rel, y_rel):
        if mask.any():
            g.loc[mask, "geometry"] = g.loc[mask, "geometry"].apply(
                lambda geom: shp_translate(
                    geom,
                    xoff=(minx + x_rel * width) - geom.centroid.x,
                    yoff=(miny + y_rel * height) - geom.centroid.y,
                )
            )

    if iso == "USA":
        m_ak = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["Alaska"])) | (
            (g["_cx"] < -1.6e7) & (g["_cy"] > 7.5e6)
        )
        m_hi = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["Hawaii"])) | (
            (g["_cx"] < -1.6e7) & (g["_cy"] < 3.0e6)
        )
        m_pr = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["Puerto Rico"])) | (
            (g["_cx"] > -7.7e6)
            & (g["_cx"] < -6.8e6)
            & (g["_cy"].between(2.0e6, 2.4e6))
        )
        m_gu = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["Guam"])) | (
            (g["_cx"] > 1.2e7) & (g["_cy"] < 2.0e6)
        )
        m_as = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["American Samoa"])) | (
            (g["_cx"] < -1.8e7) & (g["_cy"] < -1.0e6)
        )
        m_mp = g[name_col].apply(
            lambda n: _name_matches(n, TERR["USA"]["Northern Mariana Islands"])
        ) | ((g["_cx"] > 1.5e7) & (g["_cy"] < 2.0e6))
        m_vi = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["U.S. Virgin Islands"])) | (
            (g["_cx"] > -7.0e6) & (g["_cy"].between(2.0e6, 2.5e6))
        )
        m_um = g[name_col].apply(lambda n: _name_matches(n, TERR["USA"]["UM"]))

        _move(m_ak, -0.18, 0.92)
        _move(m_hi, 0.12, 0.07)
        _move(m_pr, 0.88, 0.12)
        _move(m_gu, 0.84, 0.06)
        _move(m_as, 0.80, 0.06)
        _move(m_mp, 0.90, 0.06)
        _move(m_vi, 0.94, 0.06)
        _move(m_um, 0.76, 0.06)

    if iso == "FRA":
        for nm, xy in [
            ("Guyane", (0.10, -0.05)),
            ("Guadeloupe", (0.16, -0.03)),
            ("Martinique", (0.18, -0.03)),
            ("Mayotte", (0.21, -0.03)),
            ("La Réunion", (0.24, -0.03)),
        ]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    if iso == "ESP":
        for nm, xy in [
            ("Islas Canarias", (-0.05, -0.02)),
            ("Ceuta", (0.02, -0.01)),
            ("Melilla", (0.03, -0.01)),
        ]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    if iso == "PRT":
        for nm, xy in [("Açores", (-0.08, 0.02)), ("Madeira", (-0.02, -0.02))]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    if iso == "DNK":
        for nm, xy in [("Faroe", (-0.05, 0.15)), ("Greenland", (0.05, 0.85))]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    if iso == "NOR":
        for nm, xy in [("Svalbard", (0.30, 0.90)), ("Jan Mayen", (0.05, 0.85))]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    if iso == "CHN":
        for nm, xy in [("Hong Kong", (0.90, -0.02)), ("Macao", (0.92, -0.02))]:
            m = g[name_col].apply(lambda n, nms=[nm]: _name_matches(n, nms))
            _move(m, *xy)

    g = g.to_crs(4326)
    g.drop(columns=["_cx", "_cy"], errors="ignore", inplace=True)
    return g


# ---------- GEOJSON ----------
def _get_hf_token():
    try:
        if hasattr(st, "secrets"):
            tok = st.secrets.get("HF_TOKEN")
            if tok: return str(tok)
    except Exception: pass
    return os.getenv("HF_TOKEN", "")

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_country_adm1_geojson(iso3: str):
    """
    Load ADM1 GeoJSON, then apply composite layout (USA/FRA/ESP/PRT/DNK/NOR/CHN
    get their territories pulled in as insets).
    """
    if hf_hub_download is None:
        raise FileNotFoundError("huggingface_hub is not installed/available.")

    last_err = None
    iso3u = (iso3 or "").upper()
    for repo_type in ("space", "dataset"):
        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=repo_type,
                filename=f"ADM1_geodata/{iso3u}.geojson",
                token=_get_hf_token(),
            )
            gdf = gpd.read_file(path)
            try:
                gdf["geometry"] = gdf["geometry"].buffer(0)
            except Exception:
                pass
            gdf = gdf.to_crs(4326)

            # 👇 THIS is the key: rearrange territories like in v0
            gdf_comp = _composite_adm1_layout(iso3u, gdf)

            bounds = tuple(gdf_comp.total_bounds)
            name_col = (
                "shapeName"
                if "shapeName" in gdf_comp.columns
                else ("NAME_1" if "NAME_1" in gdf_comp.columns else gdf_comp.columns[0])
            )
            return gdf_comp.__geo_interface__, bounds, name_col, gdf_comp
        except Exception as e:
            last_err = e
            continue
    raise last_err




@st.cache_data(ttl=7*24*3600, show_spinner=False)
def adm1_label_points(iso3):
    _, _, name_col, gdf = load_country_adm1_geojson(iso3)
    pts = gdf.representative_point()
    return pts.x.to_numpy(), pts.y.to_numpy(), gdf[name_col].astype(str).to_numpy(), name_col

# ---------- Elevation CSV ----------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_city_map():
    """
    Load elevation / city mapping from the HF space.

    Tries:
      - city_mapper_with_coords_v3.csv
      - city_mapper_with_coords_v2.csv

    Returns a DataFrame with columns: ['ADM0', 'ADM1', 'elevation'].
    """
    import io

    candidates = [
        "city_mapper_with_coords_v3.csv",
        "city_mapper_with_coords_v2.csv",
    ]

    for rel_path in candidates:
        url = f"{HF_SPACE_BASE}/{rel_path.lstrip('/')}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
        except Exception as e:
            _note_err(f"HF download failed for {rel_path}: {e}")
            continue

        try:
            # Decode to text and read as CSV
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
        except Exception as e:
            _note_err(f"CSV parse failed for {rel_path}: {e}")
            continue

        # Try to detect the relevant columns
        cn_country = next((c for c in df.columns if c.lower() == "country"), None)
        cn_city    = next((c for c in df.columns if c.lower() == "city"), None)
        cn_elev    = next((c for c in df.columns if "elev" in c.lower()), None)

        if not (cn_country and cn_city and cn_elev):
            _note_err(
                f"{rel_path} missing expected columns (country/city/elev); "
                f"found: {list(df.columns)}"
            )
            continue

        df = df.rename(
            columns={cn_country: "ADM0", cn_city: "ADM1", cn_elev: "elevation"}
        )
        df["ADM0"] = df["ADM0"].astype(str).str.upper().str.strip()
        df["ADM1"] = df["ADM1"].astype(str).str.strip()
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")

        return df[["ADM0", "ADM1", "elevation"]].copy()

    # If all attempts fail:
    _note_err("Could not load city_mapper_with_coords from HF space.")
    return pd.DataFrame(columns=["ADM0", "ADM1", "elevation"])

def _elevation_completeness(iso3, geojson_dict, city_map):
    try:
        feats = geojson_dict.get("features", [])
        adm1_names = [f.get("properties",{}).get("shapeName") for f in feats]
        adm1_names = [str(x) for x in adm1_names if x]
    except Exception:
        adm1_names = []
    cm_iso = city_map[city_map["ADM0"].astype(str).str.upper() == str(iso3).upper()].copy()
    if cm_iso.empty or not adm1_names: return False, 0, len(adm1_names)
    chk = pd.DataFrame({"ADM1": adm1_names})
    tmp = cm_iso[["ADM1","elevation"]].copy()
    tmp["elevation"] = pd.to_numeric(tmp["elevation"], errors="coerce")
    chk = chk.merge(tmp, on="ADM1", how="left")
    avail = int(chk["elevation"].notna().sum()); total = len(chk)
    return (avail == total and total > 0), avail, total

# ---------- fragment helper ----------
try:
    fragment = st.fragment
    _FRAG_OK = True
except Exception:
    _FRAG_OK = False
def _with_fragment(fn): return fragment(fn) if _FRAG_OK else fn

# ---------- Header / Nav ----------
try:
    import pycountry
except Exception:
    pycountry = None
_CUSTOM_COUNTRY_DISPLAY = {"CHN":"People's Republic of China","TWN":"Taipei,China","HKG":"Hong Kong, China"}

def iso3_to_name(iso):
    iso = (iso or "").upper().strip()
    if pycountry:
        try:
            c = pycountry.countries.get(alpha_3=iso)
            if c and getattr(c,"name",None): return c.name
        except Exception: pass
    return iso

def display_country_name(iso):
    iso = (iso or "").upper().strip()
    return _CUSTOM_COUNTRY_DISPLAY.get(iso, iso3_to_name(iso))

qp = st.query_params
iso3_q = (qp.get("iso3") or st.session_state.get("nav_iso3") or st.session_state.get("opt_iso3_t") or "").upper()

top_l, top_r = st.columns([0.12, 0.88])
with top_l:
    if st.button("← Home", help="Back to Home"):
        keep_iso3 = st.query_params.get("iso3","")
        st.query_params.clear()
        if keep_iso3: st.query_params.update({"iso3": keep_iso3})
        try: st.switch_page("Home_Page.py")
        except Exception: st.rerun()
st.markdown(f"### Temperature - {display_country_name(iso3_q) if iso3_q else '…'}")

# ---------- Dev toggle ----------
debug_qp = (st.query_params.get("debug", "0").lower() in {"1","true","yes"})
show_dev_debug = st.toggle("Developer debug panel", value=False, help="Paths and column checks.") if debug_qp else False

# ---------- Layout ----------
with st.spinner("Scanning available countries…"):
    countries_iso = list_available_isos()
if iso3_q and iso3_q not in countries_iso:
    countries_iso = sorted(set(countries_iso + [iso3_q]))

lc, rc = st.columns([0.34, 0.66], gap="large")

# ---------- LEFT: Map ----------
def load_latest_adm1_for_map(iso3, freq, want_debug=False):
    """
    Return the latest available ADM1 TMPA value per region for the country,
    using the precomputed ADM1 snapshot stored in the HF Space.
    Returns a DataFrame with columns ['ADM1', 'value', 'Date'].
    """
    snap = _load_temperature_adm1_snapshot()
    iso3u = str(iso3 or "").upper()
    freq_req = str(freq).title()
    df_empty = pd.DataFrame(columns=["ADM1", "value", "Date"])
    if snap is None or snap.empty:
        if want_debug:
            meta = {
                "iso3": iso3u,
                "freq_requested": freq_req,
                "rows": 0,
                "source": "snapshot_empty",
            }
            return df_empty, [("snapshot", meta)]
        return df_empty
    df = snap.copy()
    df = df[df["iso3"].astype(str).str.upper() == iso3u]
    if df.empty:
        if want_debug:
            meta = {
                "iso3": iso3u,
                "freq_requested": freq_req,
                "rows": 0,
                "source": "snapshot_no_iso",
            }
            return df_empty, [("snapshot", meta)]
        return df_empty
    dbg_source = "snapshot"
    if "freq" in df.columns:
        df_freq = df[df["freq"].astype(str).str.title() == freq_req]
        if df_freq.empty:
            df_freq = df
            dbg_source = "snapshot_any_freq"
    else:
        df_freq = df
        dbg_source = "snapshot_no_freq_col"
    if df_freq.empty:
        if want_debug:
            meta = {
                "iso3": iso3u,
                "freq_requested": freq_req,
                "rows": 0,
                "source": dbg_source,
            }
            return df_empty, [("snapshot", meta)]
        return df_empty
    out = (
        df_freq[["ADM1", "value", "Date"]]
        .dropna(subset=["ADM1", "Date", "value"])
        .sort_values(["ADM1", "Date"])
        .drop_duplicates("ADM1", keep="last")
        .reset_index(drop=True)
    )
    if want_debug:
        meta = {
            "iso3": iso3u,
            "freq_requested": freq_req,
            "rows": len(out),
            "source": dbg_source,
        }
        return out, [("snapshot", meta)]
    return out

with lc:
    country_options = ["—"] + countries_iso
    if "opt_iso3_t" not in st.session_state:
        st.session_state["opt_iso3_t"] = iso3_q if iso3_q in country_options else "—"
    iso3 = st.selectbox("Select Country", options=country_options,
                        index=country_options.index(st.session_state["opt_iso3_t"]) if st.session_state["opt_iso3_t"] in country_options else 0,
                        key="opt_iso3_t",
                        format_func=lambda v: ("Select…" if v=="—" else display_country_name(v)),
                        help="Pick a country, or arrive pre-selected via Home map.")
    iso3_cur = "" if iso3 == "—" else iso3
    if iso3_cur != st.query_params.get("iso3",""):
        st.query_params.update({"iso3": iso3_cur})
        st.rerun()

    MAP_HEIGHT = 640
    if iso3 and iso3 != "—":
        with st.spinner("Extracting data for temperature map..."):
            freq_for_map = st.session_state.get("opt_freq_t", "Monthly")
            df_latest = load_latest_adm1_for_map(iso3, freq_for_map, want_debug=False)
            if isinstance(df_latest, tuple): df_latest = df_latest[0]

            if df_latest.empty:
                sel = st.session_state.get("opt_freq_t","Monthly")
                st.warning(f"No ADM1 TMPA data found for **{iso3} — {sel}**.", icon="⚠️")
            else:
                with st.spinner("Loading ADM1 boundaries…"):
                    try:
                        geojson_dict, bounds, name_col, gdf = load_country_adm1_geojson(iso3)
                    except Exception as e:
                        st.error(f"GeoJSON load failed: {e}")
                        st.stop()

                all_adm1 = gdf[name_col].astype(str)
                gdf_norm = all_adm1.to_frame("ADM1").assign(__key=all_adm1.map(_norm_str))
                latest_norm = df_latest.assign(__key=df_latest["ADM1"].map(_norm_str))
                df_map = gdf_norm.merge(
                    latest_norm[["__key", "value", "Date"]],
                    on="__key",
                    how="left"
                ).drop(columns="__key")

                CITY_MAP = _load_city_map()
                elev_complete, elev_avail, elev_total = _elevation_completeness(iso3, geojson_dict, CITY_MAP)
                if st.session_state.get("last_iso3_for_choice_t") != iso3 or "map_data_choice_t" not in st.session_state:
                    st.session_state["map_data_choice_t"] = "Elevation" if elev_complete else "Temperature"
                    st.session_state["last_iso3_for_choice_t"] = iso3

                try:
                    cm_iso = CITY_MAP[CITY_MAP["ADM0"].astype(str).str.upper() == iso3][["ADM1","elevation"]].copy()
                    cm_iso["ADM1"] = cm_iso["ADM1"].astype(str)
                except Exception:
                    cm_iso = pd.DataFrame(columns=["ADM1","elevation"])


                choice = st.radio("Map data", ["Temperature", "Elevation"], horizontal=True,
                                    index=(1 if st.session_state.get("map_data_choice_t") == "Elevation" else 0),
                                    key="map_data_choice_t",
                                    help="Switch between latest temperature and elevation (if available).")

                if choice == "Elevation":
                    df_map = df_map.merge(cm_iso, on="ADM1", how="left")
                    color_col = "elevation"; cs_name = "Viridis"; cbar_title = "Elevation (m)"
                    hover_tmpl = "<b>%{customdata[0]}</b><br>Elevation: %{customdata[1]:.0f} m<br>As of: %{customdata[2]}<extra></extra>"
                else:
                    color_col = "value"; cs_name = "YlOrRd"; cbar_title = "Temperature (°C)"
                    hover_tmpl = "<b>%{customdata[0]}</b><br>Latest: %{customdata[1]:.2f} °C<br>As of: %{customdata[2]}<extra></extra>"

                fig = px.choropleth(
                    df_map,
                    geojson=geojson_dict,
                    locations="ADM1",
                    featureidkey=f"properties.{name_col}",
                    color=color_col,
                    projection="mercator",
                    color_continuous_scale=cs_name,
                )

                fig.update_geos(
                    projection_type="mercator",
                    fitbounds="locations",
                    showland=False,
                    showcountries=False,
                    showcoastlines=False,
                    showocean=False,
                    visible=False,
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=MAP_HEIGHT,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    hovermode="closest",
                    showlegend=False,
                    # keep your existing colorbar config
                    coloraxis_colorbar=dict(
                        title=cbar_title,
                        thickness=14,
                        len=0.92,
                        x=0.98,
                        y=0.5,
                        yanchor="middle",
                        tickfont=dict(size=10),
                        titlefont=dict(size=11),
                    ),
                )


                _date_str = pd.to_datetime(df_map["Date"], errors="coerce").dt.strftime("%Y-%m").fillna("—")
                vals = pd.to_numeric(df_map[color_col], errors="coerce")
                fig.data[0].customdata = np.stack(
                    [df_map["ADM1"].astype(str).values, vals.values, _date_str.values],
                    axis=-1,
                )
                fig.data[0].hovertemplate = hover_tmpl
                fig.update_traces(marker_line_width=0.3, marker_line_color="#999")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                latest_when = pd.to_datetime(df_map.get("Date"), errors="coerce").max()
                latest_period = latest_when.strftime("%Y-%m") if pd.notna(latest_when) else "—"
                regions_with_data = int(pd.to_numeric(df_map.get(color_col), errors="coerce").notna().sum())
                st.markdown(
                f"""
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:.5rem; margin-bottom:.5rem;">
                  <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                    Country: {display_country_name(iso3)} ({iso3})
                  </span>
                  <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                    ADM1 with data: {regions_with_data}/{len(df_map)}
                  </span>
                  <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                    Latest period: {latest_period}
                  </span>
                </div>
                """, unsafe_allow_html=True
                )

                st.markdown("**Country Map**  \nElevation data is from NASA SRTM (Feb 2000).")

# ---------- RIGHT: Global chart options + KPIs + Comparison Summary ----------
with rc:
    col_ind, col_form = st.columns([0.35, 0.65], gap="small")
    with col_ind:
        st.markdown("<div style='margin-top:0.2rem'></div>", unsafe_allow_html=True)
        indicator = st.radio("Select climate indicator", INDICATOR_LABELS, index=0, key="opt_indicator_temp",
                             help="Switch indicators.")
        if indicator == "Precipitation":
            carry_iso = st.session_state.get("opt_iso3_t","—")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso})
            try: st.switch_page("pages/2_Precipitation_Dashboard.py")
            except Exception: st.switch_page("2_Precipitation_Dashboard.py")
        elif indicator == "Humidity":
            carry_iso = st.session_state.get("opt_iso3_t","—")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso})
            try: st.switch_page("pages/3_Humidity_Dashboard.py")
            except Exception: st.switch_page("3_Humidity_Dashboard.py")
        elif indicator == "Windspeeds":
            carry_iso = st.session_state.get("opt_iso3_t","—")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso})
            try: st.switch_page("pages/4_Windspeeds_Dashboard.py")
            except Exception: st.switch_page("4_Windspeeds_Dashboard.py")

    with col_form:
        st.markdown("#### Chart Options")
        with st.form("options_form_temp", clear_on_submit=False):
            colA, colB, colC = st.columns(3)
            with colA:
                data_type = st.radio("Type", ["Historical Observations", "Projections (SSPs)"], index=0, key="opt_type_t",
                                     help="Choose historical observations or future projections.")
            with colB:
                freq = st.radio("Frequency", ["Monthly", "Seasonal", "Annual"], index=0, key="opt_freq_t",
                                help="Change the temporal aggregation for all charts.")
            with colC:
                source = st.radio("Data Source", ["CDS/CCKP", "CRU", "ERA5"], index=2, key="opt_source_t",
                                  help="Pick your preferred data provider.")
            st.form_submit_button("Apply changes", type="primary")

        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.25rem;">
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Indicator: {indicator}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Type: {('Historical' if data_type.startswith('Historical') else 'Projections')}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Frequency: {freq}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Source: {source}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Area: Country (all)</span>
            </div>
            <div style="height:10px;"></div>
            """, unsafe_allow_html=True
        )

    # KPIs (based on Country (all))
    # ===== KPIs (TMPA / TMPV) =====
    iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_t") or "")) or ""
    adm1_now = st.query_params.get("city","") or st.session_state.get("opt_adm1_t","")
    freq_sel = st.session_state.get("opt_freq_t","Monthly")
    bundle_country = load_temperature_bundle(iso3_now, freq_sel)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    kpi_container = st.container()
    with kpi_container:
        with st.spinner("Extracting data for temperature KPIs..."):
            base_tmpa = _prep_single(iso3_now, adm1_now, freq_sel, "TMPA", "TMPV", bundle=bundle_country)
            k1, k2, k3, k4 = st.columns(4)

            if base_tmpa.empty:
                with k1: render_kpi("Latest Avg Temp (TMPA)", "—", None, show_symbol=False, unit="°C")
                with k2: render_kpi("Δ vs previous point", "—", None, show_symbol=True,  unit="°C")
                with k3: render_kpi("Δ vs same period LY", "—", None, show_symbol=True,  unit="°C")
                with k4: render_kpi("Mean / σ in range", "—", None, show_symbol=False, unit="°C")
            else:
                srt = base_tmpa.sort_values("date")
                latest = float(srt["avg"].iloc[-1])
                prev   = float(srt["avg"].iloc[-2]) if len(srt) > 1 else np.nan

                def _same_period_last_year(df):
                    dmax = df["date"].max()
                    if freq_sel == "Annual":
                        tgt = dmax - pd.DateOffset(years=1)
                        yr_prev = df[df["date"].dt.year == tgt.year]
                        return float(yr_prev["avg"].iloc[-1]) if not yr_prev.empty else np.nan
                    tgt = dmax - pd.DateOffset(years=1)
                    row = df[(df["date"].dt.year == tgt.year) & (df["date"].dt.month == tgt.month)]
                    return float(row["avg"].iloc[-1]) if not row.empty else np.nan

                ly_val     = _same_period_last_year(srt)
                delta_prev = latest - prev   if np.isfinite(prev)  else np.nan
                delta_yoy  = latest - ly_val if np.isfinite(ly_val) else np.nan
                mean_v     = float(np.nanmean(srt["avg"])) if len(srt) else np.nan
                std_v      = (
                    float(np.nanmean(np.sqrt(srt["var"].clip(lower=0))))
                    if ("var" in srt.columns and srt["var"].notna().any())
                    else float(np.nanstd(srt["avg"]))
                )

                # 1) Latest Avg — no symbol
                with k1:
                    render_kpi("Latest Avg Temp (TMPA)", f"{latest:.2f} °C", None, show_symbol=False, unit="°C")

                # 2) Δ vs previous — arrow only
                with k2:
                    render_kpi(
                        "Δ vs previous point",
                        (f"{delta_prev:+.2f} °C" if np.isfinite(delta_prev) else "—"),
                        (delta_prev if np.isfinite(delta_prev) else None),
                        show_symbol=True,
                        unit="°C",
                    )

                # 3) Δ vs same period LY — arrow only
                with k3:
                    render_kpi(
                        "Δ vs same period LY",
                        (f"{delta_yoy:+.2f} °C" if np.isfinite(delta_yoy) else "—"),
                        (delta_yoy if np.isfinite(delta_yoy) else None),
                        show_symbol=True,
                        unit="°C",
                    )

                # 4) Mean / σ — no symbol
                with k4:
                    val = (
                        f"{mean_v:.2f} °C • {std_v:.2f}"
                        if np.isfinite(mean_v) and np.isfinite(std_v)
                        else "—"
                    )
                    render_kpi("Mean / σ in range", val, None, show_symbol=False, unit="°C")




    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    # Comparison summary (scrollable, searchable; ALL ADM1s)
    st.markdown("##### Comparison Summary")
    comp_container = st.container()
    with comp_container:
        with st.spinner("Extracting ADM1 comparison summary..."):
            with st.expander("What do these columns mean?"):
                st.markdown("""
- **Last date** — latest period available at the selected frequency.  
- **TMPA/TMXA/TMNA** — latest values (°C).  
- **TMPV** — variance of average temperature (unit²); if missing, blank.  
                """)
            try:
                _, _, _namecol, _gdf = load_country_adm1_geojson(iso3_now)
                all_adm1_names = sorted(_gdf[_namecol].astype(str).unique().tolist())
            except Exception:
                all_adm1_names = []

            def _last_row_for(code, freq_str, area):
                s = _prep_single(iso3_now, area, freq_str, code, None, bundle=bundle_country)
                if s.empty: return "—", np.nan
                s = s.sort_values("date")
                d = s["date"].iloc[-1]
                if freq_str == "Monthly":
                    ds = d.strftime("%Y-%m")
                elif freq_str == "Seasonal":
                    ds = f"{MONTH_TO_SEASON[int(d.month)]} {d.year}"
                else:
                    ds = d.strftime("%Y")
                return ds, float(s["avg"].iloc[-1])

            def _last_var_for(freq_str, area):
                s = _prep_single(iso3_now, area, freq_str, "TMPA", "TMPV", bundle=bundle_country)
                if s.empty: return np.nan
                s = s.sort_values("date")
                return float(s["var"].iloc[-1]) if ("var" in s.columns and s["var"].notna().any()) else np.nan

            rows = []
            for a in all_adm1_names:
                d_str, v_tmpa = _last_row_for("TMPA", freq_sel, a)
                _,    v_tmx   = _last_row_for("TMXA", freq_sel, a)
                _,    v_tmn   = _last_row_for("TMNA", freq_sel, a)
                v_var        = _last_var_for(freq_sel, a)
                rows.append({
                    "ADM1": a,
                    "Last date": d_str,
                    "TMPA (°C)": v_tmpa,
                    "TMXA (°C)": v_tmx,
                    "TMNA (°C)": v_tmn,
                    "TMPV": v_var,
                })

            df_summary = pd.DataFrame(rows)

            # Dropdown-style search
            try:
                _, _, _namecol_all_sum, _gdf_all_sum = load_country_adm1_geojson(iso3_now)
                _adm1_all_sum = sorted(_gdf_all_sum[_namecol_all_sum].astype(str).unique().tolist())
            except Exception:
                _adm1_all_sum = []

            sel_filter_adm1 = st.multiselect(
                "Search / filter ADM1s (leave empty to show all)",
                options=_adm1_all_sum,
                default=[],
                placeholder="Start typing to search…"
            )
            if sel_filter_adm1:
                df_summary = df_summary[df_summary["ADM1"].isin(sel_filter_adm1)]

            styler = (
                df_summary.style
                    .format(precision=2, na_rep="—")
                    .set_table_styles([{'selector':'th','props':[('text-align','center')]}])
                    .set_properties(**{'text-align':'center'})
            )
            st.dataframe(styler, use_container_width=True, hide_index=True, height=280)


# ---------- Divider ----------
st.markdown("---")

# ---------- Common bits for charts ----------
mapper = load_indicator_mapper()
iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_t") or "")) or ""
adm1_now = "Country (all)"
freq     = st.session_state.get("opt_freq_t","Monthly")
bundle_country = load_temperature_bundle(iso3_now, freq)
if not iso3_now:
    st.warning("Select a country to load temperature charts.")
    st.stop()

def _season_year_str(series_ts: pd.Series) -> np.ndarray:
    s = series_ts.dt.month.map(MONTH_TO_SEASON).astype(str) + " " + series_ts.dt.year.astype(str)
    return s.to_numpy(dtype=object)

def _legend_top(fig):
    # Place legend under the plot area (bottom), with extra bottom margin.
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=-0.25,        # below the x-axis
            yanchor="top",
        ),
        margin=dict(b=80)   # add bottom space so legend isn't cut off
    )


# ---------- Generic chart renderer with per-chart ADM1 multiselect ----------
def _render_chart(
    chart_key,              # str
    story_title_md,         # str
    plot_title,             # str
    avg_code,               # str
    var_code=None,          # Optional[str]
    extras=None,            # Optional[List[str]]
    show_opts=None,         # dict like {"has_form":bool, "avg":bool, "band":bool, "ext":bool}
    ylab="°C",
    layout="wide"           # "wide" = story|chart|opts (0.2/0.7/0.1), "full" = story|chart (0.2/0.8)
):
    if extras is None: extras = []
    if show_opts is None: show_opts = {}

    hide_country_key = f"{chart_key}_hide_country"
    prev_adm1_sel = st.session_state.get(f"{chart_key}_adm1s", [])
    had_adm1_prev = bool(prev_adm1_sel)


    # ADM1 options
    try:
        _, _, _namecol_all_for_chart, _gdf_all_for_chart = load_country_adm1_geojson(iso3_now)
        _adm1_all_choices = sorted(_gdf_all_for_chart[_namecol_all_for_chart].astype(str).unique().tolist())
    except Exception:
        _adm1_all_choices = []

    # Layout
    if layout == "wide":
        story_col, chart_col, opts_col = st.columns([0.2, 0.7, 0.1], gap="large")
    else:
        story_col, chart_col = st.columns([0.2, 0.8], gap="large")
        opts_col = None

    has_form = bool(opts_col and show_opts.get("has_form", False))

    # Story
    with story_col:
        st.markdown(story_title_md)

        # Display-options form (per-chart)
        if has_form:
            with opts_col:
                with st.form(f"form_{chart_key}"):
                    st.markdown("**Display options**", help="Only affects this chart.")
                    if "avg" in show_opts:
                        show_avg = st.checkbox("Show indicator", value=True, key=f"{chart_key}_avg")
                    else:
                        show_avg = True
                    if "band" in show_opts:
                        show_band = st.checkbox("Show ±1σ band", value=True, key=f"{chart_key}_band")
                    else:
                        show_band = False
                    if "ext" in show_opts:
                        show_ext = st.checkbox(
                            "Show extremes", value=bool(extras),
                            key=f"{chart_key}_ext", disabled=not extras
                        )
                    else:
                        show_ext = False

                    # Only show the hide-country toggle once the user has selected ADM1s
                    if had_adm1_prev:
                        st.checkbox(
                            "Hide Country Line",
                            value=st.session_state.get(hide_country_key, False),
                            key=hide_country_key,
                            help="Hide the Country (all) line so you can compare ADM1s directly.",
                        )
                    else:
                        # Reset if no ADM1s selected previously
                        st.session_state[hide_country_key] = False

                    st.form_submit_button("Apply changes", type="primary")
        else:
            # No side Display-options form for this chart
            show_avg, show_band, show_ext = True, False, False
            # IMPORTANT: do NOT reset hide_country here; we handle it inline below.



        # Per-chart ADM1 selector
        with chart_col:
            adm1_sel = st.multiselect(
                "Compare ADM1s (max 5)",
                options=_adm1_all_choices,
                default=st.session_state.get(f"{chart_key}_adm1s", []),
                max_selections=5,
                key=f"{chart_key}_adm1s",
                placeholder="Type to search and select ADM1s…",
                help="Adds selected ADM1 lines to THIS chart."
            )

            hide_country = False

            if not has_form:
                # Charts without a side Display-options panel (e.g. TMDR):
                # show a simple inline checkbox when ADM1s are selected.
                if adm1_sel:
                    hide_country = st.checkbox(
                        "Hide Country Line",
                        key=hide_country_key,
                        value=st.session_state.get(hide_country_key, False),
                        help="Hide the Country (all) line so you can compare ADM1s directly.",
                    )
                else:
                    # No ADM1s selected: always show baseline
                    st.session_state[hide_country_key] = False
                    hide_country = False
            else:
                # For charts with a side Display-options form (TMPA/TMXA/TMNA),
                # hide-country is controlled by the form.
                if adm1_sel:
                    hide_country = bool(st.session_state.get(hide_country_key, False))
                else:
                    st.session_state[hide_country_key] = False
                    hide_country = False

    # Series builder: decide which labels to include
    has_adm1_now = bool(adm1_sel)
    if hide_country and has_adm1_now:
        targets = [a for a in adm1_sel if a != "Country (all)"]
    else:
        targets = ["Country (all)"] + [a for a in adm1_sel if a != "Country (all)"]


    def _series_for_geo(g):
        need = [avg_code] + ([var_code] if var_code else []) + (extras or [])
        # Use preloaded Country bundle to avoid repeated HF reads
        if g in ("", "Country (all)") and isinstance(bundle_country, pd.DataFrame) and not bundle_country.empty:
            df_codes = bundle_country.copy()
        else:
            df_codes, _ = load_scope_series(iso3_now, freq, g, need)
        if df_codes.empty or avg_code not in df_codes:
            return pd.DataFrame(columns=["date","avg","var"] + (extras or []))
        out = pd.DataFrame(
            {
                "date": df_codes["date"],
                "avg": pd.to_numeric(df_codes[avg_code], errors="coerce"),
            }
        )
        if var_code and (var_code in df_codes):
            out["var"] = pd.to_numeric(df_codes[var_code], errors="coerce")
        for e in (extras or []):
            if e in df_codes:
                out[e] = pd.to_numeric(df_codes[e], errors="coerce")
        return (
            out.dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )


    sdict = {g: _series_for_geo(g) for g in targets}
    sdict = {k:v for k,v in sdict.items() if not v.empty}
    if not sdict:
        with chart_col:
            st.info("No data for current selection.")
        return

    # Global bounds
    dmin = min(s["date"].min() for s in sdict.values()).date()
    dmax = max(s["date"].max() for s in sdict.values()).date()

    # Chart
    with chart_col:
        # >>> unified picker (Seasonal -> SEASON YYYY; Annual -> YYYY) <<<
        d1, d2 = _range_picker(
            pd.concat([s["date"] for s in sdict.values()], ignore_index=True),
            key=f"rng_{chart_key}",
            freq=freq,
        )
        if d1 is None or d2 is None:
            st.info("No data in selected range.")
            return

        fig = go.Figure()
        palette = list(CBLIND.values())

        for i, (label, s) in enumerate(sdict.items()):
            s2 = s[(s["date"] >= d1) & (s["date"] <= d2)].copy()
            if s2.empty:
                continue
            color = palette[i % len(palette)]
            N = len(s2)

            # Arrays
            hdr_arr   = _hover_header_array(s2["date"], freq)         # <-- date/season header
            label_arr = np.full(N, label, dtype=object)
            avg_arr   = pd.to_numeric(s2["avg"], errors="coerce").to_numpy(dtype=float)
            sigma_arr = (np.sqrt(pd.to_numeric(s2["var"], errors="coerce").clip(lower=0)).to_numpy(float)
                         if "var" in s2.columns else np.full(N, np.nan))
            ex1_arr = np.full(N, np.nan); ex2_arr = np.full(N, np.nan)
            if extras:
                if len(extras) >= 1 and (extras[0] in s2.columns):
                    ex1_arr = pd.to_numeric(s2[extras[0]], errors="coerce").to_numpy(dtype=float)
                if len(extras) >= 2 and (extras[1] in s2.columns):
                    ex2_arr = pd.to_numeric(s2[extras[1]], errors="coerce").to_numpy(dtype=float)

            # customdata: [0]=header, [1]=ADM1, [2]=avg, [3]=sigma, [4]=ex1, [5]=ex2
            customdata = np.empty((N, 6), dtype=object)
            customdata[:,0] = hdr_arr
            customdata[:,1] = label_arr
            customdata[:,2] = avg_arr
            customdata[:,3] = sigma_arr
            customdata[:,4] = ex1_arr
            customdata[:,5] = ex2_arr

            # ±1σ band
            if show_band and np.isfinite(sigma_arr).any():
                fig.add_trace(go.Scatter(
                    x=s2["date"], y=avg_arr + sigma_arr, mode="lines",
                    line=dict(width=0), hoverinfo="skip", showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=s2["date"], y=avg_arr - sigma_arr, mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(0,114,178,0.18)", hoverinfo="skip",
                    name=f"{label} — ±1σ", showlegend=False
                ))

            # Hover template (one date header; then ADM1 block)
            block_lines = [f"Average: %{{customdata[2]:.2f}} {ylab}"]
            if np.isfinite(sigma_arr).any():
                block_lines.append(f"±1σ: %{{customdata[3]:.2f}} {ylab}")
            if show_ext and extras:
                if len(extras) >= 1:
                    block_lines.append(f"{extras[0]}: %{{customdata[4]:.2f}} {ylab}")
                if len(extras) >= 2:
                    block_lines.append(f"{extras[1]}: %{{customdata[5]:.2f}} {ylab}")

            hover_tmpl = (
                "<b>%{customdata[0]}</b><br>"
                "<b>%{customdata[1]}</b><br>" + "<br>".join(block_lines) + "<extra></extra>"
            )

            if show_avg:
                fig.add_trace(go.Scatter(
                    x=s2["date"], y=s2["avg"], mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{label} — Average",
                    customdata=customdata, hovertemplate=hover_tmpl
                ))

            if show_ext and extras:
                for ex in extras:
                    if (ex in s2.columns) and s2[ex].notna().any():
                        elab = (
                            "Max. of Max. Temp" if ex in ("TMXX","TMNX")
                            else ("Min. of Max. Temp" if ex=="TMXN"
                            else ("Min. of Min. Temp" if ex=="TMNN" else ex))
                        )
                        fig.add_trace(go.Scatter(
                            x=s2["date"], y=s2[ex], mode="lines",
                            line=dict(color=color, width=1.2, dash="dot"),
                            name=f"{label} — {elab}", hoverinfo="skip"
                        ))

        fig.update_layout(
            title=plot_title, height=420, margin=dict(l=30,r=30,t=40,b=50),
            hovermode="x unified", xaxis_title="Date", yaxis_title=ylab
        )
        _legend_top(fig)
        # axis ticks: keep yearly ticks for Seasonal/Annual (hover shows season)
        if freq == "Monthly":   fig.update_xaxes(tickformat="%b\n%Y")
        else:                   fig.update_xaxes(tickformat="%Y")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})



# ---------- Charts ----------
#st.markdown("### Temperature Indicators")

charts_container = st.container()
with charts_container:
    with st.spinner("Extracting data for indicator charts..."):
        st.markdown("### Temperature Indicators")

        # TMPA (Avg + band)
        _render_chart(
            chart_key="tmpa",
            story_title_md="**Story — Average Temperature (TMPA)**  \nTrend and variability in near-surface air temperature.",
            plot_title="Average Temperature (TMPA)",
            avg_code="TMPA", var_code="TMPV", extras=None,
            show_opts={"has_form": True, "avg": True, "band": True}, ylab="°C", layout="wide"
        )

        # TMXA (max)
        _render_chart(
            chart_key="tmxa",
            story_title_md="**Story — Average Maximum Temperature (TMXA)**  \nDaily high temperatures over time.",
            plot_title="Average Maximum Temperature (TMXA)",
            avg_code="TMXA", var_code="TMXV", extras=["TMXX","TMXN"],
            show_opts={"has_form": True, "avg": True, "band": True, "ext": True}, ylab="°C", layout="wide"
        )

        # TMNA (min)
        _render_chart(
            chart_key="tmna",
            story_title_md="**Story — Average Minimum Temperature (TMNA)**  \nNight-time / early-morning temperatures.",
            plot_title="Average Minimum Temperature (TMNA)",
            avg_code="TMNA", var_code="TMNV", extras=["TMNX","TMNN"],
            show_opts={"has_form": True, "avg": True, "band": True, "ext": True}, ylab="°C", layout="wide"
        )

        # TMDR (diurnal range)
        _render_chart(
            chart_key="tmdr",
            story_title_md="**Story — Diurnal Temperature Range (TMDR)**  \nDifference between daily highs and lows.",
            plot_title="Diurnal Temperature Range (TMDR)",
            avg_code="TMDR", var_code="TMDR_var", extras=None,
            show_opts={"has_form": True, "avg": True}, ylab="°C", layout="full"
        )

        # (keep the rest of the percentile charts code as-is below this)


# ---------- Percentiles ----------
st.markdown("---")
st.subheader("Percentiles")

# Shared percentile radio for all three charts
pct_choice = st.radio(
    "Select a percentile (applies to all charts below)",
    options=[10,20,30,40,50,60,70,80,90,100],
    horizontal=True, index=1, key="pct_t_single",
    help="Choose one percentile line to overlay per ADM1."
)

# ADM1 choices for percentile charts (each chart will have its own selector)
try:
    _, _, _namecol_all, _gdf_all = load_country_adm1_geojson(iso3_now)
    _adm1_all = sorted(_gdf_all[_namecol_all].astype(str).unique().tolist())
except Exception:
    _adm1_all = []


pct_hide_country_key = "pct_hide_country"


def _phase_from_date(dt: pd.Timestamp, freq: str) -> int:
    if freq=="Monthly": return int(dt.month)
    if freq=="Seasonal":
        m=int(dt.month)
        return 1 if m in (12,1,2) else (2 if m in (3,4,5) else (3 if m in (6,7,8) else 4))
    return 1

def _empirical_percentile_curve(s_df: pd.DataFrame, pct: int, freq: str) -> pd.DataFrame:
    if s_df is None or s_df.empty or "avg" not in s_df: return pd.DataFrame(columns=["date","p"])
    d = pd.DataFrame({"date": pd.to_datetime(s_df["date"], errors="coerce"),
                      "val":  pd.to_numeric(s_df["avg"], errors="coerce")}).dropna()
    if d.empty: return pd.DataFrame(columns=["date","p"])
    d["_phase"] = d["date"].apply(lambda x: _phase_from_date(x, freq))
    q = d.groupby("_phase")["val"].quantile(pct/100.0)
    return pd.DataFrame({"date": d["date"], "p": d["_phase"].map(q)})

def _percentile_chart(title: str, avg_code: str, chart_key: str, story: str):
    """Percentile chart with its own ADM1 selector + hide-country toggle."""
    story_col, chart_col, _ = st.columns([0.2, 0.79, 0.01], gap="large")

    hide_country_key = f"{chart_key}_hide_country"

    def _s_for(g):
        return _prep_single(iso3_now, g, freq, avg_code, None, bundle=bundle_country)

    # --- left column: story text ---
    with story_col:
        st.markdown(f"**Story — {title}**  \n{story}")

    # --- right column: per-chart ADM1 selector + hide-country toggle ---
    with chart_col:
        # Per-chart ADM1 options (from the global _adm1_all we built above)
        try:
            adm1_options = _adm1_all
        except NameError:
            adm1_options = []

        adm1_sel = st.multiselect(
            "Compare ADM1s for this percentile chart (max 5)",
            options=adm1_options,
            default=st.session_state.get(f"{chart_key}_adm1s", []),
            max_selections=5,
            key=f"{chart_key}_adm1s",
            placeholder="Type to search and select ADM1s…",
        )

        # Inline hide-country toggle for this chart
        if adm1_sel:
            hide_country = st.checkbox(
                "Hide Country Line",
                key=hide_country_key,
                value=st.session_state.get(hide_country_key, False),
                help="Hide the Country (all) line so you can compare ADM1s directly.",
            )
        else:
            # No ADM1s selected: always show baseline
            st.session_state[hide_country_key] = False
            hide_country = False

    # --- decide which geos to plot ---
    if hide_country and adm1_sel:
        compare_targets = [a for a in adm1_sel if a != "Country (all)"]
    else:
        compare_targets = ["Country (all)"] + [a for a in adm1_sel if a != "Country (all)"]

    sdict = {g: _s_for(g) for g in compare_targets}
    sdict = {k: v for k, v in sdict.items() if not v.empty}
    if not sdict:
        with chart_col:
            st.info("No data for percentiles.")
        return

    # bounds for sliders: prefer Country baseline, fall back to first series
    if "Country (all)" in sdict:
        s_base = sdict["Country (all)"]
    else:
        s_base = next(iter(sdict.values()))

    dmin, dmax = s_base["date"].min().date(), s_base["date"].max().date()

    with chart_col:
        d1, d2 = _range_picker(s_base["date"], key=f"rng_{chart_key}", freq=freq)
        if d1 is None or d2 is None:
            st.info("No data in selected range.")
            return

        fig = go.Figure()
        colors = list(CBLIND.values())

        for i, (label, s) in enumerate(sdict.items()):
            s2 = s[(s["date"] >= d1) & (s["date"] <= d2)].copy()
            if s2.empty:
                continue
            color = colors[i % len(colors)]

            # Average
            fig.add_trace(go.Scatter(
                x=s2["date"], y=s2["avg"], mode="lines",
                name=f"{label} — Avg", line=dict(color=color, width=1.6),
            ))

            # Single percentile (shared pct_choice)
            pc = _empirical_percentile_curve(s2, int(pct_choice), freq)
            if not pc.empty:
                fig.add_trace(go.Scatter(
                    x=pc["date"], y=pc["p"], mode="lines",
                    name=f"{label} — P{pct_choice}",
                    line=dict(color=color, width=1.2, dash="dot"),
                ))

        fig.update_layout(
            title=title,
            height=420,
            margin=dict(l=30, r=30, t=40, b=80),  # a bit more bottom for legend
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="°C",
        )
        _legend_top(fig)  # now places legend at the bottom

        if freq == "Monthly":
            fig.update_xaxes(tickformat="%b\n%Y")
        else:
            fig.update_xaxes(tickformat="%Y")

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})



_pct_story = "How values compare with the historical distribution for the same month/season."
mapper = load_indicator_mapper()
for _title, _code, _key in [
    (_compose_title("TMPA", mapper, "Average Temperature — Percentiles"), "TMPA", "pct_tmpa"),
    (_compose_title("TMXA", mapper, "Average Maximum Temperature — Percentiles"), "TMXA", "pct_tmxA"),
    (_compose_title("TMNA", mapper, "Average Minimum Temperature — Percentiles"), "TMNA", "pct_tmnA"),
]:
    _percentile_chart(_title, _code, _key, _pct_story)
