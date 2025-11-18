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

import os, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# === Data backend selector (local | hf | auto) ===
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None
    list_repo_files = None

def _secret_or_env(k, default=""):
    try:
        if hasattr(st, "secrets") and k in st.secrets:
            return st.secrets[k]
    except Exception:
        pass
    return os.getenv(k, default)

DATA_BACKEND_OVERRIDE = os.getenv("DATA_BACKEND", "auto")  # "local" | "hf" | "auto"
HF_REPO_ID   = _secret_or_env("HF_REPO_ID",   "pjsimba16/adb_climate_dashboard_v1")
HF_REPO_TYPE = _secret_or_env("HF_REPO_TYPE", "space")      # "space" or "dataset"
HF_TOKEN     = _secret_or_env("HF_TOKEN",     "")

def _resolve_backend() -> str:
    # allow ?backend=hf in URL, or st.secrets["DATA_BACKEND"], or env var
    qp = {}
    try:
        qp = st.query_params
    except Exception:
        pass
    if qp:
        b = str(qp.get("backend", [""])[0]).lower()
        if b in ("local","hf","auto"): return b
    if hasattr(st, "secrets"):
        v = str(st.secrets.get("DATA_BACKEND","auto")).lower()
        if v in ("local","hf","auto"): return v
    ov = str(DATA_BACKEND_OVERRIDE or "auto").lower()
    return ov if ov in ("local","hf","auto") else "auto"

DATA_BACKEND = _resolve_backend()
ROOT = Path(__file__).resolve().parents[1]  # project root (…/app)
COUNTRY_DATA_DIR = next((p for p in [
    ROOT / "country_data",
    Path.cwd() / "country_data",
    Path("/mnt/data/country_data"),
] if p.exists()), ROOT / "country_data")

def _rel_from_country_data(p: Path) -> str:
    # convert local file path to "country_data/ISO/Freq/…parquet"
    try:
        return "country_data/" + str(p.relative_to(COUNTRY_DATA_DIR)).replace("\\","/")
    except Exception:
        return None

def _read_parquet_local(p: Path, columns=None):
    return pd.read_parquet(p, columns=columns)  # let it raise if missing

def _read_parquet_hf(relpath: str, columns=None):
    if hf_hub_download is None:
        raise FileNotFoundError("huggingface_hub not available.")
    fp = hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                         filename=relpath, token=HF_TOKEN)
    return pd.read_parquet(fp, columns=columns)

def read_parquet_smart(p: Path, columns=None):
    """
    Tries local first (if backend=local or auto+exists). Otherwise loads from HF.
    'p' must be a path under COUNTRY_DATA_DIR.
    """
    use = DATA_BACKEND
    if use in ("local","auto") and p.exists():
        return _read_parquet_local(p, columns=columns)
    rel = _rel_from_country_data(p)
    if use in ("hf","auto") and rel:
        return _read_parquet_hf(rel, columns=columns)
    # last attempt: local (to surface the original error)
    return _read_parquet_local(p, columns=columns)



st.set_page_config(page_title="Temperature Dashboard", layout="wide", initial_sidebar_state="collapsed")

# === KPI styles + helper (same as precipitation) ===
st.markdown("""
<style>
.kpi { display:flex; flex-direction:column; gap:2px; }
.kpi .row { display:flex; align-items:center; gap:6px; }
.kpi .label { font-size:14px; color:#334155; display:flex; align-items:center; gap:6px; }
.kpi .value { font-size:34px; line-height:1.05; font-weight:600; }
.kpi .badge {
  display:inline-block; line-height:1; border-radius:999px; padding:2px 6px;
  border:1px solid #e5e7eb; font-size:11px; font-weight:700; color:#334155;
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
    if delta is None or not (isinstance(delta,(int,float)) and np.isfinite(delta)) or abs(delta) < 1e-12:
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
SEASON_TO_MONTH = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
MONTH_TO_SEASON = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
INDICATOR_LABELS = ("Temperature","Precipitation","Temperature Thresholds","Heatwaves","Coldwaves","Dry Conditions","Wet Conditions","Humidity","Windspeeds")
CBLIND = {"blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73","yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"}

# === Seasonal/Annual label + unified range control ===
MONTH_TO_SEASON = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}

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
    import streamlit as st
    s = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if s.empty:
        return None, None
    dmin, dmax = s.min().date(), s.max().date()

    if freq == "Seasonal":
        labels = s.map(_season_label_from_ts).tolist()
        seen, uniq, idx_map = set(), [], {}
        for dt, lb in zip(s, labels):
            if lb not in seen:
                seen.add(lb); uniq.append(lb); idx_map[lb] = pd.Timestamp(dt)
        start_lb, end_lb = st.select_slider(
            "Date range",
            options=uniq,
            value=(uniq[0], uniq[-1]),
            key=key
        )
        return idx_map[start_lb], idx_map[end_lb]

    fmt = "YYYY-MM" if freq == "Monthly" else "YYYY"
    d1, d2 = st.slider("Date range", min_value=dmin, max_value=dmax,
                       value=(dmin, dmax), format=fmt, key=key)
    return pd.Timestamp(d1), pd.Timestamp(d2)

# ---------- PATH HELPERS ----------
def _here():
    try: return Path(__file__).parent.resolve()
    except Exception: return Path.cwd()

def _probe_country_data_dir():
    here = _here()
    for p in [here / "country_data", here.parent / "country_data", Path.cwd() / "country_data", Path("/mnt/data/country_data")]:
        if p.exists() and p.is_dir(): return p
    return here.parent / "country_data"

def _probe_mapper_file():
    here = _here()
    for p in [here / "indicator_code_mapper.csv", here.parent / "indicator_code_mapper.csv",
              Path.cwd() / "indicator_code_mapper.csv", Path("/mnt/data/indicator_code_mapper.csv")]:
        if p.exists() and p.is_file(): return p
    return None

COUNTRY_DATA_DIR = _probe_country_data_dir()
MAPPER_FILE      = _probe_mapper_file()

# ---------- GENERIC HELPERS ----------
def _suffixes_for_freq(freq, adm_level):
    f = str(freq); adm = str(adm_level).upper()
    if adm == "ADM1":
        return ["_M"] if f=="Monthly" else (["_S"] if f=="Seasonal" else ["_A"])
    return ["_AM","_PM"] if f=="Monthly" else (["_AS","_PS"] if f=="Seasonal" else ["_AA"])

def _pick_col(df_cols, base_code, suffixes):
    if not base_code: return None
    want = {f"{base_code}{s}".lower() for s in suffixes}
    lower_map = {str(c).lower(): c for c in df_cols}
    for w in want:
        if w in lower_map: return lower_map[w]
    return None

def _build_date_column(df, freq):
    year = pd.to_numeric(df.get("Year"), errors="coerce")
    if str(freq) == "Monthly":
        mo = pd.to_numeric(df.get("Month"), errors="coerce")
        return pd.to_datetime(dict(year=year, month=mo, day=1), errors="coerce")
    if str(freq) == "Seasonal":
        seas = df.get("Season").astype(str)
        anchor = seas.map(SEASON_TO_MONTH).fillna(2).astype(int)
        return pd.to_datetime(dict(year=year, month=anchor, day=1), errors="coerce")
    return pd.to_datetime(dict(year=year, month=7, day=1), errors="coerce")

def _norm_str(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize('NFKD', s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _format_hover_date(ts: pd.Timestamp, freq: str) -> str:
    if freq == "Monthly":   return ts.strftime("%b %Y")
    if freq == "Seasonal":  return f"{MONTH_TO_SEASON[int(ts.month)]} {ts.year}"
    return ts.strftime("%Y")

# ---------- INDICATOR MAPPER ----------
@st.cache_data(ttl=24*3600, show_spinner=False)
def load_indicator_mapper():
    if not MAPPER_FILE: return pd.DataFrame()
    try: return pd.read_csv(MAPPER_FILE)
    except Exception: return pd.DataFrame()

def _compose_title(code, mapper, fallback):
    if isinstance(mapper, pd.DataFrame) and not mapper.empty and {"Code","Description"}.issubset(mapper.columns):
        hit = mapper.loc[mapper["Code"].astype(str).str.upper() == str(code).upper()]
        if not hit.empty:
            d = str(hit.iloc[0]["Description"])
            return d if d else fallback
    return fallback

# ---------- FILE RESOLVERS ----------
def _adm0_file(iso3, freq):
    folder = {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[str(freq)]
    p = COUNTRY_DATA_DIR / str(iso3).upper() / folder / f"{str(iso3).upper()}_ADM0_data.parquet"
    return p if p.exists() else None

def _adm1_file(iso3, adm1_name, freq):
    folder = {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[str(freq)]
    base = COUNTRY_DATA_DIR / str(iso3).upper() / folder
    if not base.exists(): return None
    candidates = [
        base / f"{adm1_name}_ADM1_data.parquet",
        base / f"{adm1_name.replace(' ', '_')}_ADM1_data.parquet",
        base / f"{adm1_name.replace('/', '-')}_ADM1_data.parquet",
    ]
    for c in candidates:
        if c.exists(): return c
    want = _norm_str(adm1_name)
    for f in sorted(base.glob("*_ADM1_data.parquet")):
        stem = f.name[:-len("_ADM1_data.parquet")]
        if _norm_str(stem) == want: return f
    return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def list_available_isos():
    if not COUNTRY_DATA_DIR.exists(): return []
    out = []
    for d in sorted(COUNTRY_DATA_DIR.iterdir()):
        try:
            if d.is_dir() and len(d.name) == 3: out.append(d.name.upper())
        except Exception: continue
    return out

# ---------- SERIES LOADERS ----------
@st.cache_data(ttl=1800, show_spinner=False)
def load_scope_series(iso3, freq, area_label, indicator_codes):
    mapper = load_indicator_mapper()
    is_country = (area_label in ("", "Country (all)"))
    suffixes = _suffixes_for_freq(freq, "ADM0" if is_country else "ADM1")
    path = _adm0_file(iso3, freq) if is_country else _adm1_file(iso3, area_label, freq)
    if path is None: return pd.DataFrame(), {}
    try: df_raw = pd.read_parquet_smart(path)
    except Exception: return pd.DataFrame(), {}
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

def _prep_single(iso3_now, adm1_now, freq, code_avg, code_var=None):
    codes = [code_avg] + ([code_var] if code_var else [])
    area = adm1_now if (adm1_now and adm1_now != "Country (all)") else "Country (all)"
    df_codes, _ = load_scope_series(iso3_now, freq, area, codes)
    if df_codes.empty or code_avg not in df_codes:
        return pd.DataFrame(columns=["date","avg","var"])
    out = pd.DataFrame({"date": df_codes["date"], "avg": pd.to_numeric(df_codes[code_avg], errors="coerce")})
    if code_var and code_var in df_codes:
        out["var"] = pd.to_numeric(df_codes[code_var], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# ---------- GEOJSON ----------
def _get_hf_token():
    try:
        if hasattr(st, "secrets"):
            tok = st.secrets.get("HF_TOKEN")
            if tok: return str(tok)
    except Exception: pass
    return os.getenv("HF_TOKEN", "")

@st.cache_data(ttl=7*24*3600, show_spinner=False)
def load_country_adm1_geojson(iso3):
    if hf_hub_download is None: raise FileNotFoundError("huggingface_hub is not available.")
    last_err = None
    for repo_type in ("space", "dataset"):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, repo_type=repo_type,
                                   filename=f"ADM1_geodata/{iso3}.geojson",
                                   token=_get_hf_token())
            gdf = gpd.read_file(path)
            try: gdf["geometry"] = gdf["geometry"].buffer(0)
            except Exception: pass
            gdf = gdf.to_crs(4326)
            bounds = tuple(gdf.total_bounds)
            name_col = "shapeName" if "shapeName" in gdf.columns else ("NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0])
            return gdf.__geo_interface__, bounds, name_col, gdf
        except Exception as e:
            last_err = e; continue
    raise last_err

@st.cache_data(ttl=7*24*3600, show_spinner=False)
def adm1_label_points(iso3):
    _, _, name_col, gdf = load_country_adm1_geojson(iso3)
    pts = gdf.representative_point()
    return pts.x.to_numpy(), pts.y.to_numpy(), gdf[name_col].astype(str).to_numpy(), name_col

# ---------- Elevation CSV ----------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_city_map():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.normpath(os.path.join(here, "..", "city_mapper_with_coords_v3.csv")),
        os.path.normpath(os.path.join(here, "..", "city_mapper_with_coords_v2.csv")),
        os.path.join(here, "city_mapper_with_coords_v3.csv"),
        os.path.join(here, "city_mapper_with_coords_v2.csv"),
        "/mnt/data/city_mapper_with_coords_v3.csv",
        "/mnt/data/city_mapper_with_coords_v2.csv",
    ]
    for fp in candidates:
        try:
            if os.path.isfile(fp):
                df = pd.read_csv(fp)
                cn_country = next((c for c in df.columns if c.lower() == "country"), None)
                cn_city    = next((c for c in df.columns if c.lower() == "city"), None)
                cn_elev    = next((c for c in df.columns if "elev" in c.lower()), None)
                if not (cn_country and cn_city and cn_elev): continue
                df = df.rename(columns={cn_country:"ADM0", cn_city:"ADM1", cn_elev:"elevation"})
                df["ADM0"] = df["ADM0"].astype(str).str.upper().str.strip()
                df["ADM1"] = df["ADM1"].astype(str).str.strip()
                df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
                return df[["ADM0","ADM1","elevation"]].copy()
        except Exception:
            continue
    return pd.DataFrame(columns=["ADM0","ADM1","elevation"])

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
@st.cache_data(ttl=900, show_spinner=False)
def load_latest_adm1_for_map(iso3, freq, want_debug=False):
    def _scan_one(iso3_in, freq_in):
        base_dir = COUNTRY_DATA_DIR / str(iso3_in).upper() / {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[str(freq_in)]
        dbg = {"iso3": iso3_in, "freq": freq_in, "base_dir": str(base_dir), "base_dir_exists": base_dir.exists(), "files": [], "hits": 0}
        if not base_dir.exists(): return pd.DataFrame(columns=["ADM1","value","Date"]), dbg
        suffixes = _suffixes_for_freq(freq_in, "ADM1")
        rows=[]
        for f in sorted(base_dir.glob("*_ADM1_data.parquet")):
            dbg["files"].append(f.name)
            try: cols = list(pd.read_parquet_smart(f, columns=None).columns)
            except Exception: continue
            col_tmpa = _pick_col(cols, "TMPA", suffixes)
            if not col_tmpa: continue
            use_cols = ["ADM1","Year", col_tmpa]
            if str(freq_in) == "Monthly":  use_cols.append("Month")
            elif str(freq_in) == "Seasonal": use_cols.append("Season")
            try: d = pd.read_parquet_smart(f, columns=use_cols)
            except Exception: continue
            d = d.rename(columns={col_tmpa:"value"})
            d["date"] = _build_date_column(d, freq_in)
            d = d.dropna(subset=["ADM1","date","value"])
            if d.empty: continue
            d = d.sort_values("date").tail(1).rename(columns={"date":"Date"})
            rows.append(d[["ADM1","value","Date"]])
        if not rows: return pd.DataFrame(columns=["ADM1","value","Date"]), dbg
        df_map = pd.concat(rows, ignore_index=True).sort_values(["ADM1","Date"]).drop_duplicates("ADM1", keep="last").reset_index(drop=True)
        dbg["hits"] = len(df_map); return df_map, dbg

    df, dbg = _scan_one(iso3, freq); tried=[(freq, dbg)]
    if df.empty and not want_debug:
        for f in ["Monthly","Seasonal","Annual"]:
            if f == freq: continue
            dfi, dbgi = _scan_one(iso3, f); tried.append((f, dbgi))
            if not dfi.empty: dfi = dfi.copy(); dfi["__actual_freq_used__"] = f; return dfi
    return (df, tried) if want_debug else df

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
            df_map = gdf_norm.merge(latest_norm[["__key","value","Date"]], on="__key", how="left").drop(columns="__key")

            CITY_MAP = _load_city_map()
            elev_complete, elev_avail, elev_total = _elevation_completeness(iso3, geojson_dict, CITY_MAP)
            if st.session_state.get("last_iso3_for_choice_t") != iso3 or "map_data_choice_t" not in st.session_state:
                st.session_state["map_data_choice_t"] = "Elevation" if elev_complete else "Temperature"
                st.session_state["last_iso3_for_choice_t"] = iso3

            try:
                cm_iso = CITY_MAP[CITY_MAP["ADM0"].astype(str).str.upper() == iso3][["ADM1","elevation"]].copy()
                cm_iso["ADM1"] = cm_iso["ADM1"].astype(str)
                df_map = df_map.merge(cm_iso, on="ADM1", how="left", suffixes=("","_cm"))
                if "elevation_cm" in df_map.columns:
                    df_map["elevation"] = df_map["elevation_cm"].combine_first(df_map.get("elevation"))
                    df_map.drop(columns=["elevation_cm"], inplace=True)
            except Exception:
                if "elevation" not in df_map.columns:
                    df_map["elevation"] = np.nan

            choice = st.radio("Map data", ["Temperature","Elevation"], horizontal=True, key="map_data_choice_t", help="Choose the data shown on the map.")
            if choice == "Elevation":
                color_col = "elevation"; cs_name = "Viridis"; cbar_title = "Elevation (m)"
                hover_tmpl = "<b>%{customdata[0]}</b><br>Elevation: %{customdata[1]:.0f} m<br>As of: %{customdata[2]}<extra></extra>"
            else:
                color_col = "value"; cs_name = "YlOrRd"; cbar_title = "Temperature (°C)"
                hover_tmpl = "<b>%{customdata[0]}</b><br>Latest: %{customdata[1]:.2f} °C<br>As of: %{customdata[2]}<extra></extra>"

            fig = px.choropleth(df_map, geojson=geojson_dict, locations="ADM1",
                                featureidkey=f"properties.{name_col}", color=color_col,
                                projection="mercator", color_continuous_scale=cs_name)
            minx, miny, maxx, maxy = bounds
            pad = 0.35
            fig.update_geos(projection_type="mercator", fitbounds="locations",
                            lonaxis_range=[minx - pad, maxx + pad],
                            lataxis_range=[miny - pad, maxy + pad],
                            showland=False, showcountries=False, showcoastlines=False, showocean=False, visible=False)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=MAP_HEIGHT,
                              paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                              hovermode="closest", showlegend=False,
                              coloraxis_colorbar=dict(title=cbar_title, thickness=14, len=0.98, y=0.92, yanchor="top", x=0.98,
                                                      tickfont=dict(size=10), titlefont=dict(size=11)))
            _date_str = pd.to_datetime(df_map["Date"], errors="coerce").dt.strftime("%Y-%m").fillna("—")
            vals = pd.to_numeric(df_map[color_col], errors="coerce")
            fig.data[0].customdata = np.stack([df_map["ADM1"].astype(str).values, vals.values, _date_str.values], axis=-1)
            fig.data[0].hovertemplate = hover_tmpl
            fig.update_traces(marker_line_width=0.3, marker_line_color="#999")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    base_tmpa = _prep_single(iso3_now, adm1_now, freq_sel, "TMPA", "TMPV")
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

        ly_val = _same_period_last_year(srt)
        delta_prev = latest - prev if np.isfinite(prev) else np.nan
        delta_yoy  = latest - ly_val if np.isfinite(ly_val) else np.nan
        mean_v     = float(np.nanmean(srt["avg"])) if len(srt) else np.nan
        std_v      = (float(np.nanmean(np.sqrt(srt["var"].clip(lower=0)))) 
                    if ("var" in srt.columns and srt["var"].notna().any()) else float(np.nanstd(srt["avg"])))

        # 1) Latest Avg — no symbol
        with k1:
            render_kpi("Latest Avg Temp (TMPA)", f"{latest:.2f} °C", None, show_symbol=False, unit="°C")

        # 2) Δ vs previous — arrow only (value text is “inert”; arrow conveys the change)
        with k2:
            render_kpi("Δ vs previous point",
                    (f"{delta_prev:+.2f} °C" if np.isfinite(delta_prev) else "—"),
                    (delta_prev if np.isfinite(delta_prev) else None),
                    show_symbol=True, unit="°C")

        # 3) Δ vs same period LY — arrow only; label adapts by frequency
        ly_label = "Δ vs same month LY" if freq_sel=="Monthly" else ("Δ vs same season LY" if freq_sel=="Seasonal" else "Δ vs same year LY")
        with k3:
            render_kpi(ly_label,
                    (f"{delta_yoy:+.2f} °C" if np.isfinite(delta_yoy) else "—"),
                    (delta_yoy if np.isfinite(delta_yoy) else None),
                    show_symbol=True, unit="°C")

        # 4) Mean / σ — no symbol
        with k4:
            val = (f"{mean_v:.2f} °C • {std_v:.2f}" if np.isfinite(mean_v) and np.isfinite(std_v) else "—")
            render_kpi("Mean / σ in range", val, None, show_symbol=False, unit="°C")



    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    # Comparison summary (scrollable, searchable; ALL ADM1s)
    st.markdown("##### Comparison Summary")
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
        s = _prep_single(iso3_now, area, freq_str, code, None)
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
        s = _prep_single(iso3_now, area, freq_str, "TMPA", "TMPV")
        if s.empty: return np.nan
        s = s.sort_values("date")
        return float(s["var"].iloc[-1]) if ("var" in s.columns and s["var"].notna().any()) else np.nan

    rows = []
    for a in all_adm1_names:
        d_str, v_tmpa = _last_row_for("TMPA", freq_sel, a)
        _,    v_tmx   = _last_row_for("TMXA", freq_sel, a)
        _,    v_tmn   = _last_row_for("TMNA", freq_sel, a)
        v_var        = _last_var_for(freq_sel, a)
        rows.append({"ADM1": a, "Last date": d_str, "TMPA (°C)": v_tmpa, "TMXA (°C)": v_tmx, "TMNA (°C)": v_tmn, "TMPV": v_var})

    df_summary = pd.DataFrame(rows)
    # Dropdown-style search (typeahead). If none selected -> show all.
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


    styler = (df_summary.style
                .format(precision=2, na_rep="—")
                .set_table_styles([{'selector':'th','props':[('text-align','center')]}])
                .set_properties(**{'text-align':'center'}))
    st.dataframe(styler, use_container_width=True, hide_index=True, height=280)

# ---------- Divider ----------
st.markdown("---")

# ---------- Common bits for charts ----------
mapper = load_indicator_mapper()
iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_t") or "")) or ""
adm1_now = "Country (all)"
freq     = st.session_state.get("opt_freq_t","Monthly")
if not iso3_now:
    st.warning("Select a country to load temperature charts.")
    st.stop()

def _season_year_str(series_ts: pd.Series) -> np.ndarray:
    s = series_ts.dt.month.map(MONTH_TO_SEASON).astype(str) + " " + series_ts.dt.year.astype(str)
    return s.to_numpy(dtype=object)

def _legend_top(fig):
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.0, xanchor="left"),
                      margin=dict(t=80))

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

    # Story
    with story_col:
        st.markdown(story_title_md)

    # Display-options form (per-chart)
    if opts_col and show_opts.get("has_form", False):
        with opts_col:
            with st.form(f"form_{chart_key}"):
                st.markdown("**Display options**", help="Only affects this chart.")
                if "avg" in show_opts:
                    show_avg = st.checkbox("Show average", value=True, key=f"{chart_key}_avg")
                else:
                    show_avg = True
                if "band" in show_opts:
                    show_band = st.checkbox("Show ±1σ band", value=True, key=f"{chart_key}_band")
                else:
                    show_band = False
                if "ext" in show_opts:
                    show_ext = st.checkbox("Show extremes", value=bool(extras), key=f"{chart_key}_ext", disabled=not extras)
                else:
                    show_ext = False
                st.form_submit_button("Apply changes", type="primary")
    else:
        show_avg, show_band, show_ext = True, False, False

    # Per-chart ADM1 selector
    with chart_col:
        adm1_sel = st.multiselect(
            "Compare ADM1s (max 5)",
            options=_adm1_all_choices,
            default=st.session_state.get(f"{chart_key}_adm1s", []),
            max_selections=5,
            key=f"{chart_key}_adm1s",
            placeholder="Type to search and select ADM1s…",
            help="Adds selected ADM1 lines to THIS chart (Country baseline is always shown)."
        )

    # Series builder
    targets = ["Country (all)"] + [a for a in adm1_sel if a != "Country (all)"]
    def _series_for_geo(g):
        need = [avg_code] + ([var_code] if var_code else []) + (extras or [])
        df_codes, _ = load_scope_series(iso3_now, freq, g, need)
        if df_codes.empty or avg_code not in df_codes:
            return pd.DataFrame(columns=["date","avg","var"] + (extras or []))
        out = pd.DataFrame({"date": df_codes["date"], "avg": pd.to_numeric(df_codes[avg_code], errors="coerce")})
        if var_code and (var_code in df_codes):
            out["var"] = pd.to_numeric(df_codes[var_code], errors="coerce")
        for e in (extras or []):
            if e in df_codes:
                out[e] = pd.to_numeric(df_codes[e], errors="coerce")
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

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
        d1, d2 = _range_picker(pd.concat([sdict["Country (all)"]["date"]], ignore_index=True), key=f"rng_{chart_key}", freq=freq)
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
st.markdown("### Temperature Indicators")

# TMPA (Avg + band)
_render_chart(
    chart_key="tmpa",
    story_title_md="**Story — Average Temperature (TMPA)**  \nTrend and variability in near-surface air temperature.",
    plot_title="Average Temperature (TMPA)",
    avg_code="TMPA", var_code="TMPV", extras=None,
    show_opts={"has_form": True, "avg": True, "band": True}, ylab="°C", layout="wide"
)

# TMXA (Avg + Extremes)
_render_chart(
    chart_key="tmxa",
    story_title_md="**Story — Average Maximum Temperature (TMXA)**  \nAverage of daily maxima; extremes show max/min if available.",
    plot_title="Average Max Temperature (TMXA) + Extremes",
    avg_code="TMXA", var_code=None, extras=["TMXX","TMXN"],
    show_opts={"has_form": True, "avg": True, "ext": True}, ylab="°C", layout="wide"
)

# TMNA (Avg + Extremes)
_render_chart(
    chart_key="tmna",
    story_title_md="**Story — Average Minimum Temperature (TMNA)**  \nAverage of daily minima; extremes show max/min if available.",
    plot_title="Average Min Temperature (TMNA) + Extremes",
    avg_code="TMNA", var_code=None, extras=["TMNX","TMNN"],
    show_opts={"has_form": True, "avg": True, "ext": True}, ylab="°C", layout="wide"
)

# TMDR (no options; fills space)
_render_chart(
    chart_key="tmdr",
    story_title_md="**Story — Diurnal Temperature Range (TMDR)**  \nDifference between daily maximum and minimum temperatures.",
    plot_title="Diurnal Temperature Range (TMDR)",
    avg_code="TMDR", var_code=None, extras=None,
    show_opts={"has_form": False}, ylab="°C", layout="full"
)

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

# One ADM1 selector shared by all percentile charts
try:
    _, _, _namecol_all, _gdf_all = load_country_adm1_geojson(iso3_now)
    _adm1_all = sorted(_gdf_all[_namecol_all].astype(str).unique().tolist())
except Exception:
    _adm1_all = []
sel_pct_adm1s = st.multiselect("Compare ADM1s for percentile charts (max 5)", options=_adm1_all,
                               default=[], max_selections=5, key="pct_adm1s_shared")

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
    story_col, chart_col, _ = st.columns([0.2, 0.79, 0.01], gap="large")

    def _s_for(g):
        return _prep_single(iso3_now, g, freq, avg_code, None)

    # Country + selected ADM1s for the shared percentile selector
    compare_targets = ["Country (all)"] + [a for a in sel_pct_adm1s if a != "Country (all)"]
    sdict = {g: _s_for(g) for g in compare_targets}
    sdict = {k:v for k,v in sdict.items() if not v.empty}
    if not sdict:
        with story_col: st.info("No data for percentiles.")
        return

    # bounds from Country (all) to keep sliders consistent
    s_base = sdict["Country (all)"]
    dmin, dmax = s_base["date"].min().date(), s_base["date"].max().date()

    with story_col:
        st.markdown(f"**Story — {title}**  \n{story}")

    with chart_col:
        d1, d2 = _range_picker(s_base["date"], key=f"rng_{chart_key}", freq=freq)
        if d1 is None or d2 is None:
            st.info("No data in selected range.")
            return

        fig = go.Figure()
        colors = list(CBLIND.values())

        for i, (label, s) in enumerate(sdict.items()):
            s2 = s[(s["date"]>=d1) & (s["date"]<=d2)].copy()
            if s2.empty: 
                continue
            color = colors[i % len(colors)]
            # Average
            fig.add_trace(go.Scatter(x=s2["date"], y=s2["avg"], mode="lines",
                                     name=f"{label} — Avg", line=dict(color=color, width=1.6)))

            # Single percentile
            pc = _empirical_percentile_curve(s2, int(pct_choice), freq)
            if not pc.empty:
                fig.add_trace(go.Scatter(x=pc["date"], y=pc["p"], mode="lines",
                                         name=f"{label} — P{pct_choice}",
                                         line=dict(color=color, width=1.2, dash="dot")))

        fig.update_layout(title=title, height=420, margin=dict(l=30,r=30,t=40,b=50),
                          hovermode="x unified", xaxis_title="Date", yaxis_title="°C")
        _legend_top(fig)
        if freq == "Monthly":   fig.update_xaxes(tickformat="%b\n%Y")
        else:                   fig.update_xaxes(tickformat="%Y")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


_pct_story = "How values compare with the historical distribution for the same month/season."
mapper = load_indicator_mapper()
for _title, _code, _key in [
    (_compose_title("TMPA", mapper, "Average Temperature — Percentiles"), "TMPA", "pct_tmpa"),
    (_compose_title("TMXA", mapper, "Average Maximum Temperature — Percentiles"), "TMXA", "pct_tmxA"),
    (_compose_title("TMNA", mapper, "Average Minimum Temperature — Percentiles"), "TMNA", "pct_tmnA"),
]:
    _percentile_chart(_title, _code, _key, _pct_story)
