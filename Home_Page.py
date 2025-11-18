# Home_Page.py — ADM0-only availability + snapshot (TMPA/HUMA/PCPA/WSPA)
import os
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Manila")
except Exception:
    LOCAL_TZ = None

try:
    import pycountry
except Exception:
    pycountry = None

# === Data backend selector (local | hf | auto) ===
import os
from pathlib import Path
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
HF_REPO_ID   = _secret_or_env("HF_REPO_ID",   "pjsimba16/adb-climate-data")
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

def _adm0_fp(iso: str, freq: str) -> Path:
    folder = {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[freq]
    return COUNTRY_DATA_DIR / iso.upper() / folder / f"{iso.upper()}_ADM0_data.parquet"

def _to_date_from_parts(df: pd.DataFrame, freq: str) -> pd.Series:
    if freq == "Monthly":
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        return pd.to_datetime(dict(year=y, month=m, day=1), errors="coerce")
    if freq == "Seasonal":
        # accept Season like "DJF","MAM","JJA","SON" or "Q1" etc.
        s = df["Season"].astype(str)
        # try quarter first
        q = s.str.extract(r"Q([1-4])", expand=False).astype(float)
        if q.notna().any():
            month = q.map({1:1,2:4,3:7,4:10})
            y = pd.to_numeric(df.get("Year"), errors="coerce")
            return pd.to_datetime(dict(year=y, month=month, day=1), errors="coerce")
        # fallback DJF/MAM/JJA/SON
        SEASON_TO_MONTH = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
        y = pd.to_numeric(df.get("Year"), errors="coerce")
        anchor = s.str.upper().map(SEASON_TO_MONTH).fillna(2).astype(int)
        return pd.to_datetime(dict(year=y, month=anchor, day=1), errors="coerce")
    y = pd.to_numeric(df["Year"], errors="coerce")
    return pd.to_datetime(dict(year=y, month=7, day=1), errors="coerce")


st.set_page_config(
    page_title="Home Page — Global Database of Subnational Climate Indicators",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styles (unchanged UI) ----------
st.markdown("""
<style>
:root { --muted:#64748b; }
h1, h2, h3 { letter-spacing:.2px; }
.subtitle { text-align:center; color:#64748b; margin-top:-.4rem; }
.card { border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#fafafa; }
.hero { padding:12px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#f8fafc; margin-bottom:.75rem; }
.badge { padding:4px 8px; border-radius:999px; background:#eef2ff; border:1px solid #e0e7ff; font-size:12px; }
.footer-box { padding:16px; border-top:1px solid #e5e7eb; margin-top:1rem; color:#64748b; }
.full-bleed { width: 100vw; margin-left: calc(-50vw + 50%); }
@media (min-width: 1400px) {
  [data-testid="stAppViewContainer"] .main .block-container { padding-left: .5rem; padding-right: .5rem; }
}
.legend-chip { display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,255,255,0.9); border:1px solid #e5e7eb; border-radius:10px; padding:6px 10px; font-size:12px; }
.legend-swatch { display:inline-block; width:10px; height:10px; background:#12a39a; border:1px solid rgba(0,0,0,.25); }
.align-with-input { height: 1.9rem; }
[data-testid="stPlotlyChart"] div, [data-testid="stPlotlyChart"] canvas { border-radius: 0 !important; }
.uc-card { border:1px solid #e5e7eb; border-radius:14px; padding:12px; background:white; height:100%; }
.uc-card h4 { margin:0 0 .25rem 0; font-size:16px; }
.uc-card p { margin:.15rem 0 0 0; font-size:13px; color:#475569; }
.panel-left{
  position: relative; border: 1px solid rgba(235,92,86,0.25);
  border-radius:16px; padding:16px 18px; margin-bottom:10px; background: transparent !important; overflow: hidden;
}
.panel-left::before{ content:""; position:absolute; inset:0; background: rgba(235,92,86,0.06); border-radius: inherit; z-index: 0; }
.panel-left > *{ position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)

def _now_label() -> str:
    try:
        now = datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.now()
        return now.strftime("%b %d, %Y %H:%M %Z")
    except Exception:
        return datetime.now().strftime("%b %d, %Y %H:%M")

def _note_err(msg: str):
    st.session_state.setdefault("hf_errors", []).append(msg)

# =========================
# CONFIG: local data + mapper
# =========================
COUNTRY_DATA_ROOT = Path("country_data")  # ← at the same level as this file

@st.cache_data(ttl=24*3600, show_spinner=False)
def _read_indicator_mapper() -> Optional[pd.DataFrame]:
    for p in (Path("/mnt/data/indicator_code_mapper.csv"),
              Path("indicator_code_mapper.csv"),
              Path("data/indicator_code_mapper.csv")):
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception as e:
                _note_err(f"Mapper read failed for {p}: {e}")
    _note_err("indicator_code_mapper.csv not found locally.")
    return None

def _suffixes_for_freq() -> Dict[str, list]:
    # Accept standard + legacy suffixes
    return {
        "Annual":   ["_A"],
        "Seasonal": ["_AS", "_PS"],
        "Monthly":  ["_AM", "_PM"],
    }

def _any_column_for_type(cols: List[str], codes_for_type: Set[str], suffixes: List[str]) -> bool:
    cset = {str(c).lower() for c in cols}
    codes_low = {str(c).lower() for c in codes_for_type}
    for sfx in suffixes:
        sfx_low = sfx.lower()
        targets = {f"{c}{sfx_low}" for c in codes_low}
        if cset.intersection(targets):
            return True
    return False

def _country_dirs(root: Path) -> List[Path]:
    if not root.exists(): return []
    return [p for p in root.iterdir() if p.is_dir() and len(p.name) == 3]

def _adm0_file(iso3: str, freq: str, root: Path) -> Path:
    # ADM0 only (per your instruction for availability & snapshot)
    return root / iso3 / freq / f"{iso3}_ADM0_data.parquet"

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
    return (t or "").strip().title()  # fallback: title-case

@st.cache_data(ttl=12*3600, show_spinner=False)
def _scan_local_availability(country_data_root: Path, mapper: Optional[pd.DataFrame]):
    """
    Returns:
      - type_to_isos: dict[type] -> set(ISO3) where ADM0 (any freq) has any column mapped to that Type
      - available_types: list of unique Types from mapper (sorted)
    """
    type_to_isos: Dict[str, Set[str]] = {}
    available_types: List[str] = []

    # Representative codes (for signal detection / fallback)
    rep_codes = {
        "Temperature": {"TMPA"},
        "Humidity": {"HUMA"},
        "Precipitation": {"PCPA"},
        "Wind speeds": {"WSPA"},
    }

    # Build codes_by_type from the mapper (case-insensitive clean)
    codes_by_type: Dict[str, Set[str]] = {}
    if mapper is not None and not mapper.empty and {"Code","Type"} <= set(mapper.columns):
        m = mapper.copy()
        m["Code"] = m["Code"].astype(str).str.strip()
        m["Type"] = m["Type"].astype(str).map(_canonical_type)
        available_types = sorted(m["Type"].dropna().unique().tolist())

        codes_by_type = {}
        for t, g in m.groupby("Type"):
            codes = set(g["Code"].dropna().astype(str).str.strip().tolist())
            # Always merge rep codes for that canonical Type
            codes |= rep_codes.get(t, set())
            if codes:
                codes_by_type[t] = codes
    else:
        # No/invalid mapper → fall back to rep codes so availability still works
        codes_by_type = rep_codes.copy()
        available_types = sorted(codes_by_type.keys())

    suff = _suffixes_for_freq()
    if not country_data_root.exists():
        return type_to_isos, available_types

    for cdir in [p for p in country_data_root.iterdir() if p.is_dir() and len(p.name) == 3]:
        iso3 = cdir.name.upper()
        for typ, codes in codes_by_type.items():
            found_for_type = False
            for freq, sfx_list in suff.items():
                f = country_data_root / iso3 / freq / f"{iso3}_ADM0_data.parquet"
                if not f.exists():
                    continue
                try:
                    cols = list(pd.read_parquet(f, columns=None).columns)
                except Exception as e:
                    _note_err(f"Failed reading {f}: {e}")
                    continue
                if _any_column_for_type(cols, codes, sfx_list):
                    found_for_type = True
                    break
            if found_for_type:
                type_to_isos.setdefault(typ, set()).add(iso3)

    return type_to_isos, available_types


INDICATOR_MAPPER = _read_indicator_mapper()
type_to_isos, AVAILABLE_INDICATORS = _scan_local_availability(COUNTRY_DATA_ROOT, INDICATOR_MAPPER)
# Ensure canonical, deduped Types (e.g., only "Wind speeds")
AVAILABLE_INDICATORS = sorted({_canonical_type(t) for t in AVAILABLE_INDICATORS})

# Keep UI usable if mapper is missing
if not AVAILABLE_INDICATORS:
    AVAILABLE_INDICATORS = ["Temperature", "Precipitation"]

# Routing table (existing + NEW pages)
INDICATOR_TO_PAGE = {
    "Temperature":    ("pages/1_Temperature_Dashboard.py", "1_Temperature_Dashboard"),
    "Precipitation":  ("pages/2_Precipitation_Dashboard.py", "2_Precipitation_Dashboard"),
    "Humidity":       ("pages/3_Humidity_Dashboard.py", "3_Humidity_Dashboard"),
    "Wind speeds":    ("pages/4_Windspeeds_Dashboard.py", "4_Windspeeds_Dashboard"),
}

iso_with_data = set().union(*type_to_isos.values()) if type_to_isos else set()
for t, s in type_to_isos.items():
    st.session_state[f"iso_{t.lower().replace(' ','_')}"] = s
st.session_state["iso_with_data"] = iso_with_data

# ---------- Title / subtitle ----------
st.markdown("<h1 style='text-align:center'>Global Database of Subnational Climate Indicators</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Built and Maintained by Roshen Fernando and Patrick Jaime Simba</div>", unsafe_allow_html=True)
st.divider()

@st.cache_data(ttl=24*3600, show_spinner=False)
def _build_global_series_for_code(code: str) -> Optional[pd.DataFrame]:
    rows = []
    for cdir in _country_dirs(COUNTRY_DATA_ROOT):
        iso3 = cdir.name.upper()
        d = _read_monthly_adm0_series(iso3, code)
        if d is not None and not d.empty:
            rows.append(d)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    # Rough sanity windows for plotting
    if code == "TMPA":
        out = out[(out["value"] > -80) & (out["value"] < 60)]
    if code == "PCPA":
        out = out[(out["value"] >= 0) & (out["value"] < 2000)]
    return out

REP_CODES = {
    "Temperature": "TMPA",
    "Precipitation": "PCPA",
    "Humidity": "HUMA",
    "Wind speeds": "WSPA",
}
# --- build the four snapshot series FIRST (place this ABOVE the hero) ---
g_temp = _build_global_series_for_code(REP_CODES["Temperature"])
g_prec = _build_global_series_for_code(REP_CODES["Precipitation"])
g_hum  = _build_global_series_for_code(REP_CODES["Humidity"])
g_wspd = _build_global_series_for_code(REP_CODES["Wind speeds"])

latest_candidates = []
for df in (g_temp, g_prec, g_hum, g_wspd):
    if df is not None and not df.empty:
        latest_candidates.append(pd.to_datetime(df["Date"]).max())

# --- Data through (works for local or HF) ---
def _latest_date_from_adm0(iso: str, freq: str, code_candidates=("TMPA","PCPA","HUMA","WSPA")) -> pd.Timestamp | None:
    p = _adm0_fp(iso, freq)
    try:
        df = read_parquet_smart(p)  # <-- backend aware
    except Exception:
        return None
    # find any available main-indicator column for this freq
    def _suffixes(level: str, freq: str):
        return (["_AM","_PM"] if freq=="Monthly" else ["_AS","_PS"] if freq=="Seasonal" else ["_A"]) if level=="ADM0" else (["_M"] if freq=="Monthly" else ["_S"] if freq=="Seasonal" else ["_A"])
    def _pick_col(cols, code, sfxs):
        low = {c.lower(): c for c in cols}
        for s in sfxs:
            k=f"{code}{s}".lower()
            if k in low: return low[k]
        return None
    col=None
    for code in code_candidates:
        col = _pick_col(list(df.columns), code, _suffixes("ADM0", freq))
        if col: break
    if not col: return None
    df2 = df.copy()
    df2["Date"] = _to_date_from_parts(df2, freq)
    df2 = df2.dropna(subset=["Date"])
    return None if df2.empty else df2["Date"].max()

def compute_data_through() -> str:
    # scan all ISO folders under country_data (or HF)
    # try Monthly first, then Seasonal, then Annual
    latest_candidates=[]
    # list ISOs (local or HF)
    isos = []
    if DATA_BACKEND in ("hf","auto") and list_repo_files is not None:
        try:
            files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
            for f in files:
                parts=f.split("/")
                if len(parts)>=4 and parts[0]=="country_data" and len(parts[1])==3:
                    isos.append(parts[1].upper())
            isos = sorted(set(isos))
        except Exception:
            pass
    if not isos:
        if COUNTRY_DATA_DIR.exists():
            isos = sorted([p.name.upper() for p in COUNTRY_DATA_DIR.iterdir() if p.is_dir() and len(p.name)==3])

    for iso in isos:
        for freq in ("Monthly","Seasonal","Annual"):
            dt = _latest_date_from_adm0(iso, freq)
            if dt is not None:
                latest_candidates.append(dt)

    if not latest_candidates:
        return "—"
    mx = max(latest_candidates)
    # render as YYYY-MM for Monthly/Seasonal anchor dates; YYYY if it's a 07-01 annual anchor
    return mx.strftime("%Y") if (mx.month==7 and mx.day==1) else mx.strftime("%Y-%m")

data_through = compute_data_through()

def _fmt_badge_dt(x):
    ts = pd.to_datetime(x, errors="coerce")
    return ts.strftime("%b %Y") if (ts is not pd.NaT and pd.notna(ts)) else "—"

badge_dt = data_through or "—"



# ===== HERO (unchanged) =====
left, right = st.columns([0.62, 0.38], gap="large")
with left:
    st.markdown(
        f"""
        <div class="panel-left">
          <h2>🌍 Explore subnational climate indicators worldwide</h2>
          <p>Click a country on the map to open its dashboard, or use Quick search to jump directly.</p>
          <div class="badgerow">
            <span class="badge">Data through: <b>{badge_dt}</b></span>
            <span class="badge" style="margin-left:6px;">Last Update: <b>{_now_label()}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with right:
    st.subheader("Custom Chart Builder")
    st.write("Create bespoke charts across countries and ADM1s, compare indicators, combine countries/ADM1s, facet, smooth, normalize and export.")
    if st.button("📈 Generate a custom chart", key="hero_custom_chart"):
        try: st.switch_page("pages/0_Custom_Chart.py")
        except Exception: st.rerun()
    st.caption("Starts a flexible chart workspace with export options.")

# ---------- First load tip ----------
if "first_load_hint" not in st.session_state:
    st.info("Tip: first load warms the cache; subsequent loads should be faster.", icon="💡")
    st.session_state["first_load_hint"] = True

# ---------- Controls (unchanged UI) ----------
if "region_scope" not in st.session_state:
    st.session_state["region_scope"] = "World"
if "default_indicator" not in st.session_state:
    st.session_state["default_indicator"] = "Temperature" if "Temperature" in AVAILABLE_INDICATORS else AVAILABLE_INDICATORS[0]

def _log_event(evt: str, payload: dict):
    st.session_state.setdefault("analytics", [])
    ts = datetime.now(LOCAL_TZ).isoformat() if LOCAL_TZ else datetime.now().isoformat()
    st.session_state["analytics"].append({"ts": ts, "event": evt, **payload})

def _reset_scope():
    st.session_state["region_scope"] = "World"
    _log_event("reset_view", {"to": "World"})

def _navigate_to_dashboard_immediate(iso3: str, indicator_type: str):
    ind = indicator_type or "Temperature"
    page_path, page_qp = INDICATOR_TO_PAGE.get(ind, INDICATOR_TO_PAGE.get("Temperature"))

    type_isos = st.session_state.get(f"iso_{ind.lower().replace(' ','_')}", set())
    if iso3 not in type_isos:
        if iso3 in st.session_state.get("iso_temperature", set()):
            ind = "Temperature"; page_path, page_qp = INDICATOR_TO_PAGE["Temperature"]
        elif iso3 in st.session_state.get("iso_precipitation", set()):
            ind = "Precipitation"; page_path, page_qp = INDICATOR_TO_PAGE["Precipitation"]
        elif iso3 in st.session_state.get("iso_humidity", set()):
            ind = "Humidity"; page_path, page_qp = INDICATOR_TO_PAGE["Humidity"]
        elif iso3 in st.session_state.get("iso_wind_speeds", set()):
            ind = "Wind speeds"; page_path, page_qp = INDICATOR_TO_PAGE["Wind speeds"]
        else:
            for tname, isos in type_to_isos.items():
                if iso3 in isos:
                    ind = tname
                    page_path, page_qp = INDICATOR_TO_PAGE.get(ind, INDICATOR_TO_PAGE.get("Temperature"))
                    break

    st.session_state["nav_iso3"] = iso3
    st.query_params.update({"page": page_qp, "iso3": iso3, "city": ""})
    try: st.switch_page(page_path)
    except Exception: st.rerun()

def _perform_nav_if_pending():
    nav = st.session_state.get("_pending_nav")
    if not nav: return
    iso3 = nav.get("iso3"); indicator = nav.get("indicator") or "Temperature"
    st.session_state["_pending_nav"] = None
    _navigate_to_dashboard_immediate(iso3, indicator)
_perform_nav_if_pending()

# ---------- Countries master ----------
if pycountry:
    all_countries = pd.DataFrame([{"iso3": c.alpha_3, "name": c.name} for c in pycountry.countries if hasattr(c, "alpha_3")])
else:
    all_countries = pd.DataFrame({"iso3": sorted(list(iso_with_data))}); all_countries["name"] = all_countries["iso3"]

all_countries["iso3"] = all_countries["iso3"].astype(str).str.upper().str.strip()
_name_overrides = {"CHN": "People's Republic of China", "TWN": "Taipe, China", "HKG": "Hong Kong, China"}
all_countries["name"] = all_countries.apply(lambda r: _name_overrides.get(r["iso3"], r.get("name", r["iso3"])), axis=1)

for t in AVAILABLE_INDICATORS:
    key = f"has_{t.lower().replace(' ','_')}"
    all_countries[key] = all_countries["iso3"].isin(type_to_isos.get(t, set()))
all_countries["has_data_any"] = False
for t in AVAILABLE_INDICATORS:
    all_countries["has_data_any"] |= all_countries[f"has_{t.lower().replace(' ','_')}"]

def _badges(iso):
    hits = [t for t in AVAILABLE_INDICATORS if iso in type_to_isos.get(t, set())]
    return " • ".join(hits) if hits else "—"
all_countries["badges"] = all_countries["iso3"].map(_badges)

# --- Continent helper (unchanged) ---
@st.cache_data(show_spinner=False)
def _continent_lookup() -> dict:
    gm = px.data.gapminder()
    base = dict(zip(gm["iso_alpha"], gm["continent"]))
    south_america = {"ARG","BOL","BRA","CHL","COL","ECU","GUY","PRY","PER","SUR","URY","VEN","FLK","GUF"}
    north_america = {"USA","CAN","MEX","GTM","BLZ","HND","SLV","NIC","CRI","PAN","CUB","DOM","HTI","JAM","TTO","BRB","BHS",
                     "ATG","DMA","GRD","KNA","LCA","VCT","ABW","BES","BMU","CUW","GLP","GRL","MTQ","MSR","PRI","SXM","SJM",
                     "TCA","VGB","VIR"}
    out = {}
    for iso, cont in base.items():
        out[iso] = "South America" if cont == "Americas" and iso in south_america else \
                   "North America" if cont == "Americas" else cont
    return out

CONTINENT_OF = _continent_lookup()

def _isos_for_region(region_name: str, all_isos: pd.Series) -> set:
    if region_name == "World": return set(all_isos.tolist())
    res = {iso for iso in all_isos if CONTINENT_OF.get(iso, None) == region_name}
    return res or set(all_isos.tolist())

# ---------- Controls row (unchanged) ----------
quick_opts = ["— Type to search —"] + sorted(all_countries["name"].tolist())
c1, c2, c3, c4 = st.columns([1, 0.32, 0.25, 0.15])
with c1:
    chosen = st.selectbox("Quick search", quick_opts, index=0,
                          help="Type a country name, or click a country on the map to navigate to its dashboard.")
with c2:
    st.selectbox("Default Indicator",
                 AVAILABLE_INDICATORS if AVAILABLE_INDICATORS else ["Temperature","Precipitation"],
                 key="default_indicator",
                 help="Which dashboard opens when you click a country or use Quick search. Does not change the map.")
with c3:
    st.selectbox("View",
                 ["World","Africa","Asia","Europe","North America","South America","Oceania"],
                 key="region_scope",
                 index=["World","Africa","Asia","Europe","North America","South America","Oceania"].index(st.session_state["region_scope"]),
                 help="Change geographic scope.")
with c4:
    st.markdown('<div class="align-with-input"></div>', unsafe_allow_html=True)
    st.button("Reset view", use_container_width=True, on_click=_reset_scope)

# Legend
st.markdown("""
<div class="legend-chip" style="margin: 6px 0 4px 0;">
  <span class="legend-swatch"></span>
  Countries with available indicators
</div>
""", unsafe_allow_html=True)

# Quick search nav
if chosen and chosen != "— Type to search —":
    row = all_countries.loc[all_countries["name"] == chosen].iloc[0]
    iso3_jump = row["iso3"]
    if iso3_jump in iso_with_data:
        _log_event("quick_search_open", {"iso3": iso3_jump, "indicator": st.session_state.get("default_indicator", "Temperature")})
        st.session_state["_pending_nav"] = {"iso3": iso3_jump, "indicator": st.session_state.get("default_indicator","Temperature")}
        st.rerun()
    else:
        st.info(f"{chosen}: No available indicators.", icon="ℹ️")

# ---------- MAP (unchanged UI) ----------
vp = components.html("""
<script>
(function(){function send(){const p={width:window.innerWidth,height:window.innerHeight};
window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:p},'*');}
window.addEventListener('resize',send);send();})();
</script>
""", height=0)
vw = int(vp["width"]) if isinstance(vp, dict) and "width" in vp else 1280
height_ratio = {"World":0.50,"Africa":0.82,"Asia":0.80,"Europe":0.90,"North America":0.82,"South America":0.94,"Oceania":0.86}
map_h = max(600, int(vw * height_ratio.get(st.session_state["region_scope"], 0.70)))

region_isos = _isos_for_region(st.session_state["region_scope"], all_countries["iso3"])
plot_df = all_countries[all_countries["iso3"].isin(region_isos)].copy()
plot_df["hovertext"] = plot_df.apply(lambda r: f"{r['name']}<br><span>Indicators: {r['badges']}</span>", axis=1)
plot_df["val"] = plot_df["has_data_any"].astype(float)

fig = go.Figure(go.Choropleth(
    locations=plot_df["iso3"], z=plot_df["val"], locationmode="ISO-3",
    colorscale=[[0.0, "#d4d4d8"], [1.0, "#12a39a"]],
    zmin=0.0, zmax=1.0, autocolorscale=False, showscale=False,
    hoverinfo="text", text=plot_df["hovertext"],
    customdata=plot_df[["iso3"]].to_numpy(),
    marker_line_width=1.6, marker_line_color="rgba(0,0,0,0.70)",
))
scope_map = {"World":"world","Africa":"africa","Asia":"asia","Europe":"europe","North America":"north america","South America":"south america","Oceania":"world"}
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=map_h,
    geo=dict(projection_type="robinson", showframe=False, showcoastlines=False, showocean=False,
             bgcolor="rgba(0,0,0,0)", scope=scope_map.get(st.session_state["region_scope"], "world"),
             fitbounds="locations" if st.session_state["region_scope"]!="World" else None,
             lataxis_range=[-60,85] if st.session_state["region_scope"]=="World" else None))

st.markdown('<div class="full-bleed">', unsafe_allow_html=True)
if plotly_events:
    events = plotly_events(fig, click_event=True, hover_event=False, override_height=map_h, override_width="100%")
else:
    st.plotly_chart(fig, use_container_width=True)
    events = []
st.markdown('</div>', unsafe_allow_html=True)

clicked_iso3 = None
if events:
    e = events[0]
    if isinstance(e, dict):
        idx = e.get("pointIndex", None)
        if idx is not None and 0 <= idx < len(plot_df):
            clicked_iso3 = str(plot_df.iloc[idx]["iso3"]).upper()
if clicked_iso3 and clicked_iso3 in iso_with_data:
    _log_event("map_click_open", {"iso3": clicked_iso3, "indicator": st.session_state.get("default_indicator", "Temperature")})
    st.session_state["_pending_nav"] = {"iso3": clicked_iso3, "indicator": st.session_state.get("default_indicator","Temperature")}
    st.rerun()

if st.query_params.get("debug", ["0"])[0] == "1":
    st.info("Debug: availability scan", icon="🔎")
    test_iso = "AFG"
    rows = []
    for freq, sfx_list in _suffixes_for_freq().items():
        f = COUNTRY_DATA_ROOT / test_iso / freq / f"{test_iso}_ADM0_data.parquet"
        exists = f.exists()
        matched_types = []
        if exists and INDICATOR_MAPPER is not None and not INDICATOR_MAPPER.empty:
            # Build codes_by_type like the scanner
            cb = {}
            m = INDICATOR_MAPPER.copy()
            m["Code"] = m["Code"].astype(str).str.strip()
            m["Type"] = m["Type"].astype(str).str.strip()
            for t, g in m.groupby("Type"):
                codes = set(g["Code"].dropna().astype(str).str.strip().tolist())
                if t in {"Temperature","Humidity","Precipitation","Wind","Wind speed","Wind speeds"}:
                    codes |= {"TMPA","HUMA","PCPA","WSPA"} if t!="Temperature" else {"TMPA"}
                if codes: cb[t] = codes
            try:
                cols = list(pd.read_parquet(f, columns=None).columns) if exists else []
                for t, codes in cb.items():
                    if _any_column_for_type(cols, codes, sfx_list):
                        matched_types.append(t)
            except Exception as e:
                matched_types = [f"err: {e}"]
        rows.append({"freq": freq, "exists": exists, "matched_types": ", ".join(sorted(set(matched_types)))})
    st.dataframe(pd.DataFrame(rows))


# =========================
# GLOBAL SNAPSHOT (beta) — ADM0 Monthly only, fixed codes
# =========================
st.divider()
st.subheader("Global snapshot (beta)")

# Representative codes per Type (fixed as requested)
REP_CODES = {
    "Temperature": "TMPA",
    "Precipitation": "PCPA",
    "Humidity": "HUMA",
    "Wind speeds": "WSPA",
}

def _monthly_adm0_path(iso3: str) -> Path:
    return COUNTRY_DATA_ROOT / iso3 / "Monthly" / f"{iso3}_ADM0_data.parquet"

def _read_monthly_adm0_series(iso3: str, code: str) -> Optional[pd.DataFrame]:
    """Read iso3 ADM0 Monthly series for a given code (prefer _AM, fall back to _PM)."""
    f = _monthly_adm0_path(iso3)
    if not f.exists(): return None
    try:
        cols = list(pd.read_parquet(f, columns=None).columns)
    except Exception as e:
        _note_err(f"Schema read failed for {f}: {e}")
        return None
    target = None
    for sfx in ("_AM","_PM"):
        cand = f"{code}{sfx}"
        if cand in cols:
            target = cand; break
    if not target: return None
    try:
        use_cols = [c for c in ("Year","Month", target) if c in cols]
        df = pd.read_parquet(f, columns=use_cols).rename(columns={target:"value"})
    except Exception as e:
        _note_err(f"Data read failed for {f}: {e}")
        return None
    if "Year" not in df.columns or "Month" not in df.columns: return None
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year","Month","value"])
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Year"].astype(int).astype(str) + "-" + df["Month"].astype(int).astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["iso3"] = iso3
    return df[["iso3","Date","value"]].sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=24*3600, show_spinner=False)
def _build_global_series_for_code(code: str) -> Optional[pd.DataFrame]:
    rows = []
    for cdir in _country_dirs(COUNTRY_DATA_ROOT):
        iso3 = cdir.name.upper()
        d = _read_monthly_adm0_series(iso3, code)
        if d is not None and not d.empty:
            rows.append(d)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    # Rough sanity windows for plotting
    if code == "TMPA":
        out = out[(out["value"] > -80) & (out["value"] < 60)]
    if code == "PCPA":
        out = out[(out["value"] >= 0) & (out["value"] < 2000)]
    return out

# Build all four series for snapshot
g_temp = _build_global_series_for_code(REP_CODES["Temperature"])
g_prec = _build_global_series_for_code(REP_CODES["Precipitation"])
g_hum  = _build_global_series_for_code(REP_CODES["Humidity"])
g_wspd = _build_global_series_for_code(REP_CODES["Wind speeds"])

def _coverage_over_time(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    d = df[["iso3","Date"]].copy()
    d["Date"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    cov = d.groupby("Date")["iso3"].nunique().reset_index(name="countries")
    return cov.sort_values("Date")

cov_t = _coverage_over_time(g_temp)
cov_p = _coverage_over_time(g_prec)
cov_h = _coverage_over_time(g_hum)
cov_w = _coverage_over_time(g_wspd)

if any(d is not None for d in (cov_t, cov_p, cov_h, cov_w)):
    cov_fig = go.Figure()
    if cov_t is not None: cov_fig.add_trace(go.Scatter(x=cov_t["Date"], y=cov_t["countries"], mode="lines", name="Temperature"))
    if cov_p is not None: cov_fig.add_trace(go.Scatter(x=cov_p["Date"], y=cov_p["countries"], mode="lines", name="Precipitation"))
    if cov_h is not None: cov_fig.add_trace(go.Scatter(x=cov_h["Date"], y=cov_h["countries"], mode="lines", name="Humidity"))
    if cov_w is not None: cov_fig.add_trace(go.Scatter(x=cov_w["Date"], y=cov_w["countries"], mode="lines", name="Wind speeds"))
    cov_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Month",
        yaxis_title="Countries with data",
    )
    st.plotly_chart(cov_fig, use_container_width=True)
else:
    st.info("Global coverage over time (ADM0 Monthly) is unavailable yet.", icon="ℹ️")

# Update "Data through" (latest of all four)
latest_candidates = []
for df in (g_temp, g_prec, g_hum, g_wspd):
    if df is not None and not df.empty and "Date" in df.columns:
        latest_candidates.append(pd.to_datetime(df["Date"], errors="coerce").max())

# drop NaT entries before taking max
latest_candidates = [d for d in latest_candidates if pd.notna(d)]
data_through = max(latest_candidates) if latest_candidates else None


# Latest datapoint histograms (two rows)
def _latest_by_country(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    last = df.sort_values(["iso3","Date"]).drop_duplicates(subset=["iso3"], keep="last")
    return last["value"]

# Row 1 — Temperature & Precipitation
hcol1, hcol2 = st.columns(2)
with hcol1:
    s = _latest_by_country(g_temp)
    if s is not None and not s.empty:
        hist_t = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30, title="Latest datapoint per country — Temperature (TMPA)")
        hist_t.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260, xaxis_title="Temperature (°C)", yaxis_title="Countries")
        st.plotly_chart(hist_t, use_container_width=True)
    else:
        st.info("Latest temperature distribution unavailable (ADM0 Monthly TMPA).", icon="ℹ️")

with hcol2:
    s = _latest_by_country(g_prec)
    if s is not None and not s.empty:
        hist_p = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30, title="Latest datapoint per country — Precipitation (PCPA)")
        hist_p.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260, xaxis_title="Precipitation (mm)", yaxis_title="Countries")
        st.plotly_chart(hist_p, use_container_width=True)
    else:
        st.info("Latest precipitation distribution unavailable (ADM0 Monthly PCPA).", icon="ℹ️")

# Row 2 — Humidity & Wind speeds
h2c1, h2c2 = st.columns(2)
with h2c1:
    s = _latest_by_country(g_hum)
    if s is not None and not s.empty:
        hist_h = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30,
                              title="Latest datapoint per country — Humidity (HUMA)")
        hist_h.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260,
                             xaxis_title="Relative humidity (%)", yaxis_title="Countries")
        st.plotly_chart(hist_h, use_container_width=True)
    else:
        st.info("Latest humidity distribution unavailable (ADM0 Monthly HUMA).", icon="ℹ️")

with h2c2:
    s = _latest_by_country(g_wspd)
    if s is not None and not s.empty:
        hist_w = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30,
                              title="Latest datapoint per country — Wind speeds (WSPA)")
        hist_w.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260,
                             xaxis_title="Wind speed (m/s)", yaxis_title="Countries")
        st.plotly_chart(hist_w, use_container_width=True)
    else:
        st.info("Latest wind speeds distribution unavailable (ADM0 Monthly WSPA).", icon="ℹ️")


# ---------- Coverage & Data sources (unchanged UI) ----------
st.divider()
k1, k2 = st.columns([1,1])
with k1:
    st.markdown(f"""
    <div class="card">
      <div style="font-size:13px;color:#64748b;">Coverage</div>
      <div style="font-size:22px;margin:.15rem 0;">
        <strong>{int(all_countries['has_data_any'].sum())}</strong> countries with at least one indicator
      </div>
      <div style="font-size:13px;color:#475569;">
        Indicators shown: {(" • ".join(AVAILABLE_INDICATORS)) if AVAILABLE_INDICATORS else "Temperature • Precipitation"}
      </div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

# ---------- Expander / Admin / Footer (unchanged UI) ----------
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
with st.expander("What’s inside this dashboard?", expanded=False):
    st.markdown("""
- **Geography:** Countries and first-level administrative regions (ADM1); selected cities for context.
- **Indicators:** Temperature and precipitation (more types supported by the data; pages added progressively).
- **Temporal frequency:** Monthly; global snapshot uses ADM0 Monthly (TMPA/PCPA/HUMA/WSPA).
- **Latency:** Updates typically published within weeks of source release.
- **Method summary:** Area-weighted aggregation of grid cells to administrative boundaries.
- **Caveats:** Administrative boundary changes, data gaps, and reanalysis corrections can affect comparability over time.
""")

if st.query_params.get("admin", ["0"])[0] == "1":
    st.divider(); st.subheader("Admin: Session analytics")
    a = st.session_state.get("analytics", [])
    st.caption("Lightweight, in-session logs. Export below. (Counts reset per session.)")
    if a:
        df_log = pd.DataFrame(a)
        st.dataframe(df_log, use_container_width=True)
        st.download_button("Download logs (CSV)", data=df_log.to_csv(index=False).encode("utf-8"),
                           file_name=f"home_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        st.markdown(f"- **View changes:** {(df_log['event']=='view_change').sum()}")
        st.markdown(f"- **Quick-search opens:** {(df_log['event']=='quick_search_open').sum()}")
        st.markdown(f"- **Map click opens:** {(df_log['event']=='map_click_open').sum()}")
        st.markdown(f"- **Resets:** {(df_log['event']=='reset_view').sum()}")
    else:
        st.info("No events logged yet in this session.", icon="ℹ️")

st.markdown("""
<div class="footer-box">
  <em>Note:</em> This page time-stamps the "Last Update" at render time (Asia/Manila).
</div>
""", unsafe_allow_html=True)
