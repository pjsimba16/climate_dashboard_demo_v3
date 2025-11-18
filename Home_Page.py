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

# === Hugging Face hub helpers (added) ===
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None
    list_repo_files = None

HF_REPO_ID   = "pjsimba16/adb-climate-data"  # <— your Space
HF_REPO_PREF = ("space", "dataset")          # try Space first; then dataset fallback

# ==== DEBUG PANEL (toggle by ?debug=1) ====
def _note_err(msg: str):
    st.session_state.setdefault("hf_errors", []).append(str(msg))

debug_on = str(st.query_params.get("debug", ["0"])[0]).lower() in {"1","true","yes"}

if debug_on:
    with st.expander("🔎 Hugging Face data debug", expanded=True):
        # 0) Show repo settings
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
            st.write("✅ huggingface_hub is installed")
        except Exception as e:
            st.error(f"❌ huggingface_hub not available: {e}")

        HF_REPO_ID = "pjsimba16/adb-climate-data"  # make sure this matches your Space
        HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
        st.write({"HF_REPO_ID": HF_REPO_ID, "has_token": bool(HF_TOKEN)})

        # 1) List some files to confirm the repo is reachable
        files = []
        try:
            files = list_repo_files(repo_id=HF_REPO_ID, repo_type="space", token=HF_TOKEN)
        except Exception as e:
            _note_err(f"list_repo_files(space) failed: {e}")
            try:
                files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.info("Used repo_type=dataset fallback.")
            except Exception as e2:
                _note_err(f"list_repo_files(dataset) failed too: {e2}")
        st.write(f"File count: {len(files)}")
        # Show a few country_data paths if any
        sample_cd = [f for f in files if f.startswith("country_data/")][:10]
        st.write("Sample country_data paths:", sample_cd)

        # 2) Pick an ISO to probe (PHL/USA/AFG exist in your data). Adjust if needed.
        probe_iso = st.text_input("Probe ISO3", value="PHL").upper().strip()
        probe_freq = st.selectbox("Probe frequency", ["Monthly", "Seasonal", "Annual"], index=0)

        # 3) Try to download one ADM0 file and read it
        folder = {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[probe_freq]
        relpath = f"country_data/{probe_iso}/{folder}/{probe_iso}_ADM0_data.parquet"
        st.write("Trying to read:", relpath)

        df_adm0 = None
        try:
            #local_fp = hf_hub_download(repo_id=HF_REPO_ID, repo_type="space", filename=relpath, token=HF_TOKEN)
            tok = _get_hf_token()
            local_fp = _hf_call_download(HF_REPO_ID, repo_type, relpath, tok)

        except Exception as e:
            _note_err(f"hf_hub_download(space) failed for {relpath}: {e}")
            try:
                #local_fp = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=relpath, token=HF_TOKEN)
                tok = _get_hf_token()
                local_fp = _hf_call_download(HF_REPO_ID, repo_type, relpath, tok)

                st.info("Used repo_type=dataset fallback.")
            except Exception as e2:
                local_fp = None
                _note_err(f"hf_hub_download(dataset) failed for {relpath}: {e2}")

        if local_fp:
            try:
                df_adm0 = pd.read_parquet(local_fp)
                st.success(f"Read OK: {relpath}")
                st.write("Columns (first 40):", list(df_adm0.columns)[:40])
                st.write("Rows:", len(df_adm0))
            except Exception as e:
                _note_err(f"pd.read_parquet failed for {relpath}: {e}")
                st.error(f"Could not read parquet: {e}")
        else:
            st.error("Could not download file from HF.")

        # 4) Verify column suffixes exist for expected codes
        if df_adm0 is not None:
            def _has_any(base, freq, cols):
                cols_low = {c.lower() for c in cols}
                suf = {"Monthly":["_AM","_PM","_M"], "Seasonal":["_AS","_PS","_S"], "Annual":["_A"]}[freq]
                for s in suf:
                    if (base + s).lower() in cols_low:
                        return True, base + s
                return False, None

            checks = {}
            for code in ["TMPA","PCPA","HUMA","WSPA"]:
                ok, found = _has_any(code, probe_freq, df_adm0.columns)
                checks[code] = (ok, found)
            st.write("Suffix checks (any column present?):", checks)

        # 5) Show recent error log
        errs = st.session_state.get("hf_errors", [])
        if errs:
            st.warning("Recent HF errors:")
            for e in errs[-10:]:
                st.code(e)
        else:
            st.info("No HF errors recorded yet.")

def _get_hf_token():
    try:
        if hasattr(st, "secrets"):
            tok = st.secrets.get("HF_TOKEN")
            if tok is not None:
                tok = str(tok).strip()
                return tok or None   # <- return None if empty
    except Exception:
        pass
    env = os.getenv("HF_TOKEN", None)
    if env is None:
        return None
    env = str(env).strip()
    return env or None               # <- return None if empty

def _hf_call_download(repo_id: str, repo_type: str, filename: str, token: str | None):
    kwargs = dict(repo_id=repo_id, repo_type=repo_type, filename=filename)
    if token:  # only include token if non-empty
        kwargs["token"] = token
    return hf_hub_download(**kwargs)

def _hf_call_list(repo_id: str, repo_type: str, token: str | None):
    kwargs = dict(repo_id=repo_id, repo_type=repo_type)
    if token:
        kwargs["token"] = token
    return list_repo_files(**kwargs)


def _hf_download(relpath: str) -> Optional[str]:
    """
    Download a file from HF Space/Dataset to local cache and return the local path.
    relpath like 'country_data/PHL/Monthly/PHL_ADM0_data.parquet'
    """
    if hf_hub_download is None:
        return None
    last_err = None
    tok = _get_hf_token()
    for repo_type in HF_REPO_PREF:
        try:
            fp = hf_hub_download(repo_id=HF_REPO_ID, repo_type=repo_type,
                                 filename=relpath, token=tok)
            
            return fp
        except Exception as e:
            last_err = e
            continue
    st.session_state.setdefault("hf_errors", []).append(f"hf_download failed for {relpath}: {last_err}")
    return None

def _hf_list_country_isos() -> List[str]:
    """
    List unique ISO3 folders under country_data/ from the HF repo.
    """
    if list_repo_files is None:
        return []
    try:
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_PREF[0], token=_get_hf_token())

    except Exception:
        # fallback repo_type
        try:
            files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_PREF[1], token=_get_hf_token())
        except Exception:
            return []
    isos = []
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "country_data" and len(parts[1]) == 3:
            isos.append(parts[1].upper())
    return sorted(set(isos))

def _hf_read_parquet(relpath: str, columns=None) -> Optional[pd.DataFrame]:
    """
    Read a parquet by repository-relative path from HF.
    """
    local = _hf_download(relpath)
    if not local:
        return None
    try:
        return pd.read_parquet(local, columns=columns)
    except Exception as e:
        st.session_state.setdefault("hf_errors", []).append(f"read_parquet({relpath}) failed: {e}")
        return None

# === Page config/UI (unchanged) ===
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
COUNTRY_DATA_ROOT = Path("country_data")  # expected local root (kept)

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
    # HF fallback for indicator_code_mapper.csv
    for rp in ("indicator_code_mapper.csv", "country_data/indicator_code_mapper.csv"):
        df = _hf_read_parquet(rp)  # will fail because CSV; try below
        if df is not None:
            return df
        # try CSV via hub
        fp = _hf_download(rp)
        if fp:
            try:
                return pd.read_csv(fp)
            except Exception as e:
                _note_err(f"Mapper hub CSV read failed for {rp}: {e}")
    _note_err("indicator_code_mapper.csv not found (local/HF).")
    return None

def _suffixes_for_freq() -> Dict[str, list]:
    # Accept standard + legacy suffixes
    return {
        "Annual":   ["_A"],
        "Seasonal": ["_AS", "_PS", "_S"],
        "Monthly":  ["_AM", "_PM", "_M"],
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
    """
    Keep local behavior; only used when local files exist.
    For HF-only deployments, we'll enumerate ISOs via _list_available_isos().
    """
    if not root.exists(): return []
    return [p for p in root.iterdir() if p.is_dir() and len(p.name) == 3]

def _adm0_file_local(iso3: str, freq: str) -> Path:
    return COUNTRY_DATA_ROOT / iso3 / freq / f"{iso3}_ADM0_data.parquet"

def _adm0_relpath(iso3: str, freq: str) -> str:
    return f"country_data/{iso3}/{freq}/{iso3}_ADM0_data.parquet"

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
def _list_available_isos() -> List[str]:
    """
    Prefer local folder list; fallback to HF repo listing.
    """
    if COUNTRY_DATA_ROOT.exists():
        out = []
        for d in sorted(COUNTRY_DATA_ROOT.iterdir()):
            try:
                if d.is_dir() and len(d.name) == 3:
                    out.append(d.name.upper())
            except Exception:
                continue
        if out:
            return out
    # HF fallback
    return _hf_list_country_isos()

@st.cache_data(ttl=12*3600, show_spinner=False)
def _scan_local_availability(country_data_root: Path, mapper: Optional[pd.DataFrame]):
    """
    Returns:
      - type_to_isos: dict[type] -> set(ISO3) where ADM0 (any freq) has any column mapped to that Type
      - available_types: list of unique Types from mapper (sorted)
    Extended to use HF when local files are missing.
    """
    type_to_isos: Dict[str, Set[str]] = {}
    available_types: List[str] = []

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

        for t, g in m.groupby("Type"):
            codes = set(g["Code"].dropna().astype(str).str.strip().tolist())
            codes |= rep_codes.get(t, set())
            if codes:
                codes_by_type[t] = codes
    else:
        # No/invalid mapper → fall back to rep codes so availability still works
        codes_by_type = rep_codes.copy()
        available_types = sorted(codes_by_type.keys())

    suff = _suffixes_for_freq()
    isos = _list_available_isos()
    if not isos:
        return type_to_isos, available_types

    for iso3 in isos:
        for typ, codes in codes_by_type.items():
            found_for_type = False
            for freq, sfx_list in suff.items():
                # try local file first
                lf = _adm0_file_local(iso3, freq)
                cols = None
                if lf.exists():
                    try:
                        cols = list(pd.read_parquet(lf, columns=None).columns)
                    except Exception as e:
                        _note_err(f"Failed reading {lf}: {e}")
                        cols = None
                if cols is None:
                    # try HF
                    rel = _adm0_relpath(iso3, freq)
                    d = _hf_read_parquet(rel, columns=None)
                    cols = list(d.columns) if d is not None else None
                if cols and _any_column_for_type(cols, codes, sfx_list):
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

# === Global series builders (kept; add HF fallback) ===
def _monthly_adm0_path(iso3: str) -> Path:
    return COUNTRY_DATA_ROOT / iso3 / "Monthly" / f"{iso3}_ADM0_data.parquet"

def _read_monthly_adm0_series(iso3: str, code: str) -> Optional[pd.DataFrame]:
    """Read iso3 ADM0 Monthly series for a given code (prefer _AM, fall back to _PM/_M)."""
    f = _monthly_adm0_path(iso3)
    df = None
    if f.exists():
        try:
            cols = list(pd.read_parquet(f, columns=None).columns)
            df_src = ("local", f)
        except Exception as e:
            _note_err(f"Schema read failed for {f}: {e}")
            cols = None
    else:
        cols = None

    if cols is None:
        # try HF
        rel = _adm0_relpath(iso3, "Monthly")
        d = _hf_read_parquet(rel, columns=None)
        if d is None:
            return None
        cols = list(d.columns)
        df = d
        df_src = ("hf", rel)

    # choose column
    target = None
    for sfx in ("_AM","_PM","_M"):
        cand = f"{code}{sfx}"
        if cand in cols:
            target = cand; break
    if not target:
        return None

    if df is None:
        # read local with needed columns
        try:
            use_cols = [c for c in ("Year","Month", target) if c in cols]
            df = pd.read_parquet(f, columns=use_cols).rename(columns={target:"value"})
        except Exception as e:
            _note_err(f"Data read failed for {f}: {e}")
            return None
    else:
        # df already from HF; subset/rename
        use_cols = [c for c in ("Year","Month", target) if c in df.columns]
        df = df[use_cols].rename(columns={target:"value"})

    if "Year" not in df.columns or "Month" not in df.columns:
        return None

    df["Year"]  = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
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
    # iterate isos from local or HF
    isos = _list_available_isos()
    for iso3 in isos:
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

def _fmt_badge_dt(x):
    ts = pd.to_datetime(x, errors="coerce")
    return ts.strftime("%b %Y") if pd.notna(ts) else "—"

data_through = max(latest_candidates) if latest_candidates else None
badge_dt = _fmt_badge_dt(data_through)

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
if debug_on:
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type="space", token=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN","")))
    except Exception:
        files = []
    isos_seen = sorted({p.split("/")[1].upper() for p in files if p.startswith("country_data/") and len(p.split("/")) >= 3})
    st.write("ISOs discovered on HF (country_data/*):", isos_seen[:30], f"...({len(isos_seen)} total)")


isos_avail = _list_available_isos()
if pycountry:
    all_countries = pd.DataFrame([{"iso3": c.alpha_3, "name": c.name} for c in pycountry.countries if hasattr(c, "alpha_3")])
    if isos_avail:
        all_countries = all_countries[all_countries["iso3"].isin(isos_avail)]
else:
    all_countries = pd.DataFrame({"iso3": isos_avail}); all_countries["name"] = all_countries["iso3"]

all_countries["iso3"] = all_countries["iso3"].astype(str).str.upper().str.strip()
_name_overrides = {"CHN": "People's Republic of China", "TWN": "Taipei, China", "HKG": "Hong Kong, China"}
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
        f = _adm0_file_local(test_iso, freq)
        exists = f.exists()
        matched_types = []
        cols = []
        if exists:
            try:
                cols = list(pd.read_parquet(f, columns=None).columns)
            except Exception as e:
                cols = [f"(local read error: {e})"]
        else:
            d = _hf_read_parquet(_adm0_relpath(test_iso, freq), columns=None)
            cols = list(d.columns) if d is not None else []
        if cols and INDICATOR_MAPPER is not None and not INDICATOR_MAPPER.empty:
            cb = {}
            m = INDICATOR_MAPPER.copy()
            m["Code"] = m["Code"].astype(str).str.strip()
            m["Type"] = m["Type"].astype(str).str.strip()
            for t, g in m.groupby("Type"):
                codes = set(g["Code"].dropna().astype(str).str.strip().tolist())
                if t in {"Temperature","Humidity","Precipitation","Wind","Wind speed","Wind speeds"}:
                    codes |= {"TMPA","HUMA","PCPA","WSPA"} if t!="Temperature" else {"TMPA"}
                if codes: cb[t] = codes
            for t, codes in cb.items():
                if _any_column_for_type(cols, codes, sfx_list):
                    matched_types.append(t)
        rows.append({"freq": freq, "local_exists": exists, "matched_types": ", ".join(sorted(set(matched_types)))})
    st.dataframe(pd.DataFrame(rows))


# =========================
# GLOBAL SNAPSHOT (beta) — ADM0 Monthly only, fixed codes
# =========================
st.divider()
st.subheader("Global snapshot (beta)")

# Build all four series for snapshot (already built above)
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
    if df is not None and not df.empty:
        latest_candidates.append(df["Date"].max())
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
