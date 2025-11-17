# pages/2_Precipitation_Dashboard.py
# Precipitation Dashboard — FULL patched version (UI parity with Temperature)
# - Map uses latest PCPA per ADM1 (with Elevation toggle like Temperature)
# - Controls/KPIs mirror Temperature page
# - ADM1 comparison + summary shown under KPIs (no toggle)
# - Charts use layout: Story(0.2) | Chart(0.7) | Options form(0.1)
# - Percentiles section: radio [10..100], layout 0.2 / 0.79 / 0.01 (no display options)
# - Hover shows: one date at top → per-ADM1 blocks
# - Seasonal hover date shows "SEASON YEAR" (e.g., JJA 2016)
# - Units in charts and KPIs: mm/day
# - Placeholder logic for percentile curves if source data not yet available

import os, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

st.set_page_config(page_title="Precipitation Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------- CONSTANTS -----------------------------
HF_REPO_ID = "pjsimba16/adb_climate_dashboard_v1"
SEASON_TO_MONTH = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
MONTH_TO_SEASON = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
INDICATOR_LABELS = (
    "Precipitation","Temperature","Temperature Thresholds","Heatwaves","Coldwaves",
    "Dry Conditions","Wet Conditions","Humidity","Windspeeds",
)
CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
}

# ----------------------------- LIGHT CSS -----------------------------
st.markdown("""
<style>
/* Larger KPI numbers + snug arrow/dash badges */
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { 
  font-size: 34px !important; line-height:1.1;
}
div[data-testid="metric-container"] > div:first-child { 
  font-size: 14px !important;
}
/* Chart options small box look */
.disp-box { border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.75); }
.kpi { display:flex; flex-direction:column; gap:2px; }
.kpi .row { display:flex; align-items:center; gap:6px; }
.kpi .label { font-size:14px; color:#334155; display:flex; align-items:center; gap:6px; }
.kpi .value { font-size:34px; line-height:1.05; font-weight:600; }
.kpi .badge {
  display:inline-block; line-height:1; border-radius:999px; padding:2px 6px;
  border:1px solid #e5e7eb; font-size:11px; font-weight:700; color:#334155;
}
.kpi .up   { background:#dcfce7; } /* green */
.kpi .down { background:#fee2e2; } /* red   */
.kpi .flat { background:#e2e8f0; } /* neutral */
</style>
""", unsafe_allow_html=True)

# ----------------------------- PATH HELPERS -----------------------------
def _here():
    try:
        return Path(__file__).parent.resolve()
    except Exception:
        return Path.cwd()

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

# ----------------------------- HELPERS -----------------------------
def _suffixes_for_freq(freq, adm_level):
    f = str(freq); adm = str(adm_level).upper()
    if adm == "ADM1":
        return ["_M"] if f=="Monthly" else (["_S"] if f=="Seasonal" else ["_A"])
    return ["_AM","_PM"] if f=="Monthly" else (["_AS","_PS"] if f=="Seasonal" else ["_A"])

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
    if freq == "Monthly":  return ts.strftime("%b %Y")
    if freq == "Seasonal": return f"{MONTH_TO_SEASON[int(ts.month)]} {ts.year}"
    return ts.strftime("%Y")

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_indicator_mapper():
    if not MAPPER_FILE: return pd.DataFrame()
    try:
        return pd.read_csv(MAPPER_FILE)
    except Exception:
        return pd.DataFrame()

def _compose_title(code, mapper, fallback):
    if isinstance(mapper, pd.DataFrame) and not mapper.empty and {"Code","Description"}.issubset(mapper.columns):
        hit = mapper.loc[mapper["Code"].astype(str).str.upper() == str(code).upper()]
        if not hit.empty:
            d = str(hit.iloc[0]["Description"])
            return d if d else fallback
    return fallback

# ----------------------------- FILE RESOLVERS -----------------------------
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
        except Exception:
            continue
    return out

# ----------------------------- SERIES LOADERS -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_scope_series(iso3, freq, area_label, indicator_codes):
    mapper = load_indicator_mapper()
    is_country = (area_label in ("", "Country (all)"))
    suffixes = _suffixes_for_freq(freq, "ADM0" if is_country else "ADM1")
    path = _adm0_file(iso3, freq) if is_country else _adm1_file(iso3, area_label, freq)
    if path is None:
        return pd.DataFrame(), {}
    try:
        df_raw = pd.read_parquet(path)
    except Exception:
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

# ----------------------------- GEO / ELEVATION -----------------------------
def _get_hf_token():
    try:
        if hasattr(st, "secrets"):
            tok = st.secrets.get("HF_TOKEN")
            if tok: return str(tok)
    except Exception:
        pass
    return os.getenv("HF_TOKEN", "")

@st.cache_data(ttl=7*24*3600, show_spinner=False)
def load_country_adm1_geojson(iso3):
    if hf_hub_download is None:
        raise FileNotFoundError("huggingface_hub is not installed/available.")
    last_err = None
    for repo_type in ("space", "dataset"):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, repo_type=repo_type,
                                   filename=f"ADM1_geodata/{iso3}.geojson",
                                   token=_get_hf_token())
            gdf = gpd.read_file(path)
            try:
                gdf["geometry"] = gdf["geometry"].buffer(0)
            except Exception:
                pass
            gdf = gdf.to_crs(4326)
            bounds = tuple(gdf.total_bounds)
            name_col = "shapeName" if "shapeName" in gdf.columns else ("NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0])
            return gdf.__geo_interface__, bounds, name_col, gdf
        except Exception as e:
            last_err = e
            continue
    raise last_err

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
                if not (cn_country and cn_city and cn_elev):
                    continue
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
    if cm_iso.empty or not adm1_names:
        return False, 0, len(adm1_names)
    chk = pd.DataFrame({"ADM1": adm1_names})
    tmp = cm_iso[["ADM1","elevation"]].copy()
    tmp["elevation"] = pd.to_numeric(tmp["elevation"], errors="coerce")
    chk = chk.merge(tmp, on="ADM1", how="left")
    avail = int(chk["elevation"].notna().sum())
    total = len(chk)
    return (avail == total and total > 0), avail, total

# ----------------------------- FRAGMENT HELPER -----------------------------
try:
    fragment = st.fragment
    _FRAG_OK = True
except Exception:
    _FRAG_OK = False

def _with_fragment(fn):
    if _FRAG_OK: return fragment(fn)
    return fn

# ----------------------------- HEADER / NAV -----------------------------
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
            if c and getattr(c,"name",None):
                return c.name
        except Exception:
            pass
    return iso

def display_country_name(iso):
    iso = (iso or "").upper().strip()
    return _CUSTOM_COUNTRY_DISPLAY.get(iso, iso3_to_name(iso))

qp = st.query_params
iso3_q = (qp.get("iso3") or st.session_state.get("nav_iso3") or st.session_state.get("opt_iso3_p") or "").upper()
city_q = qp.get("city","")

top_l, _ = st.columns([0.12, 0.88])
with top_l:
    if st.button("← Home", help="Back to Home"):
        keep_iso3 = st.query_params.get("iso3",""); keep_city = st.query_params.get("city","")
        st.query_params.clear()
        if keep_iso3: st.query_params.update({"iso3": keep_iso3})
        if keep_city: st.query_params.update({"city": keep_city})
        try:
            st.switch_page("Home_Page.py")
        except Exception:
            st.rerun()
st.markdown(f"### Precipitation - {display_country_name(iso3_q) if iso3_q else '…'}")

# Optional dev panel via ?debug=1
debug_qp = (st.query_params.get("debug", "0").lower() in {"1","true","yes"})
show_dev_debug = st.toggle("Developer debug panel", value=False,
                           help="Show resolver paths & file scans.") if debug_qp else False

# ----------------------------- LAYOUT (Map left, controls+KPIs right) -----------------------------
with st.spinner("Scanning available countries…"):
    countries_iso = list_available_isos()
if iso3_q and iso3_q not in countries_iso:
    countries_iso = sorted(set(countries_iso + [iso3_q]))

lc, rc = st.columns([0.34, 0.66], gap="large")

# =========================== LEFT: MAP ===========================
@st.cache_data(ttl=900, show_spinner=False)
def load_latest_adm1_for_map_p(iso3: str, freq: str, want_debug: bool = False):
    def _scan_one(iso3_in, freq_in):
        base_dir = COUNTRY_DATA_DIR / str(iso3_in).upper() / {"Monthly":"Monthly","Seasonal":"Seasonal","Annual":"Annual"}[str(freq_in)]
        dbg = {"iso3": iso3_in, "freq": freq_in, "base_dir": str(base_dir), "base_dir_exists": base_dir.exists(), "files": [], "hits": 0}
        if not base_dir.exists():
            return pd.DataFrame(columns=["ADM1","value","Date"]), dbg
        suffixes = _suffixes_for_freq(freq_in, "ADM1")
        rows=[]
        for f in sorted(base_dir.glob("*_ADM1_data.parquet")):
            dbg["files"].append(f.name)
            try:
                cols = list(pd.read_parquet(f, columns=None).columns)
            except Exception:
                continue
            col_pcpa = _pick_col(cols, "PCPA", suffixes)
            if not col_pcpa:
                continue
            use_cols = ["ADM1","Year", col_pcpa]
            if str(freq_in) == "Monthly":  use_cols.append("Month")
            elif str(freq_in) == "Seasonal": use_cols.append("Season")
            try:
                d = pd.read_parquet(f, columns=use_cols)
            except Exception:
                continue
            d = d.rename(columns={col_pcpa:"value"})
            d["date"] = _build_date_column(d, freq_in)
            d = d.dropna(subset=["ADM1","date","value"])
            if d.empty: continue
            d = d.sort_values("date").tail(1).rename(columns={"date":"Date"})
            rows.append(d[["ADM1","value","Date"]])
        if not rows:
            return pd.DataFrame(columns=["ADM1","value","Date"]), dbg
        df_map = (
            pd.concat(rows, ignore_index=True)
              .sort_values(["ADM1","Date"]).drop_duplicates("ADM1", keep="last")
              .reset_index(drop=True)
        )
        dbg["hits"] = len(df_map)
        return df_map, dbg

    df, dbg = _scan_one(iso3, freq)
    tried = [(freq, dbg)]
    if df.empty and not want_debug:
        for f in ["Monthly","Seasonal","Annual"]:
            if f == freq: continue
            dfi, dbgi = _scan_one(iso3, f)
            tried.append((f, dbgi))
            if not dfi.empty:
                dfi = dfi.copy(); dfi["__actual_freq_used__"] = f
                return dfi
    return (df, tried) if want_debug else df

with lc:
    country_options = ["—"] + countries_iso
    if "opt_iso3_p" not in st.session_state:
        st.session_state["opt_iso3_p"] = iso3_q if iso3_q in country_options else "—"
    iso3 = st.selectbox(
        "Select Country",
        options=country_options,
        index=country_options.index(st.session_state["opt_iso3_p"]) if st.session_state["opt_iso3_p"] in country_options else 0,
        key="opt_iso3_p",
        format_func=lambda v: ("Select…" if v=="—" else display_country_name(v)),
        help="Pick a country, or arrive pre-selected via Home map."
    )
    iso3_cur = "" if iso3 == "—" else iso3
    if iso3_cur != st.query_params.get("iso3",""):
        st.query_params.update({"iso3": iso3_cur, "city": ""})
        st.rerun()

    MAP_HEIGHT = 640
    if iso3 and iso3 != "—":
        freq_for_map = st.session_state.get("opt_freq_p", "Monthly")

        if show_dev_debug:
            with st.expander("Map scan debugger", expanded=False):
                df_latest_dbg, tried_info_dbg = load_latest_adm1_for_map_p(iso3, freq_for_map, want_debug=True)
                st.write("COUNTRY_DATA_DIR:", str(COUNTRY_DATA_DIR))
                st.write("ISO3:", iso3, "Requested Frequency:", freq_for_map)
                for f_try, info in tried_info_dbg:
                    st.write({"tried_freq": f_try, **info})

        df_latest = load_latest_adm1_for_map_p(iso3, freq_for_map, want_debug=False)
        if isinstance(df_latest, tuple): df_latest = df_latest[0]

        if df_latest.empty:
            sel = st.session_state.get("opt_freq_p","Monthly")
            st.warning(f"No ADM1 PCPA data found for **{iso3} — {sel}**.", icon="⚠️")
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
            if st.session_state.get("last_iso3_for_choice_p") != iso3 or "map_data_choice_p" not in st.session_state:
                st.session_state["map_data_choice_p"] = "Elevation" if elev_complete else "Precipitation"
                st.session_state["last_iso3_for_choice_p"] = iso3

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

            choice = st.radio("Map data", ["Precipitation","Elevation"], horizontal=True, key="map_data_choice_p",
                              help="Choose the data shown on the map.")
            if choice == "Elevation":
                color_col = "elevation"; cs_name = "Viridis"; cbar_title = "Elevation (m)"
                hover_tmpl = "<b>%{customdata[0]}</b><br>Elevation: %{customdata[1]:.0f} m<br>As of: %{customdata[2]}<extra></extra>"
            else:
                color_col = "value"; cs_name = "Blues"; cbar_title = "Avg Daily Precip (mm/day)"
                hover_tmpl = "<b>%{customdata[0]}</b><br>Latest: %{customdata[1]:.2f} mm/day<br>As of: %{customdata[2]}<extra></extra>"

            fig = px.choropleth(
                df_map, geojson=geojson_dict, locations="ADM1",
                featureidkey=f"properties.{name_col}", color=color_col,
                projection="mercator", color_continuous_scale=cs_name,
            )
            minx, miny, maxx, maxy = bounds
            pad = 0.35
            fig.update_geos(
                projection_type="mercator", fitbounds="locations",
                lonaxis_range=[minx - pad, maxx + pad],
                lataxis_range=[miny - pad, maxy + pad],
                showland=False, showcountries=False, showcoastlines=False, showocean=False, visible=False,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=MAP_HEIGHT,
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                hovermode="closest", showlegend=False,
                coloraxis_colorbar=dict(
                    title=cbar_title, thickness=14, len=0.98, y=0.92, yanchor="top", x=0.98,
                    tickfont=dict(size=10), titlefont=dict(size=11),
                )
            )
            _date_str = pd.to_datetime(df_map["Date"], errors="coerce").dt.strftime("%Y-%m").fillna("—")
            vals = pd.to_numeric(df_map[color_col], errors="coerce")
            fig.data[0].customdata = np.stack([df_map["ADM1"].astype(str).values, vals.values, _date_str.values], axis=-1)
            fig.data[0].hovertemplate = hover_tmpl
            fig.update_traces(marker_line_width=0.3, marker_line_color="#999")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Optional ADM1 labels toggle (dynamic color contrast)
            try:
                lons = gdf.representative_point().x.to_numpy()
                lats = gdf.representative_point().y.to_numpy()
                names = gdf[name_col].astype(str).to_numpy()
                arr = pd.to_numeric(df_map[color_col], errors="coerce").to_numpy(float)
                vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
                vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
                cs = px.colors.sequential.Viridis if choice=="Elevation" else px.colors.sequential.Blues
                from plotly.colors import sample_colorscale
                def _hex_to_rgb01(h):
                    s=h.lstrip("#"); s="".join(ch*2 for ch in s) if len(s)==3 else s
                    return int(s[0:2],16)/255, int(s[2:4],16)/255, int(s[4:6],16)/255
                def _lum(rgb): r,g,b=rgb; return 0.299*r+0.587*g+0.114*b
                val_map = dict(zip(df_map["ADM1"].astype(str), df_map[color_col]))
                colors=[]; denom = (vmax-vmin) if vmax!=vmin else 1.0
                for nm in names:
                    v = val_map.get(nm, np.nan)
                    if not np.isfinite(v): colors.append("black"); continue
                    t = float(np.clip((v - vmin)/denom, 0.0, 1.0))
                    col = sample_colorscale(cs, [t])[0]
                    colors.append("black" if _lum(_hex_to_rgb01(col))>0.55 else "white")
                show_labels_current = st.session_state.get(f"show_labels_p_{iso3}", False)
                if show_labels_current:
                    mw = [c=="white" for c in colors]; mb = [c=="black" for c in colors]
                    if any(mw):
                        fig.add_trace(go.Scattergeo(lon=np.array(lons)[mw], lat=np.array(lats)[mw],
                                                    text=np.array(names)[mw], mode="text",
                                                    textfont=dict(size=10, color="white"),
                                                    hoverinfo="skip", showlegend=False))
                    if any(mb):
                        fig.add_trace(go.Scattergeo(lon=np.array(lons)[mb], lat=np.array(lats)[mb],
                                                    text=np.array(names)[mb], mode="text",
                                                    textfont=dict(size=10, color="black"),
                                                    hoverinfo="skip", showlegend=False))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.toggle("Show ADM1 labels", value=show_labels_current, key=f"show_labels_p_{iso3}")
            except Exception:
                pass

            latest_when = pd.to_datetime(df_map["Date"], errors="coerce").max()
            latest_period = latest_when.strftime("%Y-%m") if pd.notna(latest_when) else "—"
            regions_with_data = int(pd.to_numeric(df_map[color_col], errors="coerce").notna().sum())
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

# =========================== RIGHT: CONTROLS + KPIs ===========================
with rc:
    # ADM1 list for chosen country
    if iso3 and iso3 != "—":
        try:
            _, _, name_col_for_list, gdf_for_list = load_country_adm1_geojson(iso3)
            _adm1_list = sorted(gdf_for_list[name_col_for_list].astype(str).unique().tolist())
        except Exception:
            _adm1_list = []
    else:
        _adm1_list = []
    adm1_options = ["Country (all)"] + _adm1_list

    url_city = st.query_params.get("city","")
    if "opt_adm1_p" not in st.session_state:
        st.session_state["opt_adm1_p"] = url_city if url_city in adm1_options else "Country (all)"

    sel_adm1 = st.selectbox(
        "Select Province/City/State",
        options=adm1_options,
        index=adm1_options.index(st.session_state["opt_adm1_p"]) if st.session_state["opt_adm1_p"] in adm1_options else 0,
        key="opt_adm1_p",
        help="Change the region focus for the charts.",
    )
    desired = "" if sel_adm1 in ("","Country (all)") else sel_adm1
    if desired != st.query_params.get("city",""):
        st.query_params.update({"city": desired})
        st.rerun()

    col_ind, col_form = st.columns([0.35, 0.65], gap="small")
    with col_ind:
        st.markdown("<div style='margin-top:0.2rem'></div>", unsafe_allow_html=True)
        indicator = st.radio("Select climate indicator", INDICATOR_LABELS, index=0, key="opt_indicator_precip",
                             help="Switch indicators.")
        if indicator == "Temperature":
            carry_iso = st.session_state.get("opt_iso3_p","—")
            carry_adm1 = st.session_state.get("opt_adm1_p","")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso,
                                    "city": "" if carry_adm1 in ("","Country (all)") else carry_adm1})
            try:
                st.switch_page("pages/1_Temperature_Dashboard.py")
            except Exception:
                st.switch_page("1_Temperature_Dashboard.py")
        elif indicator == "Humidity":
            carry_iso = st.session_state.get("opt_iso3_t","—")
            carry_adm1 = st.session_state.get("opt_adm1_t","")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso,
                                    "city": "" if carry_adm1 in ("","Country (all)") else carry_adm1})
            try:
                st.switch_page("pages/3_Humidity_Dashboard.py")
            except Exception:
                st.switch_page("3_Humidity_Dashboard.py")
        elif indicator == "Windspeeds":
            carry_iso = st.session_state.get("opt_iso3_t","—")
            carry_adm1 = st.session_state.get("opt_adm1_t","")
            st.query_params.update({"iso3": "" if carry_iso=="—" else carry_iso,
                                    "city": "" if carry_adm1 in ("","Country (all)") else carry_adm1})
            try:
                st.switch_page("pages/4_Windspeeds_Dashboard.py")
            except Exception:
                st.switch_page("4_Windspeeds_Dashboard.py")

    with col_form:
        st.markdown("#### Chart Options")
        with st.form("options_form_precip", clear_on_submit=False):
            colA, colB, colC = st.columns(3)
            with colA:
                data_type = st.radio("Type", ["Historical Observations", "Projections (SSPs)"], index=0, key="opt_type_p")
            with colB:
                freq = st.radio("Frequency", ["Monthly", "Seasonal", "Annual"], index=0, key="opt_freq_p")
            with colC:
                source = st.radio("Data Source", ["CDS/CCKP", "CRU", "ERA5"], index=2, key="opt_source_p")
            st.form_submit_button("Apply changes", type="primary")

        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.25rem;">
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Indicator: {indicator}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Type: {('Historical' if data_type.startswith('Historical') else 'Projections')}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Frequency: {freq}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Source: {source}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Area: {('Country' if sel_adm1 in ('','Country (all)') else sel_adm1)}</span>
            </div>
            <div style="height:10px;"></div>
            """, unsafe_allow_html=True
        )

    # ===== KPIs (PCPA / PCPV) with deltas — arrows only on change KPIs =====
    iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_p") or "")) or ""
    adm1_now = st.query_params.get("city","") or st.session_state.get("opt_adm1_p","")
    freq_sel = st.session_state.get("opt_freq_p","Monthly")

    def render_kpi(label: str, value_text: str, delta: float, show_symbol: bool):
        # arrow-only pill; no duplicated numbers
        if delta is None or not (isinstance(delta,(int,float)) and np.isfinite(delta)) or abs(delta) < 1e-12:
            klass, sym, tip = "flat", "—", "No change"
        else:
            if delta > 0: klass, sym, tip = "up", "↑", f"+{delta:.2f} mm/day"
            else:         klass, sym, tip = "down", "↓", f"{delta:.2f} mm/day"
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

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    base_pcpa = _prep_single(iso3_now, adm1_now, freq_sel, "PCPA", "PCPV")
    k1, k2, k3, k4 = st.columns(4)
    if base_pcpa.empty:
        with k1: render_kpi("Latest Avg Daily Precip (PCPA)", "—", None, show_symbol=False)
        with k2: render_kpi("Δ vs previous point", "—", None, show_symbol=True)
        with k3: render_kpi("Δ vs same period LY", "—", None, show_symbol=True)
        with k4: render_kpi("Mean / σ in range", "—", None, show_symbol=False)
    else:
        srt = base_pcpa.sort_values("date")
        latest = float(srt["avg"].iloc[-1])
        prev   = float(srt["avg"].iloc[-2]) if len(srt) > 1 else np.nan
        def _same_period_last_year(df):
            dmax = df["date"].max()
            if freq_sel == "Annual":
                tgt = dmax - pd.DateOffset(years=1)
                yr_prev = df[df["date"].dt.year == tgt.year]
                return yr_prev["avg"].iloc[-1] if not yr_prev.empty else np.nan
            tgt = dmax - pd.DateOffset(years=1)
            row = df[(df["date"].dt.year == tgt.year) & (df["date"].dt.month == tgt.month)]
            return row["avg"].iloc[-1] if not row.empty else np.nan
        ly_val = _same_period_last_year(srt)
        delta_prev = latest - prev if np.isfinite(prev) else np.nan
        delta_yoy  = latest - ly_val if np.isfinite(ly_val) else np.nan
        mean_v     = float(np.nanmean(srt["avg"])) if len(srt) else np.nan
        std_v      = float(np.nanstd(srt["avg"])) if "var" not in srt else float(np.nanmean(np.sqrt(srt["var"].clip(lower=0))))

        with k1: render_kpi("Latest Avg Daily Precip (PCPA)", (f"{latest:.2f} mm/day" if np.isfinite(latest) else "—"), None, show_symbol=False)
        with k2: render_kpi("Δ vs previous point", (f"{delta_prev:+.2f} mm/day" if np.isfinite(delta_prev) else "—"),
                            (delta_prev if np.isfinite(delta_prev) else None), show_symbol=True)
        ly_label = "Δ vs same month LY" if freq_sel=="Monthly" else ("Δ vs same season LY" if freq_sel=="Seasonal" else "Δ vs same year LY")
        with k3: render_kpi(ly_label, (f"{delta_yoy:+.2f} mm/day" if np.isfinite(delta_yoy) else "—"),
                            (delta_yoy if np.isfinite(delta_yoy) else None), show_symbol=True)
        with k4: render_kpi("Mean / σ in range",
                            (f"{mean_v:.2f} mm/day • {std_v:.2f}" if np.isfinite(mean_v) and np.isfinite(std_v) else "—"),
                            None, show_symbol=False)

    # small breathing room below KPIs
    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    # ===== ADM1 COMPARISON (ALWAYS SHOWN) + INFO + SUMMARY =====
    try:
        _, _, _name_col_for_list2, _gdf_for_list2 = load_country_adm1_geojson(iso3_now)
        _adm1_all_choices = sorted(_gdf_for_list2[_name_col_for_list2].astype(str).unique().tolist())
    except Exception:
        _adm1_all_choices = []

    _main_label = ("Country (all)" if adm1_now in ("", "Country (all)") else adm1_now)
    _pool = (["Country (all)"] if _main_label != "Country (all)" else []) + [x for x in _adm1_all_choices if x != _main_label]

    st.markdown("**Compare with other ADM1s or full country (up to 5)**")
    sel_compare = st.multiselect(
        "These selections will be added to all charts below.",
        options=_pool, max_selections=5, key="sel_compare_p",
        help="The main selection is always included."
    )
    geo_list = [_main_label] + [g for g in sel_compare if g != _main_label][:5]

    with st.expander("What do these columns mean?"):
        st.markdown("""
- **Last date** — latest period available for each ADM1 at the selected frequency.
- **Latest (mm/day)** — average daily precipitation at the latest period.
- **Δ prev (mm/day)** — change from the immediately previous period.
- **Δ YoY/YoS (mm/day)** — change from the same month/season last year (or previous year for annual).
- **Mean (mm/day)** — arithmetic mean over the displayed series.
- **σ (mm/day)** — typical variability; from variance column if available, else std. dev. of the series.
- **Outlier** — z-score flag on the last point vs a recent window (≥2σ).
        """)

    def _freq_window_for_outliers(freq_str: str) -> int:
        return 12 if freq_str=="Monthly" else (8 if freq_str=="Seasonal" else 5)

    def _kpi_for_geo(iso3_in: str, geo_label: str, freq_str: str) -> dict:
        s = _prep_single(iso3_in, geo_label, freq_str, "PCPA", "PCPV")
        if s.empty:
            return {"ADM1": geo_label, "Last date": "—", "Latest (mm/day)": np.nan,
                    "Δ prev (mm/day)": np.nan, "Δ YoY/YoS (mm/day)": np.nan,
                    "Mean (mm/day)": np.nan, "σ (mm/day)": np.nan, "Outlier": "—"}
        s = s.sort_values("date")
        latest = s.iloc[-1]
        last_val = float(latest["avg"])
        prev_val = float(s.iloc[-2]["avg"]) if len(s) > 1 else np.nan
        if freq_str == "Annual":
            tgt_yr = latest["date"].year - 1
            prev_row = s[s["date"].dt.year == tgt_yr]
        else:
            tgt = latest["date"] - pd.DateOffset(years=1)
            prev_row = s[(s["date"].dt.year == tgt.year) & (s["date"].dt.month == tgt.month)]
        yoy_val = float(prev_row["avg"].iloc[-1]) if not prev_row.empty else np.nan
        mean_v = float(np.nanmean(s["avg"]))
        sigmas = np.sqrt(s["var"].clip(lower=0)) if "var" in s.columns else None
        sigma_v = float(np.nanmean(sigmas)) if (sigmas is not None and sigmas.notna().any()) else float(np.nanstd(s["avg"]))

        win = _freq_window_for_outliers(freq_str)
        tail = s.tail(win)["avg"].to_numpy(dtype=float)
        outlier_flag = "—"
        if len(tail) >= 3 and np.nanstd(tail) > 0:
            z = (tail[-1] - np.nanmean(tail)) / np.nanstd(tail)
            if np.isfinite(z) and abs(z) >= 2.0:
                outlier_flag = f"{z:+.2f}σ"

        if freq_str == "Monthly":
            last_date = latest["date"].strftime("%Y-%m")
        elif freq_str == "Annual":
            last_date = latest["date"].strftime("%Y")
        else:
            last_date = f"{MONTH_TO_SEASON[int(latest['date'].month)]} {latest['date'].year}"

        return {
            "ADM1": geo_label, "Last date": last_date, "Latest (mm/day)": round(last_val, 2),
            "Δ prev (mm/day)": (round(last_val - prev_val, 2) if np.isfinite(prev_val) else np.nan),
            "Δ YoY/YoS (mm/day)": (round(last_val - yoy_val, 2) if np.isfinite(yoy_val) else np.nan),
            "Mean (mm/day)": round(mean_v, 2), "σ (mm/day)": round(sigma_v, 2), "Outlier": outlier_flag
        }

    if geo_list:
        st.markdown("##### Comparison Summary")
        rows = [_kpi_for_geo(iso3_now, g, freq_sel) for g in geo_list]
        summary = pd.DataFrame(rows)
        styler = (
            summary.style
                .format(precision=2, na_rep="—")
                .set_table_styles([{'selector':'th','props':[('text-align','center')]}])
                .set_properties(**{'text-align':'center'})
        )
        st.dataframe(styler, use_container_width=True, hide_index=True)

# ----------------------------- DIVIDER -----------------------------
st.markdown("---")

mapper = load_indicator_mapper()
iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_p") or "")) or ""
adm1_now = st.query_params.get("city","") or st.session_state.get("opt_adm1_p","")
freq     = st.session_state.get("opt_freq_p","Monthly")
if not iso3_now:
    st.warning("Select a country to load precipitation charts.")
    st.stop()

def _season_year_str(series_ts: pd.Series) -> np.ndarray:
    s = series_ts.dt.month.map(MONTH_TO_SEASON).astype(str) + " " + series_ts.dt.year.astype(str)
    return s.to_numpy(dtype=object)

# ----------------------------- GENERIC CHART RENDERER -----------------------------
def set_legend_top(fig: "go.Figure"):
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=1.02, yanchor="bottom",
            x=0.0,  xanchor="left",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemwidth=30
        ),
        margin=dict(t=80)  # headroom for the legend
    )

def render_chart_with_controls_p(
    story_title_md: str,
    plot_title: str,
    avg_code: str,
    var_code: str,
    extras: list[str],
    hover_labels_by_trace: dict[str, str],
    chart_key: str,
    units: str = "mm/day",
):
    main_label = ("Country (all)" if adm1_now in ("", "Country (all)") else adm1_now)
    comp_list = st.session_state.get("sel_compare_p", []) or []
    geo_list_local = [main_label] + [g for g in comp_list if g != main_label][:5]

    def _series_for_geo(g):
        need = [avg_code] + ([var_code] if var_code else []) + (extras or [])
        df_codes, _ = load_scope_series(iso3_now, freq, g, need)
        if df_codes.empty or avg_code not in df_codes:
            return pd.DataFrame(columns=["date","avg","var"] + (extras or []))
        out = pd.DataFrame({"date": df_codes["date"], "avg": pd.to_numeric(df_codes[avg_code], errors="coerce")})
        if var_code and var_code in df_codes:
            out["var"] = pd.to_numeric(df_codes[var_code], errors="coerce")
        for e in (extras or []):
            if e in df_codes:
                out[e] = pd.to_numeric(df_codes[e], errors="coerce")
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    sdict = {g: _series_for_geo(g) for g in geo_list_local}
    sdict = {k:v for k,v in sdict.items() if not v.empty}
    if not sdict:
        left, _, _ = st.columns([0.2,0.7,0.1], gap="large")
        with left: st.info("No data for current selection.")
        return

    dmin = min(s["date"].min() for s in sdict.values()).date()
    dmax = max(s["date"].max() for s in sdict.values()).date()

    story_col, chart_col, opts_col = st.columns([0.2, 0.7, 0.1], gap="large")

    with story_col:
        st.markdown(story_title_md)

    with opts_col:
        with st.form(f"form_{chart_key}"):
            st.markdown("**Display options**", help="Only affects this chart.")
            show_avg  = st.checkbox("Show average", value=True, key=f"{chart_key}_avg")
            show_band = st.checkbox("Show ±1σ band", value=True, key=f"{chart_key}_band")
            show_ext  = st.checkbox("Show extremes", value=bool(extras), key=f"{chart_key}_ext", disabled=not extras)
            apply_btn = st.form_submit_button("Apply changes", type="primary")
            if apply_btn: st.rerun()

    with chart_col:
        d1, d2 = st.slider(
            "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax),
            format="YYYY-MM", key=f"rng_{chart_key}"
        )

        fig = go.Figure()
        palette = list(CBLIND.values())

        for i, (label, s) in enumerate(sdict.items()):
            s2 = s[(s["date"] >= pd.to_datetime(d1)) & (s["date"] <= pd.to_datetime(d2))].copy()
            if s2.empty: 
                continue
            color = palette[i % len(palette)]

            # Prepare customdata (shape-safe)
            N = len(s2)
            label_arr = np.full(N, label, dtype=object)
            avg_arr   = pd.to_numeric(s2["avg"], errors="coerce").to_numpy(dtype=float)

            if "var" in s2.columns:
                sigma_arr = np.sqrt(pd.to_numeric(s2["var"], errors="coerce").clip(lower=0)).to_numpy(dtype=float)
            else:
                sigma_arr = np.full(N, np.nan)

            ex1_arr = np.full(N, np.nan)
            ex2_arr = np.full(N, np.nan)
            if extras:
                if len(extras) >= 1 and extras[0] in s2.columns:
                    ex1_arr = pd.to_numeric(s2[extras[0]], errors="coerce").to_numpy(dtype=float)
                if len(extras) >= 2 and extras[1] in s2.columns:
                    ex2_arr = pd.to_numeric(s2[extras[1]], errors="coerce").to_numpy(dtype=float)

            season_hdr = _season_year_str(s2["date"]) if freq == "Seasonal" else np.full(N, "", dtype=object)

            # [adm1, avg, sigma, ex1, ex2, season_hdr]
            customdata = np.empty((N, 6), dtype=object)
            customdata[:, 0] = label_arr
            customdata[:, 1] = avg_arr
            customdata[:, 2] = sigma_arr
            customdata[:, 3] = ex1_arr
            customdata[:, 4] = ex2_arr
            customdata[:, 5] = season_hdr

            # Band (no hover)
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

            # AVG line (single hover block per ADM1)
            if show_avg:
                # --- build hover template (avoid f-strings for %{...} parts) ---
                if freq == "Seasonal":
                    header = "%{customdata[5]}"          # e.g., "JJA 2016"
                elif freq == "Monthly":
                    header = "%{x|%b %Y}"                # e.g., "Jan 2016"
                else:
                    header = "%{x|%Y}"                   # e.g., "2016"

                unit_txt = " " + units

                block_lines = [
                    hover_labels_by_trace.get('avg', 'Average') + ": %{customdata[1]:.2f}" + unit_txt
                ]

                # show σ if present
                if "var" in s2.columns:
                    block_lines.append("±1σ: %{customdata[2]:.2f}" + unit_txt)

                # extras (e.g., extremes)
                if extras:
                    if len(extras) >= 1 and extras[0] in s2.columns:
                        block_lines.append(hover_labels_by_trace.get(extras[0], extras[0]) + ": %{customdata[3]:.2f}" + unit_txt)
                    if len(extras) >= 2 and extras[1] in s2.columns:
                        block_lines.append(hover_labels_by_trace.get(extras[1], extras[1]) + ": %{customdata[4]:.2f}" + unit_txt)

                hover_tmpl = (
                    "<b>" + header + "</b><br>"
                    "<b>%{customdata[0]}</b><br>"
                    + "<br>".join(block_lines) +
                    "<extra></extra>"
                )

                fig.add_trace(go.Scatter(
                    x=s2["date"], y=s2["avg"], mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{label} — {hover_labels_by_trace.get('avg','Average')}",
                    customdata=customdata,
                    hovertemplate=hover_tmpl
                ))


            # Extremes (no hover)
            if show_ext and extras:
                for j, ex in enumerate(extras):
                    if ex in s2.columns and s2[ex].notna().any():
                        elab = hover_labels_by_trace.get(ex, ex)
                        fig.add_trace(go.Scatter(
                            x=s2["date"], y=s2[ex], mode="lines",
                            line=dict(color=color, width=1.2, dash="dot"),
                            name=f"{label} — {elab}",
                            hoverinfo="skip"
                        ))

        fig.update_layout(
            title=plot_title, height=420, margin=dict(l=30,r=30,t=40,b=50),
            hovermode="x unified", xaxis_title="Date", yaxis_title=units
        )
        set_legend_top(fig)
        if freq == "Monthly":
            fig.update_xaxes(tickformat="%b\n%Y")
        elif freq == "Seasonal":
            fig.update_xaxes(tickformat="%Y")
        else:
            fig.update_xaxes(tickformat="%Y")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ----------------------------- CHARTS -----------------------------
st.markdown("### Precipitation Indicators")

# PCPA (+PCPV)
render_chart_with_controls_p(
    story_title_md="**Story — Average Daily Precipitation (PCPA)**  \nTrend and variability in daily precipitation.",
    plot_title="Average Daily Precipitation (PCPA)",
    avg_code="PCPA", var_code="PCPV", extras=None,
    hover_labels_by_trace={"avg":"Avg. Daily Precip"},
    chart_key="pcpa", units="mm/day"
)

# PCPS (if you store a 'spread' / standard precip measure as PCPS; otherwise still plotted if present)
render_chart_with_controls_p(
    story_title_md="**Story — Precipitation Secondary Measure (PCPS)**  \nAdditional precipitation indicator (if available).",
    plot_title="Precipitation — Secondary (PCPS)",
    avg_code="PCPS", var_code=None, extras=None,
    hover_labels_by_trace={"avg":"PCPS"},
    chart_key="pcps", units="mm/day"
)

# PCPX (max)
render_chart_with_controls_p(
    story_title_md="**Story — Maximum Daily Precipitation (PCPX)**  \nObserved maximums in daily precipitation.",
    plot_title="Maximum Daily Precipitation (PCPX)",
    avg_code="PCPX", var_code=None, extras=None,
    hover_labels_by_trace={"avg":"Max Daily Precip"},
    chart_key="pcpx", units="mm/day"
)

# PCPN (min)
render_chart_with_controls_p(
    story_title_md="**Story — Minimum Daily Precipitation (PCPN)**  \nObserved minimums in daily precipitation.",
    plot_title="Minimum Daily Precipitation (PCPN)",
    avg_code="PCPN", var_code=None, extras=None,
    hover_labels_by_trace={"avg":"Min Daily Precip"},
    chart_key="pcpn", units="mm/day"
)

# ----------------------------- PERCENTILES -----------------------------
st.markdown("---")
st.subheader("Percentiles")

pct_choice = st.radio(
    "Select a percentile (applies to all charts below)",
    options=[10,20,30,40,50,60,70,80,90,100],
    horizontal=True, index=1, key="pct_p_single",
    help="Choose one percentile line to overlay per ADM1."
)

def _phase_from_date(dt: pd.Timestamp, freq: str) -> int:
    if freq=="Monthly": return int(dt.month)
    if freq=="Seasonal":
        m=int(dt.month)
        return 1 if m in (12,1,2) else (2 if m in (3,4,5) else (3 if m in (6,7,8) else 4))
    return 1

def _empirical_percentile_curve(s_df: pd.DataFrame, pct: int, freq: str) -> pd.DataFrame:
    if s_df is None or s_df.empty or "avg" not in s_df:
        return pd.DataFrame(columns=["date","p"])
    d = pd.DataFrame({"date": pd.to_datetime(s_df["date"], errors="coerce"),
                      "val":  pd.to_numeric(s_df["avg"], errors="coerce")}).dropna()
    if d.empty: return pd.DataFrame(columns=["date","p"])
    d["_phase"] = d["date"].apply(lambda x: _phase_from_date(x, freq))
    q = d.groupby("_phase")["val"].quantile(pct/100.0)
    return pd.DataFrame({"date": d["date"], "p": d["_phase"].map(q)})

def _percentile_chart_p(title: str, avg_code: str, chart_key: str, story: str, units="mm/day"):
    # layout: 0.2 / 0.79 / 0.01 (no options form)
    story_col, chart_col, _ = st.columns([0.2, 0.79, 0.01], gap="large")

    main_label = ("Country (all)" if adm1_now in ("", "Country (all)") else adm1_now)
    comp_list = st.session_state.get("sel_compare_p", []) or []
    geo_list_local = [main_label] + [g for g in comp_list if g != main_label][:5]

    def _s_for(g):
        return _prep_single(iso3_now, g, freq, avg_code, None)
    sdict = {g: _s_for(g) for g in geo_list_local}
    sdict = {k:v for k,v in sdict.items() if not v.empty}

    # Placeholder logic if nothing available yet
    if not sdict:
        with story_col: st.markdown(f"**Story — {title}**  \n{story}")
        with chart_col:
            st.info("Percentile data not available yet — placeholder line shown.")
            # simple placeholder (flat line at 0 across an arbitrary short timeline)
            dates = pd.date_range("2000-01-01", periods=24, freq="MS")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=np.zeros(len(dates)), mode="lines", name=f"P{pct_choice} (placeholder)"))
            fig.update_layout(title=title, height=420, margin=dict(l=30,r=30,t=40,b=50),
                              hovermode="x unified", xaxis_title="Date", yaxis_title=units)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        return

    dmin = min(s["date"].min() for s in sdict.values()).date()
    dmax = max(s["date"].max() for s in sdict.values()).date()

    with story_col:
        st.markdown(f"**Story — {title}**  \n{story}")

    with chart_col:
        d1, d2 = st.slider(
            "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax),
            format="YYYY-MM", key=f"rng_{chart_key}"
        )
        fig = go.Figure()
        colors = list(CBLIND.values())

        for i, (label, s) in enumerate(sdict.items()):
            s2 = s[(s["date"]>=pd.to_datetime(d1)) & (s["date"]<=pd.to_datetime(d2))].copy()
            if s2.empty: 
                continue
            color = colors[i % len(colors)]

            # Average
            fig.add_trace(go.Scatter(
                x=s2["date"], y=s2["avg"], mode="lines",
                name=f"{label} — Avg",
                line=dict(color=color, width=1.6),
            ))

            # Single percentile
            pc = _empirical_percentile_curve(s2, int(pct_choice), freq)
            if not pc.empty:
                fig.add_trace(go.Scatter(
                    x=pc["date"], y=pc["p"], mode="lines",
                    name=f"{label} — P{pct_choice}",
                    line=dict(color=color, width=1.2, dash="dot"),
                ))
            else:
                # placeholder for this ADM1 only
                d_placeholder = s2["date"]
                fig.add_trace(go.Scatter(
                    x=d_placeholder, y=np.zeros(len(d_placeholder)), mode="lines",
                    name=f"{label} — P{pct_choice} (placeholder)",
                    line=dict(color=color, width=1.2, dash="dot"),
                ))

        fig.update_layout(
            title=title, height=420, margin=dict(l=30,r=30,t=40,b=50),
            hovermode="x unified", xaxis_title="Date", yaxis_title=units
        )
        set_legend_top(fig)
        if freq == "Monthly":   fig.update_xaxes(tickformat="%b\n%Y")
        elif freq == "Seasonal": fig.update_xaxes(tickformat="%Y")
        else:                    fig.update_xaxes(tickformat="%Y")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

_pct_story = "How values compare with the historical distribution for the same month/season."
mapper = load_indicator_mapper()
_percentile_chart_map = [
    (_compose_title("PCPA", mapper, "Average Daily Precipitation — Percentiles"), "PCPA", "pct_pcpa"),
    (_compose_title("PCPS", mapper, "Precipitation Secondary — Percentiles"), "PCPS", "pct_pcps"),
    (_compose_title("PCPX", mapper, "Maximum Daily Precipitation — Percentiles"), "PCPX", "pct_pcpx"),
    (_compose_title("PCPN", mapper, "Minimum Daily Precipitation — Percentiles"), "PCPN", "pct_pcpn"),
]
for _title, _code, _key in _percentile_chart_map:
    _percentile_chart_p(_title, _code, _key, _pct_story, units="mm/day")
