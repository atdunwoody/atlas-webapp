import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Mapping

import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import folium
import fiona

st.set_page_config(layout="wide")

# -------------------------
# Path utilities & validation
# -------------------------
def resolve_gpkg_path(raw_path: str) -> Path:
    if not raw_path:
        raise FileNotFoundError("No GeoPackage path provided.")
    p = Path(raw_path.replace("\\", "/")).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"GPKG not found: {p}\n(Working directory: {Path.cwd()})")
    if not p.is_file():
        raise FileNotFoundError(f"Path is not a file: {p}")
    return p

# -------------------------
# Data loading & validation
# -------------------------
@st.cache_data
def list_layers(gpkg_path: str) -> List[str]:
    p = resolve_gpkg_path(gpkg_path)
    layers = fiona.listlayers(str(p))
    if not layers:
        raise ValueError(f"No layers found in GeoPackage: {p}")
    return layers

@st.cache_data
def load_layer(gpkg_path: str, layer: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    p = resolve_gpkg_path(gpkg_path)
    gdf = gpd.read_file(str(p), layer=layer)
    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty in {p}.")

    # Add a stable row id BEFORE any reprojection/processing so we can join for display
    if "_rowid" not in gdf.columns:
        gdf = gdf.reset_index(drop=True).assign(_rowid=lambda d: d.index.astype("int64"))

    if gdf.crs is None:
        st.warning("Input layer has no CRS. Assuming EPSG:4326 for display; writing preserves as-is.")
        gdf_wgs84 = gdf
    else:
        gdf_wgs84 = gdf.to_crs(epsg=4326)

    return gdf, gdf_wgs84

def ensure_numeric(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        raise ValueError(f"Field '{name}' has no numeric values.")
    return s

# -------------------------
# Case-insensitive column resolver
# -------------------------
EXPECTED: Mapping[str, tuple[str, ...]] = {
    "Geomorphic": ("geomorphic",),
    "PScore": ("pscore", "p_score", "p-score"),
    "UScore": ("uscore", "u_score", "u-score", "uscore_value"),
    "CurrCond": ("currcond", "current_condition", "current_cond", "curr_cond"),
    "CurrTemp": ("currtemp", "current_temperature", "current_temp", "curr_temp"),
    "Basin": ("basin", "basin_name"),
    "_rowid": ("_rowid",),
}

def resolve_columns_ci(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping from canonical names -> actual df columns (case-insensitive)."""
    lowmap = {c.lower(): c for c in df.columns}
    out: Dict[str, str] = {}
    missing: list[str] = []
    for canon, aliases in EXPECTED.items():
        found = None
        for a in aliases:
            if a in lowmap:
                found = lowmap[a]
                break
        if found is None:
            # also try exact (already matched) and canonical itself
            if canon in df.columns:
                found = canon
            elif canon.lower() in lowmap:
                found = lowmap[canon.lower()]
        if found is None:
            missing.append(canon)
        else:
            out[canon] = found
    if missing:
        # We only require the scoring fields and Basin; _rowid is injected above.
        must_have = {"Geomorphic", "PScore", "UScore", "CurrCond", "CurrTemp", "Basin"}
        truly_missing = sorted(must_have.intersection(missing))
        if truly_missing:
            raise ValueError(
                "Missing required fields (case-insensitive match): " + ", ".join(truly_missing)
            )
    return out

# -------------------------
# UI helpers (paired slider + input)
# -------------------------
def _pair_weight_control(label: str, key: str, default: int = 20) -> int:
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(default)
    if input_key not in st.session_state:
        st.session_state[input_key] = int(default)

    def sync_from_slider():
        st.session_state[input_key] = int(st.session_state[slider_key])

    def sync_from_input():
        v = int(st.session_state[input_key])
        v = max(0, min(100, v))
        st.session_state[input_key] = v
        st.session_state[slider_key] = v

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(label, 0, 100, step=1, key=slider_key, on_change=sync_from_slider)
    with c2:
        st.number_input(" ", 0, 100, step=1, key=input_key,
                        on_change=sync_from_input, label_visibility="hidden")
    return int(st.session_state[input_key])

def weights_section() -> Dict[str, int]:
    st.subheader("Component Weights (must sum to 100)")
    cA, cB, cC = st.columns(3)
    with cA:
        geom_w = _pair_weight_control("Geomorphic weight", "geom_w", 20)
        p_w    = _pair_weight_control("PScore weight", "p_w", 20)
    with cB:
        u_w    = _pair_weight_control("UScore weight", "u_w", 20)
        cc_w   = _pair_weight_control("CurrCond weight", "cc_w", 20)
    with cC:
        ct_w   = _pair_weight_control("CurrTemp weight", "ct_w", 20)

    total = geom_w + p_w + u_w + cc_w + ct_w
    remaining = 100 - total

    # Status banner
    if remaining == 0:
        st.success("Remaining points: 0 (weights valid)")
    elif remaining > 0:
        st.info(f"Remaining points: {remaining} (allocate these)")
    else:
        st.markdown(
            f"<div style='padding:8px;border:1px solid #cc0000;background:#ffe6e6;color:#cc0000;'>"
            f"Remaining points: {remaining} (over by {-remaining}). Adjust weights so they sum to 100."
            f"</div>",
            unsafe_allow_html=True,
        )

    return {
        "Geomorphic_weight": geom_w,
        "PScore_Weight": p_w,
        "UScore_Weight": u_w,
        "CurrCond_Weight": cc_w,
        "CurrTemp_Weight": ct_w,
        "remaining": remaining,
    }

# -------------------------
# Scoring logic
# -------------------------
def compute_scores(gdf: gpd.GeoDataFrame, colmap: Dict[str, str], weights: Dict[str, int]) -> gpd.GeoDataFrame:
    """
    Weighted_Score = Geomorphic_w*(Geomorphic/25) + ... (weights sum to 100).
    Weighted_Tier rules depend on Basin.
    """
    geom = ensure_numeric(gdf[colmap["Geomorphic"]], "Geomorphic")
    ps   = ensure_numeric(gdf[colmap["PScore"]],     "PScore")
    us   = ensure_numeric(gdf[colmap["UScore"]],     "UScore")
    cc   = ensure_numeric(gdf[colmap["CurrCond"]],   "CurrCond")
    ct   = ensure_numeric(gdf[colmap["CurrTemp"]],   "CurrTemp")
    basin_col = colmap["Basin"]

    gw = float(weights["Geomorphic_weight"])
    pw = float(weights["PScore_Weight"])
    uw = float(weights["UScore_Weight"])
    cw = float(weights["CurrCond_Weight"])
    tw = float(weights["CurrTemp_Weight"])

    weighted_score = (gw * (geom / 25.0) +
                      pw * (ps   / 25.0) +
                      uw * (us   / 25.0) +
                      cw * (cc   / 25.0) +
                      tw * (ct   / 25.0))

    out = gdf.copy()
    out["Weighted_Score"] = weighted_score.astype("float64")

    def tier_for_row(basin: str, score: float) -> Optional[int]:
        if pd.isna(score) or basin is None:
            return None
        b = str(basin).strip()
        if b == "Upper Grande Ronde":
            if score < 65:
                return 1
            elif score > 85:
                return 3
            else:
                return 2
        elif b == "Catherine Creek":
            if score < 50:
                return 1
            elif score > 75:
                return 3
            else:
                return 2
        else:
            return None

    out["Weighted_Tier"] = pd.Series(
        [tier_for_row(b, s) for b, s in zip(out[basin_col], out["Weighted_Score"])],
        dtype="Int64"
    )
    return out

# -------------------------
# Map rendering by Weighted_Tier
# -------------------------
def create_tier_map(gdf_wgs84: gpd.GeoDataFrame) -> folium.Map:
    tier_colors = {1: "#d7191c", 2: "#fdae61", 3: "#ffffbf"}
    default_fill = "#dddddd"

    def style_function(feature):
        t = feature["properties"].get("Weighted_Tier", None)
        color = tier_colors.get(t, default_fill)
        return {"fillColor": color, "color": "#333333", "weight": 0.6, "fillOpacity": 0.8}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf_wgs84.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    fields, aliases = [], []
    for col, alias in (("Weighted_Tier", "Weighted Tier:"), ("Weighted_Score", "Weighted Score:"), ("Basin", "Basin:")):
        if col in gdf_wgs84.columns:
            fields.append(col)
            aliases.append(alias)

    folium.GeoJson(
        gdf_wgs84,
        name="Weighted Priority",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 10px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="margin-bottom:6px;"><b>Weighted Tier (1 = highest priority)</b></div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#d7191c;border:1px solid #444;margin-right:6px;"></span> 1</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#fdae61;border:1px solid #444;margin-right:6px;"></span> 2</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#ffffbf;border:1px solid #444;margin-right:6px;"></span> 3</div>
      <div style="margin-top:6px;color:#666;">Hover to see Weighted Score</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m

# -------------------------
# App
# -------------------------
def main() -> None:
    st.title("Weighted Restoration Prioritization")

    uploaded = st.file_uploader("Upload a GeoPackage (optional)", type=["gpkg"])
    gpkg_default = "data/outputs/base_bsr_with_temp.gpkg"
    gpkg_input = st.text_input("GeoPackage path", value=gpkg_default,
                               help="If not uploading, provide a path relative to app root.")

    if uploaded is not None:
        tmp_path = Path("/tmp") / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        gpkg_path = str(tmp_path)
        st.info(f"Using uploaded file: {tmp_path.name}")
    else:
        gpkg_path = gpkg_input

    # Layers
    try:
        layers = list_layers(gpkg_path)
    except Exception as e:
        st.error(str(e))
        st.stop()
    sel_layer = st.selectbox("Select layer:", layers, index=0)

    # Load
    try:
        gdf_src, gdf_wgs84 = load_layer(gpkg_path, sel_layer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Resolve columns (case-insensitive). If something is missing, show but DO NOT permanently disable button.
    try:
        colmap = resolve_columns_ci(gdf_src)
        missing_msg = None
    except ValueError as e:
        colmap = {}
        missing_msg = str(e)
        st.warning(missing_msg)

    # Weights
    weights = weights_section()
    remaining = weights.pop("remaining")

    # Output target
    st.subheader("Output")
    out_col1, out_col2 = st.columns([2, 1])
    with out_col1:
        out_gpkg = st.text_input(
            "Output GeoPackage path",
            value=str(Path(gpkg_path).with_name(Path(gpkg_path).stem + "_weighted.gpkg")),
            help="Creates/overwrites this GeoPackage."
        )
    with out_col2:
        out_layer = st.text_input("Output layer name", value=sel_layer)

    # --- Enable/disable logic (FIXED) ---
    # Enabled IFF: weights sum to 100 AND required fields resolve successfully.
    disabled_reason = None
    if remaining != 0:
        disabled_reason = "Weights must sum to 100."
    elif missing_msg:
        disabled_reason = missing_msg

    compute_btn = st.button(
        "Compute Weighted_Score & Weighted_Tier and Save",
        type="primary",
        disabled=(disabled_reason is not None),
        help=(disabled_reason or "Writes fields to output GeoPackage/layer.")
    )

    if compute_btn:
        try:
            # Compute on source CRS df
            gdf_scored = compute_scores(gdf_src, colmap, weights)

            # Write to output
            out_path = Path(out_gpkg).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Replace (single-layer overwrite semantics)
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            gdf_scored.to_file(out_path, layer=out_layer, driver="GPKG")
            st.success(f"Wrote fields to {out_path} (layer '{out_layer}').")

            # Join to WGS84 copy for display via _rowid
            w_keep = gdf_scored[["_rowid", "Weighted_Score", "Weighted_Tier"]]
            gdf_disp = gdf_wgs84.drop(columns=[c for c in ["Weighted_Score", "Weighted_Tier"] if c in gdf_wgs84.columns], errors="ignore")
            gdf_disp = gdf_disp.merge(w_keep, on="_rowid", how="left")

            m = create_tier_map(gdf_disp)
            st_folium(m, use_container_width=True, height=700)

        except Exception as e:
            st.error(f"Failed to compute/write: {e}")
            st.stop()
    else:
        # Preview existing if present
        has_existing = all(c in gdf_wgs84.columns for c in ["Weighted_Score", "Weighted_Tier"])
        if has_existing and remaining == 0 and not disabled_reason:
            st.caption("Displaying current Weighted_Tier from the layer (no recompute this run).")
            m = create_tier_map(gdf_wgs84)
            st_folium(m, use_container_width=True, height=700)
        elif disabled_reason:
            st.info(disabled_reason)

if __name__ == "__main__":
    main()
