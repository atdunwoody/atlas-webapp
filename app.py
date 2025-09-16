import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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
    """Normalize and validate a path across OSes."""
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
    """Return available layers in a GeoPackage."""
    p = resolve_gpkg_path(gpkg_path)
    try:
        layers = fiona.listlayers(str(p))
    except Exception as e:
        raise ValueError(f"Unable to read layers from GPKG: {e}") from e
    if not layers:
        raise ValueError(f"No layers found in GeoPackage: {p}")
    return layers


@st.cache_data
def load_layer(gpkg_path: str, layer: str) -> gpd.GeoDataFrame:
    """Read a layer and reproject to EPSG:4326 for web mapping (preserve original in session)."""
    p = resolve_gpkg_path(gpkg_path)
    try:
        gdf = gpd.read_file(str(p), layer=layer)
    except Exception as e:
        raise ValueError(f"Failed to read layer '{layer}' from {p}: {e}") from e

    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty in {p}.")

    # Keep original CRS for writing; make a WGS84 copy for web map
    if gdf.crs is None:
        st.warning("Input layer has no CRS. Assuming it is EPSG:4326 for display; writing preserves as-is.")
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
# UI helpers
# -------------------------
def _pair_weight_control(label: str, key: str, default: int = 20) -> int:
    """
    A slider (0–100) paired with a number input (box).
    Bi-directional sync via st.session_state.
    """
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    # Initialize once
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
        st.slider(
            label, min_value=0, max_value=100, step=1,
            key=slider_key, on_change=sync_from_slider
        )
    with c2:
        st.number_input(" ", min_value=0, max_value=100, step=1,
                        key=input_key, on_change=sync_from_input, label_visibility="hidden")
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

    # Visual status
    if remaining == 0:
        st.success("Remaining points: 0 (weights valid)")
    elif remaining > 0:
        st.info(f"Remaining points: {remaining} (allocate these)")
    else:
        # Color cue when over budget
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
REQUIRED_FIELDS = ["Geomorphic", "PScore", "UScore", "CurrCond", "CurrTemp", "Basin"]

def compute_scores(gdf: gpd.GeoDataFrame, weights: Dict[str, int]) -> gpd.GeoDataFrame:
    """
    Compute Weighted_Score and Weighted_Tier.
    Weighted_Score = sum( weight * (component/25) ), where weight in [0..100] and sum(weights)=100.
    Tier rules depend on Basin.
    """
    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required fields in layer: {', '.join(missing)}")

    # Coerce numeric fields
    geom = ensure_numeric(gdf["Geomorphic"], "Geomorphic")
    ps   = ensure_numeric(gdf["PScore"], "PScore")
    us   = ensure_numeric(gdf["UScore"], "UScore")
    cc   = ensure_numeric(gdf["CurrCond"], "CurrCond")
    ct   = ensure_numeric(gdf["CurrTemp"], "CurrTemp")

    # Extract weights
    gw = float(weights["Geomorphic_weight"])
    pw = float(weights["PScore_Weight"])
    uw = float(weights["UScore_Weight"])
    cw = float(weights["CurrCond_Weight"])
    tw = float(weights["CurrTemp_Weight"])

    # Weighted score (0–100) assuming component scores are 0–25
    weighted_score = (gw * (geom / 25.0) +
                      pw * (ps   / 25.0) +
                      uw * (us   / 25.0) +
                      cw * (cc   / 25.0) +
                      tw * (ct   / 25.0))

    out = gdf.copy()
    out["Weighted_Score"] = weighted_score.astype("float64")

    # Tier mapping by Basin
    def tier_for_row(basin: str, score: float) -> Optional[int]:
        if pd.isna(score) or basin is None:
            return None
        b = str(basin).strip()
        if b == "Upper Grande Ronde":
            # 1: <65, 2: 65–85, 3: >85 (interpretation: 65 < x < 85 is Tier 2; tie-breaks go to higher tier)
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
            return None  # Unmapped basin -> no tier

    out["Weighted_Tier"] = [
        tier_for_row(basin, sc) for basin, sc in zip(out["Basin"], out["Weighted_Score"])
    ]
    # Use Int64 (nullable) for nicer writes
    out["Weighted_Tier"] = pd.Series(out["Weighted_Tier"], dtype="Int64")
    return out


# -------------------------
# Map rendering by Weighted_Tier
# -------------------------
def create_tier_map(gdf_wgs84: gpd.GeoDataFrame) -> folium.Map:
    """
    Category map for Weighted_Tier with 3 colors (1=highest priority).
    Tooltip shows Weighted_Score.
    """
    # Color map: 1 (high) -> strong red, 2 -> orange, 3 -> pale yellow
    tier_colors = {1: "#d7191c", 2: "#fdae61", 3: "#ffffbf"}
    default_fill = "#dddddd"

    def style_function(feature):
        props = feature["properties"]
        tier = props.get("Weighted_Tier", None)
        color = tier_colors.get(tier, default_fill)
        return {"fillColor": color, "color": "#333333", "weight": 0.6, "fillOpacity": 0.8}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf_wgs84.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    # Tooltip content
    tooltip_fields = []
    tooltip_aliases = []

    if "Weighted_Tier" in gdf_wgs84.columns:
        tooltip_fields.append("Weighted_Tier")
        tooltip_aliases.append("Weighted Tier:")
    if "Weighted_Score" in gdf_wgs84.columns:
        tooltip_fields.append("Weighted_Score")
        tooltip_aliases.append("Weighted Score:")

    if "Basin" in gdf_wgs84.columns:
        tooltip_fields.append("Basin")
        tooltip_aliases.append("Basin:")

    gj = folium.GeoJson(
        gdf_wgs84,
        name="Weighted Priority",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True),
    )
    gj.add_to(m)

    # Legend
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
# Streamlit UI
# -------------------------
def main() -> None:
    st.title("Weighted Restoration Prioritization")

    uploaded = st.file_uploader("Upload a GeoPackage (optional)", type=["gpkg"])
    gpkg_default = "data/outputs/base_bsr_with_temp_scaled_joined.gpkg"
    gpkg_input = st.text_input(
        "GeoPackage path",
        value=gpkg_default,
        help="If not uploading, provide a path relative to app root."
    )

    if uploaded is not None:
        tmp_path = Path(st.secrets.get("TMP_DIR", "/tmp")) / uploaded.name  # falls back to /tmp
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        gpkg_path = str(tmp_path)
        st.info(f"Using uploaded file: {tmp_path.name}")
    else:
        gpkg_path = gpkg_input

    # List layers
    try:
        layers = list_layers(gpkg_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    sel_layer = st.selectbox("Select layer:", layers, index=0)

    # Load selected layer
    try:
        gdf_src, gdf_wgs84 = load_layer(gpkg_path, sel_layer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Weights
    weights = weights_section()
    remaining = weights.pop("remaining")

    # Output GPKG target
    st.subheader("Output")
    out_col1, out_col2 = st.columns([2, 1])
    with out_col1:
        out_gpkg = st.text_input(
            "Output GeoPackage path",
            value=str(Path(gpkg_path).with_name(Path(gpkg_path).stem + "_weighted.gpkg")),
            help="Will create or update this GeoPackage."
        )
    with out_col2:
        out_layer = st.text_input(
            "Output layer name",
            value=sel_layer,
            help="Layer to write (overwrites if it already exists)."
        )

    # Compute & save
    disabled_reason = None
    if remaining != 0:
        disabled_reason = "Weights must sum to 100."
    else:
        # Check required fields existence before enabling
        missing = [f for f in REQUIRED_FIELDS if f not in gdf_src.columns]
        if missing:
            disabled_reason = f"Missing fields: {', '.join(missing)}"

    compute_btn = st.button(
        "Compute Weighted_Score & Weighted_Tier and Save",
        type="primary",
        disabled=disabled_reason is not None,
        help=disabled_reason or "Writes fields to output GeoPackage/layer."
    )

    if compute_btn:
        try:
            # Compute on source CRS dataframe for writing
            gdf_scored = compute_scores(gdf_src, weights)

            # Write to output GPKG
            out_path = Path(out_gpkg).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing layer if present (overwrite semantics)
            mode = "w"
            if out_path.exists():
                # If the container exists and has the layer, we will overwrite by writing to new file
                # using driver='GPKG' with mode='w' only if creating fresh; otherwise, we need to drop layer.
                # Simplest: if file exists and we want to overwrite the layer, we can write to a temp then replace.
                # For clarity here, we remove the whole file when writing a new single layer.
                out_path.unlink(missing_ok=True)
            gdf_scored.to_file(out_path, layer=out_layer, driver="GPKG")

            st.success(f"Wrote fields to {out_path} (layer '{out_layer}').")

            # For map display: use WGS84 version with the new fields merged for visualization
            gdf_wgs84_disp = gdf_wgs84.copy()
            gdf_wgs84_disp = gdf_wgs84_disp.drop(columns=[c for c in ["Weighted_Score", "Weighted_Tier"] if c in gdf_wgs84_disp.columns], errors="ignore")
            gdf_wgs84_disp = gdf_wgs84_disp.merge(
                gdf_scored[["Weighted_Score", "Weighted_Tier", gdf_scored.index.name or "index"]].reset_index(),
                on=(gdf_scored.index.name or "index"),
                how="left"
            )

            m = create_tier_map(gdf_wgs84_disp)
            st_folium(m, use_container_width=True, height=700)

        except Exception as e:
            st.error(f"Failed to compute/write: {e}")
            st.stop()
    else:
        # If not computed yet (or invalid weights), show guidance and a preview-only map (if fields already exist)
        has_existing = all(c in gdf_wgs84.columns for c in ["Weighted_Score", "Weighted_Tier"])
        if has_existing and remaining == 0:
            st.caption("Displaying current Weighted_Tier from the layer (no recompute this run).")
            m = create_tier_map(gdf_wgs84)
            st_folium(m, use_container_width=True, height=700)
        else:
            st.info("Adjust weights so they sum to 100, then click Compute to create fields and map the tiers.")


if __name__ == "__main__":
    main()
