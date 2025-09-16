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
# Data loading
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
    """Read a layer and reproject to EPSG:4326 for web mapping."""
    p = resolve_gpkg_path(gpkg_path)
    try:
        gdf = gpd.read_file(str(p), layer=layer)
    except Exception as e:
        raise ValueError(f"Failed to read layer '{layer}' from {p}: {e}") from e

    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty in {p}.")

    if gdf.crs is None:
        st.warning("Input layer has no CRS. Assuming it is already in EPSG:4326 for display.")
        gdf_wgs84 = gdf
    else:
        gdf_wgs84 = gdf.to_crs(epsg=4326)

    return gdf_wgs84


# -------------------------
# UI helpers: linked slider + number box with sum constraint
# -------------------------
def _init_weight_state(keys: List[str], default_each: int = 20) -> None:
    """Initialize session_state for weight sliders and numeric boxes."""
    for k in keys:
        s_key, n_key = f"{k}_slider", f"{k}_num"
        if s_key not in st.session_state:
            st.session_state[s_key] = default_each
        if n_key not in st.session_state:
            st.session_state[n_key] = default_each


def _link_slider_to_box(key: str):
    """Callback: when slider moves, update its paired number input."""
    sk, nk = f"{key}_slider", f"{key}_num"
    st.session_state[nk] = st.session_state[sk]


def _link_box_to_slider(key: str):
    """Callback: when number changes, update its paired slider and clamp to [0,100]."""
    sk, nk = f"{key}_slider", f"{key}_num"
    val = max(0, min(100, int(st.session_state[nk])))
    st.session_state[nk] = val
    st.session_state[sk] = val


def weight_inputs() -> Dict[str, int]:
    """
    Render 5 weight controls (slider + numeric input) and return their integer values.
    Sliders range 0–100; total must sum to 100 to enable Compute.
    """
    labels = [
        ("Geomorphic_weight", "Geomorphic"),
        ("PScore_Weight", "PScore"),
        ("UScore_Weight", "UScore"),
        ("CurrCond_Weight", "CurrCond"),
        ("CurrTemp_Weight", "CurrTemp"),
    ]
    keys = [k for k, _ in labels]
    _init_weight_state(keys)

    st.markdown("### Weights (must sum to **100**)")
    cols = st.columns(5)
    for (key, label), col in zip(labels, cols):
        with col:
            st.slider(
                label,
                min_value=0, max_value=100, step=1,
                key=f"{key}_slider",
                on_change=_link_slider_to_box, args=(key,),
                help=f"Weight for {label} (0–100)",
            )
            st.number_input(
                "pts", min_value=0, max_value=100, step=1,
                key=f"{key}_num",
                on_change=_link_box_to_slider, args=(key,),
                label_visibility="collapsed",
                help="Type exact points",
            )

    # Collect values and show remaining
    weights = {k: int(st.session_state[f"{k}_slider"]) for k in keys}
    total = sum(weights.values())
    remaining = 100 - total

    # Status line with color
    if remaining == 0:
        st.success("Remaining points: 0 ✓ (ready to compute)")
    elif remaining > 0:
        st.info(f"Remaining points: {remaining}")
    else:
        st.error(f"Over budget by {-remaining} (reduce weights to proceed)")

    # Simple guidance when over/under
    st.caption("Tip: You can type exact values in the small boxes under each slider.")

    return weights


# -------------------------
# Scoring & tiering
# -------------------------
REQUIRED_FIELDS = ["Geomorphic", "PScore", "UScore", "CurrCond", "CurrTemp", "Basin"]


def compute_weighted_fields(gdf: gpd.GeoDataFrame, weights: Dict[str, int]) -> gpd.GeoDataFrame:
    """
    Add in-memory fields:
      - Weighted_Score = sum(Weight_i * Score_i/25)  (scores are 0–25 per component)
      - Weighted_Tier based on Basin-specific thresholds
    Does not write to disk.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Defensive: coerce numeric score fields; non-numeric -> NaN -> treated as 0
    for f in ["Geomorphic", "PScore", "UScore", "CurrCond", "CurrTemp"]:
        gdf[f] = pd.to_numeric(gdf[f], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Compute Weighted_Score exactly as specified (weights are 0–100 and must sum to 100)
    ws = (
        weights["Geomorphic_weight"] * (gdf["Geomorphic"] / 25.0)
        + weights["PScore_Weight"] * (gdf["PScore"] / 25.0)
        + weights["UScore_Weight"] * (gdf["UScore"] / 25.0)
        + weights["CurrCond_Weight"] * (gdf["CurrCond"] / 25.0)
        + weights["CurrTemp_Weight"] * (gdf["CurrTemp"] / 25.0)
    )
    gdf = gdf.copy()
    gdf["Weighted_Score"] = ws.astype(float).round(2).clip(lower=0.0, upper=100.0)

    # Basin-specific tiering
    def tier_row(basin: str, score: float) -> int:
        b = str(basin).strip()
        s = float(score)
        if b == "Upper Grande Ronde":
            if s < 65:
                return 1
            elif 65 < s < 85:
                return 2
            else:
                return 3
        elif b == "Catherine Creek":
            if s < 50:
                return 1
            elif 50 < s < 75:
                return 2
            else:
                return 3
        else:
            # If other basins occur, default to mid priority to avoid misleading extremes
            return 2

    gdf["Weighted_Tier"] = [tier_row(b, s) for b, s in zip(gdf["Basin"], gdf["Weighted_Score"])]
    return gdf


# -------------------------
# Map rendering (Weighted_Tier)
# -------------------------
def map_weighted_tier(gdf: gpd.GeoDataFrame) -> folium.Map:
    """
    Folium map of Weighted_Tier with:
      - 1 (highest priority) -> red, 2 -> orange, 3 -> green
      - Tooltip shows Basin, Weighted_Tier, Weighted_Score (2 d.p.)
    """
    # Priority color scheme (1 highest → warm; 3 lowest → cool)
    tier_colors = {1: "#b10026", 2: "#fd8d3c", 3: "#31a354"}

    def style_fn(feature):
        t = feature["properties"].get("Weighted_Tier", None)
        color = tier_colors.get(int(t) if t is not None else 2, "#fd8d3c")
        return {"fillColor": color, "color": "#333333", "weight": 0.6, "fillOpacity": 0.8}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    fields = [f for f in ["Basin", "Weighted_Tier", "Weighted_Score"] if f in gdf.columns]
    aliases = [f"{f}:" for f in fields]

    gj = folium.GeoJson(
        gdf,
        name="Priority",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 10px; z-index: 9999;
                background: white; padding: 8px 10px; border: 1px solid #bbb;
                border-radius: 4px; font-size: 12px;">
      <div style="margin-bottom:4px;"><b>Weighted Tier (Priority)</b></div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#b10026;border:1px solid #333;margin-right:6px;"></span> 1 (Highest)</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#fd8d3c;border:1px solid #333;margin-right:6px;"></span> 2</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#31a354;border:1px solid #333;margin-right:6px;"></span> 3 (Lowest)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


# -------------------------
# Streamlit app
# -------------------------
def main() -> None:
    st.title("BSR Weighted Scoring & Priority Map")

    uploaded = st.file_uploader("Upload a GeoPackage (optional)", type=["gpkg"])
    gpkg_default = "data/outputs/base_bsr_with_temp.gpkg"
    gpkg_input = st.text_input(
        "GeoPackage path",
        value=gpkg_default,
        help="If not uploading, provide a path relative to app root."
    )

    if uploaded is not None:
        tmp_path = Path("/tmp") / uploaded.name
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
        gdf = load_layer(gpkg_path, sel_layer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Check required fields present
    missing = [f for f in REQUIRED_FIELDS if f not in gdf.columns]
    if missing:
        st.error(
            "This app expects the following fields to exist in the selected layer:\n\n"
            + ", ".join(REQUIRED_FIELDS)
            + "\n\nMissing: " + ", ".join(missing)
        )
        st.stop()

    # Weight UI
    weights = weight_inputs()
    total = sum(weights.values())
    ready = (total == 100)

    # Compute button (disabled unless weights sum to 100)
    st.markdown("---")
    compute_clicked = st.button(
        "Compute Weighted Score & Map",
        type="primary",
        disabled=not ready,
        help="Enabled when weights sum to exactly 100"
    )

    if compute_clicked and ready:
        try:
            gdf_scored = compute_weighted_fields(gdf, weights)
        except Exception as e:
            st.error(f"Failed to compute weighted fields: {e}")
            st.stop()

        m = map_weighted_tier(gdf_scored)
        st_folium(m, use_container_width=True, height=720)

        with st.expander("Preview of computed attributes (first 10 rows)"):
            st.dataframe(
                gdf_scored[["Basin", "Geomorphic", "PScore", "UScore", "CurrCond", "CurrTemp",
                            "Weighted_Score", "Weighted_Tier"]].head(10)
            )
        st.caption("Note: Fields are added in memory only; the GeoPackage is not modified.")
    else:
        st.info("Adjust weights so the total equals 100, then click **Compute**.")


if __name__ == "__main__":
    main()
