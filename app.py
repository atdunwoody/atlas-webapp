import math
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import branca.colormap as bcm
import fiona

st.set_page_config(layout="wide")

# -------------------------
# Data loading & validation
# -------------------------
@st.cache_data
def list_layers(gpkg_path: str) -> List[str]:
    """
    Return available layers in a GeoPackage.
    Raises FileNotFoundError / ValueError with informative message.
    """
    p = Path(gpkg_path)
    if not p.exists():
        raise FileNotFoundError(f"GPKG not found: {gpkg_path}")
    try:
        layers = fiona.listlayers(gpkg_path)
    except Exception as e:
        raise ValueError(f"Unable to read layers from GPKG: {e}") from e
    if not layers:
        raise ValueError("No layers found in the provided GeoPackage.")
    return layers

@st.cache_data
def load_layer(gpkg_path: str, layer: str) -> gpd.GeoDataFrame:
    """
    Read a layer from a GPKG and reproject to EPSG:4326 for web mapping.
    Preserves source CRS internally; converts only for display.
    """
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception as e:
        raise ValueError(f"Failed to read layer '{layer}' from GPKG: {e}") from e

    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty.")

    if gdf.crs is None:
        st.warning("Input layer has no CRS. Assuming it is already in EPSG:4326 for display.")
        gdf_wgs84 = gdf
    else:
        gdf_wgs84 = gdf.to_crs(epsg=4326)

    # Ensure the two temperature fields exist (we'll allow plotting if selected exists;
    # null/“hatched” condition uses both if present).
    missing = [f for f in ("S30_2040D_median", "S32_2080D_median") if f not in gdf.columns]
    if missing:
        st.warning(f"Missing expected fields in layer '{layer}': {', '.join(missing)}")

    return gdf_wgs84

def numeric_min_max(series: pd.Series) -> Tuple[float, float]:
    """Return finite (min, max) from a numeric series ignoring NaNs/inf, else raise."""
    s = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    if s.empty:
        raise ValueError("Selected field has no numeric values.")
    return float(s.min()), float(s.max())

# -------------------------
# Map rendering
# -------------------------
def create_map(
    gdf: gpd.GeoDataFrame,
    field: str,
    threshold: Optional[float],
    *,
    hatch_on_missing: bool = True
) -> folium.Map:
    """
    Build a Folium map with:
      - Color by `field`
      - Grey style where value < threshold
      - "Hatched" approximation (dashed outline + low-opacity fill) where either median field is null
    """
    # Color scale limits from non-null numeric values of selected field
    vmin, vmax = numeric_min_max(gdf[field])

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Precompute masks
    s30 = pd.to_numeric(gdf.get("S30_2040D_median"), errors="coerce")
    s32 = pd.to_numeric(gdf.get("S32_2080D_median"), errors="coerce")
    missing_temp_mask = (s30.isna()) | (s32.isna())
    field_vals = pd.to_numeric(gdf[field], errors="coerce")

    def style_function(feature):
        props = feature["properties"]
        v = props.get(field, None)

        # Detect missing temp (either S30 or S32 is NaN)
        is_missing_temp = False
        if hatch_on_missing and ("S30_2040D_median" in props or "S32_2080D_median" in props):
            s30_v = props.get("S30_2040D_median", None)
            s32_v = props.get("S32_2080D_median", None)
            # Treat non-numeric as missing
            try:
                s30_num = float(s30_v) if s30_v is not None else float("nan")
            except Exception:
                s30_num = float("nan")
            try:
                s32_num = float(s32_v) if s32_v is not None else float("nan")
            except Exception:
                s32_num = float("nan")
            is_missing_temp = (math.isnan(s30_num) or math.isnan(s32_num))

        # Hatch approximation (dashed outline + low-opacity light fill)
        if is_missing_temp:
            return {
                "fillColor": "#ffffff",
                "color": "#000000",
                "weight": 1.0,
                "dashArray": "4 4",
                "fillOpacity": 0.05,
            }

        # Grey if below threshold
        try:
            v_num = float(v)
        except (TypeError, ValueError):
            v_num = float("nan")

        if (threshold is not None) and not math.isnan(v_num) and (v_num < threshold):
            return {
                "fillColor": "#bdbdbd",
                "color": "#666666",
                "weight": 0.5,
                "fillOpacity": 0.8,
            }

        # Normal color mapping
        if math.isnan(v_num):
            # If selected field itself is NaN but temp medians exist, show as very light grey
            return {
                "fillColor": "#eeeeee",
                "color": "#999999",
                "weight": 0.5,
                "fillOpacity": 0.6,
            }

        color = mcolors.rgb2hex(cmap(norm(v_num)))
        return {
            "fillColor": color,
            "color": "#000000",
            "weight": 0.5,
            "fillOpacity": 0.8,
        }

    # Map
    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    # Fit to data bounds (lonmin, latmin, lonmax, latmax)
    x_min, y_min, x_max, y_max = gdf.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    # Main layer
    gj = folium.GeoJson(
        gdf,
        name="Polygons",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[fld for fld in [field, "S30_2040D_median", "S32_2080D_median"] if fld in gdf.columns],
            aliases=[f"{field}:", "S30_2040D_median:", "S32_2080D_median:"][:sum([fld in gdf.columns for fld in [field, 'S30_2040D_median', 'S32_2080D_median']])],
            sticky=True,
        ),
    )
    gj.add_to(m)

    # Colorbar legend
    colormap = bcm.LinearColormap(
        colors=[mcolors.rgb2hex(cmap(x)) for x in [0.0, 0.25, 0.5, 0.75, 1.0]],
        vmin=vmin,
        vmax=vmax,
        caption=field,
    )
    colormap.add_to(m)

    # Add simple legend note for hatch & grey
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 10px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="margin-bottom:4px;"><b>Symbology</b></div>
      <div><span style="display:inline-block;width:14px;height:14px;border:1px dashed #000;background:#fff;opacity:0.6;margin-right:6px;"></span> No temperature data (one or both medians missing)</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#bdbdbd;border:1px solid #666;margin-right:6px;"></span> Below threshold</div>
    </div>
    """
    folium.map.CustomPane("legend").add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("BSR Analysis Map Viewer (GPKG)")

    # Path input (editable) — default to a plausible workspace path
    gpkg_path = st.text_input(
        "GeoPackage path",
        value=r"data\outputs\All_Fish_Dist_with_temp.gpkg",
        help="Enter path to a GeoPackage with multiple layers."
    )

    if not gpkg_path:
        st.stop()

    # List layers
    try:
        layers = list_layers(gpkg_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    sel_layer = st.selectbox("Select layer to display:", layers, index=0)

    # Load selected layer
    try:
        gdf = load_layer(gpkg_path, sel_layer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Field selection restricted to the two requested fields if present
    candidate_fields = [f for f in ["S30_2040D_median", "S32_2080D_median"] if f in gdf.columns]
    if not candidate_fields:
        st.error("Neither 'S30_2040D_median' nor 'S32_2080D_median' found in this layer.")
        st.stop()

    sel_field = st.selectbox("Select temperature field:", candidate_fields, index=0)

    # Threshold slider on the selected field
    try:
        vmin, vmax = numeric_min_max(gdf[sel_field])
        default_val = float(pd.to_numeric(gdf[sel_field], errors="coerce").dropna().median())
    except Exception as e:
        st.error(f"Cannot compute slider bounds for '{sel_field}': {e}")
        st.stop()

    threshold = st.slider(
        f"Threshold for {sel_field} (polygons below render grey):",
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(default_val),
        step=(vmax - vmin) / 100.0 if vmax > vmin else 1.0,
    )

    # Render map
    m = create_map(gdf, sel_field, threshold)
    st_folium(m, use_container_width=True, height=700)

if __name__ == "__main__":
    main()
