import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import branca.colormap as bcm

st.set_page_config(layout="wide")

ALLOWED_FIELDS = ("S30_2040D_median", "S32_2080D_median")

import sqlite3
from pathlib import Path
import geopandas as gpd
import numpy as np

def list_gpkg_layers(gpkg_path: str) -> list[str]:
    """
    Return feature layer names from a GeoPackage by querying gpkg_contents.
    No GDAL/Fiona dependency.
    """
    p = Path(gpkg_path)
    if not p.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")
    with sqlite3.connect(str(p)) as conn:
        rows = conn.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        ).fetchall()
    if not rows:
        raise ValueError(f"No feature layers found in {gpkg_path}")
    return [r[0] for r in rows]

def read_layer(gpkg_path: str, layer: str):
    """
    Read a single GPKG layer; prefer pyogrio if present, else fall back.
    Reprojects to EPSG:4326 for web mapping.
    """
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
    except Exception:
        # Fall back to default engine without forcing Fiona import here
        gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty.")
    try:
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        raise RuntimeError("Failed to reproject to EPSG:4326.") from e

def _is_null(v) -> bool:
    """True if v is None/NaN-like without importing pandas in the style callback."""
    return v is None or (isinstance(v, float) and np.isnan(v))

@st.cache_data(show_spinner=False)
def list_layers(gpkg_path: str) -> list[str]:
    """Return layer names in the GeoPackage."""
    try:
        return list(list_gpkg_layers(gpkg_path))
    except Exception as e:
        raise RuntimeError(f"Failed to list layers in '{gpkg_path}'.") from e

@st.cache_data(show_spinner=False)
def load_layer(gpkg_path: str, layer: str) -> gpd.GeoDataFrame:
    """Load a single layer and project to EPSG:4326 for web mapping."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception as e:
        raise RuntimeError(f"Failed to read layer '{layer}' from '{gpkg_path}'.") from e
    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty.")
    try:
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        raise RuntimeError("Failed to reproject layer to EPSG:4326.") from e

def create_map(
    gdf: gpd.GeoDataFrame,
    field: str,
    threshold: float | None,
    *,
    extent=((44.95588104611764, -118.74003621179101), (45.88223081618968, -117.48221656753968))
) -> folium.Map:
    """
    Build a folium map with styling:
      - White if either ALLOWED_FIELDS is null (when present in the layer).
      - Grey if selected field < threshold.
      - Viridis ramp otherwise.
    """
    # Compute normalization on valid values only (ignore NaNs)
    vals = gdf[field].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
    valid = vals[~np.isnan(vals)]
    if valid.size == 0:
        raise ValueError(f"No valid numeric data in field '{field}' to visualize.")
    vmin, vmax = float(valid.min()), float(valid.max())
    if np.isclose(vmin, vmax):
        # Avoid divide-by-zero; make a tiny range
        vmax = vmin + 1e-6

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    has_s30 = ALLOWED_FIELDS[0] in gdf.columns
    has_s32 = ALLOWED_FIELDS[1] in gdf.columns

    def style_function(feature):
        props = feature["properties"]

        # White if either metric is null (only check fields that exist in this layer)
        s30 = props.get(ALLOWED_FIELDS[0]) if has_s30 else None
        s32 = props.get(ALLOWED_FIELDS[1]) if has_s32 else None
        if (has_s30 and _is_null(s30)) or (has_s32 and _is_null(s32)):
            return {"fillColor": "#ffffff", "color": "black", "weight": 0.5, "fillOpacity": 0.8}

        v = props.get(field)
        if _is_null(v):
            # Defensive: selected field missing/NaN -> white
            return {"fillColor": "#ffffff", "color": "black", "weight": 0.5, "fillOpacity": 0.8}

        try:
            v = float(v)
        except Exception:
            return {"fillColor": "#ffffff", "color": "black", "weight": 0.5, "fillOpacity": 0.8}

        # Grey if below threshold
        if threshold is not None and v < threshold:
            return {"fillColor": "lightgrey", "color": "black", "weight": 0.5, "fillOpacity": 0.8}

        color = mcolors.rgb2hex(cmap(norm(v)))
        return {"fillColor": color, "color": "black", "weight": 0.5, "fillOpacity": 0.8}

    m = folium.Map()
    m.fit_bounds(extent)

    tooltip_fields = [f for f in ALLOWED_FIELDS if f in gdf.columns]
    if field not in tooltip_fields:
        tooltip_fields = [field] + tooltip_fields

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[f"{f}:" for f in tooltip_fields],
            sticky=True,
        ),
        name="Polygons",
    ).add_to(m)

    # Colorbar for the selected field
    colormap = bcm.LinearColormap(
        colors=[mcolors.rgb2hex(cmap(x)) for x in [0, 0.25, 0.5, 0.75, 1.0]],
        vmin=vmin,
        vmax=vmax,
        caption=field,
    )
    colormap.add_to(m)

    # Simple HTML legend note for white/grey
    folium.map.Marker(
        [extent[1][0], extent[0][1]],
        icon=folium.DivIcon(
            html=(
                "<div style='background: rgba(255,255,255,0.85); padding:6px; border:1px solid #666; "
                "font-size:12px; line-height:1.3;'>"
                "<b>Legend notes</b><br>"
                "<span style='display:inline-block;width:12px;height:12px;background:#ffffff;border:1px solid #000;margin-right:6px;'></span>"
                "No data in S30/S32<br>"
                "<span style='display:inline-block;width:12px;height:12px;background:lightgrey;border:1px solid #000;margin-right:6px;'></span>"
                "Below threshold"
                "</div>"
            )
        ),
    ).add_to(m)

    return m

def main():
    st.title("BSR Analysis Map Viewer")

    # --- Path input (edit default as needed) ---
    gpkg_path = st.text_input(
        "GeoPackage path:",
        r"data\outputs\All_Fish_Dist_with_temp.gpkg",
        help="Enter the path to the input GeoPackage.",
    )
    if not gpkg_path:
        st.stop()

    # --- Layer selection ---
    try:
        layers = list_layers(gpkg_path)
    except Exception as e:
        st.error(str(e))
        st.stop()
    if not layers:
        st.error("No layers found in the GeoPackage.")
        st.stop()

    layer = st.selectbox("Select a layer:", layers)

    # --- Load layer ---
    try:
        gdf = load_layer(gpkg_path, layer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # --- Field selection (only allow S30/S32 and only those present) ---
    present_fields = [f for f in ALLOWED_FIELDS if f in gdf.columns]
    if not present_fields:
        st.error(
            f"Neither {ALLOWED_FIELDS[0]} nor {ALLOWED_FIELDS[1]} is present in layer '{layer}'."
        )
        st.stop()

    field = st.selectbox("Metric to visualize:", present_fields)

    # --- Threshold slider on selected field ---
    vals = gdf[field].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        st.error(f"Field '{field}' contains no valid numeric data.")
        st.stop()

    min_v, max_v = float(vals.min()), float(vals.max())
    default_v = float(vals.median())
    threshold = st.slider(
        f"Grey features below this {field} value:",
        min_value=min_v,
        max_value=max_v,
        value=default_v,
        step=(max_v - min_v) / 100 if max_v > min_v else 0.01,
    )

    # --- Build and show map ---
    try:
        m = create_map(gdf, field, threshold)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st_folium(m, width=1000, height=700)

if __name__ == "__main__":
    # Assumption: the threshold slider applies to the SELECTED metric (not `Ch_Miles`).
    main()
