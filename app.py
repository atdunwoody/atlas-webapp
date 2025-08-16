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
# Path utilities & validation
# -------------------------
def resolve_gpkg_path(raw_path: str) -> Path:
    """
    Normalize a path from user input across OSes.
    - Converts Windows backslashes to POSIX.
    - Resolves relative to CWD.
    - Validates existence.
    """
    if not raw_path:
        raise FileNotFoundError("No GeoPackage path provided.")
    p = Path(raw_path.replace("\\", "/")).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"GPKG not found: {p}\n(Working directory: {Path.cwd()})"
        )
    if not p.is_file():
        raise FileNotFoundError(f"Path is not a file: {p}")
    return p


# -------------------------
# Data loading & validation
# -------------------------
@st.cache_data
def list_layers(gpkg_path: str) -> List[str]:
    """
    Return available layers in a GeoPackage.
    Raises FileNotFoundError / ValueError with informative message.
    """
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
    """
    Read a layer from a GPKG and reproject to EPSG:4326 for web mapping.
    Preserves source CRS internally; converts only for display.
    """
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
    """Folium choropleth with grey-below-threshold and white for missing temp medians."""
    vmin, vmax = numeric_min_max(gdf[field])
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def style_function(feature):
        props = feature["properties"]
        v = props.get(field, None)

        # detect missing temp medians
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return float("nan")

        s30_num = to_float(props.get("S30_2040D_median", None))
        s32_num = to_float(props.get("S32_2080D_median", None))
        is_missing_temp = hatch_on_missing and (math.isnan(s30_num) or math.isnan(s32_num))

        if is_missing_temp:
            return {"fillColor": "#ffffff", "color": "#000000", "weight": 1.0, "dashArray": "4 4", "fillOpacity": 0.05}

        v_num = to_float(v)
        if (threshold is not None) and not math.isnan(v_num) and (v_num < threshold):
            return {"fillColor": "#bdbdbd", "color": "#666666", "weight": 0.5, "fillOpacity": 0.8}

        if math.isnan(v_num):
            return {"fillColor": "#eeeeee", "color": "#999999", "weight": 0.5, "fillOpacity": 0.6}

        color = mcolors.rgb2hex(cmap(norm(v_num)))
        return {"fillColor": color, "color": "#000000", "weight": 0.5, "fillOpacity": 0.8}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

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

    bcm.LinearColormap(
        colors=[mcolors.rgb2hex(cmap(x)) for x in [0.0, 0.25, 0.5, 0.75, 1.0]],
        vmin=vmin, vmax=vmax, caption=field
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 10px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="margin-bottom:4px;"><b>Symbology</b></div>
      <div><span style="display:inline-block;width:14px;height:14px;border:1px dashed #000;background:#fff;opacity:0.6;margin-right:6px;"></span> No temperature data</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#bdbdbd;border:1px solid #666;margin-right:6px;"></span> Below threshold</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


# -------------------------
# Streamlit UI
# -------------------------
def main() -> None:
    st.title("BSR Analysis Map Viewer")

    # Optional upload (works on Streamlit Cloud); else use path input
    uploaded = st.file_uploader("Upload a GeoPackage (optional)", type=["gpkg"])
    gpkg_default = "data/base_bsr_with_temp.gpkg"  # use forward slashes for Linux
    gpkg_input = st.text_input("GeoPackage path", value=gpkg_default,
                               help="If not uploading, provide a path relative to app root.")

    gpkg_path: Optional[str] = None
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