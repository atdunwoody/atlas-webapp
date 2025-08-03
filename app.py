import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import branca.colormap as bcm

st.set_page_config(layout="wide")

@st.cache_data
def load_shapefile(path):
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return gdf

def get_numeric_columns(gdf):
    return [col for col in gdf.columns if gdf[col].dtype.kind in 'ifc']

def create_map(gdf, field, threshold=None):
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=gdf[field].min(), vmax=gdf[field].max())

    def style_function(feature):
        value = feature['properties'][field]
        if threshold is not None and feature['properties'].get("Ch_Miles", float("inf")) < threshold:
            return {
                'fillColor': 'lightgrey',
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7,
            }
        color = mcolors.rgb2hex(cmap(norm(value)))
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }

    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12)
    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[field], aliases=[f'{field}:'], sticky=True)
    ).add_to(m)

    # Add colorbar legend
    colormap = bcm.LinearColormap(
        colors=[mcolors.rgb2hex(cmap(x)) for x in [0, 0.25, 0.5, 0.75, 1.0]],
        vmin=gdf[field].min(),
        vmax=gdf[field].max(),
        caption=field
    )
    colormap.add_to(m)

    return m

def main():
    st.title("BSR Analysis Map Viewer")

    shapefile_path = "data/BSR_Analysis_DMS.shp"
    gdf = load_shapefile(shapefile_path)

    numeric_fields = get_numeric_columns(gdf)

    if not numeric_fields:
        st.error("No numeric fields found in shapefile.")
        return

    selected_field = st.selectbox("Select a field to visualize:", numeric_fields)

    if "Ch_miles" not in gdf.columns:
        st.warning("'Ch_miles' field not found. Grey-masking will be skipped.")
        threshold = None
    else:
        threshold = st.slider("Threshold for Ch_miles (polygons below this will be grey):",
                              float(gdf["Ch_miles"].min()), float(gdf["Ch_miles"].max()),
                              float(gdf["Ch_miles"].mean()))

    m = create_map(gdf, selected_field, threshold)
    st_folium(m, width=1000, height=700)

if __name__ == "__main__":
    main()
