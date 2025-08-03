import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

@st.cache_data
def load_shapefile(path):
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return gdf

def get_numeric_columns(gdf):
    return [col for col in gdf.columns if gdf[col].dtype.kind in 'ifc']

def create_map(gdf, field):
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=gdf[field].min(), vmax=gdf[field].max())

    def style_function(feature):
        value = feature['properties'][field]
        color = mcolors.rgb2hex(cmap(norm(value)))
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }

    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=12)

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[field], aliases=[f'{field}:'], sticky=True)
    ).add_to(m)

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

    m = create_map(gdf, selected_field)
    st_folium(m, width=1000, height=700)

if __name__ == "__main__":
    main()
