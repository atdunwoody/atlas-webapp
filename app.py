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
        ch_val = feature['properties'].get('Ch_miles', None)
        if threshold is not None and ch_val is not None and ch_val < threshold:
            color = 'grey'
        else:
            color = mcolors.rgb2hex(cmap(norm(value)))
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }

    m = folium.Map(tiles="CartoDB positron", zoom_start=12)

    geojson = folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[field, "Ch_miles"], aliases=[f'{field}:', 'Ch_miles:'], sticky=True)
    )
    geojson.add_to(m)

    # Auto-zoom to bounds
    bounds = [[gdf.bounds.miny.min(), gdf.bounds.minx.min()],
              [gdf.bounds.maxy.max(), gdf.bounds.maxx.max()]]
    m.fit_bounds(bounds)

    # Add colorbar legend
    colormap = bcm.LinearColormap(
        colors=[mcolors.rgb2hex(cmap(i)) for i in [0, 0.5, 1]],
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

    ch_slider = None
    if 'Ch_miles' in gdf.columns:
        min_val, max_val = float(gdf['Ch_miles'].min()), float(gdf['Ch_miles'].max())
        ch_slider = st.slider("Ch_miles threshold (values below will be greyed out):", 
                              min_value=min_val, max_value=max_val, value=min_val)

    m = create_map(gdf, selected_field, threshold=ch_slider)
    st_folium(m, width=1000, height=700)

if __name__ == "__main__":
    main()
