import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import folium_static
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functions import *

@st.cache_data
def load_data():
	gdf = gpd.read_feather('./data/df_full.feather').to_crs('EPSG:4326')
	gdf = gdf.reset_index(drop = True)
	
	nodes = gpd.read_feather('./data/nodes.feather').to_crs('EPSG:4326')
	nodes = nodes.reset_index().rename(columns = {'osmid' : 'knooppunt'})
	return (gdf, nodes)

def style_function(feature):
	cmap = cm.RdYlGn  # Choose a continuous colormap (11 colors)
	value = feature['properties']['Score']  # Get the value from your column
	color = mcolors.rgb2hex(cmap(value))  # Map the value to a color
	return {'color': color}

def calculate_new_column(gdf, ovl, bomen, water, monumenten, wegen, parken):
    # Add your calculation logic here, e.g., using min_value and max_value
	gdf['Score'] = (gdf['score_ovl']*ovl + 
					gdf['score_bomen']*bomen + 
					gdf['score_water']*water + 
					gdf['score_monumenten']*monumenten + 
					gdf['score_wegen']*wegen + 
					gdf['score_park']*parken)

	return gdf	

#@st.cache_resource
def create_map(_gdf, _nodes, _df_route = None, route = False):
	m = folium.Map(location=[_gdf['geometry'].centroid.y.mean(), _gdf['geometry'].centroid.x.mean()], zoom_start=14)

	folium.GeoJson(
		_gdf,
		name='Score',
		style_function=style_function).add_to(m)	
	
	folium.GeoJson(
		_nodes,
		name='Nodes',
		marker = folium.CircleMarker(radius = 2, # Radius in metres
                                           weight = 0, #outline weight
                                           fill_color = '#000000', 
                                           fill_opacity = 1),
		tooltip=folium.GeoJsonTooltip(fields=['knooppunt'], labels=True, sticky=True)
		).add_to(m)	
	
	if route: 
		folium.GeoJson(
			_df_route,
			name='Route').add_to(m)	
		
	return m

def calculate_route(gdf, start, end, min, max):

	a = gdf.sort_values(['u','v']).pivot(index = 'u', columns = 'v', values = 'length').fillna(100000).values
	s = gdf.sort_values(['u','v']).pivot(index = 'u', columns = 'v', values = 'Score').fillna(0).values
	s=s*a

	s_norm = s.copy()
	s_norm -= np.min(s_norm)
	s_norm /= np.max(s_norm)
 
	best_solution, distance, score, runtime = looproutes_ant_colony_optimization(a,s,s_norm,start,end,min,max)
	
	#route = route_corine = [2921, 2775, 3095, 3097, 2139, 2068, 2019, 2007, 2006, 1977, 1980, 1969, 1978, 1941, 1926, 1853, 1825, 1847, 2498, 1738, 1681, 1682, 1658, 1623, 1615, 1581, 1558, 1557, 1560, 2537, 1539, 2757, 1523, 1498, 1495, 1497, 1479, 1446, 1441, 1445, 1420, 1438, 1419, 1384, 1346, 1321, 1302, 1263, 1253, 1222, 1220, 1221, 2866, 1276, 1184, 1168, 1183, 1199, 1227, 2655, 1193, 2913]
	df_route = pd.DataFrame({'u' : best_solution})
	df_route['v'] = df_route.u.shift(-1)
	df_route = df_route.dropna()
	df_route = gdf.merge(df_route)
	return df_route

def main():
	# Title and description
	st.title("Loopplezier kaart")
	
	(gdf, nodes) = load_data()
	
	# Sidebar with sliders
	st.sidebar.header("Map Settings")

	df_route = None
	route = False
	calculate_triggered = False
	
	ovl = st.sidebar.number_input("Score openbare verlichting", -10,10,0,1,  key="ovl")
	bomen = st.sidebar.number_input("Score bomen", -10,10,0,1, key="bomen")
	water = st.sidebar.number_input("Score water", -10,10,0,1, key="water")
	monumenten = st.sidebar.number_input("Score monumenten", -10,10,0,1, key="monumenten")
	wegen = st.sidebar.number_input("Score drukke wegen", -10,10,0,1, key="wegen")
	parken = st.sidebar.number_input("Score parken", -10,10,0,1, key="parken")
	calculate_button = st.sidebar.button("Calculate")
	
	start = st.sidebar.number_input("Start knooppunt", 0,3100,0,1,  key="start")
	end = st.sidebar.number_input("Eind knooppunt", 0,3100,0,1,  key="end")
	min_dist = st.sidebar.number_input("Minimale afstand", 0,10000,0,100,  key="min_dist")
	max_dist = st.sidebar.number_input("Maximale afstand", 0,10000,0,100,  key="max_dist")
	add_route = st.sidebar.button("Add route")
		
	if calculate_button:
		gdf = calculate_new_column(gdf, ovl, bomen, water, monumenten, wegen, parken)
		calculate_triggered = True

	if add_route:
		gdf = calculate_new_column(gdf, ovl, bomen, water, monumenten, wegen, parken)
		df_route = calculate_route(gdf, start, end, min_dist, max_dist)
		route = True
		calculate_triggered = True
	
	if calculate_triggered:
		 folium_static(create_map(gdf, nodes, df_route, route), width=800, height=600)

	calculate_triggered = False
	route = False
	
# Run the app
if __name__ == '__main__':
	main()