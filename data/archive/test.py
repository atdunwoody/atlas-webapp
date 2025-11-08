from pathlib import Path
import geopandas as gpd

# ------------------------
# Input paths
# ------------------------
path1 = Path(r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\outputs\base_bsr_with_temp_scaled_joined.gpkg")
path2 = Path(r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\outputs\base_bsr_scaled_scores.gpkg")

# ------------------------
# Read GeoPackages
# ------------------------
gdf1 = gpd.read_file(path1)
gdf2 = gpd.read_file(path2)

print(gdf1.columns)
# ------------------------
# Select fields to join
# ------------------------
fields_to_join = ["BSR", "CurrCond_RCAT_Linear", "CurrCond_RCAT_Quad "]
df_join = gdf1[fields_to_join].copy()
# ------------------------
# Merge on BSR
# ------------------------
gdf_merged = gdf2.merge(df_join, on="BSR", how="left")

# ------------------------
# Save output (overwrite path2 or specify new output)
# ------------------------
output_path = path2  # or specify new file if you prefer
gdf_merged.to_file(output_path, driver="GPKG")

print(f"Joined fields written to {output_path}")
