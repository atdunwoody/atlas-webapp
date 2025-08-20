import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 60)
print("RUNNING UPDATED VERSION - Script last modified: 8/20/2025 3:10 PM")
print("=" * 60)

def analyze_temp_anomalies(input_file, basin_name):
    """
    Analyze temperature anomalies for a given basin.
    
    Parameters:
    input_file (str): Path to input geopackage
    basin_name (str): Name of the basin for identification
    
    Returns:
    gpd.GeoDataFrame: GeoDataFrame with anomaly analysis results
    """
    print(f"\nProcessing {basin_name} basin...")
    
    # Load geopackage
    gdf = gpd.read_file(input_file)
    
    # Keep relevant columns and drop missing values
    df = gdf[['MEDIAN', 'DA_km2', 'elevation', 'geometry']].dropna()
    
    # Add basin identifier
    df['basin'] = basin_name
    
    # Log-transform drainage area (avoid log(0))
    df['log_DA'] = np.log(df['DA_km2'] + 1e-6)
    
    # Define predictors and response
    X = sm.add_constant(df[['log_DA', 'elevation']])
    y = df['MEDIAN']
    
    # Fit linear regression
    model = sm.OLS(y, X).fit()
    
    # Predict expected temperatures
    df['pred_temp'] = model.predict(X)
    
    # Calculate residuals
    df['residual'] = df['MEDIAN'] - df['pred_temp']
    
    # Standardize residuals (z-scores)
    df['resid_z'] = (df['residual'] - df['residual'].mean()) / df['residual'].std()
    
    # Flag anomalies
    df['anomaly'] = np.where(df['resid_z'] > 1, 'Warm anomaly',
                      np.where(df['resid_z'] < -1, 'Cool anomaly', 'Normal')) #Anomaly 'boundary' set to +/- 1 STD to show trends. Weren't very many before.
    
    # Print model summary
    print(f"Model summary for {basin_name}:")
    print(model.summary())
    print(f"Number of records processed: {len(df)}")
    print(f"Anomalies found: {(df['anomaly'] != 'Normal').sum()}")
      
    return gpd.GeoDataFrame(df, geometry='geometry')

# Define basin configurations
basins = {
    "CC": r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\inputs\CRITFC FLIR\CCR_2010_FLIR_CRITFC_elev-DA.gpkg",
    "UGR": r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\inputs\CRITFC FLIR\UGR_2010_FLIR_CRITFC_elev-DA.gpkg"
}

# Process all basins and collect results
all_results = []

for basin_name, input_file in basins.items():
    result = analyze_temp_anomalies(input_file, basin_name)
    all_results.append(result)

# Merge all results into a single GeoDataFrame
print("\nMerging results from all basins...")
merged_gdf = pd.concat(all_results, ignore_index=True)

# Output file path for merged results
output_file = r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\outputs\Temp_Anomalies\combined_stream_temp_anomalies.gpkg"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Remove existing file if it exists to avoid overwrite error
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Removed existing file: {output_file}")

# Save merged results
merged_gdf.to_file(output_file, driver="GPKG")

print(f"\nAnalysis complete!")
print(f"Total records processed: {len(merged_gdf)}")
print(f"Results saved to: {output_file}")
print(f"\nSummary by basin:")
for basin in merged_gdf['basin'].unique():
    basin_data = merged_gdf[merged_gdf['basin'] == basin]
    anomalies = (basin_data['anomaly'] != 'Normal').sum()
    print(f"  {basin}: {len(basin_data)} records, {anomalies} anomalies")

# Create visualizations
print("\nCreating visualizations...")

# Load BSR boundaries for map background
bsr_file = r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\inputs\base_bsr.gpkg"
try:
    bsr_gdf = gpd.read_file(bsr_file)
    print(f"Loaded BSR boundaries: {len(bsr_gdf)} features")
except FileNotFoundError:
    print(f"Warning: BSR file not found at {bsr_file}")
    bsr_gdf = None

# Create the anomaly map
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot BSR boundaries first (if available)
if bsr_gdf is not None:
    bsr_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)

# Define colors for anomalies
color_map = {'Warm anomaly': 'red', 'Cool anomaly': 'blue', 'Normal': 'none'}
edge_colors = {'Warm anomaly': 'red', 'Cool anomaly': 'blue', 'Normal': 'lightgray'}

# Plot each anomaly type
for anomaly_type in ['Warm anomaly', 'Cool anomaly', 'Normal']:
    subset = merged_gdf[merged_gdf['anomaly'] == anomaly_type]
    if len(subset) > 0:
        subset.plot(ax=ax, 
                   color=color_map[anomaly_type], 
                   edgecolor=edge_colors[anomaly_type],
                   alpha=0.7 if anomaly_type != 'Normal' else 0.1,
                   linewidth=0.5,
                   markersize=15 if anomaly_type != 'Normal' else 8,
                   label=f'{anomaly_type} (n={len(subset)})')

ax.set_title('Stream Temperature Anomalies (1 STD) by Basin', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.axis('off')

# Set equal aspect ratio for proper geographic display
ax.set_aspect('equal')

# Save the map
map_output = r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\outputs\Temp_Anomalies\anomaly_map.png"
os.makedirs(os.path.dirname(map_output), exist_ok=True)
plt.tight_layout()
plt.savefig(map_output, dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print(f"Map saved to: {map_output}")

# Create linear fit and residuals plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Observed vs Predicted temperatures
for basin in merged_gdf['basin'].unique():
    basin_data = merged_gdf[merged_gdf['basin'] == basin]
    ax1.scatter(basin_data['pred_temp'], basin_data['MEDIAN'], 
               alpha=0.6, label=f'{basin} (n={len(basin_data)})', s=20)

# Add 1:1 line
min_temp = min(merged_gdf['MEDIAN'].min(), merged_gdf['pred_temp'].min())
max_temp = max(merged_gdf['MEDIAN'].max(), merged_gdf['pred_temp'].max())
ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.7, label='1:1 line')
ax1.set_xlabel('Predicted Temperature (°C)')
ax1.set_ylabel('Observed Temperature (°C)')
ax1.set_title('Observed vs Predicted Stream Temperatures')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs Predicted
for basin in merged_gdf['basin'].unique():
    basin_data = merged_gdf[merged_gdf['basin'] == basin]
    ax2.scatter(basin_data['pred_temp'], basin_data['residual'], 
               alpha=0.6, label=f'{basin}', s=20)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
ax2.axhline(y=2*merged_gdf['residual'].std(), color='r', linestyle=':', alpha=0.5, label='±2σ')
ax2.axhline(y=-2*merged_gdf['residual'].std(), color='r', linestyle=':', alpha=0.5)
ax2.set_xlabel('Predicted Temperature (°C)')
ax2.set_ylabel('Residuals (°C)')
ax2.set_title('Residuals vs Predicted Temperatures')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Histogram of standardized residuals
ax3.hist(merged_gdf['resid_z'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.7, label='Mean')
ax3.axvline(x=2, color='r', linestyle=':', alpha=0.7, label='±2σ thresholds')
ax3.axvline(x=-2, color='r', linestyle=':', alpha=0.7)
ax3.set_xlabel('Standardized Residuals (z-score)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Standardized Residuals')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Temperature vs Elevation colored by basin
for basin in merged_gdf['basin'].unique():
    basin_data = merged_gdf[merged_gdf['basin'] == basin]
    scatter = ax4.scatter(basin_data['elevation'], basin_data['MEDIAN'], 
                         c=basin_data['resid_z'], cmap='RdBu_r', 
                         alpha=0.6, s=20, label=f'{basin}')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Standardized Residuals')
ax4.set_xlabel('Elevation (m)')
ax4.set_ylabel('Stream Temperature (°C)')
ax4.set_title('Temperature vs Elevation (colored by residuals)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the analysis plots
plots_output = r"C:\Users\DominiqueShore\Documents\git\Atlas\atlas-webapp\data\outputs\Temp_Anomalies\analysis_plots.png"
plt.savefig(plots_output, dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print(f"Analysis plots saved to: {plots_output}")

print(f"\nVisualizations saved:")
print(f"  Map: {map_output}")
print(f"  Analysis plots: {plots_output}")