import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib import cm
from matplotlib.colors import Normalize
import json
import os
import argparse
from GeoValuator import DATA_DIR

def load_visualization_data():
    """Load data saved by the main sampling script"""
    data_dir = os.path.join(DATA_DIR, "interim")
    
    points_file = os.path.join(data_dir, "sampled_points_muen.csv")
    geo_file = os.path.join(data_dir, "districts_geo_muen.geojson")
    config_file = os.path.join(data_dir, "visualization_config_muen.json")
    
    if not all(os.path.exists(f) for f in [points_file, geo_file, config_file]):
        raise FileNotFoundError("Visualization data not found. Run the main sampling script first.")
    
    # Load data
    df = pd.read_csv(points_file)
    gdf = gpd.read_file(geo_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return df, gdf, config

def create_data_distribution_heatmap(df, gdf, config, output_path):
    """
    Creates a heatmap showing data point distribution across districts
    """
    # Count points per district
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'point_count']
    
    # Merge with original geodataframe
    gdf_with_counts = gdf.merge(district_counts, 
                               left_on=config['column_name'], 
                               right_on='district', 
                               how='left')
    
    # Fill NaN values with 0 for districts with no points
    gdf_with_counts['point_count'] = gdf_with_counts['point_count'].fillna(0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Normalize point counts for coloring
    norm = Normalize(vmin=0, vmax=gdf_with_counts['point_count'].max())
    cmap = cm.get_cmap('YlOrRd')  # Yellow to Red colormap
    
    # Plot districts with color based on point count
    gdf_with_counts.plot(column='point_count', 
                        ax=ax, 
                        cmap=cmap, 
                        norm=norm,
                        edgecolor='black', 
                        linewidth=0.5,
                        legend=True,
                        legend_kwds={'label': 'Number of Data Points',
                                    'orientation': 'horizontal',
                                    'shrink': 0.8,
                                    'pad': 0.01})
    
    # Add point count labels to each district
    for _, row in gdf_with_counts.iterrows():
        if row['point_count'] > 0:  # Only label districts with points
            centroid = row['geometry'].centroid
            ax.annotate(text=str(int(row['point_count'])), 
                       xy=(centroid.x, centroid.y),
                       ha='center', 
                       va='center',
                       fontsize=8,
                       fontweight='bold',
                       color='black',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', 
                                alpha=0.7,
                                edgecolor='none'))
    
    # Add basemap
    try:
        # Ensure CRS is in web mercator for basemap compatibility
        if gdf_with_counts.crs != 'EPSG:3857':
            gdf_web_mercator = gdf_with_counts.to_crs(epsg=3857)
        else:
            gdf_web_mercator = gdf_with_counts
            
        ctx.add_basemap(ax, crs=gdf_web_mercator.crs.to_string(), 
                       source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Could not add basemap: {e}, using plain background")
    
    # Customize plot
    ax.set_title(f'Data Point Distribution - {config["city"]}\nTotal Points: {len(df)} (Target: {config["total_points_target"]})', 
                fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    # Add statistics text box
    stats_text = f"""Statistics:
- Total districts: {len(gdf_with_counts)}
- Districts with data: {len(gdf_with_counts[gdf_with_counts['point_count'] > 0])}
- Districts without data: {len(gdf_with_counts[gdf_with_counts['point_count'] == 0])}
- Average points per district: {gdf_with_counts['point_count'].mean():.1f}
- Max points in district: {gdf_with_counts['point_count'].max()}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top',
           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_path}")
    return gdf_with_counts

def create_price_distribution_chart(df, config, output_path):
    """
    Creates a bar chart showing point distribution across price bins
    """
    # Count points per price bin
    bin_counts = df['price_bin'].value_counts().sort_index()
    
    # Get average rent price for each bin
    bin_prices = df.groupby('price_bin')['rent_price'].mean()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar chart for point counts
    bars = ax1.bar(bin_counts.index.astype(str), bin_counts.values, 
                  alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Price Bin')
    ax1.set_ylabel('Number of Points', color='black')
    ax1.set_title('Point Distribution Across Price Bins')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for average rent prices
    ax2 = ax1.twinx()
    ax2.plot(bin_prices.index.astype(str), bin_prices.values, 
            color='red', marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('Average Rent Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Price distribution chart saved to: {output_path}")

def main():
    
    try:
        # Load data
        df, gdf, config = load_visualization_data()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(DATA_DIR, "outputs", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create heatmap
        heatmap_path = os.path.join(output_dir, f"data_distribution_heatmap_muen.pdf")
        create_data_distribution_heatmap(df, gdf, config, heatmap_path)
        
        # Create price distribution chart
        price_chart_path = os.path.join(output_dir, f"price_distribution_muen.pdf")
        create_price_distribution_chart(df, config, price_chart_path)
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()