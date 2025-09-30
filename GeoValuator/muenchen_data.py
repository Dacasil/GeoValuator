import osmnx as ox
import os
import pandas as pd
import argparse
import yaml
import numpy as np
from GeoValuator import CONFIG_DIR
from GeoValuator import DATA_DIR
from GeoValuator import FIGURES_DIR
from tqdm import tqdm
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from rtree import index
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
from random import uniform
import time
import folium
from folium.plugins import FastMarkerCluster
import base64, hmac, hashlib
from urllib.parse import urlparse


################ Functions ################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", help="Google Street View API Key", type=str, required=True)
    parser.add_argument("--secret", help="Google Street View API Secret", type=str, required=True)
    parser.add_argument("--config", help="Name of the used config.yml file", type=str, required=True)
    parser.add_argument("--resume", help="Resume from checkpoint", action="store_true")
    return parser.parse_args()

def sign_url(input_url: str, secret: str) -> str:
    """
    Sign a Google Maps API request with a URL Signing Secret
    """
    decoded_secret = base64.urlsafe_b64decode(secret)
    parsed_url = urlparse(input_url)
    url_to_sign = f"{parsed_url.path}?{parsed_url.query}"
    signature = hmac.new(decoded_secret, url_to_sign.encode("utf-8"), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode("utf-8")

    return f"https://{parsed_url.netloc}{url_to_sign}&signature={encoded_signature}"

def create_district_dict(config):
    """Creates a dictionary for easy access on district data"""
    return {district['name']: district for district in config['region']['districts']}

def get_neighborhood(lat, lon, gdf_filtered, ROW_NAME, COLUMN_NAME):
    """
    Returns the Neighborhood for a given latitude and longitude.
    Uses the pre-filtered GeoDataFrame.
    """
    point = Point(lon, lat)
    for _, row in gdf_filtered.iterrows():
        if row[ROW_NAME].contains(point):
            return row[COLUMN_NAME]
    return "Region/District not found in the data"

def save_checkpoint(df, successful_df, failed_coordinates, checkpoint_path):
    """Save the current state to resume later"""
    checkpoint_data = {
        'remaining_coordinates': df.to_dict('records'),
        'successful_downloads': successful_df.to_dict('records'),
        'failed_coordinates': failed_coordinates
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {len(successful_df)} successful, {len(failed_coordinates)} failed")

def load_checkpoint(checkpoint_path):
    """Load previous state if checkpoint exists"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['remaining_coordinates'])
            successful_df = pd.DataFrame(data['successful_downloads'])
            failed_coordinates = data['failed_coordinates']
            print(f"Checkpoint loaded: {len(successful_df)} successful, {len(failed_coordinates)} failed")
            return df, successful_df, failed_coordinates
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, None
    return None, None, None

def download_images_for_coordinates(df, api_key, secret, max_retries=1, checkpoint_path=None, checkpoint_interval=50):
    """
    Download Street View images with checkpointing, signed with API key + Secret
    """
    base_url = 'https://maps.googleapis.com/maps/api/streetview'
    meta_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
    
    successful_downloads = []
    failed_coordinates = []
    
    print("Start image download process")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        lat, lon, district = row['latitude'], row['longitude'], row['district']
        
        # Create district folder
        district_folder = os.path.join(DATA_DIR, "processed", "muenchen_images", district)
        os.makedirs(district_folder, exist_ok=True)
        
        # Skip if already exists
        image_filename = os.path.join(district_folder, f'{lat},{lon}.jpg')
        if os.path.exists(image_filename):
            print(f"Image already exists for {lat},{lon}, skipping")
            successful_downloads.append(row)
            continue
        heading = int(uniform(0, 360))
        
        # Construct unsigned URLs
        params = f"size=640x640&location={lat},{lon}&heading={heading}&pitch=20&key={api_key}"
        unsigned_url = f"{base_url}?{params}"
        
        meta_params = f"location={lat},{lon}&key={api_key}"
        unsigned_meta_url = f"{meta_url}?{meta_params}"
        
        attempts = 0
        success = False

        while attempts < max_retries and not success:
            attempts += 1
            try:
                # Sign metadata URL
                signed_meta_url = sign_url(unsigned_meta_url, secret)
                meta_response = requests.get(signed_meta_url, timeout=30).json()
                status = meta_response.get('status')

                if status == 'REQUEST_DENIED':
                    print(f"API denied request for {lat},{lon}")
                    time.sleep(uniform(0.75, 1.5))
                    continue

                elif status == 'OK':
                    # Sign image URL
                    signed_url = sign_url(unsigned_url, secret)
                    response = requests.get(signed_url, timeout=30)
                    response.raise_for_status()
                    
                    with open(image_filename, "wb") as file:
                        file.write(response.content)
                    success = True
                    successful_downloads.append(row)
                    time.sleep(uniform(0.1, 0.5))

                else:
                    print(f"Attempt {attempts} failed with status: {status} for {lat},{lon}")
                    time.sleep(uniform(1, 2))

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempts} failed with network error: {e} for {lat},{lon}")
                time.sleep(uniform(2, 5))

        if not success:
            failed_coordinates.append((lat, lon, district))
            print(f"Failed to download image for {lat},{lon} in {district} after {max_retries} attempts")
        
        # Save checkpoint
        if checkpoint_path and idx % checkpoint_interval == 0:
            remaining_df = df.iloc[idx+1:].copy()
            successful_df = pd.DataFrame(successful_downloads)
            save_checkpoint(remaining_df, successful_df, failed_coordinates, checkpoint_path)
    
    # Final checkpoint
    if checkpoint_path:
        successful_df = pd.DataFrame(successful_downloads)
        save_checkpoint(pd.DataFrame(), successful_df, failed_coordinates, checkpoint_path)
    
    return pd.DataFrame(successful_downloads), failed_coordinates


def get_more_coordinates_for_district(district_name, district_geometry, num_needed, gdf_filtered, district_dict, scaler, bins, num_bins):
    """
    Get additional coordinates for a district if downloads failed
    """
    try:
        district_G = ox.graph_from_polygon(district_geometry, network_type="drive")
        if len(district_G.nodes) == 0:
            print(f"    No streets found in {district_name} for additional sampling")
            return []
            
        district_Gp = ox.project_graph(district_G)
        district_Gp_undirected = district_Gp.to_undirected()
        
        # Sample additional points
        additional_points = ox.utils_geo.sample_points(district_Gp_undirected, n=num_needed)
        points_wgs = additional_points.to_crs(epsg=4326)
        new_coords = list(zip(points_wgs.geometry.y, points_wgs.geometry.x))
        
        new_rows = []
        for lat, lon in new_coords:
            rent_price = district_dict[district_name]['rent_price']
            log_rent = np.log1p(rent_price)
            normalized_price = scaler.transform([[log_rent]])[0][0]
            bin_id = np.digitize(normalized_price, bins) - 1
            bin_id = max(0, min(bin_id, num_bins-1))
            
            new_rows.append({
                'district': district_name, 
                'latitude': lat, 
                'longitude': lon, 
                'rent_price': rent_price,
                'normalized_price': normalized_price,
                'price_bin': bin_id
            })
        
        return new_rows
        
    except Exception as e:
        print(f"Error getting additional coordinates for {district_name}: {e}")
        return []

def create_coordinate_map(coordinates):
    center_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    center_lon = sum(coord[1] for coord in coordinates) / len(coordinates)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Convert coords to list of lists
    locations = [[lat, lon] for lat, lon in coordinates]
    
    FastMarkerCluster(locations).add_to(m)

    map_dir = os.path.join(FIGURES_DIR, "my_coordinates_map_muen.html")
    m.save(map_dir)


################ Data Extraction ################

#### Get all the district data ####

args = get_args()
API_KEY = args.key
SECRET_KEY = args.secret

CONFIG_PATH = os.path.join(CONFIG_DIR, args.config)

# Load config
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Convert config to a dict
district_dict = create_district_dict(config)
district_names = list(district_dict.keys())

# Get the city Polygon data
data = config['data_name']
DATA_PATH_EXTERNAL = os.path.join(DATA_DIR, "external", data)
ROW_NAME = config['row_name']
COLUMN_NAME = config['districts_name']

# Extract region & data details
Region_config = config['region']
country = Region_config['country']
city = Region_config['city']
districts = [district['name'] for district in Region_config['districts']]
rent_prices = [district['rent_price'] for district in Region_config['districts']]
folder_name = config['data_folder_name']

# Number of coordinates to extract
NUMBER = config['number']

gdf = gpd.read_file(DATA_PATH_EXTERNAL)
if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs('EPSG:4326')

gdf_filtered = gdf[gdf[config['districts_name']].isin(district_names)]

combined_polygon = unary_union(gdf_filtered.geometry)
G = ox.graph_from_polygon(combined_polygon, network_type="drive")
Gp = ox.project_graph(G)
Gp_undirected = Gp.to_undirected()

# Create a temporary dataframe with district info to calculate bins
district_info = []
for district_name in tqdm(district_names):
    district_geometry = gdf_filtered[gdf_filtered[COLUMN_NAME] == district_name].geometry.iloc[0]
    rent_price = district_dict[district_name]['rent_price']
    district_info.append({
        'name': district_name,
        'geometry': district_geometry,
        'rent_price': rent_price
    })

district_df = pd.DataFrame(district_info)


#### Create the bins to order the districts ####

# Log transformation first
district_df['log_rent'] = np.log1p(district_df['rent_price'])
scaler = MinMaxScaler()
district_df['normalized_price'] = scaler.fit_transform(district_df[['log_rent']].values)

# Create bins for stratified sampling
num_bins = 2
district_df['price_bin'] = pd.qcut(district_df['normalized_price'], q=num_bins, labels=False)
# Bin edges
bins = pd.qcut(district_df['normalized_price'], q=num_bins, retbins=True)[1]

# Calculate target samples per bin
target_samples_per_bin = NUMBER // num_bins
remaining_samples = NUMBER % num_bins  # For uneven distribution

# Group districts by their price bin
binned_districts = {}
for bin_id in tqdm(range(num_bins)):
    bin_df = district_df[district_df['price_bin'] == bin_id].copy()
    if len(bin_df) > 0:
        # Create spatial index for this bin
        idx = index.Index()
        spatial_data = []
        for i, (_, row) in enumerate(bin_df.iterrows()):
            idx.insert(i, row.geometry.bounds)
            spatial_data.append({
                'name': row['name'],
                'geometry': row['geometry'],
                'rent_price': row['rent_price'],
                'normalized_price': row['normalized_price']
            })
        binned_districts[bin_id] = {
            'df': bin_df,
            'index': idx,
            'spatial_data': spatial_data,
            'combined_polygon': unary_union(bin_df.geometry.tolist())
        }

#### Plot the bin distribution ####

# Count districts in each bin
bin_counts = district_df['price_bin'].value_counts().sort_index()
bin_ranges = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(len(bins)-1)]

for i, count in bin_counts.items():
    print(f"Bin {i}: {count} districts")

# Create the plot
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(bin_counts)), bin_counts.values, color='lightcoral', edgecolor='black')

# Customize the plot
plt.xlabel('Price Bin Ranges (Normalized Prices)', fontsize=12)
plt.ylabel('Number of Districts', fontsize=12)
plt.title('Distribution of Districts Across Price Bins', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Set x-axis labels to show bin ranges
plt.xticks(range(len(bin_counts)), bin_ranges, rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "bin_plot_muen.pdf"))


#### Get the coordinates from each district ####

# Sample points for each bin
sampled_points = []

for bin_id, bin_data in tqdm(binned_districts.items()):
    if len(bin_data['df']) == 0:
        continue
        
    print(f"Processing bin {bin_id} with {len(bin_data['df'])} districts")
    
    # Calculate samples needed for this bin
    samples_this_bin = target_samples_per_bin
    if bin_id < remaining_samples:  # Distribute leftover samples
        samples_this_bin += 1
    
    # Calculate samples per district in this bin
    districts_in_bin = bin_data['df']
    samples_per_district = max(1, samples_this_bin // len(districts_in_bin))
    remaining_district_samples = samples_this_bin % len(districts_in_bin)
    
    # Sample from each district in this bin
    for district_idx, (_, district_row) in tqdm(enumerate(districts_in_bin.iterrows())):
        district_samples = samples_per_district
        if district_idx < remaining_district_samples:
            district_samples += 1
            
        print(f"Sampling {district_samples} points from {district_row['name']}")
        
        try:
            # Get graph for this specific district
            district_G = ox.graph_from_polygon(district_row['geometry'], network_type="drive")
            if len(district_G.nodes) == 0:
                print(f"No streets found in {district_row['name']}, skipping")
                continue
                
            district_Gp = ox.project_graph(district_G)
            district_Gp_undirected = district_Gp.to_undirected()
            
            # Sample points from this district
            district_points = ox.utils_geo.sample_points(district_Gp_undirected, n=district_samples)
            points_wgs = district_points.to_crs(epsg=4326)
            latlon_coords = list(zip(points_wgs.geometry.y, points_wgs.geometry.x))
            
            for lat, lon in latlon_coords:
                sampled_points.append({
                    'district': district_row['name'], 
                    'latitude': lat, 
                    'longitude': lon, 
                    'rent_price': district_row['rent_price'],
                    'normalized_price': district_row['normalized_price'],
                    'price_bin': bin_id
                })
                    
        except Exception as e:
            print(f"Error sampling from {district_row['name']}: {e}")

# Create the initial dataframe with all sampled coordinates
df = pd.DataFrame(sampled_points)

# Supplement with random sampling if there are not enough coordinates
if len(df) < NUMBER:
    remaining_points = NUMBER - len(df)
    print(f"Supplementing with {remaining_points} random points to reach target")
    additional_points = ox.utils_geo.sample_points(Gp_undirected, n=remaining_points)
    additional_points_wgs = additional_points.to_crs(epsg=4326)
    additional_latlon_coords = list(zip(additional_points_wgs.geometry.y, additional_points_wgs.geometry.x))
    
    for lat, lon in additional_latlon_coords:
        district = get_neighborhood(lat, lon, gdf_filtered, ROW_NAME, COLUMN_NAME)
        if district in district_dict:
            rent_price = district_dict[district]['rent_price']
            log_rent = np.log1p(rent_price)
            normalized_price = scaler.transform([[log_rent]])[0][0]
            
            bin_id = np.digitize(normalized_price, bins) - 1
            bin_id = max(0, min(bin_id, num_bins-1))
            
            df = pd.concat([df, pd.DataFrame([{
                'district': district, 
                'latitude': lat, 
                'longitude': lon, 
                'rent_price': rent_price,
                'normalized_price': normalized_price,
                'price_bin': bin_id
            }])], ignore_index=True)

# Save the coordinate data first
DATA_PATH_INTERIM = os.path.join(DATA_DIR, "interim")
points_file = os.path.join(DATA_PATH_INTERIM, "sampled_points_muen.csv")
df.to_csv(points_file, index=False)

# Create coordinate map
latlon_coords = df[['latitude', 'longitude']].values
create_coordinate_map(latlon_coords)


#### Download the images from Google street view for each coordinate ####

checkpoint_path = os.path.join(DATA_PATH_INTERIM, "download_checkpoint_muen.json")

# Only try to load checkpoint if --resume flag is provided
if args.resume:
    remaining_df, successful_df, failed_coordinates = load_checkpoint(checkpoint_path)
    if remaining_df is not None:
        print("Resuming from checkpoint.")
        df = remaining_df
        initial_successful = successful_df
    else:
        print("No checkpoint found or error loading checkpoint. Starting fresh download.")
        initial_successful = pd.DataFrame()
        failed_coordinates = []
else:
    print("Starting fresh download.")
    initial_successful = pd.DataFrame()
    failed_coordinates = []

# Download images for all coordinates
new_successful_df, new_failed_coordinates = download_images_for_coordinates(
    df, API_KEY, SECRET_KEY, checkpoint_path=checkpoint_path, checkpoint_interval=25
)

# Combine with previous successful downloads if resuming
final_df = pd.concat([initial_successful, new_successful_df], ignore_index=True) if not initial_successful.empty else new_successful_df
failed_coordinates.extend(new_failed_coordinates)

# Remove checkpoint after successful completion
if os.path.exists(checkpoint_path) and len(failed_coordinates) == 0:
    os.remove(checkpoint_path)
    print("Checkpoint removed after successful completion")

# Set maximum retry attempts for failed coordinates
max_retries = 3
retry_count = 0

# Get new images from the districts if coordinates failed
while failed_coordinates and retry_count < max_retries:
    retry_count += 1
    print(f"\nAttempting to get replacement coordinates for {len(failed_coordinates)} failed downloads (Attempt {retry_count}/{max_retries})")
    
    # Group failed coordinates by district
    failed_by_district = {}
    for lat, lon, district in failed_coordinates:
        if district not in failed_by_district:
            failed_by_district[district] = 0
        failed_by_district[district] += 1
    
    replacement_coordinates = []
    for district, count in failed_by_district.items():
        district_geometry = gdf_filtered[gdf_filtered[COLUMN_NAME] == district].geometry.iloc[0]
        new_coords = get_more_coordinates_for_district(
            district, district_geometry, count, gdf_filtered, 
            district_dict, scaler, bins, num_bins
        )
        replacement_coordinates.extend(new_coords)
        print(f"Got {len(new_coords)} replacement coordinates for {district}")
    
    if replacement_coordinates:
        replacement_df = pd.DataFrame(replacement_coordinates)
        replacement_success_df, replacement_failed = download_images_for_coordinates(
            replacement_df, API_KEY, SECRET_KEY, checkpoint_path=checkpoint_path
        )
        final_df = pd.concat([final_df, replacement_success_df], ignore_index=True)
        failed_coordinates = replacement_failed
        print(f"Added {len(replacement_success_df)} successful replacement downloads")
        print(f"Still have {len(failed_coordinates)} failed downloads after attempt {retry_count}")
    else:
        print("No replacement coordinates could be generated")
        break

# Final report
if failed_coordinates:
    print(f"\nWarning: {len(failed_coordinates)} coordinates still failed after {max_retries} attempts")
    # Save final checkpoint with remaining failures
    save_checkpoint(pd.DataFrame(), final_df, failed_coordinates, checkpoint_path)
else:
    print(f"\nSuccess: All coordinates downloaded after {retry_count} attempts")
    # Remove checkpoint if it exists
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


#### Save the final coordinate data ####

# # Save the final dataframe with only successful downloads
final_points_file = os.path.join(DATA_PATH_INTERIM, "sampled_points_muen.csv")
final_df.to_csv(final_points_file, index=False)

# Save district geometry information
district_geo_file = os.path.join(DATA_PATH_INTERIM, "districts_geo_muen.geojson")
gdf_filtered.to_file(district_geo_file, driver='GeoJSON')

# Save configuration information for the visualization script
viz_config = {
    'column_name': COLUMN_NAME,
    'row_name': ROW_NAME,
    'total_points_target': NUMBER,
    'num_bins': num_bins,
    'city': city
}

config_file = os.path.join(DATA_PATH_INTERIM, "visualization_config_muen.json")
with open(config_file, 'w') as f:
    json.dump(viz_config, f, indent=2)
