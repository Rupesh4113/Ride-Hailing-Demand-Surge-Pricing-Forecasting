import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd
import zipfile
import io

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHICAGO_TAXI_API_URL = "https://data.cityofchicago.org/resource/wrvz-psew.json"

def download_chicago_taxi_data(limit=50000):
    """
    Downloads Chicago Taxi Trip records from the Socrata API.
    """
    print("Attempting to download Chicago Taxi Trip data...")
    params = {
        "$limit": limit,
        "$where": "pickup_latitude is not null AND pickup_longitude is not null AND dropoff_latitude is not null AND dropoff_longitude is not null",
        "$order": "trip_start_timestamp DESC"
    }
    
    try:
        response = requests.get(CHICAGO_TAXI_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("No data returned from API")
        
        df = pd.DataFrame(data)
        
        # Rename or typecast columns to expected types
        df['pickup_latitude'] = pd.to_numeric(df['pickup_latitude'])
        df['pickup_longitude'] = pd.to_numeric(df['pickup_longitude'])
        df['dropoff_latitude'] = pd.to_numeric(df['dropoff_latitude'])
        df['dropoff_longitude'] = pd.to_numeric(df['dropoff_longitude'])
        df['fare'] = pd.to_numeric(df['fare'], errors='coerce').fillna(0.0)
        df['trip_miles'] = pd.to_numeric(df['trip_miles'], errors='coerce').fillna(0.0)
        
        # Parse dates
        df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])
        df['trip_end_timestamp'] = pd.to_datetime(df['trip_end_timestamp'])
        
        print(f"Successfully downloaded {len(df)} records from Chicago Taxi API.")
        return df
        
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Falling back to generating realistic synthetic data...")
        return generate_synthetic_taxi_data(num_records=100000)

def generate_synthetic_taxi_data(num_records=100000):
    """
    Generates highly realistic spatio-temporal synthetic taxi trip data for Chicago.
    """
    print(f"Generating {num_records} synthetic taxi trips...")
    np.random.seed(42)
    
    centers = {
        'loop': {'lat': 41.8781, 'lon': -87.6298, 'weight': 0.60, 'spread': 0.02},
        'ohare': {'lat': 41.9742, 'lon': -87.9073, 'weight': 0.20, 'spread': 0.01},
        'midway': {'lat': 41.7868, 'lon': -87.7522, 'weight': 0.10, 'spread': 0.01},
        'north_side': {'lat': 41.9484, 'lon': -87.6553, 'weight': 0.10, 'spread': 0.015}
    }
    
    center_keys = list(centers.keys())
    center_weights = [centers[k]['weight'] for k in center_keys]
    start_centers = np.random.choice(center_keys, size=num_records, p=center_weights)
    
    pickup_lats = []
    pickup_lons = []
    for center_key in start_centers:
        c = centers[center_key]
        pickup_lats.append(np.random.normal(c['lat'], c['spread']))
        pickup_lons.append(np.random.normal(c['lon'], c['spread']))
        
    pickup_lats = np.array(pickup_lats)
    pickup_lons = np.array(pickup_lons)
    
    end_centers = np.random.choice(center_keys, size=num_records, p=[0.5, 0.2, 0.1, 0.2])
    dropoff_lats = []
    dropoff_lons = []
    for center_key in end_centers:
        c = centers[center_key]
        dropoff_lats.append(np.random.normal(c['lat'], c['spread']))
        dropoff_lons.append(np.random.normal(c['lon'], c['spread']))
        
    dropoff_lats = np.array(dropoff_lats)
    dropoff_lons = np.array(dropoff_lons)
    
    start_date = datetime(2025, 10, 1)
    
    hour_probs = np.array([
        0.02, 0.01, 0.005, 0.005, 0.01, 0.03, # 00:00 - 05:00
        0.06, 0.08, 0.09, 0.06, 0.04, 0.04,  # 06:00 - 11:00
        0.05, 0.05, 0.05, 0.06, 0.08, 0.09,  # 12:00 - 17:00
        0.08, 0.06, 0.04, 0.03, 0.025, 0.02  # 18:00 - 23:00
    ])
    hour_probs /= hour_probs.sum()
    
    hours = np.random.choice(24, size=num_records, p=hour_probs)
    minutes = np.random.choice(60, size=num_records)
    days = np.random.randint(0, 14, size=num_records)
    
    timestamps = [start_date + timedelta(days=int(d), hours=int(h), minutes=int(m)) 
                  for d, h, m in zip(days, hours, minutes)]
    
    trip_miles = np.abs(np.random.normal(4.5, 3.0, size=num_records)) + 0.5
    trip_durations_min = trip_miles * np.random.normal(3.0, 0.5, size=num_records) + 2.0
    
    end_timestamps = [t + timedelta(minutes=float(dur)) for t, dur in zip(timestamps, trip_durations_min)]
    
    fares = trip_miles * 2.5 + 3.25 + np.random.normal(0, 2.0, size=num_records)
    fares = np.clip(fares, 3.25, None)
    
    df = pd.DataFrame({
        'trip_start_timestamp': timestamps,
        'trip_end_timestamp': end_timestamps,
        'pickup_latitude': pickup_lats,
        'pickup_longitude': pickup_lons,
        'dropoff_latitude': dropoff_lats,
        'dropoff_longitude': dropoff_lons,
        'fare': fares,
        'trip_miles': trip_miles
    })
    
    print("Synthetic data generation complete.")
    return df

def download_nyc_taxi_zones():
    """
    Downloads and extracts NYC Taxi Zones shapefile from TLC.
    """
    zones_dir = os.path.join(DATA_DIR, "taxi_zones")
    shp_path = os.path.join(zones_dir, "taxi_zones", "taxi_zones.shp")
    
    if os.path.exists(shp_path):
        return shp_path
        
    print("Downloading NYC Taxi Zones shapefile from cloud...")
    os.makedirs(zones_dir, exist_ok=True)
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(zones_dir)
    print("Extracted NYC Taxi Zones shapefile.")
    return shp_path

def download_nyc_tlc_cloud_data(year=2024, month=1):
    """
    Downloads NYC TLC Green Taxi dataset from AWS Cloud and maps to pickup/dropoff coordinates.
    """
    print(f"Downloading NYC TLC Green Taxi data for {year}-{month:02d} from cloud...")
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    
    try:
        df = pd.read_parquet(url)
        print(f"Successfully downloaded {len(df)} records from NYC TLC Cloud.")
        
        # Resolve LocationID to Coordinates
        shp_path = download_nyc_taxi_zones()
        gdf = gpd.read_file(shp_path)
        
        # Calculate centroids correctly using CRS projection properties
        gdf['centroid'] = gdf.geometry.centroid
        gdf_centroids = gdf.set_geometry('centroid').to_crs(epsg=4326)
        
        zone_coords = pd.DataFrame({
            'LocationID': gdf_centroids['LocationID'],
            'lat': gdf_centroids.geometry.y,
            'lon': gdf_centroids.geometry.x
        })
        
        # Match pickup zones
        df = df.merge(
            zone_coords.rename(columns={'LocationID': 'PULocationID', 'lat': 'pickup_latitude', 'lon': 'pickup_longitude'}),
            on='PULocationID',
            how='left'
        )
        # Match dropoff zones
        df = df.merge(
            zone_coords.rename(columns={'LocationID': 'DOLocationID', 'lat': 'dropoff_latitude', 'lon': 'dropoff_longitude'}),
            on='DOLocationID',
            how='left'
        )
        
        # Drop rows with missing spatial information
        df = df.dropna(subset=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'])
        
        # Rename and slice columns to conform to pipeline format
        df.rename(columns={
            'lpep_pickup_datetime': 'trip_start_timestamp',
            'lpep_dropoff_datetime': 'trip_end_timestamp',
            'trip_distance': 'trip_miles',
            'fare_amount': 'fare'
        }, inplace=True)
        
        # Select columns
        keep_cols = [
            'trip_start_timestamp', 'trip_end_timestamp',
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude',
            'fare', 'trip_miles'
        ]
        df = df[keep_cols]
        
        print(f"Formatted and resolved {len(df)} trips with geographic coordinates.")
        return df
        
    except Exception as e:
        print(f"Failed to download/map NYC TLC cloud dataset: {e}")
        print("Falling back to Chicago synthetic data...")
        return generate_synthetic_taxi_data()

def get_data(source='synthetic', force_download=False):
    """
    Retrieves dataset based on source: synthetic, chicago_api, or nyc_cloud.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    cache_path = os.path.join(DATA_DIR, f"taxi_trips_{source}.parquet")
    
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached dataset from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        if source == 'nyc_cloud':
            df = download_nyc_tlc_cloud_data()
        elif source == 'chicago_api':
            df = download_chicago_taxi_data()
        else:
            df = generate_synthetic_taxi_data()
            
        df.to_parquet(cache_path, index=False)
        print(f"Saved dataset to {cache_path}")
        
    return df

if __name__ == "__main__":
    df = get_data(source='nyc_cloud')
    print(df.head())
    print(df.info())
