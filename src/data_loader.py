import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHICAGO_TAXI_API_URL = "https://data.cityofchicago.org/resource/wrvz-psew.json"

def download_chicago_taxi_data(limit=50000):
    """
    Downloads Chicago Taxi Trip records from the Socrata API.
    """
    print("Attempting to download Chicago Taxi Trip data...")
    # Get recent taxi trips that have coordinate coordinates
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
    
    # Coordinates boundary for Chicago (Loop, O'Hare, Midway, North Side)
    # Chicago Center: 41.8781° N, 87.6298° W
    # O'Hare: 41.9742° N, 87.9073° W
    # Midway: 41.7868° N, 87.7522° W
    
    centers = {
        'loop': {'lat': 41.8781, 'lon': -87.6298, 'weight': 0.60, 'spread': 0.02},
        'ohare': {'lat': 41.9742, 'lon': -87.9073, 'weight': 0.20, 'spread': 0.01},
        'midway': {'lat': 41.7868, 'lon': -87.7522, 'weight': 0.10, 'spread': 0.01},
        'north_side': {'lat': 41.9484, 'lon': -87.6553, 'weight': 0.10, 'spread': 0.015}
    }
    
    # Sample starting centers
    center_keys = list(centers.keys())
    center_weights = [centers[k]['weight'] for k in center_keys]
    start_centers = np.random.choice(center_keys, size=num_records, p=center_weights)
    
    # Generate pickup coords
    pickup_lats = []
    pickup_lons = []
    for center_key in start_centers:
        c = centers[center_key]
        pickup_lats.append(np.random.normal(c['lat'], c['spread']))
        pickup_lons.append(np.random.normal(c['lon'], c['spread']))
        
    pickup_lats = np.array(pickup_lats)
    pickup_lons = np.array(pickup_lons)
    
    # Generate dropoffs - mostly Loop or other hubs
    end_centers = np.random.choice(center_keys, size=num_records, p=[0.5, 0.2, 0.1, 0.2])
    dropoff_lats = []
    dropoff_lons = []
    for center_key in end_centers:
        c = centers[center_key]
        dropoff_lats.append(np.random.normal(c['lat'], c['spread']))
        dropoff_lons.append(np.random.normal(c['lon'], c['spread']))
        
    dropoff_lats = np.array(dropoff_lats)
    dropoff_lons = np.array(dropoff_lons)
    
    # Generate temporal features spanning 14 days
    start_date = datetime(2025, 10, 1)
    
    # Probability of trips by hour (rush hour spikes, night lulls)
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
    
    # Assemble timestamps
    timestamps = [start_date + timedelta(days=int(d), hours=int(h), minutes=int(m)) 
                  for d, h, m in zip(days, hours, minutes)]
    
    # Durations and distances
    trip_miles = np.abs(np.random.normal(4.5, 3.0, size=num_records)) + 0.5
    trip_durations_min = trip_miles * np.random.normal(3.0, 0.5, size=num_records) + 2.0
    
    end_timestamps = [t + timedelta(minutes=float(dur)) for t, dur in zip(timestamps, trip_durations_min)]
    
    # Fares
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

def get_data(force_download=False):
    """
    Retrieves dataset: checks if already cached locally, downloads it, or generates synthetic fallback.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    cache_path = os.path.join(DATA_DIR, "taxi_trips.parquet")
    
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached dataset from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        df = download_chicago_taxi_data()
        df.to_parquet(cache_path, index=False)
        print(f"Saved dataset to {cache_path}")
        
    return df

if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(df.info())
