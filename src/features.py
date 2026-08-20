import pandas as pd
import numpy as np
import os
from spatial_utils import latlng_to_h3, get_h3_neighbors

def preprocess_and_aggregate(df, resolution=8, time_interval='30min'):
    """
    Maps pickups and dropoffs to H3 cells, floors timestamps, 
    and aggregates demand and supply counts.
    """
    print(f"Aggregating trips to H3 resolution {resolution} and {time_interval} intervals...")
    
    # 1. Map coordinates to H3
    df['pickup_h3'] = df.apply(lambda r: latlng_to_h3(r['pickup_latitude'], r['pickup_longitude'], resolution), axis=1)
    df['dropoff_h3'] = df.apply(lambda r: latlng_to_h3(r['dropoff_latitude'], r['dropoff_longitude'], resolution), axis=1)
    
    # 2. Floor timestamps
    df['pickup_time'] = df['trip_start_timestamp'].dt.floor(time_interval)
    df['dropoff_time'] = df['trip_end_timestamp'].dt.floor(time_interval)
    
    # 3. Aggregate pickups (Demand)
    demand = df.groupby(['pickup_time', 'pickup_h3']).size().reset_index(name='pickup_count')
    demand.rename(columns={'pickup_time': 'timestamp', 'pickup_h3': 'h3_index'}, inplace=True)
    
    # 4. Aggregate dropoffs (Supply proxy)
    supply = df.groupby(['dropoff_time', 'dropoff_h3']).size().reset_index(name='dropoff_count')
    supply.rename(columns={'dropoff_time': 'timestamp', 'dropoff_h3': 'h3_index'}, inplace=True)
    
    # 5. Build dense cartesian grid to handle zero-demand intervals
    # Filter for active H3 cells to keep grid size reasonable (e.g. cells with >= 15 total pickups)
    active_h3 = demand.groupby('h3_index')['pickup_count'].sum()
    active_h3 = active_h3[active_h3 >= 15].index.tolist()
    print(f"Filtering down to {len(active_h3)} active H3 hexagons out of {demand['h3_index'].nunique()} total.")
    
    min_time = df['pickup_time'].min()
    max_time = df['pickup_time'].max()
    time_range = pd.date_range(start=min_time, end=max_time, freq=time_interval)
    
    grid = pd.MultiIndex.from_product([time_range, active_h3], names=['timestamp', 'h3_index']).to_frame().reset_index(drop=True)
    
    # Merge demand and supply
    grid = grid.merge(demand, on=['timestamp', 'h3_index'], how='left')
    grid = grid.merge(supply, on=['timestamp', 'h3_index'], how='left')
    
    grid['pickup_count'] = grid['pickup_count'].fillna(0).astype(int)
    grid['dropoff_count'] = grid['dropoff_count'].fillna(0).astype(int)
    
    return grid, active_h3

def engineer_features(grid):
    """
    Creates temporal, lag, rolling window, and spatial neighbor features.
    All features for timestamp t are based on information from t-1 and earlier.
    """
    print("Engineering features...")
    df = grid.sort_values(by=['h3_index', 'timestamp']).reset_index(drop=True)
    
    # Ensure correct time indexing for sorting-based shift/rolling operations
    # Shift operations are safe because we grouped by h3_index and have a dense, sorted grid
    
    # Lags (using t-1, t-2, t-48)
    df['demand_lag_1'] = df.groupby('h3_index')['pickup_count'].shift(1)
    df['demand_lag_2'] = df.groupby('h3_index')['pickup_count'].shift(2)
    df['demand_lag_48'] = df.groupby('h3_index')['pickup_count'].shift(48)
    
    df['supply_lag_1'] = df.groupby('h3_index')['dropoff_count'].shift(1)
    
    # Rolling windows on lagged demand (to prevent data leakage, roll over shifted demand)
    lagged_demand = df.groupby('h3_index')['pickup_count'].shift(1)
    
    df['demand_roll_mean_3'] = lagged_demand.groupby(df['h3_index']).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['demand_roll_std_3'] = lagged_demand.groupby(df['h3_index']).rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    
    df['demand_roll_mean_6'] = lagged_demand.groupby(df['h3_index']).rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
    df['demand_roll_std_6'] = lagged_demand.groupby(df['h3_index']).rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Cyclical temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Spatial lag features (average neighboring demand at t-1)
    print("Calculating spatial lag features (neighboring cell demand)...")
    
    # Pivot to easily lookup any cell's demand at any timestamp
    pivot_demand = df.pivot(index='timestamp', columns='h3_index', values='pickup_count').fillna(0)
    
    # Shift the pivot table to represent t-1 demand
    pivot_demand_lag1 = pivot_demand.shift(1).fillna(0)
    
    # Get neighbors for each cell
    unique_cells = df['h3_index'].unique()
    neighbor_map = {cell: get_h3_neighbors(cell, 1) for cell in unique_cells}
    
    # Compute mean neighboring demand
    spatial_lag_dict = {}
    for cell in unique_cells:
        neighbors = neighbor_map[cell]
        # Filter neighbors to only those present in our active cells list
        valid_neighbors = [n for n in neighbors if n in pivot_demand_lag1.columns]
        if valid_neighbors:
            spatial_lag_dict[cell] = pivot_demand_lag1[valid_neighbors].mean(axis=1)
        else:
            spatial_lag_dict[cell] = pd.Series(0.0, index=pivot_demand_lag1.index)
            
    # Convert dict to df and melt back to merge
    spatial_lag_df = pd.DataFrame(spatial_lag_dict).reset_index()
    spatial_lag_melted = spatial_lag_df.melt(id_vars='timestamp', var_name='h3_index', value_name='spatial_demand_lag_1')
    
    df = df.merge(spatial_lag_melted, on=['timestamp', 'h3_index'], how='left')
    
    # Drop rows with NaN (due to lags)
    df = df.dropna().reset_index(drop=True)
    
    # Rename target
    df.rename(columns={'pickup_count': 'target_demand'}, inplace=True)
    
    return df

def generate_features_pipeline():
    """
    Orchestrates the entire feature preparation pipeline.
    """
    from data_loader import get_data
    raw_df = get_data()
    
    grid, active_h3 = preprocess_and_aggregate(raw_df)
    features_df = engineer_features(grid)
    
    # Save features and metadata
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    features_path = os.path.join(data_dir, "features.parquet")
    features_df.to_parquet(features_path, index=False)
    
    print(f"Saved feature dataset to {features_path} with shape {features_df.shape}")
    return features_df

if __name__ == "__main__":
    df = generate_features_pipeline()
    print("\nFeature Columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df[['timestamp', 'h3_index', 'target_demand', 'demand_lag_1', 'spatial_demand_lag_1', 'supply_lag_1']].head())
