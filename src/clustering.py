import hdbscan
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def fit_pickup_clusters(df, sample_size=30000, min_cluster_size=150, min_samples=15):
    """
    Applies HDBSCAN to find spatial pickup hotspot clusters in coordinate data.
    Uses Haversine distance for correct geographic clustering.
    """
    print("Running HDBSCAN clustering on pickup locations...")
    
    # If dataset is large, sample it to speed up clustering and avoid OOM
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()
        
    # HDBSCAN Haversine expects coordinates in (latitude, longitude) converted to radians
    coords_rad = np.radians(df_sample[['pickup_latitude', 'pickup_longitude']].values)
    
    # Earth radius in kilometers to interpret distance metric if needed
    # earth_radius_km = 6371.0088
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(coords_rad)
    df_sample['cluster_label'] = labels
    
    # Summary stats
    unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    print(f"Discovered {unique_clusters} spatial clusters. Noise points: {noise_count} ({noise_count/len(labels)*100:.1f}%)")
    
    # Calculate cluster centroids
    cluster_centroids = {}
    for c_id in set(labels):
        if c_id == -1:
            continue
        c_subset = df_sample[df_sample['cluster_label'] == c_id]
        cluster_centroids[c_id] = {
            'lat': c_subset['pickup_latitude'].mean(),
            'lon': c_subset['pickup_longitude'].mean(),
            'size': len(c_subset)
        }
        
    return clusterer, df_sample, cluster_centroids

def plot_clusters(df_sample, centroids, output_path=None):
    """
    Generates a scatter plot of the pickup clusters and saves it to disk.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot noise in light grey
    noise = df_sample[df_sample['cluster_label'] == -1]
    plt.scatter(noise['pickup_longitude'], noise['pickup_latitude'], c='lightgrey', s=1, alpha=0.3, label='Noise')
    
    # Plot clusters with distinct colors
    clustered = df_sample[df_sample['cluster_label'] != -1]
    if not clustered.empty:
        scatter = plt.scatter(
            clustered['pickup_longitude'], 
            clustered['pickup_latitude'], 
            c=clustered['cluster_label'], 
            cmap='tab20', 
            s=4, 
            alpha=0.6
        )
        
    # Plot centroids
    for c_id, c_data in centroids.items():
        plt.plot(c_data['lon'], c_data['lat'], marker='x', color='red', markersize=10, markeredgewidth=2)
        plt.text(c_data['lon'] + 0.005, c_data['lat'], f"Cluster {c_id}", color='black', fontsize=9, weight='bold')
        
    plt.title("HDBSCAN Spatial Clustering of Taxi Pickups (Chicago)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    from data_loader import get_data
    df = get_data()
    clusterer, df_sample, centroids = fit_pickup_clusters(df)
    plot_clusters(df_sample, centroids, output_path="data/pickup_clusters.png")
