import h3
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

def latlng_to_h3(lat, lng, resolution=8):
    """
    Converts latitude and longitude to H3 hex ID string.
    """
    return h3.latlng_to_cell(lat, lng, resolution)

def get_h3_centroid(h3_id):
    """
    Returns the centroid coordinates (latitude, longitude) of an H3 hex.
    """
    return h3.cell_to_latlng(h3_id)

def h3_to_boundary_polygon(h3_id):
    """
    Converts an H3 hexagon into a Shapely Polygon (with coordinates flipped to longitude, latitude).
    """
    boundary = h3.cell_to_boundary(h3_id)
    # Shapely expects coordinates as (longitude, latitude)
    flipped_boundary = [(lng, lat) for lat, lng in boundary]
    return Polygon(flipped_boundary)

def get_h3_neighbors(h3_id, k=1):
    """
    Returns a list of neighboring H3 hex IDs at ring distance k (excluding self).
    """
    disk = h3.grid_disk(h3_id, k)
    neighbors = [cell for cell in disk if cell != h3_id]
    return neighbors

def create_geopandas_gdf(h3_ids, data=None):
    """
    Creates a GeoDataFrame from a list/series of H3 hex IDs.
    Optional: Include associated data dictionary or DataFrame.
    """
    polygons = [h3_to_boundary_polygon(h3_id) for h3_id in h3_ids]
    
    if data is not None:
        gdf = gpd.GeoDataFrame(data, geometry=polygons, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(index=h3_ids, geometry=polygons, crs="EPSG:4326")
        gdf['h3_index'] = h3_ids
        
    return gdf

if __name__ == "__main__":
    test_lat, test_lng = 41.8781, -87.6298
    h3_id = latlng_to_h3(test_lat, test_lng)
    print(f"H3 ID for ({test_lat}, {test_lng}): {h3_id}")
    print(f"Centroid: {get_h3_centroid(h3_id)}")
    print(f"Boundary polygon: {h3_to_boundary_polygon(h3_id)}")
    print(f"Neighbors (k=1): {get_h3_neighbors(h3_id, 1)}")
    
    gdf = create_geopandas_gdf([h3_id])
    print(f"GDF Head:\n{gdf.head()}")
