"""Satellite image retrieval module for fetching Bing Maps tiles.
Modified from Aerial-Satellite-Imagery-Retrieval/main.py.
This module provides functions to calculate tile positions and download
satellite imagery based on GPS coordinates.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import requests

def calculate_pixel_position(
    latitude: float, longitude: float, level: int
) -> Tuple[int, int]:
    """Calculates the global pixel position for a given lat/lon and zoom level.
    Args:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
        level: Zoom level.
    Returns:
        A tuple of (pixel_x, pixel_y).
    """
    map_size = 256 * 2**level
    latitude = min(max(latitude, -85.05112878), 85.05112878)
    longitude = min(max(longitude, 0.0), 180.0)
    sin_latitude = math.sin(latitude * math.pi / 180)
    pixel_x = ((longitude + 180) / 360) * map_size
    pixel_y = (
        0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)
    ) * map_size
    pixel_x = min(max(pixel_x, 0), map_size - 1)
    pixel_y = min(max(pixel_y, 0), map_size - 1)
    return (int(pixel_x), int(pixel_y))

def calculate_tile_position(pixel_position: Tuple[int, int]) -> Tuple[int, int]:
    """Determines the tile coordinates containing a specific pixel position.
    Args:
        pixel_position: Global pixel coordinates (x, y).
    Returns:
        A tuple of (tile_x, tile_y).
    """
    tile_x = math.floor(pixel_position[0] / 256.0)
    tile_y = math.floor(pixel_position[1] / 256.0)
    return (int(tile_x), int(tile_y))

def calculate_quad_key(tile_position: Tuple[int, int], level: int) -> str:
    """Calculates the Bing Maps QuadKey for a specific tile.
    Args:
        tile_position: Tile coordinates (x, y).
        level: Zoom level.
    Returns:
        The QuadKey string.
    """
    tile_x = tile_position[0]
    tile_y = tile_position[1]
    quad_key = ""
    i = level
    while i > 0:
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quad_key += str(digit)
        i -= 1
    return quad_key

def pixel_xy_to_latlong(pixel_x: int, pixel_y: int, level: int) -> Tuple[float, float]:
    """Converts global pixel coordinates back to latitude and longitude.
    Args:
        pixel_x: X pixel coordinate.
        pixel_y: Y pixel coordinate.
        level: Zoom level.
    Returns:
        A tuple of (latitude, longitude).
    """
    map_size = 256 * 2**level
    x = (pixel_x / map_size) - 0.5
    y = 0.5 - (pixel_y / map_size)
    latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    longitude = 360 * x
    return latitude, longitude

def get_tile_bounds(
    tile_x: int, tile_y: int, level: int
) -> Tuple[float, float, float, float]:
    """Retrieves geodetic bounds for a specific tile.
    Args:
        tile_x: X tile coordinate.
        tile_y: Y tile coordinate.
        level: Zoom level.
    Returns:
        A tuple of (top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon).
    """
    pixel_x1 = tile_x * 256
    pixel_y1 = tile_y * 256
    pixel_x2 = (tile_x + 1) * 256
    pixel_y2 = (tile_y + 1) * 256
    lat1, lon1 = pixel_xy_to_latlong(pixel_x1, pixel_y1, level)
    lat2, lon2 = pixel_xy_to_latlong(pixel_x2, pixel_y2, level)
    return lat1, lon1, lat2, lon2

def download_image(quad_key: str) -> Optional[np.ndarray]:
    """Downloads a satellite tile image from Bing Maps.
    Args:
        quad_key: The QuadKey for the desired tile.
    Returns:
        The image as a numpy array or None if the download fails.
    """
    url = "http://h0.ortho.tiles.virtualearth.net/tiles/a" + quad_key + ".jpeg?g=131"
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        print(f"Failed to download tile {quad_key}: Status {response.status_code}")
        return None
    except Exception as e:
        print(f"Error downloading tile {quad_key}: {e}")
        return None

def download_tiles(
    upper_left_tile: Tuple[int, int],
    lower_right_tile: Tuple[int, int],
    level: int,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Downloads a range of tiles and saves them to a directory.
    Args:
        upper_left_tile: Starting tile coordinates (x, y).
        lower_right_tile: Ending tile coordinates (x, y).
        level: Zoom level.
        output_dir: Directory to save the images.
    Returns:
        A list of dictionaries containing metadata for each downloaded tile.
    """
    tiles_metadata = []
    x_range = range(upper_left_tile[0], lower_right_tile[0] + 1)
    y_range = range(upper_left_tile[1], lower_right_tile[1] + 1)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tiles from ({upper_left_tile}) to ({lower_right_tile})...")
    for y in y_range:
        for x in x_range:
            quad_key = calculate_quad_key((x, y), level)
            filename = f"tile_{level}_{x}_{y}.jpg"
            file_path = output_dir_path / filename
            if not file_path.exists():
                image = download_image(quad_key)
                if image is not None:
                    if image.shape == (256, 256, 3):
                        cv2.imwrite(str(file_path), image)
                    else:
                        print(
                            f"Warning: Tile {quad_key} has unexpected shape {image.shape}"
                        )
                        image = cv2.resize(image, (256, 256))
                        cv2.imwrite(str(file_path), image)
                else:
                    continue
            lat1, lon1, lat2, lon2 = get_tile_bounds(x, y, level)
            tiles_metadata.append(
                {
                    "Filename": filename,
                    "Top_left_lat": lat1,
                    "Top_left_lon": lon1,
                    "Bottom_right_lat": lat2,
                    "Bottom_right_long": lon2,
                    "TileX": x,
                    "TileY": y,
                    "Level": level,
                    "QuadKey": quad_key,
                }
            )
    return tiles_metadata

def retrieve_map_tiles(
    lat1: float, lon1: float, lat2: float, lon2: float, level: int, output_dir: str
) -> List[Dict[str, Any]]:
    """Retrieves and prepares all satellite tiles covering a geodetic box.
    Args:
        lat1: Top latitude.
        lon1: Left longitude.
        lat2: Bottom latitude.
        lon2: Right longitude.
        level: Zoom level.
        output_dir: Destination directory.
    Returns:
        List of tile metadata.
    """
    print(
        f"Retrieving map tiles for bbox: ({lat1:.4f}, {lon1:.4f}) - "
        f"({lat2:.4f}, {lon2:.4f}) at Level {level}"
    )
    p1 = calculate_pixel_position(lat1, lon1, level)
    p2 = calculate_pixel_position(lat2, lon2, level)
    t1 = calculate_tile_position(p1)
    t2 = calculate_tile_position(p2)
    min_tx = min(t1[0], t2[0])
    max_tx = max(t1[0], t2[0])
    min_ty = min(t1[1], t2[1])
    max_ty = max(t1[1], t2[1])
    return download_tiles((min_tx, min_ty), (max_tx, max_ty), level, output_dir)
