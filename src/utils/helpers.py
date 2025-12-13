"""
Utility helper functions for GPS calculations and localization.

This module provides essential functions for:
- Coordinate system conversions (GPS to pixel)
- Distance calculations (Haversine formula)
- Localization error computation
"""

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Earth's mean radius in meters (WGS84)
EARTH_RADIUS_METERS: float = 6371000.0


def latlon_to_pixel(
    lat: float,
    lon: float,
    map_metadata: Dict[str, Any],
    map_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Convert geographic coordinates to pixel coordinates within a map tile.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        map_metadata: Dictionary containing map corner coordinates.
            Required keys: 'Top_left_lat', 'Bottom_right_lat',
                          'Bottom_right_long', 'Top_left_lon'
        map_shape: Map image dimensions as (height, width).

    Returns:
        Pixel coordinates as numpy array [x, y], or None if coordinates
        are outside the map bounds (with 5% tolerance buffer).
    """
    height, width = map_shape[:2]

    required_keys = ['Top_left_lat', 'Bottom_right_lat', 'Bottom_right_long', 'Top_left_lon']
    if not all(key in map_metadata for key in required_keys):
        return None

    lat_range = map_metadata['Top_left_lat'] - map_metadata['Bottom_right_lat']
    lon_range = map_metadata['Bottom_right_long'] - map_metadata['Top_left_lon']

    if abs(lat_range) < 1e-9 or abs(lon_range) < 1e-9 or width <= 0 or height <= 0:
        return None

    lat_fraction = (map_metadata['Top_left_lat'] - lat) / lat_range
    lon_fraction = (lon - map_metadata['Top_left_lon']) / lon_range

    pixel_x = lon_fraction * width
    pixel_y = lat_fraction * height

    # Allow 5% buffer outside bounds
    buffer = 0.05
    if (-buffer * width <= pixel_x <= (1 + buffer) * width and
            -buffer * height <= pixel_y <= (1 + buffer) * height):
        pixel_x_clipped = max(0.0, min(float(width - 1), pixel_x))
        pixel_y_clipped = max(0.0, min(float(height - 1), pixel_y))
        return np.array([pixel_x_clipped, pixel_y_clipped])

    return None


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate the great-circle distance between two GPS coordinates.

    Uses the Haversine formula to compute the shortest distance over
    the Earth's surface.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in meters. Returns infinity on calculation error.
    """
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
            math.radians, [lat1, lon1, lat2, lon2]
        )

        delta_lon = lon2_rad - lon1_rad
        delta_lat = lat2_rad - lat1_rad

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)

        # Clamp for numerical stability
        a = max(0.0, min(1.0, a))
        c = 2 * math.asin(math.sqrt(a))

        return c * EARTH_RADIUS_METERS
    except Exception:
        return float('inf')


def calculate_predicted_gps(
    map_metadata: Dict[str, Any],
    normalized_center_xy: Optional[Tuple[float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert normalized map coordinates to GPS coordinates.

    Args:
        map_metadata: Dictionary containing map corner coordinates.
        normalized_center_xy: Normalized coordinates (0-1 range) as (x, y).

    Returns:
        Tuple of (predicted_lat, predicted_lon), or (None, None) on error.
    """
    if normalized_center_xy is None:
        return None, None

    center_x_norm, center_y_norm = normalized_center_xy

    required_keys = ['Top_left_lat', 'Bottom_right_lat', 'Bottom_right_long', 'Top_left_lon']
    if not all(key in map_metadata for key in required_keys):
        return None, None

    try:
        lat_diff = map_metadata['Top_left_lat'] - map_metadata['Bottom_right_lat']
        pred_lat = map_metadata['Top_left_lat'] - center_y_norm * lat_diff

        lon_diff = map_metadata['Bottom_right_long'] - map_metadata['Top_left_lon']
        pred_lon = map_metadata['Top_left_lon'] + center_x_norm * lon_diff

        return pred_lat, pred_lon
    except Exception:
        return None, None


def calculate_location_and_error(
    query_metadata: Dict[str, Any],
    map_metadata: Dict[str, Any],
    query_shape: Tuple[int, int, int],
    map_shape: Tuple[int, int, int],
    homography: Optional[np.ndarray]
) -> Optional[Tuple[float, float]]:
    """
    Calculate the normalized center of predicted query location within the map.

    Transforms the query image center through the homography matrix to find
    the corresponding location in the map tile.

    Args:
        query_metadata: Query image metadata dictionary.
        map_metadata: Map image metadata dictionary.
        query_shape: Query image dimensions as (H, W, C).
        map_shape: Map image dimensions as (H, W, C).
        homography: 3x3 homography matrix (Query -> Map), or None.

    Returns:
        Normalized center coordinates (x, y) in range [0, 1], or None on error.

    Note:
        Coordinates outside [0, 1] indicate the predicted location is
        outside the map tile bounds.
    """
    if homography is None:
        return None

    query_h, query_w = query_shape[:2]
    if query_w <= 0 or query_h <= 0:
        print(f"Warning: Invalid query shape {query_shape} for center calculation.")
        return None

    # Transform query center to map coordinates
    query_center = np.array([[[query_w / 2.0, query_h / 2.0]]], dtype=np.float32)

    try:
        predicted_location = cv2.perspectiveTransform(query_center, homography)
        if predicted_location is None:
            raise ValueError("perspectiveTransform returned None")

        pred_pixel = predicted_location[0, 0]
        map_h, map_w = map_shape[:2]

        if map_w <= 0 or map_h <= 0:
            print(f"Warning: Invalid map dimensions ({map_w}x{map_h}).")
            return None

        norm_x = pred_pixel[0] / map_w
        norm_y = pred_pixel[1] / map_h
        normalized_center = (norm_x, norm_y)

        # Warn if significantly outside bounds
        if not (-0.5 <= norm_x <= 1.5 and -0.5 <= norm_y <= 1.5):
            map_filename = map_metadata.get('Filename', 'N/A')
            print(f"Warning: Normalized center ({norm_x:.3f}, {norm_y:.3f}) "
                  f"outside [0,1] bounds for map {map_filename}.")

        return normalized_center

    except (cv2.error, ValueError, TypeError) as e:
        print(f"Warning: cv2.perspectiveTransform failed: {e}")
        return None