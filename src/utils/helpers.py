"""Utility helper functions for geographic calculations and visual positioning.

This module provides essential functions for:
    - Coordinate system conversions (geographic to pixel)
    - Distance calculations (Haversine formula)
    - Positioning error computation
    - Mercator projection for Bing Maps tile system
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.utils.tile_system import TileSystem

    _HAS_TILESYSTEM = True
except ImportError:
    try:
        from .tile_system import TileSystem

        _HAS_TILESYSTEM = True
    except ImportError:

        class TileSystem:
            """Fallback TileSystem class when the actual module is not found."""

            @staticmethod
            def latlong_to_pixel_xy(*args, **kwargs):
                """Returns dummy pixel coordinates.

                Returns:
                    Tuple of (0, 0) as fallback coordinates.
                """
                return (0, 0)

            @staticmethod
            def pixel_xy_to_latlong(*args, **kwargs):
                """Returns dummy geographic coordinates.

                Returns:
                    Tuple of (0, 0) as fallback coordinates.
                """
                return (0, 0)

        _HAS_TILESYSTEM = False
        print("WARNING: TileSystem not found. Falling back to linear interpolation.")

EARTH_RADIUS_METERS: float = 6371000.0


def latlon_to_pixel(
    lat: float,
    lon: float,
    map_metadata: Dict[str, Any],
    map_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Converts geographic coordinates to pixel coordinates within a map tile.

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
    required_keys = [
        "Top_left_lat",
        "Bottom_right_lat",
        "Bottom_right_long",
        "Top_left_lon",
    ]

    if not all(key in map_metadata for key in required_keys):
        return None

    lat_range = map_metadata["Top_left_lat"] - map_metadata["Bottom_right_lat"]
    lon_range = map_metadata["Bottom_right_long"] - map_metadata["Top_left_lon"]

    if abs(lat_range) < 1e-9 or abs(lon_range) < 1e-9 or width <= 0 or height <= 0:
        return None

    lat_fraction = (map_metadata["Top_left_lat"] - lat) / lat_range
    lon_fraction = (lon - map_metadata["Top_left_lon"]) / lon_range

    pixel_x = lon_fraction * width
    pixel_y = lat_fraction * height

    buffer = 0.05
    if (
        -buffer * width <= pixel_x <= (1 + buffer) * width
        and -buffer * height <= pixel_y <= (1 + buffer) * height
    ):
        pixel_x_clipped = max(0.0, min(float(width - 1), pixel_x))
        pixel_y_clipped = max(0.0, min(float(height - 1), pixel_y))
        return np.array([pixel_x_clipped, pixel_y_clipped])

    return None


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculates the great-circle distance between two geographic coordinates.

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
        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        a = max(0.0, min(1.0, a))
        c = 2 * math.asin(math.sqrt(a))
        return c * EARTH_RADIUS_METERS
    except Exception:
        return float("inf")


def calculate_predicted_gps(
    map_metadata: Dict[str, Any],
    normalized_center_xy: Optional[Tuple[float, float]],
    map_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Converts normalized map coordinates to geographic coordinates.

    Uses Bing Maps TileSystem for accurate Mercator projection when available.
    Falls back to linear interpolation if TileSystem is not available.

    Args:
        map_metadata: Dictionary containing map corner coordinates and Level.
        normalized_center_xy: Normalized coordinates (0-1 range) as (x, y).
        map_shape: Map image dimensions as (H, W, C) for Mercator calculation.

    Returns:
        Tuple of (predicted_lat, predicted_lon), or (None, None) on error.
    """
    if normalized_center_xy is None:
        return None, None

    center_x_norm, center_y_norm = normalized_center_xy
    required_keys = [
        "Top_left_lat",
        "Bottom_right_lat",
        "Bottom_right_long",
        "Top_left_lon",
    ]

    if not all(key in map_metadata for key in required_keys):
        return None, None

    try:
        if _HAS_TILESYSTEM and "Level" in map_metadata and map_shape is not None:
            return _calculate_gps_mercator(
                map_metadata, normalized_center_xy, map_shape
            )

        lat_diff = map_metadata["Top_left_lat"] - map_metadata["Bottom_right_lat"]
        pred_lat = map_metadata["Top_left_lat"] - center_y_norm * lat_diff
        lon_diff = map_metadata["Bottom_right_long"] - map_metadata["Top_left_lon"]
        pred_lon = map_metadata["Top_left_lon"] + center_x_norm * lon_diff

        return pred_lat, pred_lon
    except Exception as e:
        print(f"WARNING: GPS calculation failed: {e}")
        return None, None


def _calculate_gps_mercator(
    map_metadata: Dict[str, Any],
    normalized_center_xy: Tuple[float, float],
    map_shape: Tuple[int, ...],
) -> Tuple[float, float]:
    """Calculates geographic coordinates using Mercator projection.

    Uses the Bing Maps TileSystem for accurate coordinate conversion.

    Args:
        map_metadata: Dictionary with 'Top_left_lat', 'Top_left_lon', 'Level'.
        normalized_center_xy: Normalized (x, y) in range [0, 1].
        map_shape: Map image dimensions as (H, W, C).

    Returns:
        Tuple of (latitude, longitude) in degrees.
    """
    level = int(map_metadata["Level"])
    nw_lat = float(map_metadata["Top_left_lat"])
    nw_lon = float(map_metadata["Top_left_lon"])

    nw_pixel_x, nw_pixel_y = TileSystem.latlong_to_pixel_xy(nw_lat, nw_lon, level)

    map_h, map_w = map_shape[:2]
    center_x_norm, center_y_norm = normalized_center_xy

    local_pixel_x = center_x_norm * map_w
    local_pixel_y = center_y_norm * map_h

    global_pixel_x = nw_pixel_x + local_pixel_x
    global_pixel_y = nw_pixel_y + local_pixel_y

    pred_lat, pred_lon = TileSystem.pixel_xy_to_latlong(
        global_pixel_x, global_pixel_y, level
    )

    return pred_lat, pred_lon


def is_stable_homography(
    H: Optional[np.ndarray],
    query_shape: Tuple[int, int],
) -> bool:
    """Checks if a homography is physically plausible for this application.

    Args:
        H: 3x3 homography matrix, or None.
        query_shape: (height, width) of the query image.

    Returns:
        True if plausible, False otherwise.
    """
    if H is None or H.shape != (3, 3):
        return False

    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        return False

    det = np.linalg.det(H)
    if abs(det) < 1e-9:
        return False

    h, w = query_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
        -1, 1, 2
    )

    try:
        proj_corners = cv2.perspectiveTransform(corners, H).squeeze()
        area = 0.5 * np.abs(
            np.dot(proj_corners[:, 0], np.roll(proj_corners[:, 1], 1))
            - np.dot(proj_corners[:, 1], np.roll(proj_corners[:, 0], 1))
        )
        if area < 100 or area > (h * w * 25):
            return False
        return True
    except Exception:
        return False


def calculate_location_and_error(
    query_metadata: Dict[str, Any],
    map_metadata: Dict[str, Any],
    query_shape: Tuple[int, ...],
    map_shape: Tuple[int, ...],
    homography: Optional[np.ndarray],
) -> Optional[Tuple[float, float]]:
    """Calculates the normalized center of predicted query location within the map.

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
    """
    h_q, w_q = query_shape[0], query_shape[1]
    h_m, w_m = map_shape[0], map_shape[1]

    if not is_stable_homography(homography, (h_q, w_q)):
        return None

    query_center = np.array([[[w_q / 2.0, h_q / 2.0]]], dtype=np.float32)

    try:
        predicted_location = cv2.perspectiveTransform(query_center, homography)
        if predicted_location is None:
            return None

        pred_pixel = predicted_location[0, 0]
        norm_x = pred_pixel[0] / w_m
        norm_y = pred_pixel[1] / h_m
        normalized_center = (float(norm_x), float(norm_y))

        if not (-0.2 <= norm_x <= 1.2 and -0.2 <= norm_y <= 1.2):
            return None

        return normalized_center
    except (cv2.error, ValueError, TypeError, IndexError):
        return None
