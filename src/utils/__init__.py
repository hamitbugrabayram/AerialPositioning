"""Utility modules for the Satellite Visual Positioning framework.

This package provides:
    - helpers: GPS and positioning calculation utilities (haversine distance,
      coordinate conversion, error computation)
    - preprocessing: Image preprocessing and camera modeling (perspective warp,
      resize, camera intrinsics)
    - visualization: Match visualization utilities for debugging and analysis
    - tile_system: Web Mercator tile utilities for projection and retrieval
    - providers: Satellite imagery providers (ESRI, Google)
"""

from src.matchers.base import BaseMatcher, MatchResult

from .helpers import (calculate_location_and_error, calculate_predicted_gps,
                      haversine_distance, latlon_to_pixel)
from .preprocessing import CameraModel, QueryPreprocessor
from .visualization import create_match_visualization

__all__ = [
    "BaseMatcher",
    "MatchResult",
    "haversine_distance",
    "calculate_predicted_gps",
    "calculate_location_and_error",
    "latlon_to_pixel",
    "CameraModel",
    "QueryPreprocessor",
    "create_match_visualization",
]
