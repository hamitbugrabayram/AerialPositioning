"""
Utility modules for the Satellite Visual Localization framework.
This package provides:
- helpers: GPS and localization calculation utilities
- preprocessing: Image preprocessing and camera modeling
- visualization: Match visualization utilities
"""
from src.matchers.base import BaseMatcher, MatchResult
from .helpers import (
    haversine_distance,
    calculate_predicted_gps,
    calculate_location_and_error,
    latlon_to_pixel,
)
from .preprocessing import (
    CameraModel,
    QueryPreprocessor,
    compute_resize_dimensions,
    get_intrinsic_matrix,
    euler_to_rotation_matrix,
)
from .visualization import create_match_visualization
__all__ = [
    'BaseMatcher',
    'MatchResult',
    'haversine_distance',
    'calculate_predicted_gps',
    'calculate_location_and_error',
    'latlon_to_pixel',
    'CameraModel',
    'QueryPreprocessor',
    'compute_resize_dimensions',
    'get_intrinsic_matrix',
    'euler_to_rotation_matrix',
    'create_match_visualization',
]
