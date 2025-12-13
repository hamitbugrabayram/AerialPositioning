"""
Utility modules for the Satellite Visual Localization framework.

This package provides:
- base_matcher: Abstract base class for matcher pipelines
- helpers: GPS and localization calculation utilities
- preprocessing: Image preprocessing and camera modeling
- visualization: Match visualization utilities
"""

from .base_matcher import BaseMatcher, MatchResult
from .helpers import (
    haversine_distance,
    calculate_predicted_gps,
    calculate_location_and_error,
    latlon_to_pixel,
)
from .preprocessing import (
    CameraModel,
    QueryPreprocessor,
    QueryProcessor,  # Backward compatibility alias
    compute_resize_dimensions,
    get_intrinsic_matrix,
    euler_to_rotation_matrix,
)
from .visualization import create_match_visualization

__all__ = [
    # Base classes
    'BaseMatcher',
    'MatchResult',
    # Helpers
    'haversine_distance',
    'calculate_predicted_gps',
    'calculate_location_and_error',
    'latlon_to_pixel',
    # Preprocessing
    'CameraModel',
    'QueryPreprocessor',
    'QueryProcessor',
    'compute_resize_dimensions',
    'get_intrinsic_matrix',
    'euler_to_rotation_matrix',
    # Visualization
    'create_match_visualization',
]
