"""Configuration models for the satellite localization system.
This module contains data classes for configuration and results management.
"""
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml
@dataclass
class QueryResult:
    """Container for per-query localization results.
    Attributes:
        query_filename: Name of the query image file.
        best_map_filename: Name of the best matching map tile.
        inliers: Number of RANSAC inliers for best match.
        outliers: Number of RANSAC outliers for best match.
        time: Matching time in seconds.
        gt_latitude: Ground truth latitude.
        gt_longitude: Ground truth longitude.
        predicted_latitude: Predicted latitude from localization.
        predicted_longitude: Predicted longitude from localization.
        error_meters: Localization error in meters.
        success: Whether localization was successful.
    """
    query_filename: str
    best_map_filename: Optional[str] = None
    inliers: int = -1
    outliers: int = -1
    time: float = 0.0
    gt_latitude: Optional[float] = None
    gt_longitude: Optional[float] = None
    predicted_latitude: Optional[float] = None
    predicted_longitude: Optional[float] = None
    error_meters: float = float('inf')
    success: bool = False
@dataclass
class LocalizationConfig:
    """Parsed localization configuration.
    Attributes:
        matcher_type: Type of matcher to use (lightglue, superglue, etc.).
        device: Compute device (cuda, cpu).
        data_paths: Dictionary of data directory paths.
        preprocessing: Preprocessing configuration.
        camera_model: Camera model parameters for warping.
        matcher_weights: Matcher-specific weight paths.
        matcher_params: Matcher-specific parameters.
        ransac_params: RANSAC configuration.
        localization_params: Localization process parameters.
    """
    matcher_type: str
    device: str
    data_paths: Dict[str, str]
    preprocessing: Dict[str, Any]
    camera_model: Optional[Dict[str, Any]]
    matcher_weights: Dict[str, Any]
    matcher_params: Dict[str, Any]
    ransac_params: Dict[str, Any]
    localization_params: Dict[str, Any]
    @classmethod
    def from_yaml(cls, config_path: str) -> 'LocalizationConfig':
        """Load configuration from YAML file.
        Args:
            config_path: Path to the YAML configuration file.
        Returns:
            Parsed LocalizationConfig instance.
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        localization_params = config.get('localization_params')
        if localization_params is None:
            localization_params = config.get('benchmark_params', {})
            if 'benchmark_params' in config:
                warnings.warn(
                    "'benchmark_params' is deprecated. "
                    "Please use 'localization_params' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
        return cls(
            matcher_type=config.get('matcher_type', 'lightglue'),
            device=config.get('device', 'cuda'),
            data_paths=config.get('data_paths', {}),
            preprocessing=config.get('preprocessing', {'enabled': False}),
            camera_model=config.get('camera_model'),
            matcher_weights=config.get('matcher_weights', {}),
            matcher_params=config.get('matcher_params', {}),
            ransac_params=config.get('ransac_params', {}),
            localization_params=localization_params,
        )
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline initialization.
        Returns:
            Configuration as dictionary.
        """
        return {
            'matcher_type': self.matcher_type,
            'device': self.device,
            'data_paths': self.data_paths,
            'preprocessing': self.preprocessing,
            'camera_model': self.camera_model,
            'matcher_weights': self.matcher_weights,
            'matcher_params': self.matcher_params,
            'ransac_params': self.ransac_params,
            'benchmark_params': self.localization_params,
        }
