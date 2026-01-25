"""Configuration models for the satellite visual positioning system.

This module contains data classes for configuration and results management,
providing structured access to positioning parameters and output data.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass
class QueryResult:
    """Container for per-query positioning results.

    Attributes:
        query_filename: Name of the query image file.
        best_map_filename: Name of the best matching map tile.
        inliers: Number of RANSAC inliers for best match.
        outliers: Number of RANSAC outliers for best match.
        time: Matching time in seconds.
        gt_latitude: Ground truth latitude.
        gt_longitude: Ground truth longitude.
        predicted_latitude: Predicted latitude from positioning.
        predicted_longitude: Predicted longitude from positioning.
        error_meters: Positioning error in meters.
        success: Whether positioning was successful.
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
    error_meters: float = float("inf")
    success: bool = False


@dataclass
class PositioningConfig:
    """Parsed positioning configuration.

    Attributes:
        matcher_type: Type of matcher to use (lightglue, superglue, etc.).
        device: Compute device (cuda, cpu).
        data_paths: Dictionary of data directory paths.
        preprocessing: Preprocessing configuration.
        camera_model: Camera model parameters for warping.
        matcher_weights: Matcher-specific weight paths.
        matcher_params: Matcher-specific parameters.
        ransac_params: RANSAC configuration.
        positioning_params: Positioning process parameters.
        tile_provider: Tile provider configuration.
    """

    matcher_type: str
    device: str
    data_paths: Dict[str, str]
    preprocessing: Dict[str, Any]
    camera_model: Optional[Dict[str, Any]]
    matcher_weights: Dict[str, Any]
    matcher_params: Dict[str, Any]
    ransac_params: Dict[str, Any]
    positioning_params: Dict[str, Any]
    tile_provider: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> "PositioningConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            PositioningConfig: Parsed PositioningConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        positioning_params = config.get("positioning_params", {})

        return cls(
            matcher_type=config.get("matcher_type", "gim"),
            device=config.get("device", "cuda"),
            data_paths=config.get("data_paths", {}),
            preprocessing=config.get(
                "preprocessing",
                {
                    "enabled": True,
                    "steps": ["resize", "warp"],
                    "resize_target": [1024],
                    "save_processed": True,
                    "target_gimbal_pitch": -90.0,
                    "target_gimbal_roll": 0.0,
                    "target_gimbal_yaw": 0.0,
                    "adaptive_yaw": False,
                },
            ),
            camera_model=config.get(
                "camera_model", {"focal_length": 4.5, "hfov_deg": 82.9}
            ),
            matcher_weights=config.get(
                "matcher_weights",
                {
                    "gim_model_type": "lightglue",
                    "gim_weights_path": "matchers/gim/weights/gim_lightglue_100h.ckpt",
                    "lightglue_features": "superpoint",
                    "loftr_weights_path": "matchers/LoFTR/weights/outdoor_ds.ckpt",
                },
            ),
            matcher_params=config.get("matcher_params", {}),
            ransac_params=config.get(
                "ransac_params",
                {
                    "confidence": 0.999,
                    "max_iter": 10000,
                    "method": "RANSAC",
                    "reproj_threshold": 5.0,
                },
            ),
            positioning_params=positioning_params
            or {"min_inliers_for_success": 75, "save_visualization": True},
            tile_provider=config.get(
                "tile_provider", {"name": "esri", "cache_dir": "satellite"}
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline initialization.

        Returns:
            Dict[str, Any]: Configuration as dictionary.
        """
        return {
            "matcher_type": self.matcher_type,
            "device": self.device,
            "data_paths": self.data_paths,
            "preprocessing": self.preprocessing,
            "camera_model": self.camera_model,
            "matcher_weights": self.matcher_weights,
            "matcher_params": self.matcher_params,
            "ransac_params": self.ransac_params,
            "positioning_params": self.positioning_params,
            "tile_provider": self.tile_provider,
        }
