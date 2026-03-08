"""Configuration models for the satellite visual positioning system.

This module contains data classes for configuration and results management,
providing structured access to positioning parameters and output data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class QueryResult:
    """Container for per-query positioning results.

    Attributes:
        query_filename: Name of the query image file.
        best_map_filename: Name of the best matching map tile.
        inliers: Number of RANSAC inliers for best match.
        outliers: Number of RANSAC outliers for best match.
        matched_features: Number of tentative matched correspondences.
        gt_latitude: Ground truth latitude.
        gt_longitude: Ground truth longitude.
        predicted_latitude: Predicted latitude from positioning.
        predicted_longitude: Predicted longitude from positioning.
        error_meters: Positioning error in meters.
        success: Whether positioning was successful.
        search_radius_m: Search radius in meters.
        candidate_maps: Number of maps in search radius.
        evaluated_maps: Number of maps evaluated.
        search_center_latitude: Center latitude of search.
        search_center_longitude: Center longitude of search.
        failure_reason: Reason for failure, if any.

    """

    query_filename: str
    best_map_filename: Optional[str] = None
    inliers: int = -1
    outliers: int = -1
    matched_features: int = 0
    gt_latitude: Optional[float] = None
    gt_longitude: Optional[float] = None
    predicted_latitude: Optional[float] = None
    predicted_longitude: Optional[float] = None
    error_meters: float = float("inf")
    success: bool = False
    search_radius_m: Optional[float] = None
    candidate_maps: int = 0
    evaluated_maps: int = 0
    search_center_latitude: Optional[float] = None
    search_center_longitude: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class PositioningConfig:
    """Parsed positioning configuration.

    Attributes:
        matcher_type: Type of matcher to use
            (lightglue, gim, loftr, minima, orb).
        device: Compute device (cuda, cpu).
        data_paths: Dictionary of data directory paths.
        preprocessing: Preprocessing configuration.
        camera_model: Optional camera intrinsics; None when unavailable.
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

    @staticmethod
    def _require_mapping(container: Dict[str, Any], key: str) -> Dict[str, Any]:
        """Returns a required mapping value or raises a config error."""
        value = container.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Config key '{key}' must be a mapping.")
        return value

    @staticmethod
    def _require_keys(mapping: Dict[str, Any], keys: List[str], prefix: str) -> None:
        """Ensures required keys are present in a mapping."""
        missing = [key for key in keys if key not in mapping]
        if missing:
            missing_text = ", ".join(f"{prefix}.{key}" for key in missing)
            raise ValueError(f"Missing required config key(s): {missing_text}")

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> None:
        """Validates required config structure and matcher-specific fields."""
        cls._require_keys(
            config,
            [
                "matcher_type",
                "device",
                "preprocessing",
                "matcher_weights",
                "matcher_params",
                "ransac_params",
                "positioning_params",
                "tile_provider",
            ],
            "config",
        )

        preprocessing = cls._require_mapping(config, "preprocessing")
        cls._require_keys(preprocessing, ["save_processed"], "preprocessing")

        matcher_weights = cls._require_mapping(config, "matcher_weights")
        matcher_params = cls._require_mapping(config, "matcher_params")
        ransac_params = cls._require_mapping(config, "ransac_params")
        positioning_params = cls._require_mapping(config, "positioning_params")
        tile_provider = cls._require_mapping(config, "tile_provider")

        cls._require_keys(
            ransac_params,
            ["confidence", "max_iter", "method", "reproj_threshold"],
            "ransac_params",
        )
        cls._require_keys(tile_provider, ["name", "cache_dir"], "tile_provider")
        cls._require_keys(
            positioning_params,
            [
                "min_inliers_for_success",
                "save_visualization",
                "save_frame_sequence",
                "sample_interval",
                "map_context",
                "pair_logging",
                "adaptive_search",
            ],
            "positioning_params",
        )

        map_context = cls._require_mapping(positioning_params, "map_context")
        cls._require_keys(
            map_context,
            ["enabled", "coverage_factor", "max_grid", "save_context_maps"],
            "positioning_params.map_context",
        )
        pair_logging = cls._require_mapping(positioning_params, "pair_logging")
        cls._require_keys(
            pair_logging,
            ["enabled", "save_failed", "save_matched", "max_unique_pairs"],
            "positioning_params.pair_logging",
        )
        adaptive_search = cls._require_mapping(positioning_params, "adaptive_search")
        cls._require_keys(
            adaptive_search,
            [
                "strategy",
                "initial_radius_m",
                "max_radius_m",
                "skip_penalty_m",
                "synthetic_ins_noise_sigma_m",
                "synthetic_ins_noise_max_m",
            ],
            "positioning_params.adaptive_search",
        )

        matcher_type = str(config["matcher_type"]).lower()
        required_weights_by_matcher = {
            "gim": ["gim_model_type", "gim_weights_path"],
            "lightglue": ["lightglue_features"],
            "loftr": ["loftr_weights_path"],
            "minima": [
                "minima_method",
                "minima_weights_dir",
                "minima_xoftr_ckpt",
                "minima_loftr_ckpt",
                "minima_sp_lg_ckpt",
            ],
            "orb": [],
        }
        required_params_by_matcher = {
            "gim": [
                "dkm_h",
                "dkm_w",
                "gim_lightglue_max_keypoints",
                "gim_lightglue_filter_threshold",
                "resize_max",
                "dfactor",
            ],
            "lightglue": ["extractor_max_keypoints"],
            "loftr": ["match_thr", "temp_bug_fix", "resize"],
            "minima": ["match_threshold", "fine_threshold", "loftr_threshold"],
            "orb": [
                "max_features",
                "scale_factor",
                "nlevels",
                "edge_threshold",
                "patch_size",
                "ratio_test",
                "max_matches",
                "fast_threshold",
                "min_descriptor_matches",
                "resize_max",
                "use_clahe",
            ],
        }
        if matcher_type not in required_params_by_matcher:
            raise ValueError(f"Unsupported matcher_type '{matcher_type}'.")

        cls._require_keys(
            matcher_weights,
            required_weights_by_matcher[matcher_type],
            "matcher_weights",
        )
        matcher_param_block = cls._require_mapping(matcher_params, matcher_type)
        cls._require_keys(
            matcher_param_block,
            required_params_by_matcher[matcher_type],
            f"matcher_params.{matcher_type}",
        )

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

        if not isinstance(config, dict):
            raise ValueError("Config root must be a mapping.")
        cls._validate_config(config)
        positioning_params = config["positioning_params"]
        return cls(
            matcher_type=config["matcher_type"],
            device=config["device"],
            data_paths=config.get("data_paths", {}),
            preprocessing=config["preprocessing"],
            camera_model=config.get("camera_model"),
            matcher_weights=config["matcher_weights"],
            matcher_params=config["matcher_params"],
            ransac_params=config["ransac_params"],
            positioning_params=positioning_params,
            tile_provider=config["tile_provider"],
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
