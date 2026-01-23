"""Core positioning engine for image matching and coordinate estimation.

This module provides the PositioningEngine class which handles image
preprocessing, matching against satellite tiles, and georeferencing.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import cv2
import numpy as np
import pandas as pd

from src.models.config import PositioningConfig
from src.utils.helpers import PositioningError


class PositioningEngine:
    """Handles the core computational steps of visual positioning."""

    def __init__(self, config: PositioningConfig, pipeline: Any, preprocessor: Any):
        """Initializes the engine with necessary components.

        Args:
            config: The positioning configuration.
            pipeline: The initialized matcher pipeline.
            preprocessor: The image preprocessor.
        """
        self.config = config
        self.pipeline = pipeline
        self.preprocessor = preprocessor
        self._processed_images: Dict[Path, np.ndarray] = {}

        self.haversine_distance = None
        self.calculate_predicted_gps = None
        self.calculate_location_and_error = None

    def inject_helpers(self, haversine, predicted_gps, location_and_error):
        """Injects helper functions to avoid circular imports."""
        self.haversine_distance = haversine
        self.calculate_predicted_gps = predicted_gps
        self.calculate_location_and_error = location_and_error

    def preprocess_query(
        self, query_path: Path, query_row: pd.Series, temp_dir: Optional[Path]
    ) -> Tuple[Path, Optional[Tuple[int, ...]]]:
        """Applies configured preprocessing steps to a query image."""
        if self.preprocessor is None:
            img = cv2.imread(str(query_path))
            shape = img.shape if img is not None else None
            return query_path, shape

        img_original = cv2.imread(str(query_path))
        if img_original is None:
            raise RuntimeError(f"Failed to read image at {query_path}")

        processed = self.preprocessor(img_original, query_row.to_dict())
        shape = processed.shape

        if processed.shape == img_original.shape and np.array_equal(processed, img_original):
            return query_path, shape

        save_processed = bool(self.config.preprocessing.get("save_processed", False))
        if not save_processed:
            self._processed_images[query_path] = processed
            return query_path, shape

        if temp_dir:
            name = f"{Path(query_path.name).stem}_processed{Path(query_path.name).suffix}"
            processed_path = temp_dir / name
            cv2.imwrite(str(processed_path), processed)
            return processed_path, shape

        return query_path, img_original.shape

    def match_query_to_map(
        self,
        query_path: Path,
        query_shape: Tuple[int, ...],
        query_row: pd.Series,
        map_row: pd.Series,
        results_dir: Path,
        min_inliers: int,
        save_viz: bool,
    ) -> Optional[Dict[str, Any]]:
        """Matches a query image against a specific satellite tile."""
        map_filename = str(map_row["Filename"])
        map_path = Path(self.config.data_paths["map_dir"]) / map_filename
        if not map_path.is_file() or self.pipeline is None:
            return None

        map_img = cv2.imread(str(map_path))
        if map_img is None:
            return None

        match_results = self.pipeline.match(query_path, map_path)
        inliers_mask = match_results.get("inliers")
        num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0
        
        ransac_successful = (
            bool(match_results.get("success", False)) and int(num_inliers) >= min_inliers
        )

        try:
            pos_res = self.compute_positioning(
                ransac_successful, match_results.get("homography"), 
                query_row, map_row, query_shape, map_img.shape
            )
        except Exception as e:
            if self.config.matcher_params.get("verbose"):
                print(f"    Tile positioning failed: {e}")
            return None

        if pos_res["success"] or (save_viz and ransac_successful):
            results_dir.mkdir(exist_ok=True)
            if save_viz and ransac_successful:
                self._save_viz(results_dir, query_path, map_path, match_results)

        if not pos_res["success"]:
            return None

        return {
            "map_filename": map_filename,
            "inliers": int(num_inliers),
            "outliers": len(match_results.get("mkpts0", [])) - int(num_inliers),
            "time": match_results.get("time", 0.0),
            "pred_lat": pos_res["pred_lat"],
            "pred_lon": pos_res["pred_lon"],
            "error_meters": pos_res["error_meters"],
        }

    def compute_positioning(
        self,
        ransac_successful: bool,
        homography: Optional[np.ndarray],
        query_row: pd.Series,
        map_row: pd.Series,
        query_shape: Tuple[int, ...],
        map_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """Calculates geographic position from the estimated homography."""
        res = {"pred_lat": None, "pred_lon": None, "error_meters": float("inf"), "success": False}
        
        if not ransac_successful or homography is None:
            return res

        from typing import Callable
        calc_loc = cast(Callable, self.calculate_location_and_error)
        calc_gps = cast(Callable, self.calculate_predicted_gps)
        dist_fn = cast(Callable, self.haversine_distance)
        
        if calc_loc is None or calc_gps is None or dist_fn is None:
            return res

        norm_center = calc_loc(
            query_row.to_dict(), map_row.to_dict(), query_shape, map_shape, homography
        )
        if norm_center is None:
            return res

        plat, plon = calc_gps(map_row.to_dict(), norm_center, map_shape)
        glat, glon = query_row.get("Latitude"), query_row.get("Longitude")

        if plat is not None and glat is not None and glon is not None:
            err = dist_fn(float(glat), float(glon), float(plat), float(plon))
            res.update({"pred_lat": plat, "pred_lon": plon, "error_meters": err, "success": True})
        return res

    def _save_viz(self, results_dir, q_path, m_path, match_results):
        """Saves match visualization."""
        out_path = results_dir / f"{q_path.stem}_vs_{m_path.stem}_match.png"
        if hasattr(self.pipeline, "visualize_matches"):
            try:
                self.pipeline.visualize_matches(
                    q_path, m_path, match_results["mkpts0"], match_results["mkpts1"],
                    match_results["inliers"], out_path, homography=match_results.get("homography")
                )
            except Exception as e:
                print(f"  WARNING: Visualization failed: {e}")
