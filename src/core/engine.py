
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
from src.utils.tile_system import TileSystem

from src.utils.logger import get_logger
_logger = get_logger(__name__)


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

        self.haversine_distance = None
        self.calculate_predicted_gps = None
        self.calculate_location_and_error = None
        self._map_context_cache: Dict[str, Dict[str, Any]] = {}
        self._pair_log_cache: set[str] = set()
        self._pair_log_seq_by_query: Dict[str, int] = {}
        self._query_variants_cache: Dict[str, list[Tuple[Path, Tuple[int, ...]]]] = {}

    def _multi_variant_enabled(self) -> bool:
        """Returns whether multiple preprocessing variants are enabled."""
        p_cfg = self.config.preprocessing if isinstance(self.config.preprocessing, dict) else {}
        mv_cfg = p_cfg.get("multi_variant", {})
        if isinstance(mv_cfg, bool):
            return mv_cfg
        if isinstance(mv_cfg, dict):
            return bool(mv_cfg.get("enabled", True))
        return True

    def _pair_logging_config(self) -> Dict[str, Any]:
        """Returns normalized pair logging configuration."""
        pair_cfg = self.config.positioning_params.get("pair_logging", {})
        if isinstance(pair_cfg, bool):
            return {"enabled": pair_cfg, "save_failed": True, "save_matched": True}
        if isinstance(pair_cfg, dict) and pair_cfg:
            try:
                max_unique_pairs = int(pair_cfg.get("max_unique_pairs", 0))
            except Exception:
                max_unique_pairs = 0
            return {
                "enabled": bool(pair_cfg.get("enabled", False)),
                "save_failed": bool(pair_cfg.get("save_failed", True)),
                "save_matched": bool(pair_cfg.get("save_matched", True)),
                "max_unique_pairs": max_unique_pairs,
            }

        legacy_cfg = self.config.positioning_params.get("failed_pair_logging", {})
        if isinstance(legacy_cfg, bool):
            enabled = legacy_cfg
        else:
            enabled = bool(legacy_cfg.get("enabled", False))
        return {
            "enabled": enabled,
            "save_failed": True,
            "save_matched": False,
            "max_unique_pairs": 0,
        }

    def _should_log_pair(self, status: str) -> bool:
        """Checks whether a pair should be logged for given status."""
        cfg = self._pair_logging_config()
        if not bool(cfg.get("enabled", False)):
            return False
        if status == "matched":
            return bool(cfg.get("save_matched", True))
        return bool(cfg.get("save_failed", True))

    def _save_pair_log(
        self,
        query_path: Path,
        map_path: Path,
        status: str,
        inliers: int,
        matcher_success: bool,
        error_meters: Optional[float] = None,
    ) -> None:
        """Saves side-by-side diagnostics for query-map match attempts."""
        if not self._should_log_pair(status):
            return

        key = f"{query_path.name}|{map_path.name}"
        if key in self._pair_log_cache:
            return

        max_pairs = int(self._pair_logging_config().get("max_unique_pairs", 0))
        if max_pairs > 0 and len(self._pair_log_cache) >= max_pairs:
            return

        query_img = cv2.imread(str(query_path))
        map_img = cv2.imread(str(map_path))
        if query_img is None or map_img is None:
            return

        pair_dir = Path(self.config.data_paths["output_dir"]) / "pair_logs"
        pair_dir.mkdir(parents=True, exist_ok=True)

        target_h = max(query_img.shape[0], map_img.shape[0])

        def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
            if img.shape[0] == h:
                return img
            w = int(round(img.shape[1] * h / img.shape[0]))
            interp = cv2.INTER_AREA if h < img.shape[0] else cv2.INTER_LINEAR
            return cv2.resize(img, (w, h), interpolation=interp)

        q_view = resize_to_h(query_img, target_h)
        m_view = resize_to_h(map_img, target_h)
        gap = 16
        header_h = 90
        canvas_w = q_view.shape[1] + gap + m_view.shape[1]
        canvas_h = header_h + target_h
        canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
        canvas[header_h:, : q_view.shape[1]] = q_view
        canvas[header_h:, q_view.shape[1] + gap :] = m_view

        cv2.putText(
            canvas,
            f"PAIR LOG | status={status}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"query={query_path.name} | map={map_path.name}",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"matcher_success={matcher_success} | inliers={inliers} | "
            f"error_m={error_meters if error_meters is not None else 'na'}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        out_name = (
            f"{self._pair_log_seq_by_query.get(query_path.name, 0) + 1:04d}"
            f"__{Path(query_path.name).stem}__{Path(map_path.name).stem}"
            f"__{status}.jpg"
        )
        out_path = pair_dir / out_name
        cv2.imwrite(str(out_path), canvas)
        self._pair_log_cache.add(key)
        self._pair_log_seq_by_query[query_path.name] = (
            self._pair_log_seq_by_query.get(query_path.name, 0) + 1
        )

    def _save_failed_pair(
        self,
        query_path: Path,
        map_path: Path,
        reason: str,
        inliers: int,
        matcher_success: bool,
    ) -> None:
        """Backward-compatible wrapper for failed pair logging."""
        self._save_pair_log(
            query_path,
            map_path,
            reason,
            inliers,
            matcher_success,
            None,
        )

    def _map_context_enabled(self) -> bool:
        """Returns whether contextual map composition is enabled."""
        map_context = self.config.positioning_params.get("map_context", {})
        if isinstance(map_context, bool):
            return map_context
        return bool(map_context.get("enabled", False))

    def _map_context_edge_pixels(self) -> int:
        """Returns edge size (in pixels) contributed by neighbors per side."""
        map_context = self.config.positioning_params.get("map_context", {})
        if isinstance(map_context, dict):
            try:
                edge = int(map_context.get("edge_pixels", 128))
            except Exception:
                edge = 128
        else:
            edge = 128
        return max(0, min(128, edge))

    def _compose_half_neighbor_context(
        self, map_row: pd.Series
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        """Builds a context map with center tile + nearest neighbor halves.

        The output image is 512x512 and includes:
        - full center tile in the middle (256x256)
        - left/right/top/bottom neighbor near halves
        - corner neighbor near quadrants

        Returns:
            Tuple containing context image path, context image array,
            and effective map metadata matching the composed view.
        """
        map_filename = str(map_row["Filename"])
        cache_hit = self._map_context_cache.get(map_filename)
        if cache_hit is not None:
            cached_path = Path(str(cache_hit["path"]))
            cached_img = cv2.imread(str(cached_path))
            if cached_img is not None:
                return cached_path, cached_img, dict(cache_hit["metadata"])

        map_dir = Path(self.config.data_paths["map_dir"])
        output_dir = Path(self.config.data_paths["output_dir"])
        context_dir = output_dir / ".tmp_context_maps"
        context_dir.mkdir(parents=True, exist_ok=True)

        tile_x = int(cast(Any, map_row["TileX"]))
        tile_y = int(cast(Any, map_row["TileY"]))
        level = int(cast(Any, map_row["Level"]))
        provider = str(map_row.get("Provider", self.config.tile_provider.get("name", "esri")))

        def tile_path(tx: int, ty: int) -> Path:
            return map_dir / f"tile_{provider}_{level}_{tx}_{ty}.jpg"

        def load_or_blank(tx: int, ty: int) -> np.ndarray:
            candidate = tile_path(tx, ty)
            if not candidate.exists():
                return np.full((256, 256, 3), 230, dtype=np.uint8)
            img = cv2.imread(str(candidate))
            if img is None:
                return np.full((256, 256, 3), 230, dtype=np.uint8)
            if img.shape[:2] != (256, 256):
                return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            return img

        center = load_or_blank(tile_x, tile_y)
        left = load_or_blank(tile_x - 1, tile_y)
        right = load_or_blank(tile_x + 1, tile_y)
        top = load_or_blank(tile_x, tile_y - 1)
        bottom = load_or_blank(tile_x, tile_y + 1)
        top_left = load_or_blank(tile_x - 1, tile_y - 1)
        top_right = load_or_blank(tile_x + 1, tile_y - 1)
        bottom_left = load_or_blank(tile_x - 1, tile_y + 1)
        bottom_right = load_or_blank(tile_x + 1, tile_y + 1)

        edge = self._map_context_edge_pixels()
        panel_size = 256 + 2 * edge
        panel = np.full((panel_size, panel_size, 3), 240, dtype=np.uint8)
        panel[edge : edge + 256, edge : edge + 256] = center
        if edge > 0:
            panel[edge : edge + 256, 0:edge] = left[:, 256 - edge : 256]
            panel[edge : edge + 256, edge + 256 : edge + 256 + edge] = right[:, 0:edge]
            panel[0:edge, edge : edge + 256] = top[256 - edge : 256, :]
            panel[edge + 256 : edge + 256 + edge, edge : edge + 256] = bottom[0:edge, :]
            panel[0:edge, 0:edge] = top_left[256 - edge : 256, 256 - edge : 256]
            panel[0:edge, edge + 256 : edge + 256 + edge] = top_right[256 - edge : 256, 0:edge]
            panel[edge + 256 : edge + 256 + edge, 0:edge] = bottom_left[0:edge, 256 - edge : 256]
            panel[edge + 256 : edge + 256 + edge, edge + 256 : edge + 256 + edge] = bottom_right[0:edge, 0:edge]

        out_name = f"{Path(map_filename).stem}_ctx_half_neighbors.jpg"
        out_path = context_dir / out_name
        cv2.imwrite(str(out_path), panel)

        center_px_x, center_px_y = TileSystem.tile_xy_to_pixel_xy(tile_x, tile_y)
        nw_px_x = center_px_x - edge
        nw_px_y = center_px_y - edge
        se_px_x = center_px_x + 256 + edge
        se_px_y = center_px_y + 256 + edge
        nw_lat, nw_lon = TileSystem.pixel_xy_to_latlong(nw_px_x, nw_px_y, level)
        se_lat, se_lon = TileSystem.pixel_xy_to_latlong(se_px_x, se_px_y, level)

        effective_metadata = map_row.to_dict()
        effective_metadata["Top_left_lat"] = float(nw_lat)
        effective_metadata["Top_left_lon"] = float(nw_lon)
        effective_metadata["Bottom_right_lat"] = float(se_lat)
        effective_metadata["Bottom_right_long"] = float(se_lon)
        effective_metadata["Filename"] = out_name

        self._map_context_cache[map_filename] = {
            "path": str(out_path),
            "metadata": effective_metadata,
        }

        return out_path, panel, effective_metadata

    def _prepare_map_for_matching(
        self, map_row: pd.Series
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        """Returns matcher-ready map image path/data and effective metadata."""
        map_filename = str(map_row["Filename"])
        map_path = Path(self.config.data_paths["map_dir"]) / map_filename
        map_img = cv2.imread(str(map_path))
        if map_img is None:
            raise RuntimeError(f"Failed to read map image at {map_path}")

        if not self._map_context_enabled():
            return map_path, map_img, map_row.to_dict()

        required_cols = {"TileX", "TileY", "Level"}
        if not required_cols.issubset(set(map_row.index)):
            return map_path, map_img, map_row.to_dict()

        return self._compose_half_neighbor_context(map_row)

    def inject_helpers(self, haversine, predicted_gps, location_and_error):
        """Injects helper functions to avoid circular imports.

        Args:
            haversine: Distance function implementation.
            predicted_gps: Coordinate prediction function implementation.
            location_and_error: Homography-to-location conversion function.

        Returns:
            None.
        """
        self.haversine_distance = haversine
        self.calculate_predicted_gps = predicted_gps
        self.calculate_location_and_error = location_and_error

    def preprocess_query(
        self, query_path: Path, query_row: pd.Series, temp_dir: Optional[Path]
    ) -> Tuple[Path, Optional[Tuple[int, ...]]]:
        """Applies configured preprocessing steps to a query image.

        Args:
            query_path: Query image file path.
            query_row: Query metadata row.
            temp_dir: Directory used for processed-image persistence.

        Returns:
            A tuple of `(path_for_matcher, query_shape)`.
        """
        query_key = str(query_row.get("Filename", query_path.name))

        if self.preprocessor is None:
            img = cv2.imread(str(query_path))
            shape = img.shape if img is not None else None
            if shape is not None:
                self._query_variants_cache[query_key] = [(query_path, shape)]
            return query_path, shape

        img_original = cv2.imread(str(query_path))
        if img_original is None:
            raise RuntimeError(f"Failed to read image at {query_path}")

        metadata = query_row.to_dict()
        variant_images: list[Tuple[np.ndarray, str]] = []

        def _safe_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        if self._multi_variant_enabled() and hasattr(self.preprocessor, "_apply_resize"):
            variant_images.append((img_original, "default"))

            yaw_candidates: list[Tuple[str, float]] = []
            seen_yaw_candidates: set[Tuple[str, int]] = set()
            yaw_sources = [
                ("phi1", metadata.get("Gimball_Yaw_Phi1")),
                ("phi2", metadata.get("Gimball_Yaw_Phi2")),
                ("phi1", metadata.get("Phi1")),
                ("phi2", metadata.get("Phi2")),
                ("phi1", metadata.get("Gimball_Yaw")),
            ]
            for yaw_tag, raw_value in yaw_sources:
                yaw_value = _safe_float(raw_value)
                if yaw_value is not None:
                    key = (yaw_tag, int(round(yaw_value * 1000.0)))
                    if key in seen_yaw_candidates:
                        continue
                    seen_yaw_candidates.add(key)
                    yaw_candidates.append((yaw_tag, yaw_value))

            try:
                resized_only = self.preprocessor._apply_resize(img_original)
                variant_images.append((resized_only, "resized"))
            except Exception:
                pass

            ordered_candidates: list[Tuple[str, float]] = []
            phi1_value = next((v for t, v in yaw_candidates if t == "phi1"), None)
            phi2_value = next((v for t, v in yaw_candidates if t == "phi2"), None)
            if phi1_value is not None:
                ordered_candidates.append(("phi1", phi1_value))
            if phi2_value is not None:
                ordered_candidates.append(("phi2", phi2_value))

            for yaw_tag, yaw_value in ordered_candidates:
                try:
                    yaw_meta = dict(metadata)
                    yaw_meta["Gimball_Yaw"] = yaw_value
                    yaw_processed = self.preprocessor(img_original, yaw_meta)
                    variant_images.append((yaw_processed, f"preprocessed_{yaw_tag}"))
                except Exception:
                    pass
        else:
            processed = self.preprocessor(img_original, metadata)
            variant_images.append((processed, "default"))

        if temp_dir:
            prepared: list[Tuple[Path, Tuple[int, ...]]] = []

            for image, tag in variant_images:
                if image is None or image.size == 0:
                    continue

                name = f"{Path(query_path.name).stem}_{tag}{Path(query_path.name).suffix}"
                processed_path = temp_dir / name
                cv2.imwrite(str(processed_path), image)
                prepared.append((processed_path, image.shape))

            if not prepared:
                prepared = [(query_path, img_original.shape)]

            self._query_variants_cache[query_key] = prepared
            return prepared[0][0], prepared[0][1]

        raise RuntimeError(
            "Processed image cannot be matched without a temporary directory."
        )

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
        """Matches a query image against a specific satellite tile.

        Args:
            query_path: Query image path.
            query_shape: Query image shape.
            query_row: Query metadata row.
            map_row: Candidate map tile metadata row.
            results_dir: Per-query output directory.
            min_inliers: Minimum inlier threshold for a valid match.
            save_viz: Whether to save match visualizations.

        Returns:
            Match summary dictionary if successful, otherwise `None`.
        """
        map_filename = str(map_row["Filename"])
        if self.pipeline is None:
            return None

        try:
            map_match_path, map_img, effective_map_row = self._prepare_map_for_matching(
                map_row
            )
        except Exception:
            return None

        query_key = str(query_row.get("Filename", query_path.name))
        variants = self._query_variants_cache.get(
            query_key, [(query_path, query_shape)]
        )

        best_variant_path = query_path
        best_variant_shape = query_shape
        best_match_results: Optional[Dict[str, Any]] = None
        best_num_inliers = -1

        for v_path, v_shape in variants:
            current_results = self.pipeline.match(v_path, map_match_path)
            current_mask = current_results.get("inliers")
            current_inliers = (
                int(np.sum(current_mask)) if current_mask is not None else 0
            )
            if current_inliers > best_num_inliers:
                best_num_inliers = current_inliers
                best_match_results = current_results
                best_variant_path = v_path
                best_variant_shape = v_shape

        if best_match_results is None:
            return None

        match_results = best_match_results
        inliers_mask = match_results.get("inliers")
        num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0

        ransac_successful = (
            bool(match_results.get("success", False))
            and int(num_inliers) >= min_inliers
        )

        if not ransac_successful:
            reason = "ransac_failed_or_low_inliers"
            self._save_failed_pair(
                best_variant_path,
                map_match_path,
                reason,
                int(num_inliers),
                bool(match_results.get("success", False)),
            )
            return None

        try:
            pos_res = self.compute_positioning(
                ransac_successful,
                match_results.get("homography"),
                query_row,
                pd.Series(effective_map_row),
                best_variant_shape,
                map_img.shape,
            )
        except Exception as e:
            if self.config.matcher_params.get("verbose"):
                _logger.info(f"    Tile positioning failed: {e}")
            self._save_failed_pair(
                best_variant_path,
                map_match_path,
                "positioning_exception",
                int(num_inliers),
                True,
            )
            return None

        if save_viz and ransac_successful:
            results_dir.mkdir(exist_ok=True)
            self._save_viz(results_dir, query_path, map_match_path, match_results)

        if not pos_res["success"]:
            self._save_failed_pair(
                best_variant_path,
                map_match_path,
                "positioning_unsuccessful",
                int(num_inliers),
                True,
            )
            return None

        self._save_pair_log(
            best_variant_path,
            map_match_path,
            "matched",
            int(num_inliers),
            True,
            float(pos_res.get("error_meters", float("inf"))),
        )

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
        """Calculates geographic position from the estimated homography.

        Args:
            ransac_successful: Whether geometric estimation passed.
            homography: Estimated homography matrix.
            query_row: Query metadata row.
            map_row: Map tile metadata row.
            query_shape: Query image shape.
            map_shape: Map image shape.

        Returns:
            Positioning result dictionary with success and error metadata.
        """
        res = {
            "pred_lat": None,
            "pred_lon": None,
            "error_meters": float("inf"),
            "success": False,
        }

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
            res.update(
                {
                    "pred_lat": plat,
                    "pred_lon": plon,
                    "error_meters": err,
                    "success": True,
                }
            )
        return res

    def _save_viz(self, results_dir, q_path, m_path, match_results):
        """Saves match visualization.

        Args:
            results_dir: Destination directory.
            q_path: Query image path.
            m_path: Map image path.
            match_results: Matcher output dictionary.

        Returns:
            None.
        """
        out_path = results_dir / f"{q_path.stem}_vs_{m_path.stem}_match.png"
        if hasattr(self.pipeline, "visualize_matches"):
            try:
                self.pipeline.visualize_matches(
                    q_path,
                    m_path,
                    match_results["mkpts0"],
                    match_results["mkpts1"],
                    match_results["inliers"],
                    out_path,
                    homography=match_results.get("homography"),
                )
            except Exception as e:
                _logger.info(f"  WARNING: Visualization failed: {e}")
