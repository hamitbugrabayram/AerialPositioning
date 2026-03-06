
"""Core positioning engine for image matching and coordinate estimation.

This module provides the PositioningEngine class which handles image
preprocessing, matching against satellite tiles, and georeferencing.
"""

import math
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

    def _pair_logging_config(self) -> Dict[str, Any]:
        """Returns normalised pair-logging configuration.

        Reads the ``pair_logging`` key from ``positioning_params``.
        Accepts a plain *bool* for quick toggling or a *dict* with
        ``enabled``, ``save_failed``, ``save_matched``, and optional
        ``max_unique_pairs`` keys.

        Falls back to the legacy ``failed_pair_logging`` key when the
        modern key is absent.

        Returns:
            Dictionary with ``enabled``, ``save_failed``,
            ``save_matched``, and ``max_unique_pairs``.
        """
        pair_cfg = self.config.positioning_params.get("pair_logging", {})
        if isinstance(pair_cfg, bool):
            return {
                "enabled": pair_cfg,
                "save_failed": True,
                "save_matched": True,
                "max_unique_pairs": None,
            }
        if isinstance(pair_cfg, dict) and pair_cfg:
            raw_max = pair_cfg.get("max_unique_pairs")
            max_unique_pairs = None
            if raw_max is not None:
                try:
                    max_unique_pairs = max(0, int(raw_max))
                except (TypeError, ValueError):
                    max_unique_pairs = None
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
            "max_unique_pairs": None,
        }

    def _should_log_pair(self, status: str) -> bool:
        """Checks whether a pair should be logged for given status."""
        cfg = self._pair_logging_config()
        if not bool(cfg.get("enabled", False)):
            return False

        max_unique_pairs = cfg.get("max_unique_pairs")
        if isinstance(max_unique_pairs, int) and max_unique_pairs >= 0:
            if len(self._pair_log_cache) >= max_unique_pairs:
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
        match_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves rich diagnostics for query-map match attempts.

        When ``match_results`` is provided (contains ``mkpts0``, ``mkpts1``,
        ``inliers``, and optionally ``homography``), the pair log shows:

        * Inlier keypoints and match lines (green).
        * Homography polygon projected onto the map (cyan).
        * Projected query center as a crosshair.

        Falls back to a plain side-by-side image when no match data is
        available.
        """
        if not self._should_log_pair(status):
            return

        key = f"{query_path.name}|{map_path.name}"
        if key in self._pair_log_cache:
            return

        pair_dir = Path(self.config.data_paths["output_dir"]) / "pair_logs"
        pair_dir.mkdir(parents=True, exist_ok=True)

        seq = self._pair_log_seq_by_query.get(query_path.name, 0) + 1
        out_name = (
            f"{seq:04d}__{Path(query_path.name).stem}"
            f"__{Path(map_path.name).stem}__{status}.jpg"
        )
        out_path = pair_dir / out_name

        has_match_data = (
            match_results is not None
            and match_results.get("mkpts0") is not None
            and len(match_results["mkpts0"]) > 0
        )
        if has_match_data:
            try:
                from src.utils.visualization import create_match_visualization

                mkpts0 = match_results["mkpts0"]
                mkpts1 = match_results["mkpts1"]
                inlier_mask = match_results.get("inliers")
                if inlier_mask is None:
                    inlier_mask = np.zeros(len(mkpts0), dtype=bool)
                homography = match_results.get("homography")

                err_str = (
                    f"{error_meters:.1f}m" if error_meters is not None else "n/a"
                )
                text_lines = [
                    f"{status}  |  inliers={inliers}  |  error={err_str}",
                    f"query={query_path.name}  |  map={map_path.name}",
                ]

                create_match_visualization(
                    image0_path=query_path,
                    image1_path=map_path,
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    inliers_mask=inlier_mask,
                    output_path=out_path,
                    text_info=text_lines,
                    show_outliers=True,
                    target_height=720,
                    homography=homography,
                )

                self._pair_log_cache.add(key)
                self._pair_log_seq_by_query[query_path.name] = seq
                return
            except Exception:
                pass 

        query_img = cv2.imread(str(query_path))
        map_img = cv2.imread(str(map_path))
        if query_img is None or map_img is None:
            return

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
            canvas, f"PAIR LOG | status={status}",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"query={query_path.name} | map={map_path.name}",
            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA,
        )
        err_str = error_meters if error_meters is not None else "na"
        cv2.putText(
            canvas,
            f"matcher_success={matcher_success} | inliers={inliers} | error_m={err_str}",
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA,
        )

        cv2.imwrite(str(out_path), canvas)
        self._pair_log_cache.add(key)
        self._pair_log_seq_by_query[query_path.name] = seq

    def _save_failed_pair(
        self,
        query_path: Path,
        map_path: Path,
        reason: str,
        inliers: int,
        matcher_success: bool,
        match_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Backward-compatible wrapper for failed pair logging."""
        self._save_pair_log(
            query_path,
            map_path,
            reason,
            inliers,
            matcher_success,
            None,
            match_results=match_results,
        )

    def _map_context_enabled(self) -> bool:
        """Returns whether contextual map composition is enabled."""
        map_context = self.config.positioning_params.get("map_context", {})
        if isinstance(map_context, bool):
            return map_context
        return bool(map_context.get("enabled", False))

    def _save_map_context_maps(self) -> bool:
        """Returns whether stitched context maps should be persisted."""
        map_context = self.config.positioning_params.get("map_context", {})
        if isinstance(map_context, dict):
            return bool(map_context.get("save_context_maps", False))
        return False

    def _map_context_coverage_factor(self) -> float:
        """Returns the altitude-to-ground-coverage multiplier.

        The drone's approximate ground coverage in meters is estimated as
        ``altitude * coverage_factor``.  Typical values:

        * 1.5 -- conservative, suits narrow FOV cameras (~60-70 deg).
        * 2.0 -- moderate, suits most consumer drones (~75-85 deg).
        * 2.5 -- generous, suits wide-angle cameras (>85 deg).
        """
        map_context = self.config.positioning_params.get("map_context", {})
        if isinstance(map_context, dict):
            try:
                return max(0.5, float(map_context.get("coverage_factor", 2.0)))
            except Exception:
                return 2.0
        return 2.0

    def _calculate_adaptive_grid_size(
        self,
        altitude_m: float,
        latitude: float,
        level: int,
    ) -> int:
        """Determines the tile grid size needed to cover the drone's view.

        Estimates the ground coverage from the flight altitude and a
        configurable ``coverage_factor``, then computes how many tiles at
        the given zoom level and latitude are required to contain that area.

        The result is clamped to ``map_context.max_grid`` (default 5) so
        that the composite stays small enough for the matcher's internal
        resize (typically 768 px) to preserve useful detail.

        Args:
            altitude_m: Drone altitude in meters above ground.
            latitude: Query latitude in degrees (affects tile size in meters).
            level: Tile zoom level.

        Returns:
            Odd integer grid size (1, 3, 5, ...).
        """
        map_ctx = self.config.positioning_params.get("map_context", {})
        max_grid = 5
        if isinstance(map_ctx, dict):
            try:
                max_grid = int(map_ctx.get("max_grid", 5))
            except (TypeError, ValueError):
                max_grid = 5

        if altitude_m <= 0:
            return min(3, max_grid)

        coverage_m = altitude_m * self._map_context_coverage_factor()

        tile_coverage_m = TileSystem.ground_resolution(latitude, level) * 256.0
        if tile_coverage_m <= 0:
            return min(3, max_grid)

        grid_size = math.ceil(coverage_m / tile_coverage_m)
        if grid_size < 1:
            grid_size = 1
        if grid_size > max_grid:
            grid_size = max_grid
        if grid_size % 2 == 0:
            grid_size += 1
        return grid_size

    def _compose_grid_context(
        self, map_row: pd.Series, grid_size: int
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        """Builds a context map by stitching a NxN grid of tiles.

        Creates a composite image from an NxN grid of tiles centered on the
        target tile. For example, with grid_size=3 the output is 768x768
        (3x3 tiles of 256px each), and with grid_size=5 it is 1280x1280.

        Args:
            map_row: Map tile metadata row.
            grid_size: NxN grid dimension (must be odd and >= 1).

        Returns:
            Tuple containing context image path, context image array,
            and effective map metadata matching the composed view.
        """
        if grid_size < 1:
            grid_size = 1
        if grid_size % 2 == 0:
            grid_size += 1

        map_filename = str(map_row["Filename"])
        cache_key = f"{map_filename}_grid{grid_size}"

        cache_hit = self._map_context_cache.get(cache_key)
        if cache_hit is not None:
            cached_path = Path(str(cache_hit["path"]))
            cached_img = cv2.imread(str(cached_path))
            if cached_img is not None:
                return cached_path, cached_img, dict(cache_hit["metadata"])

        map_dir = Path(self.config.data_paths["map_dir"])
        output_dir = Path(self.config.data_paths["output_dir"])
        context_dir_name = "context_maps" if self._save_map_context_maps() else ".tmp_context_maps"
        context_dir = output_dir / context_dir_name
        context_dir.mkdir(parents=True, exist_ok=True)

        tile_x = int(cast(Any, map_row["TileX"]))
        tile_y = int(cast(Any, map_row["TileY"]))
        level = int(cast(Any, map_row["Level"]))
        provider = str(
            map_row.get(
                "Provider", self.config.tile_provider.get("name", "esri")
            )
        )

        half = grid_size // 2

        def load_or_blank(tx: int, ty: int) -> np.ndarray:
            path = map_dir / f"tile_{provider}_{level}_{tx}_{ty}.jpg"
            if not path.exists():
                return np.full((256, 256, 3), 230, dtype=np.uint8)
            img = cv2.imread(str(path))
            if img is None:
                return np.full((256, 256, 3), 230, dtype=np.uint8)
            if img.shape[:2] != (256, 256):
                return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            return img

        panel_size = grid_size * 256
        panel = np.full((panel_size, panel_size, 3), 230, dtype=np.uint8)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                tile_img = load_or_blank(tile_x + dx, tile_y + dy)
                py = (dy + half) * 256
                px = (dx + half) * 256
                panel[py : py + 256, px : px + 256] = tile_img

        out_name = f"{Path(map_filename).stem}_ctx_grid_{grid_size}x{grid_size}.jpg"
        out_path = context_dir / out_name
        cv2.imwrite(str(out_path), panel)

        nw_px_x, nw_px_y = TileSystem.tile_xy_to_pixel_xy(
            tile_x - half, tile_y - half
        )
        se_px_x = nw_px_x + panel_size
        se_px_y = nw_px_y + panel_size
        nw_lat, nw_lon = TileSystem.pixel_xy_to_latlong(nw_px_x, nw_px_y, level)
        se_lat, se_lon = TileSystem.pixel_xy_to_latlong(se_px_x, se_px_y, level)

        effective_metadata = map_row.to_dict()
        effective_metadata["Top_left_lat"] = float(nw_lat)
        effective_metadata["Top_left_lon"] = float(nw_lon)
        effective_metadata["Bottom_right_lat"] = float(se_lat)
        effective_metadata["Bottom_right_long"] = float(se_lon)
        effective_metadata["Filename"] = out_name

        self._map_context_cache[cache_key] = {
            "path": str(out_path),
            "metadata": effective_metadata,
        }

        return out_path, panel, effective_metadata

    def _prepare_map_for_matching(
        self,
        map_row: pd.Series,
        query_row: Optional[pd.Series] = None,
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        """Returns matcher-ready map image path/data and effective metadata.

        When map context is enabled, computes an adaptive grid size based
        on the drone's altitude and latitude, then stitches neighbouring
        tiles into a composite that covers the drone's estimated ground
        footprint.

        Args:
            map_row: Map tile metadata row.
            query_row: Optional query metadata row.  When provided,
                altitude and latitude are read to compute the adaptive
                grid size.  Falls back to a 3x3 grid when absent.
        """
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

        altitude = 0.0
        level = int(cast(Any, map_row["Level"]))
        if query_row is not None:
            try:
                altitude = float(query_row.get("Altitude", 0.0))
            except (TypeError, ValueError):
                altitude = 0.0

        try:
            latitude = (
                float(map_row["Top_left_lat"]) + float(map_row["Bottom_right_lat"])
            ) / 2.0
        except (KeyError, TypeError, ValueError):
            latitude = 0.0

        grid = self._calculate_adaptive_grid_size(altitude, latitude, level)
        if grid <= 1:
            return map_path, map_img, map_row.to_dict()
        return self._compose_grid_context(map_row, grid)

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

        yaw = metadata.get("Gimball_Yaw_Phi1")
        if yaw is None:
            yaw = metadata.get("Phi1")
        if yaw is None:
            yaw = metadata.get("Phi2")
        if yaw is None:
            yaw = metadata.get("Gimball_Yaw", 0.0)
        metadata["Gimball_Yaw"] = yaw
        processed = self.preprocessor(img_original, metadata)
        variant_images.append((processed, "preprocessed"))

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

        Tries every preprocessed variant of the query against the
        (optionally context-expanded) map tile.  Returns the best
        match summary when RANSAC succeeds and positioning is valid.

        The returned dictionary includes rich geometry data
        (``homography``, ``effective_map_metadata``, ``query_shape``,
        ``map_shape``) consumed by ``CompositeFrameRenderer``.

        Args:
            query_path: Query image path.
            query_shape: Query image shape.
            query_row: Query metadata row.
            map_row: Candidate map tile metadata row.
            results_dir: Per-query output directory.
            min_inliers: Minimum inlier threshold for a valid match.
            save_viz: Whether to save match visualizations.

        Returns:
            Match summary dictionary if successful, otherwise ``None``.
        """
        map_filename = str(map_row["Filename"])
        if self.pipeline is None:
            return None

        try:
            map_match_path, map_img, effective_map_row = self._prepare_map_for_matching(
                map_row, query_row=query_row
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
                match_results=match_results,
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
                match_results=match_results,
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
                match_results=match_results,
            )
            return None

        self._save_pair_log(
            best_variant_path,
            map_match_path,
            "matched",
            int(num_inliers),
            True,
            float(pos_res.get("error_meters", float("inf"))),
            match_results=match_results,
        )

        return {
            "map_filename": map_filename,
            "inliers": int(num_inliers),
            "outliers": len(match_results.get("mkpts0", [])) - int(num_inliers),
            "time": match_results.get("time", 0.0),
            "pred_lat": pos_res["pred_lat"],
            "pred_lon": pos_res["pred_lon"],
            "error_meters": pos_res["error_meters"],
            "homography": match_results.get("homography"),
            "effective_map_metadata": effective_map_row,
            "query_shape": best_variant_shape,
            "map_shape": map_img.shape,
            "query_variant_path": str(best_variant_path),
            "mkpts0": match_results.get("mkpts0"),
            "mkpts1": match_results.get("mkpts1"),
            "inliers_mask": inliers_mask,
            "map_match_path": str(map_match_path),
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
