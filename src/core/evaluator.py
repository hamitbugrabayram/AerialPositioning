"""Visual positioning and visualization module.

This module handles the primary visual positioning pipeline, including
adaptive search radius management and path visualization for GNSS-free
coordinate estimation.
"""

import math
import random
import time
from pathlib import Path
from typing import Any, List, Optional, cast

import pandas as pd

from src.core.runner import PositioningRunner
from src.models.config import PositioningConfig, QueryResult
from src.utils.composite_frames import CompositeFrameRenderer, FrameData
from src.utils.logger import get_logger
from src.utils.plotting import TrajectoryVisualizer

_logger = get_logger(__name__)


class Evaluator(PositioningRunner):
    """Orchestrates visual positioning with configurable search strategy.

    Two strategies are available (selected via ``strategy`` key):

    * **ins_simulation** *(default)* -- A simulated INS propagates the
      search center using GT-derived displacement vectors (proxy for
      IMU dead-reckoning) plus per-step Gaussian noise.  On success
      the INS snaps to the visual fix; on failure the frame is skipped
      and the radius grows linearly.
    * **adaptive_radius** -- The search center stays at the last
      successful match.  On failure the frame is skipped and the
      radius grows linearly.  No INS propagation.

    Attributes:
        full_query_df: Complete query dataframe with all frames.
        sampled_query_df: Subsampled query dataframe at fixed intervals.
        frames_dir: Directory for saving frame visualizations.
        sample_interval: Number of frames between positioning checkpoints.
        strategy: Search strategy name (``ins_simulation`` or ``adaptive_radius``).
        initial_radius_m: Starting radius for adaptive search.
        max_radius_m: Maximum allowed search radius.
        skip_penalty_m: Metres added to radius after each skipped frame.
        ins_noise_sigma_m: Per-step INS noise standard deviation (metres).
        ins_noise_max_m: Per-step INS noise hard cap (metres).
    """

    _VALID_STRATEGIES = ("ins_simulation", "adaptive_radius")

    _M_PER_DEG_LAT = 111_320.0

    def __init__(self, config: PositioningConfig):
        """Initializes the evaluator with configuration.

        Args:
            config: Global positioning configuration object.
        """
        super().__init__(config)
        self.full_query_df: Optional[pd.DataFrame] = None
        self.sampled_query_df: Optional[pd.DataFrame] = None
        self.frames_dir: Optional[Path] = None
        self.save_frame_sequence = bool(
            self.config.positioning_params.get("save_frame_sequence", False)
        )
        self.sample_interval: int = int(
            self.config.positioning_params.get("sample_interval", 1)
        )
        if self.sample_interval <= 0:
            self.sample_interval = 1

        search_conf = self.config.positioning_params.get("adaptive_search", {})
        self.strategy = str(
            search_conf.get("strategy", "ins_simulation")
        ).lower()
        if self.strategy not in self._VALID_STRATEGIES:
            _logger.warning(
                f"Unknown strategy '{self.strategy}', "
                f"falling back to 'ins_simulation'."
            )
            self.strategy = "ins_simulation"
        self.initial_radius_m = float(search_conf.get("initial_radius_m", 1000.0))
        self.max_radius_m = float(search_conf.get("max_radius_m", 2000.0))
        self.skip_penalty_m = float(search_conf.get("skip_penalty_m", 200.0))
        self.ins_noise_sigma_m = float(search_conf.get("ins_noise_sigma_m", 30.0))
        self.ins_noise_max_m = float(search_conf.get("ins_noise_max_m", 100.0))

        _logger.info(f"Search strategy: {self.strategy}")

        self.visualizer = TrajectoryVisualizer(config)
        self.composite_renderer: Optional[CompositeFrameRenderer] = None

    def run_trajectory(self) -> None:
        """Executes the visual positioning pipeline with sampling and prediction.

        Returns:
            None.
        """
        self._validate_paths()
        self._setup_output_directory()
        self._initialize_preprocessor()
        self._initialize_pipeline()

        from src.core.engine import PositioningEngine
        from src.utils.statistics import ResultManager

        self.engine = PositioningEngine(self.config, self.pipeline, self.preprocessor)
        if self.output_dir is not None and self.assets_dir is not None:
            self.result_manager = ResultManager(self.output_dir, self.assets_dir)
        self._load_helpers()

        if self.assets_dir is not None and self.save_frame_sequence:
            self.frames_dir = self.assets_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)

        self.visualizer.set_context(
            output_dir=self.output_dir,
            frames_dir=self.frames_dir,
            assets_dir=self.assets_dir,
        )

        self._load_metadata()
        self.visualizer.set_context(map_df=self.map_df)

        if self.query_df is not None:
            self.query_df = self.query_df.sort_values(by="Filename").reset_index(
                drop=True
            )
            self.full_query_df = self.query_df.copy()
            self.visualizer.set_context(full_query_df=self.full_query_df)

            self.sampled_query_df = self.query_df.iloc[
                :: self.sample_interval
            ].reset_index(drop=True)

            num_checkpoints = (
                len(self.sampled_query_df) if self.sampled_query_df is not None else 0
            )
            _logger.info(
                f"Checkpoints: {num_checkpoints} (Interval: {self.sample_interval})"
            )

        if (
            self.engine is not None
            and self.full_query_df is not None
            and self.map_df is not None
            and self.output_dir is not None
        ):
            pair_cfg = self.engine._pair_logging_config()
            if pair_cfg.get("enabled", False):
                try:
                    self.composite_renderer = CompositeFrameRenderer(
                        self.config,
                        self.map_df,
                        self.full_query_df,
                        self.output_dir,
                    )
                except Exception as exc:
                    _logger.warning(f"Composite renderer init failed: {exc}")

        results = self._process_eval_queries()
        self._save_results(results)
        self._cleanup_temp_processed_queries()
        self.visualizer.generate_trajectory_plot(results)
        _logger.info("Positioning Complete")

    def _ins_noise_degrees(self, lat: float) -> tuple:
        """Generates a random per-step INS noise vector in degrees.

        The noise magnitude is drawn from a half-normal distribution
        (|N(0, sigma)|) and hard-capped at ``ins_noise_max_m``.  The
        direction is uniformly random.

        Args:
            lat: Current latitude (used for longitude scaling).

        Returns:
            Tuple ``(d_lat, d_lon)`` in degrees.
        """
        mag_m = min(abs(random.gauss(0, self.ins_noise_sigma_m)),
                     self.ins_noise_max_m)
        bearing = random.uniform(0, 2 * math.pi)
        d_lat = (mag_m * math.cos(bearing)) / self._M_PER_DEG_LAT
        cos_lat = math.cos(math.radians(lat)) or 1e-9
        d_lon = (mag_m * math.sin(bearing)) / (self._M_PER_DEG_LAT * cos_lat)
        return d_lat, d_lon

    def _prepare_eval_context(self):
        """Prepares temp directory, min_inliers, and save_viz flag.

        Returns:
            Tuple ``(temp_dir, min_inliers, save_viz)``.
        """
        temp_dir: Optional[Path] = None
        if self.assets_dir:
            if self.config.preprocessing.get("save_processed", False):
                temp_dir = self.assets_dir / "processed_queries"
            else:
                temp_dir = self.assets_dir / ".tmp_processed_queries"
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        min_inliers = int(
            self.config.positioning_params.get("min_inliers_for_success", 10)
        )
        save_viz = bool(self.config.positioning_params.get("save_visualization", False))
        return temp_dir, min_inliers, save_viz

    def _handle_result(
        self,
        results: List[QueryResult],
        result: QueryResult,
        idx_int: int,
        total_frames: int,
        current_radius: float,
        consecutive_skips: int,
        query_row: Optional[pd.Series] = None,
        match_details: Optional[dict] = None,
    ) -> None:
        """Appends the result, renders composite frame, and logs progress.

        When a :class:`CompositeFrameRenderer` is active, a
        :class:`FrameData` bundle is assembled from *result* and
        *match_details* and forwarded to the renderer.

        Args:
            results: Mutable list of accumulated results.
            result: Current frame result.
            idx_int: Zero-based frame index.
            total_frames: Total number of frames in the evaluation.
            current_radius: Current adaptive search radius in metres.
            consecutive_skips: Running count of consecutive failures.
            query_row: Original query metadata row (optional).
            match_details: Rich match dictionary returned by the engine
                containing homography, tile metadata, etc. (optional).
        """
        results.append(result)
        if self.save_frame_sequence:
            self.visualizer.generate_trajectory_plot(results, frame_idx=idx_int)

        if self.composite_renderer is not None:
            q_path = str(
                Path(self.config.data_paths["query_dir"]) / result.query_filename
            )
            fd = FrameData(
                frame_idx=idx_int,
                query_filename=result.query_filename,
                query_image_path=q_path,
                gt_lat=result.gt_latitude or 0.0,
                gt_lon=result.gt_longitude or 0.0,
                search_center_lat=result.search_center_latitude or 0.0,
                search_center_lon=result.search_center_longitude or 0.0,
                search_radius_m=result.search_radius_m or self.initial_radius_m,
                success=result.success,
                predicted_lat=result.predicted_latitude,
                predicted_lon=result.predicted_longitude,
                error_meters=(
                    result.error_meters if result.success else None
                ),
                inliers=result.inliers if result.success else 0,
            )
            if match_details:
                fd.homography = match_details.get("homography")
                fd.effective_map_metadata = match_details.get(
                    "effective_map_metadata"
                )
                fd.query_shape = match_details.get("query_shape")
                fd.map_shape = match_details.get("map_shape")
                fd.preprocessed_image_path = match_details.get(
                    "query_variant_path"
                )
                fd.mkpts0 = match_details.get("mkpts0")
                fd.mkpts1 = match_details.get("mkpts1")
                fd.inliers_mask = match_details.get("inliers_mask")
                fd.map_match_path = match_details.get("map_match_path")
            self.composite_renderer.render_frame(fd, results)

        success_count = sum(1 for r in results if r.success)
        status = "OK" if result.success else "SKIP"
        _logger.info(
            f"Frame: {idx_int + 1}/{total_frames} | "
            f"{status} | Radius: {current_radius:.0f}m | "
            f"Skips: {consecutive_skips} | "
            f"Success: {success_count}/{idx_int + 1}"
        )

    def _handle_exception(
        self,
        results: List[QueryResult],
        query_row: pd.Series,
        exc: Exception,
    ) -> None:
        """Appends a failure result for an unhandled exception."""
        results.append(
            QueryResult(
                query_filename=str(query_row.get("Filename")),
                success=False,
                failure_reason=f"evaluator_exception: {exc}",
            )
        )

    def _process_eval_queries(self) -> List[QueryResult]:
        """Dispatches to the selected search strategy.

        Returns:
            List of sampled-frame query results.
        """
        if self.sampled_query_df is None or self.sampled_query_df.empty:
            return []

        if self.strategy == "ins_simulation":
            return self._process_ins_simulation()
        return self._process_adaptive_radius()

    def _process_ins_simulation(self) -> List[QueryResult]:
        """INS-guided skip-and-grow search.

        A simulated INS propagates the search center each frame using
        GT-derived displacement plus Gaussian noise.  Each frame is
        attempted **once**:

        * **Success** -- INS snaps to the matched position (visual
          correction), radius resets to ``initial_radius_m``.
        * **Failure** -- frame is skipped, radius grows by
          ``skip_penalty_m`` (capped at ``max_radius_m``), INS keeps
          dead-reckoning.

        Returns:
            List of sampled-frame query results.
        """
        assert self.sampled_query_df is not None
        temp_dir, min_inliers, save_viz = self._prepare_eval_context()

        results: List[QueryResult] = []

        ins_lat = float(self.sampled_query_df.iloc[0]["Latitude"])
        ins_lon = float(self.sampled_query_df.iloc[0]["Longitude"])

        prev_gt_lat = ins_lat
        prev_gt_lon = ins_lon

        current_radius = self.initial_radius_m
        consecutive_skips = 0

        total_start = time.time()
        total_frames = len(self.sampled_query_df)

        for idx, query_row in self.sampled_query_df.iterrows():
            try:
                idx_int = int(cast(Any, idx))

                curr_gt_lat = float(query_row["Latitude"])
                curr_gt_lon = float(query_row["Longitude"])

                if idx_int > 0:
                    delta_lat = curr_gt_lat - prev_gt_lat
                    delta_lon = curr_gt_lon - prev_gt_lon
                    n_lat, n_lon = self._ins_noise_degrees(ins_lat)
                    ins_lat += delta_lat + n_lat
                    ins_lon += delta_lon + n_lon

                prev_gt_lat = curr_gt_lat
                prev_gt_lon = curr_gt_lon

                result, match_details = self._match_with_adaptive_radius(
                    query_row, idx_int, temp_dir, min_inliers, save_viz,
                    ins_lat, ins_lon, current_radius,
                )

                if result.success:
                    ins_lat = float(
                        result.predicted_latitude
                        if result.predicted_latitude is not None
                        else ins_lat
                    )
                    ins_lon = float(
                        result.predicted_longitude
                        if result.predicted_longitude is not None
                        else ins_lon
                    )
                    current_radius = self.initial_radius_m
                    consecutive_skips = 0
                else:
                    consecutive_skips += 1
                    current_radius = min(
                        current_radius + self.skip_penalty_m,
                        self.max_radius_m,
                    )

                self._handle_result(
                    results, result, idx_int, total_frames,
                    current_radius, consecutive_skips,
                    query_row=query_row,
                    match_details=match_details,
                )

            except Exception as e:
                self._handle_exception(results, query_row, e)
                consecutive_skips += 1
                current_radius = min(
                    current_radius + self.skip_penalty_m,
                    self.max_radius_m,
                )
                continue

        total_time = time.time() - total_start
        _logger.info(f"Total Processing Time: {total_time:.2f}s")
        return results

    def _process_adaptive_radius(self) -> List[QueryResult]:
        """Plain skip-and-grow search without INS propagation.

        The search center stays at the last successfully matched
        position.  Each frame is attempted **once**:

        * **Success** -- search center moves to the matched position,
          radius resets to ``initial_radius_m``.
        * **Failure** -- frame is skipped, radius grows by
          ``skip_penalty_m`` (capped at ``max_radius_m``), search
          center unchanged.

        Returns:
            List of sampled-frame query results.
        """
        assert self.sampled_query_df is not None
        temp_dir, min_inliers, save_viz = self._prepare_eval_context()

        results: List[QueryResult] = []

        search_lat = float(self.sampled_query_df.iloc[0]["Latitude"])
        search_lon = float(self.sampled_query_df.iloc[0]["Longitude"])

        current_radius = self.initial_radius_m
        consecutive_skips = 0

        total_start = time.time()
        total_frames = len(self.sampled_query_df)

        for idx, query_row in self.sampled_query_df.iterrows():
            try:
                idx_int = int(cast(Any, idx))

                result, match_details = self._match_with_adaptive_radius(
                    query_row, idx_int, temp_dir, min_inliers, save_viz,
                    search_lat, search_lon, current_radius,
                )

                if result.success:
                    search_lat = float(
                        result.predicted_latitude
                        if result.predicted_latitude is not None
                        else search_lat
                    )
                    search_lon = float(
                        result.predicted_longitude
                        if result.predicted_longitude is not None
                        else search_lon
                    )
                    current_radius = self.initial_radius_m
                    consecutive_skips = 0
                else:
                    consecutive_skips += 1
                    current_radius = min(
                        current_radius + self.skip_penalty_m,
                        self.max_radius_m,
                    )

                self._handle_result(
                    results, result, idx_int, total_frames,
                    current_radius, consecutive_skips,
                    query_row=query_row,
                    match_details=match_details,
                )

            except Exception as e:
                self._handle_exception(results, query_row, e)
                consecutive_skips += 1
                current_radius = min(
                    current_radius + self.skip_penalty_m,
                    self.max_radius_m,
                )
                continue

        total_time = time.time() - total_start
        _logger.info(f"Total Processing Time: {total_time:.2f}s")
        return results

    def _match_with_adaptive_radius(
        self,
        query_row: pd.Series,
        idx: int,
        temp_dir: Optional[Path],
        min_inliers: int,
        save_viz: bool,
        search_lat: float,
        search_lon: float,
        radius: float,
    ) -> tuple:
        """Attempts to match a query image within the given adaptive radius.

        Args:
            query_row: Pandas Series containing query metadata.
            idx: Index of the current query.
            temp_dir: Optional directory for saving processed queries.
            min_inliers: Minimum inliers required for success.
            save_viz: Whether to save match visualizations.
            search_lat: Search center latitude.
            search_lon: Search center longitude.
            radius: Current adaptive search radius in meters.

        Returns:
            Tuple of ``(QueryResult, Optional[Dict])`` where the second
            element carries rich match details (homography, tile
            metadata, etc.) when a match succeeded.
        """
        query_filename = str(query_row["Filename"])
        gt_lat = float(cast(Any, query_row).get("Latitude", 0.0))
        gt_lon = float(cast(Any, query_row).get("Longitude", 0.0))

        res = QueryResult(
            query_filename=query_filename,
            gt_latitude=gt_lat,
            gt_longitude=gt_lon,
            search_center_latitude=search_lat,
            search_center_longitude=search_lon,
            search_radius_m=radius,
        )

        best_match_details: Optional[dict] = None
        base_details: dict = {}

        q_path = Path(self.config.data_paths["query_dir"]) / query_filename
        if not q_path.is_file() or self.engine is None:
            res.failure_reason = "query_file_missing_or_engine_unavailable"
            return res, best_match_details

        try:
            q_match, q_shape = self.engine.preprocess_query(q_path, query_row, temp_dir)
            base_details["query_variant_path"] = str(q_match)
        except Exception as e:
            res.failure_reason = f"preprocess_failed: {e}"
            return res, best_match_details

        if q_shape is None or self.map_df is None or self.output_dir is None:
            res.failure_reason = "query_shape_or_metadata_unavailable"
            return res, best_match_details

        rel_maps = self._filter_maps_by_reference(
            search_lat, search_lon, self.map_df, radius
        )
        if not rel_maps.empty:
            try:
                center_lats = (
                    pd.to_numeric(rel_maps["Top_left_lat"], errors="coerce")
                    + pd.to_numeric(rel_maps["Bottom_right_lat"], errors="coerce")
                ) / 2.0
                center_lons = (
                    pd.to_numeric(rel_maps["Top_left_lon"], errors="coerce")
                    + pd.to_numeric(rel_maps["Bottom_right_long"], errors="coerce")
                ) / 2.0
                distances = self._haversine_np(
                    search_lat,
                    search_lon,
                    center_lats.to_numpy(dtype=float),
                    center_lons.to_numpy(dtype=float),
                )
                rel_maps = rel_maps.assign(_dist_m=distances).sort_values(
                    by="_dist_m"
                )
                rel_maps = rel_maps.drop(columns=["_dist_m"])
            except Exception:
                pass
        res.candidate_maps = int(len(rel_maps))
        res_dir = self.output_dir / Path(query_filename).stem

        for _, m_row in rel_maps.iterrows():
            res.evaluated_maps += 1
            try:
                m_res = self.engine.match_query_to_map(
                    q_match,
                    q_shape,
                    query_row,
                    m_row,
                    res_dir,
                    min_inliers,
                    save_viz,
                )
                if m_res and self._is_better(m_res, res):
                    self._update(res, m_res)
                    best_match_details = m_res
            except Exception as e:
                if self.config.matcher_params.get("verbose"):
                    _logger.warning(f"Tile positioning failed: {e}")
                continue

        if not res.success and not res.failure_reason:
            res.failure_reason = "no_valid_match_in_candidate_maps"

        if best_match_details is not None:
            for k, v in base_details.items():
                best_match_details.setdefault(k, v)
            return res, best_match_details
        if base_details:
            return res, base_details
        return res, best_match_details
