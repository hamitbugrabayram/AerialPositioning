"""Visual positioning and visualization module.

This module handles the primary visual positioning pipeline, including
adaptive search radius management and path visualization for GNSS-free
coordinate estimation.
"""

import time
from pathlib import Path
from typing import Any, List, Optional, cast

import pandas as pd

from src.core.runner import PositioningRunner
from src.models.config import PositioningConfig, QueryResult
from src.utils.logger import get_logger
from src.utils.plotting import TrajectoryVisualizer

_logger = get_logger(__name__)

_logger = get_logger(__name__)


class Evaluator(PositioningRunner):
    """Orchestrates visual positioning with adaptive search.

    Uses an exponential backoff strategy for search radius management:
    on successful match the radius cools down toward the initial value;
    on failure the radius grows by a configurable factor, up to a hard
    ceiling. The search center always tracks the last successfully
    matched position.

    Attributes:
        full_query_df: Complete query dataframe with all frames.
        sampled_query_df: Subsampled query dataframe at fixed intervals.
        frames_dir: Directory for saving frame visualizations.
        sample_interval: Number of frames between positioning checkpoints.
        initial_radius_m: Starting radius for adaptive search.
        max_radius_m: Maximum allowed search radius.
        growth_factor: Multiplier for radius upon match failure.
        cooldown_factor: Decay factor for radius upon match success.
    """

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
        self.initial_radius_m = float(search_conf.get("initial_radius_m", 1000.0))
        self.max_radius_m = float(search_conf.get("max_radius_m", 10000.0))
        self.growth_factor = float(search_conf.get("growth_factor", 1.5))
        self.cooldown_factor = float(search_conf.get("cooldown_factor", 0.5))

        self.visualizer = TrajectoryVisualizer(config)

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

        results = self._process_eval_queries()
        self._save_results(results)
        self._cleanup_temp_processed_queries()
        self.visualizer.generate_trajectory_plot(results)
        _logger.info("Positioning Complete")

    def _process_eval_queries(self) -> List[QueryResult]:
        """Processes sampled frames using adaptive radius search.

        The algorithm maintains a dynamic search radius that starts at
        ``initial_radius_m``. After each frame:

        * **Success** -- the radius decays toward ``initial_radius_m``
          by the ``cooldown_factor``, and the search center snaps to the
          matched position.
        * **Failure** -- the radius is multiplied by ``growth_factor``
          (capped at ``max_radius_m``), and the search center remains at
          the last known position.

        Returns:
            List of sampled-frame query results.
        """
        results: List[QueryResult] = []
        if self.sampled_query_df is None:
            return results

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

        last_match_lat = float(self.sampled_query_df.iloc[0]["Latitude"])
        last_match_lon = float(self.sampled_query_df.iloc[0]["Longitude"])
        current_radius = self.initial_radius_m
        consecutive_failures = 0

        total_start = time.time()
        total_frames = len(self.sampled_query_df)

        for idx, query_row in self.sampled_query_df.iterrows():
            try:
                idx_int = int(cast(Any, idx))
                search_lat = last_match_lat
                search_lon = last_match_lon

                # Search dynamically within the same frame up to max_radius_m
                while True:
                    result = self._match_with_adaptive_radius(
                        query_row,
                        idx_int,
                        temp_dir,
                        min_inliers,
                        save_viz,
                        search_lat,
                        search_lon,
                        current_radius,
                    )

                    if result.success:
                        last_match_lat = float(
                            result.predicted_latitude
                            if result.predicted_latitude is not None
                            else search_lat
                        )
                        last_match_lon = float(
                            result.predicted_longitude
                            if result.predicted_longitude is not None
                            else search_lon
                        )
                        excess = current_radius - self.initial_radius_m
                        if excess > 0:
                            current_radius = (
                                self.initial_radius_m + excess * self.cooldown_factor
                            )
                        consecutive_failures = 0
                        break  # Found a match, move to next frame
                    else:
                        consecutive_failures += 1
                        if current_radius >= self.max_radius_m:
                            # Reached max radius, stop searching this frame
                            break
                        
                        _logger.info(
                            f"Frame: {idx_int + 1}/{total_frames} failed at "
                            f"{current_radius:.0f}m. Expanding search radius..."
                        )
                        current_radius = min(
                            current_radius * self.growth_factor, self.max_radius_m
                        )

                results.append(result)
                if self.save_frame_sequence:
                    self.visualizer.generate_trajectory_plot(results, frame_idx=idx_int)

                success_count = sum(1 for r in results if r.success)
                _logger.info(
                    f"Frame: {idx_int + 1}/{total_frames} | "
                    f"Radius: {current_radius:.0f}m | "
                    f"Success: {success_count}/{idx_int + 1}"
                )

            except Exception as e:
                results.append(
                    QueryResult(
                        query_filename=str(query_row.get("Filename")),
                        success=False,
                        failure_reason=f"evaluator_exception: {e}",
                    )
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
    ) -> QueryResult:
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
            QueryResult for this frame.
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

        q_path = Path(self.config.data_paths["query_dir"]) / query_filename
        if not q_path.is_file() or self.engine is None:
            res.failure_reason = "query_file_missing_or_engine_unavailable"
            return res

        try:
            q_match, q_shape = self.engine.preprocess_query(q_path, query_row, temp_dir)
        except Exception as e:
            res.failure_reason = f"preprocess_failed: {e}"
            return res

        if q_shape is None or self.map_df is None or self.output_dir is None:
            res.failure_reason = "query_shape_or_metadata_unavailable"
            return res

        rel_maps = self._filter_maps_by_reference(
            search_lat, search_lon, self.map_df, radius
        )
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
            except Exception as e:
                if self.config.matcher_params.get("verbose"):
                    _logger.warning(f"Tile positioning failed: {e}")
                continue

        if not res.success and not res.failure_reason:
            res.failure_reason = "no_valid_match_in_candidate_maps"
        return res
