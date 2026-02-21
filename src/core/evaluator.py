"""Visual positioning and visualization module.

This module handles the primary visual positioning pipeline, including
displacement-based prediction and path visualization for GNSS-free
coordinate estimation.
"""

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import pandas as pd

from src.core.runner import PositioningRunner
from src.models.config import PositioningConfig, QueryResult
from src.utils.plotting import TrajectoryVisualizer


class Evaluator(PositioningRunner):
    """Orchestrates visual positioning with displacement prediction.

    This class extends PositioningRunner to provide trajectory-based
    evaluation with displacement prediction between sampled frames.

    Attributes:
        full_query_df: Complete query dataframe with all frames.
        sampled_query_df: Subsampled query dataframe at fixed intervals.
        frames_dir: Directory for saving frame visualizations.
        sample_interval: Number of frames between positioning checkpoints.
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
        radius_levels = self.config.positioning_params.get(
            "radius_levels", [1000.0, 2000.0, 3000.0]
        )
        self.radius_levels: List[float] = [float(x) for x in radius_levels]
        if not self.radius_levels:
            self.radius_levels = [1000.0, 2000.0, 3000.0]
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
            print(f"Checkpoints: {num_checkpoints} (Interval: {self.sample_interval})")

        results = self._process_eval_queries()
        self._save_results(results)
        self._cleanup_temp_processed_queries()
        self.visualizer.generate_trajectory_plot(results)
        print("\nPositioning Complete")

    def _process_eval_queries(self) -> List[QueryResult]:
        """Processes images with GT-based displacement prediction.

        Returns:
            List of sampled-frame query results.
        """
        results: List[QueryResult] = []
        if self.sampled_query_df is None or self.full_query_df is None:
            return results

        temp_dir: Optional[Path] = None
        if self.assets_dir and self.config.preprocessing.get("enabled", False):
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

        total_start = time.time()
        total_frames = len(self.sampled_query_df)

        for idx, query_row in self.sampled_query_df.iterrows():
            try:
                filename = str(query_row["Filename"])
                idx_int = int(cast(Any, idx))

                if idx_int == 0:
                    search_lat, search_lon = last_match_lat, last_match_lon
                else:
                    prev_filename = str(self.sampled_query_df.iloc[idx_int - 1]["Filename"])
                    curr_gt = self.full_query_df[self.full_query_df["Filename"] == filename].iloc[0]
                    prev_gt = self.full_query_df[self.full_query_df["Filename"] == prev_filename].iloc[0]

                    d_lat = float(cast(Any, curr_gt)["Latitude"]) - float(cast(Any, prev_gt)["Latitude"])
                    d_lon = float(cast(Any, curr_gt)["Longitude"]) - float(cast(Any, prev_gt)["Longitude"])

                    search_lat = last_match_lat + d_lat
                    search_lon = last_match_lon + d_lon

                result, radius = self._try_match_with_increasing_radius(
                    query_row, idx_int, temp_dir, min_inliers, save_viz, search_lat, search_lon
                )

                if result.success:
                    last_match_lat = float(result.predicted_latitude) if result.predicted_latitude is not None else search_lat
                    last_match_lon = float(result.predicted_longitude) if result.predicted_longitude is not None else search_lon
                else:
                    last_match_lat, last_match_lon = search_lat, search_lon
                    result.success = False

                results.append(result)
                if self.save_frame_sequence:
                    self.visualizer.generate_trajectory_plot(results, frame_idx=idx_int)

                success_count = sum(1 for r in results if r.success)
                print(f"\rFrame: {idx_int+1}/{total_frames} | Radius: {radius}m | Success: {success_count}/{idx_int+1}", end="", flush=True)

            except Exception as e:
                results.append(
                    QueryResult(
                        query_filename=str(query_row.get("Filename")),
                        success=False,
                        failure_reason=f"evaluator_exception: {e}",
                    )
                )
                continue

        print()
        total_time = time.time() - total_start
        print(f"Total Processing Time: {total_time:.2f}s")
        return results

    def _try_match_with_increasing_radius(
        self,
        query_row: pd.Series,
        idx: int,
        temp_dir: Optional[Path],
        min_inliers: int,
        save_viz: bool,
        ref_lat: float,
        ref_lon: float,
    ) -> Tuple[QueryResult, float]:
        """Tries matching with progressively increasing search radius.

        Attempts matching at 1000m, 2000m, and 3000m radii. If all fail,
        returns a failed result.

        Args:
            query_row: Pandas Series containing query metadata.
            idx: Index of the current query.
            temp_dir: Optional directory for saving processed queries.
            min_inliers: Minimum inliers required for success.
            save_viz: Whether to save match visualizations.
            ref_lat: Reference latitude for search window.
            ref_lon: Reference longitude for search window.

        Returns:
            Tuple containing QueryResult and the radius used.
        """
        final_result = QueryResult(query_filename=str(query_row.get("Filename")), success=False)

        for radius in self.radius_levels:
            final_result = self._process_single_query_with_custom_radius(
                query_row, idx, temp_dir, min_inliers, save_viz, ref_lat, ref_lon, radius
            )
            final_result.search_radius_m = radius
            if final_result.success:
                return final_result, radius

        final_result.failure_reason = final_result.failure_reason or "all_radius_levels_failed"
        return final_result, self.radius_levels[-1]

    def _process_single_query_with_custom_radius(
        self,
        query_row: pd.Series,
        idx: int,
        temp_dir: Optional[Path],
        min_inliers: int,
        save_viz: bool,
        ref_lat: float,
        ref_lon: float,
        radius: float,
    ) -> QueryResult:
        """Processes a single query with custom search radius.

        Args:
            query_row: Query metadata row.
            idx: Query index in sampled sequence.
            temp_dir: Optional directory for processed query images.
            min_inliers: Minimum inlier threshold.
            save_viz: Whether to save match visualizations.
            ref_lat: Reference latitude for candidate filtering.
            ref_lon: Reference longitude for candidate filtering.
            radius: Search radius in meters.

        Returns:
            Per-query best match result.
        """
        query_filename = str(query_row["Filename"])
        gt_lat = float(cast(Any, query_row).get("Latitude", 0.0))
        gt_lon = float(cast(Any, query_row).get("Longitude", 0.0))
        res = QueryResult(
            query_filename=query_filename,
            gt_latitude=gt_lat,
            gt_longitude=gt_lon,
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

        rel_maps = self._filter_maps_by_custom_ref(ref_lat, ref_lon, self.map_df, radius)
        res.candidate_maps = int(len(rel_maps))
        res_dir = self.output_dir / Path(query_filename).stem

        for _, m_row in rel_maps.iterrows():
            res.evaluated_maps += 1
            try:
                m_res = self.engine.match_query_to_map(
                    q_match, q_shape, query_row, m_row, res_dir, min_inliers, save_viz
                )
                if m_res and self._is_better(m_res, res):
                    self._update(res, m_res)
            except Exception as e:
                if self.config.matcher_params.get("verbose"):
                    print(f"    Tile positioning failed: {e}")
                continue
        if not res.success and not res.failure_reason:
            res.failure_reason = "no_valid_match_in_candidate_maps"
        return res

    def _filter_maps_by_custom_ref(
        self,
        lat: float,
        lon: float,
        map_df: pd.DataFrame,
        radius: float,
    ) -> pd.DataFrame:
        """Filters map tiles by distance from a reference point.

        Args:
            lat: Reference latitude.
            lon: Reference longitude.
            map_df: Map metadata dataframe.
            radius: Search radius in meters.

        Returns:
            Filtered map metadata dataframe.
        """
        if self.engine is None:
            return map_df
        return self._filter_maps_by_reference(lat, lon, map_df, radius)
