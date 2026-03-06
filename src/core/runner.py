
"""Positioning orchestration logic for aerial position estimation.

This module contains the PositioningRunner class which coordinates the
loading of metadata, preprocessing of images, execution of matching algorithms,
and calculation of positioning accuracy.
"""

import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.engine import PositioningEngine
from src.core.factory import PipelineFactory
from src.models.config import PositioningConfig, QueryResult
from src.utils.statistics import ResultManager

from src.utils.logger import get_logger
_logger = get_logger(__name__)


class PositioningRunner:
    """Orchestrates the visual positioning process.

    This class manages the lifecycle of a positioning run, including
    initialization of matchers, data loading, result computation, and reporting.

    Attributes:
        config (PositioningConfig): Global positioning configuration object.
        pipeline (Any): The instantiated matching pipeline.
        preprocessor (Any): The image preprocessing module.
        engine (PositioningEngine): The core positioning engine.
        result_manager (ResultManager): Handles result storage and statistics.
        query_df (Optional[pd.DataFrame]): DataFrame containing query metadata.
        map_df (Optional[pd.DataFrame]): DataFrame containing map tile metadata.
        output_dir (Optional[Path]): Directory for saving results.
        assets_dir (Optional[Path]): Directory for saving visual assets.
    """

    def __init__(self, config: PositioningConfig):
        """Initializes the runner with the given configuration.

        Args:
            config: Global positioning configuration object.
        """
        self.config = config
        self.pipeline = None
        self.preprocessor = None
        self.engine = None
        self.result_manager = None
        self.query_df: Optional[pd.DataFrame] = None
        self.map_df: Optional[pd.DataFrame] = None
        self.output_dir: Optional[Path] = None
        self.assets_dir: Optional[Path] = None
        self._helpers_loaded = False

    def _load_helpers(self) -> bool:
        """Lazily imports utility functions and injects them into the engine.

        Returns:
            `True` when helper injection is ready.
        """
        if self._helpers_loaded:
            return True
        from src.utils.helpers import (calculate_location_and_error,
                                       calculate_predicted_gps,
                                       haversine_distance)

        if self.engine:
            self.engine.inject_helpers(
                haversine_distance,
                calculate_predicted_gps,
                calculate_location_and_error,
            )
        self._helpers_loaded = True
        return True

    def run(self) -> None:
        """Executes the complete positioning pipeline.

        Returns:
            None.
        """
        self._validate_paths()
        self._setup_output_directory()
        self._initialize_preprocessor()
        self._initialize_pipeline()

        self.engine = PositioningEngine(self.config, self.pipeline, self.preprocessor)
        if self.output_dir and self.assets_dir:
            self.result_manager = ResultManager(self.output_dir, self.assets_dir)
        self._load_helpers()

        self._load_metadata()
        results = self._process_queries()
        self._save_results(results)
        self._cleanup_temp_processed_queries()
        self._cleanup_temp_context_maps()
        _logger.info("\nComplete")

    def _validate_paths(self) -> None:
        """Ensures all necessary data paths are provided in the config.

        Returns:
            None.
        """
        paths = self.config.data_paths
        req = ["query_dir", "map_dir", "output_dir", "query_metadata", "map_metadata"]
        missing = [p for p in req if not paths.get(p)]
        if missing:
            raise ValueError(f"Missing required paths in config: {missing}")

    def _setup_output_directory(self) -> None:
        """Sets up the output directory structure.

        Returns:
            None.
        """
        self.output_dir = Path(self.config.data_paths["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir

    def _initialize_preprocessor(self) -> None:
        """Initializes the QueryPreprocessor if enabled.

        Returns:
            None.
        """
        from src.utils.preprocessing import CameraModel, QueryPreprocessor

        cam_model = None
        if self.config.camera_model:
            v_keys = set(CameraModel.__annotations__.keys())
            params = {k: v for k, v in self.config.camera_model.items() if k in v_keys}
            cam_model = CameraModel(**params)
        self.preprocessor = QueryPreprocessor(
            camera_model=cam_model,
            device=self.config.device if hasattr(self.config, "device") else "cpu",
        )

    def _initialize_pipeline(self) -> None:
        """Initializes the matcher pipeline through the PipelineFactory.

        Returns:
            None.
        """
        self.pipeline = PipelineFactory.create(self.config)
        if self.pipeline is None:
            raise RuntimeError("PipelineFactory returned None.")

    def _load_metadata(self) -> None:
        """Loads and cleans query and map metadata from CSV files.

        Returns:
            None.
        """
        paths = self.config.data_paths
        try:
            self.query_df = pd.read_csv(paths["query_metadata"], skipinitialspace=True)
            self.map_df = pd.read_csv(paths["map_metadata"], skipinitialspace=True)
            self.query_df.columns = self.query_df.columns.str.strip()
            self.map_df.columns = self.map_df.columns.str.strip()
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load metadata CSV files: {e}"
            ) from e
        self._validate_metadata_columns()

    def _validate_metadata_columns(self) -> None:
        """Validates that all required columns are present in the metadata.

        Returns:
            None.
        """
        if self.query_df is None or self.map_df is None:
            return
        rq = [
            "Filename",
            "Latitude",
            "Longitude",
            "Gimball_Yaw",
            "Gimball_Pitch",
            "Gimball_Roll",
        ]
        rm = [
            "Filename",
            "Top_left_lat",
            "Top_left_lon",
            "Bottom_right_lat",
            "Bottom_right_long",
        ]
        mq = [c for c in rq if c not in self.query_df.columns]
        mm = [c for c in rm if c not in self.map_df.columns]
        if mq:
            raise ValueError(f"CRITICAL: Query metadata missing columns: {mq}")
        if mm:
            raise ValueError(f"CRITICAL: Map metadata missing columns: {mm}")

    def _process_queries(self) -> List[QueryResult]:
        """Iterates over all query images and performs positioning.

        Returns:
            List of per-query positioning outputs.
        """
        results: List[QueryResult] = []
        if self.query_df is None or self.engine is None:
            return results
        p_cfg = self.config.preprocessing
        temp_dir = None
        if self.assets_dir is not None:
            if p_cfg.get("save_processed", False):
                temp_dir = self.assets_dir / "processed_queries"
            else:
                temp_dir = self.assets_dir / ".tmp_processed_queries"
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)
        m_inl = int(self.config.positioning_params.get("min_inliers_for_success", 10))
        s_viz = bool(self.config.positioning_params.get("save_visualization", False))
        for _, row in self.query_df.iterrows():
            try:
                results.append(self._process_single_query(row, temp_dir, m_inl, s_viz))
            except Exception as e:
                _logger.info(f"\n  CRITICAL ERROR on query {row.get('Filename')}: {e}")
                results.append(
                    QueryResult(
                        query_filename=str(row.get("Filename")),
                        success=False,
                        failure_reason=f"runner_exception: {e}",
                    )
                )
        return results

    def _process_single_query(
        self, row, temp_dir, min_inliers, save_viz
    ) -> QueryResult:
        """Processes a single query through the engine.

        Args:
            row: Query metadata row.
            temp_dir: Optional processed-image output directory.
            min_inliers: Minimum inlier threshold.
            save_viz: Whether to save visualization artifacts.

        Returns:
            Per-query best result.
        """
        fname = str(row["Filename"])
        res = QueryResult(
            query_filename=fname,
            gt_latitude=float(row["Latitude"]),
            gt_longitude=float(row["Longitude"]),
        )
        q_path = Path(self.config.data_paths["query_dir"]) / fname
        if not q_path.is_file() or self.engine is None:
            res.failure_reason = "query_file_missing_or_engine_unavailable"
            return res
        try:
            q_for_match, q_shape = self.engine.preprocess_query(q_path, row, temp_dir)
        except Exception as e:
            res.failure_reason = f"preprocess_failed: {e}"
            return res
        if q_shape is None or self.map_df is None or self.output_dir is None:
            res.failure_reason = "query_shape_or_metadata_unavailable"
            return res
        res_dir = self.output_dir / Path(fname).stem
        rel_maps = self._filter_relevant_maps(row, self.map_df)
        res.candidate_maps = int(len(rel_maps))
        for _, m_row in rel_maps.iterrows():
            res.evaluated_maps += 1
            m_res = self.engine.match_query_to_map(
                q_for_match, q_shape, row, m_row, res_dir, min_inliers, save_viz
            )
            if m_res and self._is_better(m_res, res):
                self._update(res, m_res)
        if not res.success and not res.failure_reason:
            res.failure_reason = "no_valid_match_in_candidate_maps"
        return res

    def _filter_relevant_maps(self, row, map_df, radius: float = 600.0) -> pd.DataFrame:
        """Filters map tiles based on proximity.

        Args:
            row: Query metadata row.
            map_df: Map metadata dataframe.
            radius: Search radius in meters.

        Returns:
            Filtered map metadata dataframe.
        """
        if self.engine is None:
            return map_df
        return self._filter_maps_by_reference(
            float(row["Latitude"]),
            float(row["Longitude"]),
            map_df,
            radius,
        )

    def _filter_maps_by_reference(
        self, lat: float, lon: float, map_df: pd.DataFrame, radius: float
    ) -> pd.DataFrame:
        """Filters map tiles by distance to a reference coordinate.

        Args:
            lat: Reference latitude.
            lon: Reference longitude.
            map_df: Map metadata dataframe.
            radius: Search radius in meters.

        Returns:
            Filtered dataframe containing only tiles within the radius.
        """
        if map_df.empty:
            return map_df
        try:
            center_lats = (
                pd.to_numeric(map_df["Top_left_lat"], errors="coerce")
                + pd.to_numeric(map_df["Bottom_right_lat"], errors="coerce")
            ) / 2.0
            center_lons = (
                pd.to_numeric(map_df["Top_left_lon"], errors="coerce")
                + pd.to_numeric(map_df["Bottom_right_long"], errors="coerce")
            ) / 2.0

            valid = center_lats.notna() & center_lons.notna()
            if not valid.any():
                return map_df.iloc[0:0]

            distances = self._haversine_np(
                lat,
                lon,
                center_lats[valid].to_numpy(dtype=float),
                center_lons[valid].to_numpy(dtype=float),
            )
            selected_indices = center_lats[valid].index[distances <= radius]
            return map_df.loc[selected_indices]
        except Exception:
            return map_df

    @staticmethod
    def _haversine_np(
        lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray
    ) -> np.ndarray:
        """Computes vectorized haversine distance in meters.

        Args:
            lat1: Reference latitude.
            lon1: Reference longitude.
            lat2_arr: Candidate latitude array.
            lon2_arr: Candidate longitude array.

        Returns:
            Distance array in meters.
        """
        earth_radius = 6371000.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2_arr)
        lon2_rad = np.radians(lon2_arr)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        return earth_radius * c

    def _is_better(self, new, curr) -> bool:
        """Compares a new match against the current best.

        Args:
            new: Candidate match dictionary.
            curr: Current best query result.

        Returns:
            `True` if the candidate should replace current best result.
        """
        return (
            not curr.success
            or new["inliers"] > curr.inliers
            or (
                new["inliers"] == curr.inliers
                and new["error_meters"] < curr.error_meters
            )
        )

    def _update(self, res, m_res) -> None:
        """Updates result object.

        Args:
            res: QueryResult instance to mutate.
            m_res: Match dictionary from engine.

        Returns:
            None.
        """
        res.best_map_filename, res.inliers = m_res["map_filename"], m_res["inliers"]
        res.outliers, res.time = m_res["outliers"], m_res["time"]
        res.predicted_latitude, res.predicted_longitude = (
            m_res["pred_lat"],
            m_res["pred_lon"],
        )
        res.error_meters, res.success = m_res["error_meters"], True
        res.failure_reason = None

    def _save_results(self, results: List[QueryResult]) -> None:
        """Delegates result saving to the ResultManager.

        Args:
            results: Final query results.

        Returns:
            None.
        """
        if self.result_manager:
            q_len = len(self.query_df) if self.query_df is not None else 0
            self.result_manager.save_results(results, q_len)

    def _cleanup_temp_processed_queries(self) -> None:
        """Removes runtime-only processed files when persistence is disabled.

        Returns:
            None.
        """
        if self.assets_dir is None:
            return
        if self.config.preprocessing.get("save_processed", False):
            return
        tmp_dir = self.assets_dir / ".tmp_processed_queries"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cleanup_temp_context_maps(self) -> None:
        """Removes runtime-only stitched context maps when persistence is disabled."""
        if self.assets_dir is None or self.engine is None:
            return
        if self.engine._save_map_context_maps():
            return
        tmp_dir = self.assets_dir / ".tmp_context_maps"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
