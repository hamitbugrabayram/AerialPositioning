"""Positioning orchestration logic for aerial position estimation.

This module contains the PositioningRunner class which coordinates the
loading of metadata, preprocessing of images, execution of matching algorithms,
and calculation of positioning accuracy.
"""

from pathlib import Path
from typing import Any, List, Optional, cast

import pandas as pd

from src.core.engine import PositioningEngine
from src.core.factory import PipelineFactory
from src.models.config import PositioningConfig, QueryResult
from src.utils.statistics import ResultManager


class PositioningRunner:
    """Orchestrates the visual positioning process.

    This class manages the lifecycle of a positioning run, including
    initialization of matchers, data loading, result computation, and reporting.
    """

    def __init__(self, config: PositioningConfig):
        """Initializes the runner with the given configuration."""
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
        """Lazily imports utility functions and injects them into the engine."""
        if self._helpers_loaded:
            return True
        from src.utils.helpers import (
            calculate_location_and_error,
            calculate_predicted_gps,
            haversine_distance,
        )
        if self.engine:
            self.engine.inject_helpers(
                haversine_distance, calculate_predicted_gps, calculate_location_and_error
            )
        self._helpers_loaded = True
        return True

    def run(self) -> None:
        """Executes the complete positioning pipeline."""
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
        print("\nComplete")

    def _validate_paths(self) -> None:
        """Ensures all necessary data paths are provided in the config."""
        paths = self.config.data_paths
        req = ["query_dir", "map_dir", "output_dir", "query_metadata", "map_metadata"]
        missing = [p for p in req if not paths.get(p)]
        if missing:
            raise ValueError(f"Missing required paths in config: {missing}")

    def _setup_output_directory(self) -> None:
        """Sets up the output directory structure."""
        self.output_dir = Path(self.config.data_paths["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "output"
        self.assets_dir.mkdir(exist_ok=True)

    def _initialize_preprocessor(self) -> None:
        """Initializes the QueryPreprocessor if enabled."""
        pcfg = self.config.preprocessing
        if not pcfg.get("enabled", False):
            return
        from src.utils.preprocessing import CameraModel, QueryPreprocessor

        cam_model = None
        if self.config.camera_model:
            v_keys = set(CameraModel.__annotations__.keys())
            params = {k: v for k, v in self.config.camera_model.items() if k in v_keys}
            cam_model = CameraModel(**params)
        self.preprocessor = QueryPreprocessor(
            processings=pcfg.get("steps", []),
            resize_target=pcfg.get("resize_target"),
            camera_model=cam_model,
            target_gimbal_yaw=pcfg.get("target_gimbal_yaw", 0.0),
            target_gimbal_pitch=pcfg.get("target_gimbal_pitch", -90.0),
            target_gimbal_roll=pcfg.get("target_gimbal_roll", 0.0),
            adaptive_yaw=pcfg.get("adaptive_yaw", False),
        )

    def _initialize_pipeline(self) -> None:
        """Initializes the matcher pipeline through the PipelineFactory."""
        self.pipeline = PipelineFactory.create(self.config)
        if self.pipeline is None:
            raise RuntimeError("PipelineFactory returned None.")

    def _load_metadata(self) -> None:
        """Loads and cleans query and map metadata from CSV files."""
        paths = self.config.data_paths
        try:
            self.query_df = pd.read_csv(paths["query_metadata"], skipinitialspace=True)
            self.map_df = pd.read_csv(paths["map_metadata"], skipinitialspace=True)
            self.query_df.columns = self.query_df.columns.str.strip()
            self.map_df.columns = self.map_df.columns.str.strip()
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load metadata CSV files: {e}") from e
        self._validate_metadata_columns()

    def _validate_metadata_columns(self) -> None:
        """Validates that all required columns are present in the metadata."""
        if self.query_df is None or self.map_df is None:
            return
        rq = ["Filename", "Latitude", "Longitude"]
        rm = [
            "Filename",
            "Top_left_lat",
            "Top_left_lon",
            "Bottom_right_lat",
            "Bottom_right_long",
        ]
        p_cfg = self.config.preprocessing
        if p_cfg.get("enabled") and "warp" in p_cfg.get("steps", []):
            rq.extend(["Gimball_Yaw", "Gimball_Pitch", "Gimball_Roll", "Flight_Yaw"])
        mq = [c for c in rq if c not in self.query_df.columns]
        mm = [c for c in rm if c not in self.map_df.columns]
        if mq:
            raise ValueError(f"CRITICAL: Query metadata missing columns: {mq}")
        if mm:
            raise ValueError(f"CRITICAL: Map metadata missing columns: {mm}")

    def _process_queries(self) -> List[QueryResult]:
        """Iterates over all query images and performs positioning."""
        results: List[QueryResult] = []
        if self.query_df is None or self.engine is None:
            return results
        p_cfg = self.config.preprocessing
        temp_dir = None
        if self.assets_dir is not None and p_cfg.get("save_processed"):
            temp_dir = self.assets_dir / "processed_queries"
        if temp_dir:
            temp_dir.mkdir(exist_ok=True)
        m_inl = int(self.config.positioning_params.get("min_inliers_for_success", 10))
        s_viz = bool(self.config.positioning_params.get("save_visualization", False))
        for _, row in self.query_df.iterrows():
            try:
                results.append(self._process_single_query(row, temp_dir, m_inl, s_viz))
            except Exception as e:
                print(f"\n  CRITICAL ERROR on query {row.get('Filename')}: {e}")
                results.append(QueryResult(query_filename=str(row.get('Filename')), success=False))
        return results

    def _process_single_query(self, row, temp_dir, min_inliers, save_viz) -> QueryResult:
        """Processes a single query through the engine."""
        fname = str(row["Filename"])
        res = QueryResult(
            query_filename=fname,
            gt_latitude=float(row["Latitude"]),
            gt_longitude=float(row["Longitude"]),
        )
        q_path = Path(self.config.data_paths["query_dir"]) / fname
        if not q_path.is_file() or self.engine is None:
            return res
        try:
            q_for_match, q_shape = self.engine.preprocess_query(q_path, row, temp_dir)
        except Exception:
            return res
        if q_shape is None or self.map_df is None or self.output_dir is None:
            return res
        res_dir = self.output_dir / Path(fname).stem
        rel_maps = self._filter_relevant_maps(row, self.map_df)
        for _, m_row in rel_maps.iterrows():
            m_res = self.engine.match_query_to_map(
                q_for_match, q_shape, row, m_row, res_dir, min_inliers, save_viz
            )
            if m_res and self._is_better(m_res, res):
                self._update(res, m_res)
        return res

    def _filter_relevant_maps(self, row, map_df, radius: float = 600.0) -> pd.DataFrame:
        """Filters map tiles based on proximity."""
        if self.engine is None or self.engine.haversine_distance is None:
            return map_df
        indices = []
        for idx, m_row in map_df.iterrows():
            m_lat = (float(m_row["Top_left_lat"]) + float(m_row["Bottom_right_lat"])) / 2
            m_lon = (float(m_row["Top_left_lon"]) + float(m_row["Bottom_right_long"])) / 2
            dist = self.engine.haversine_distance(
                float(row["Latitude"]), float(row["Longitude"]), m_lat, m_lon
            )
            if dist <= radius:
                indices.append(idx)
        return map_df.loc[indices]

    def _is_better(self, new, curr) -> bool:
        """Compares a new match against the current best."""
        return (
            not curr.success
            or new["inliers"] > curr.inliers
            or (new["inliers"] == curr.inliers and new["error_meters"] < curr.error_meters)
        )

    def _update(self, res, m_res) -> None:
        """Updates result object."""
        res.best_map_filename, res.inliers = m_res["map_filename"], m_res["inliers"]
        res.outliers, res.time = m_res["outliers"], m_res["time"]
        res.predicted_latitude, res.predicted_longitude = m_res["pred_lat"], m_res["pred_lon"]
        res.error_meters, res.success = m_res["error_meters"], True

    def _save_results(self, results: List[QueryResult]) -> None:
        """Delegates result saving to the ResultManager."""
        if self.result_manager:
            q_len = len(self.query_df) if self.query_df is not None else 0
            self.result_manager.save_results(results, q_len)
