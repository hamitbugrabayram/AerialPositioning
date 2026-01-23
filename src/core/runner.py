"""Positioning orchestration logic for aerial position estimation.

This module contains the PositioningRunner class which coordinates the
loading of metadata, preprocessing of images, execution of matching algorithms,
and calculation of positioning accuracy.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.core.factory import PipelineFactory
from src.models.config import PositioningConfig, QueryResult


class PositioningRunner:
    """Orchestrates the visual positioning process.

    This class manages the lifecycle of a positioning run, including
    initialization of matchers, data loading, result computation, and reporting.

    Attributes:
        config: Parsed positioning configuration.
        pipeline: Initialized matcher pipeline.
        preprocessor: Image preprocessor instance.
        query_df: Metadata for query images.
        map_df: Metadata for satellite tiles.
        output_dir: Directory where results are stored.
    """

    def __init__(self, config: PositioningConfig):
        """Initializes the runner with the given configuration.

        Args:
            config: The parsed configuration object.
        """
        self.config = config
        self.pipeline = None
        self.preprocessor = None
        self.query_df: Optional[pd.DataFrame] = None
        self.map_df: Optional[pd.DataFrame] = None
        self.output_dir: Optional[Path] = None
        self._processed_images: Dict[Path, np.ndarray] = {}
        self._helpers_loaded = False
        self._haversine_distance = None
        self._calculate_predicted_gps = None
        self._calculate_location_and_error = None

    def _load_helpers(self) -> bool:
        """Lazily imports utility functions to avoid circular dependencies.

        Returns:
            bool: True if helpers loaded successfully, False otherwise.
        """
        if self._helpers_loaded:
            return True
        try:
            from src.utils.helpers import (
                calculate_location_and_error,
                calculate_predicted_gps,
                haversine_distance,
            )

            self._haversine_distance = haversine_distance
            self._calculate_predicted_gps = calculate_predicted_gps
            self._calculate_location_and_error = calculate_location_and_error
            self._helpers_loaded = True
            return True
        except ImportError as e:
            print(f"WARNING: Failed to import helper functions: {e}")
            return False

    def run(self) -> None:
        """Executes the complete positioning pipeline."""
        print("\n[Aerial Positioning System]")
        if not self._load_helpers():
            raise RuntimeError("Required helper functions not available.")
        self._validate_paths()
        self._setup_output_directory()
        self._initialize_preprocessor()
        self._initialize_pipeline()
        self._load_metadata()
        results = self._process_queries()
        self._save_results(results)
        print("\n[Complete]")

    def _validate_paths(self) -> None:
        """Ensures all necessary data paths are provided in the config.

        Raises:
            ValueError: If required paths are missing from configuration.
        """
        paths = self.config.data_paths
        required = [
            "query_dir",
            "map_dir",
            "output_dir",
            "query_metadata",
            "map_metadata",
        ]
        missing = [p for p in required if not paths.get(p)]
        if missing:
            raise ValueError(f"Missing required paths in config: {missing}")

    def _setup_output_directory(self) -> None:
        """Sets up the output directory based on the configuration."""
        paths = self.config.data_paths
        self.output_dir = Path(paths["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")

    def _get_matcher_suffix(self) -> str:
        """Generates a suffix string based on the chosen matcher and its weights.

        Returns:
            str: Suffix string for output file naming.
        """
        matcher_type = self.config.matcher_type
        if matcher_type == "gim":
            model_type = self.config.matcher_weights.get("gim_model_type", "unknown")
            return f"_{model_type}"
        if matcher_type == "loftr":
            weights_path = self.config.matcher_weights.get(
                "loftr_weights_path", "unknown.ckpt"
            )
            return f"_{Path(weights_path).stem}"
        if matcher_type == "minima":
            method = self.config.matcher_weights.get("minima_method", "xoftr")
            return f"_{method}"
        return ""

    def _initialize_preprocessor(self) -> None:
        """Initializes the QueryPreprocessor if preprocessing is enabled.

        Raises:
            RuntimeError: If preprocessing modules fail to import or initialize.
            ValueError: If warp step is enabled without camera model.
        """
        preprocess_config = self.config.preprocessing
        if not preprocess_config.get("enabled", False):
            return
        try:
            from src.utils.preprocessing import CameraModel, QueryPreprocessor
        except ImportError:
            raise RuntimeError("Preprocessing enabled but modules failed to import.")
        camera_model = None
        if self.config.camera_model:
            try:
                valid_keys = set(CameraModel.__annotations__.keys())
                cam_params = {
                    k: v for k, v in self.config.camera_model.items() if k in valid_keys
                }
                camera_model = CameraModel(**cam_params)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize camera model: {e}")
        elif "warp" in preprocess_config.get("steps", []):
            raise ValueError("Warp step requires camera_model configuration.")
        try:
            self.preprocessor = QueryPreprocessor(
                processings=preprocess_config.get("steps", []),
                resize_target=preprocess_config.get("resize_target"),
                camera_model=camera_model,
                target_gimbal_yaw=preprocess_config.get("target_gimbal_yaw", 0.0),
                target_gimbal_pitch=preprocess_config.get("target_gimbal_pitch", -90.0),
                target_gimbal_roll=preprocess_config.get("target_gimbal_roll", 0.0),
                adaptive_yaw=preprocess_config.get("adaptive_yaw", False),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize preprocessor: {e}")

    def _initialize_pipeline(self) -> None:
        """Initializes the matcher pipeline through the PipelineFactory.

        Raises:
            RuntimeError: If pipeline initialization fails.
        """
        try:
            self.pipeline = PipelineFactory.create(self.config)
            if self.pipeline:
                print(f"Matcher: {self.pipeline.name}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize pipeline: {e}") from e

    def _load_metadata(self) -> None:
        """Loads and cleans query and map metadata from CSV files.

        Raises:
            RuntimeError: If metadata files cannot be loaded.
        """
        paths = self.config.data_paths
        try:
            self.query_df = pd.read_csv(paths["query_metadata"], skipinitialspace=True)
            self.map_df = pd.read_csv(paths["map_metadata"], skipinitialspace=True)
            self.query_df.columns = self.query_df.columns.str.strip()
            self.map_df.columns = self.map_df.columns.str.strip()
            print(f"Queries: {len(self.query_df)}, Maps: {len(self.map_df)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}") from e
        self._validate_metadata_columns()

    def _validate_metadata_columns(self) -> None:
        """Validates that all required columns are present in the metadata DataFrames.

        Raises:
            ValueError: If required columns are missing from metadata.
        """
        if self.query_df is None or self.map_df is None:
            return
        required_query = ["Filename", "Latitude", "Longitude"]
        required_map = [
            "Filename",
            "Top_left_lat",
            "Top_left_lon",
            "Bottom_right_lat",
            "Bottom_right_long",
        ]
        steps = self.config.preprocessing.get("steps", [])
        if self.config.preprocessing.get("enabled") and "warp" in steps:
            required_query.extend(
                ["Gimball_Yaw", "Gimball_Pitch", "Gimball_Roll", "Flight_Yaw"]
            )
        missing_query = [c for c in required_query if c not in self.query_df.columns]
        missing_map = [c for c in required_map if c not in self.map_df.columns]
        if missing_query:
            raise ValueError(f"Query metadata missing columns: {missing_query}")
        if missing_map:
            raise ValueError(f"Map metadata missing columns: {missing_map}")

    def _process_queries(self) -> List[QueryResult]:
        """Iterates over all query images and performs positioning.

        Returns:
            List[QueryResult]: A list of QueryResult objects containing
                estimated positions.
        """
        results = []
        if self.query_df is None:
            return results
        save_processed = self.config.preprocessing.get("save_processed", False)
        if save_processed and self.output_dir:
            temp_dir = self.output_dir / "processed_queries"
            temp_dir.mkdir(exist_ok=True)
        else:
            temp_dir = None
        min_inliers = self.config.positioning_params.get("min_inliers_for_success", 10)
        save_viz = self.config.positioning_params.get("save_visualization", False)
        total_start = time.time()
        for idx, query_row in self.query_df.iterrows():
            current_idx = int(idx)
            result = self._process_single_query(
                query_row, current_idx, temp_dir, min_inliers, save_viz
            )
            results.append(result)
        total_time = time.time() - total_start
        print(f"Processing time: {total_time:.2f}s")
        return results

    def _filter_relevant_maps(
        self, query_row: pd.Series, map_df: pd.DataFrame, radius_meters: float = 600.0
    ) -> pd.DataFrame:
        """Filters satellite tiles based on initial coordinate proximity.

        Args:
            query_row: Metadata for the query image.
            map_df: DataFrame containing all satellite tiles.
            radius_meters: Search radius for filtering.

        Returns:
            pd.DataFrame: A filtered DataFrame of relevant satellite tiles.
        """
        q_lat = query_row.get("Latitude")
        q_lon = query_row.get("Longitude")
        if q_lat is None or q_lon is None or self._haversine_distance is None:
            return map_df
        relevant_indices = []
        for idx, map_row in map_df.iterrows():
            try:
                m_lat = (
                    float(map_row["Top_left_lat"]) + float(map_row["Bottom_right_lat"])
                ) / 2
                m_lon = (
                    float(map_row["Top_left_lon"]) + float(map_row["Bottom_right_long"])
                ) / 2
                dist = self._haversine_distance(
                    float(q_lat), float(q_lon), m_lat, m_lon
                )
                if dist <= radius_meters:
                    relevant_indices.append(idx)
            except (ValueError, TypeError):
                continue
        return map_df.loc[relevant_indices]

    def _process_single_query(
        self,
        query_row: pd.Series,
        idx: int,
        temp_dir: Optional[Path],
        min_inliers: int,
        save_viz: bool,
    ) -> QueryResult:
        """Handles matching and positioning for a single query image.

        Args:
            query_row: The metadata series for the query.
            idx: Index of the query.
            temp_dir: Directory for storing intermediate preprocessed images.
            min_inliers: Threshold for considering a match successful.
            save_viz: Whether to generate visualization images.

        Returns:
            QueryResult: The positioning result.
        """
        query_filename = str(query_row["Filename"])
        query_path = Path(self.config.data_paths["query_dir"]) / query_filename
        result = QueryResult(
            query_filename=query_filename,
            gt_latitude=query_row.get("Latitude"),
            gt_longitude=query_row.get("Longitude"),
        )
        if not query_path.is_file():
            print("  WARNING: Query image not found, skipping")
            return result
        query_for_match, query_shape = self._preprocess_query(
            query_path, query_row, temp_dir
        )
        if query_shape is None or self.map_df is None or self.output_dir is None:
            print("  WARNING: Invalid query shape, skipping")
            return result
        query_results_dir = self.output_dir / Path(query_filename).stem
        relevant_maps = self._filter_relevant_maps(query_row, self.map_df)
        print(
            f"  Matching against {len(relevant_maps)} relevant maps "
            f"(out of {len(self.map_df)})"
        )
        for _, map_row in relevant_maps.iterrows():
            match_result = self._match_query_to_map(
                query_for_match,
                query_shape,
                query_row,
                map_row,
                query_results_dir,
                min_inliers,
                save_viz,
            )
            if match_result is not None:
                if self._is_better_match(match_result, result):
                    self._update_result(result, match_result)
        if result.success:
            print(
                f"  Best: {result.best_map_filename} "
                f"({result.inliers} inliers, {result.error_meters:.2f}m error)"
            )
        else:
            print("  No successful positioning")
        return result

    def _update_result(self, result: QueryResult, match_result: Dict[str, Any]) -> None:
        """Updates the best result with data from a new successful match.

        Args:
            result: The current QueryResult to update.
            match_result: Dictionary containing new match data.
        """
        result.best_map_filename = match_result["map_filename"]
        result.inliers = match_result["inliers"]
        result.outliers = match_result["outliers"]
        result.time = match_result["time"]
        result.predicted_latitude = match_result["pred_lat"]
        result.predicted_longitude = match_result["pred_lon"]
        result.error_meters = match_result["error_meters"]
        result.success = True

    def _preprocess_query(
        self, query_path: Path, query_row: pd.Series, temp_dir: Optional[Path]
    ) -> Tuple[Path, Optional[Tuple[int, ...]]]:
        """Applies configured preprocessing steps to a query image.

        Args:
            query_path: Path to the query image file.
            query_row: Metadata for the query image.
            temp_dir: Directory for saving preprocessed images.

        Returns:
            Tuple[Path, Optional[Tuple[int, ...]]]: Tuple of
                (path to use for matching, image shape).
        """
        if self.preprocessor is None:
            img = cv2.imread(str(query_path))
            shape = img.shape if img is not None else None
            return query_path, shape
        img_original = cv2.imread(str(query_path))
        if img_original is None:
            return query_path, None
        processed = self.preprocessor(img_original, query_row.to_dict())
        shape = processed.shape
        if processed.shape == img_original.shape and np.array_equal(
            processed, img_original
        ):
            return query_path, shape
        save_processed = self.config.preprocessing.get("save_processed", False)
        if not save_processed:
            self._processed_images[query_path] = processed
            return query_path, shape
        if temp_dir:
            processed_name = (
                f"{Path(query_path.name).stem}_processed{Path(query_path.name).suffix}"
            )
            processed_path = temp_dir / processed_name
            try:
                cv2.imwrite(str(processed_path), processed)
                return processed_path, shape
            except Exception as e:
                print(f"  WARNING: Failed to save processed image: {e}")
        return query_path, img_original.shape

    def _match_query_to_map(
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
            query_path: Path to the query image.
            query_shape: Shape of the query image.
            query_row: Query metadata.
            map_row: Map tile metadata.
            results_dir: Directory for saving results.
            min_inliers: Minimum inliers for success.
            save_viz: Whether to save visualization.

        Returns:
            Optional[Dict[str, Any]]: Match result dictionary or None if failed.
        """
        map_filename = str(map_row["Filename"])
        map_path = Path(self.config.data_paths["map_dir"]) / map_filename
        if not map_path.is_file() or self.pipeline is None:
            return None
        map_img = cv2.imread(str(map_path))
        if map_img is None:
            return None
        map_shape = map_img.shape
        try:
            match_results = self.pipeline.match(query_path, map_path)
        except Exception as e:
            print(f"  ERROR matching with {map_filename}: {e}")
            return None
        match_time = match_results.get("time", 0)
        homography = match_results.get("homography")
        inliers_mask = match_results.get("inliers")
        mkpts0 = match_results.get("mkpts0", np.array([]))
        mkpts1 = match_results.get("mkpts1", np.array([]))
        num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0
        num_total = len(mkpts0)
        num_outliers = num_total - num_inliers
        ransac_successful = (
            match_results.get("success", False) and num_inliers >= min_inliers
        )
        positioning_result = self._compute_positioning(
            ransac_successful, homography, query_row, map_row, query_shape, map_shape
        )
        if positioning_result["success"] or (save_viz and ransac_successful):
            results_dir.mkdir(exist_ok=True)
            if save_viz and ransac_successful:
                self._save_visualization(
                    results_dir,
                    query_path,
                    map_path,
                    mkpts0,
                    mkpts1,
                    inliers_mask,
                    homography,
                )
        if not positioning_result["success"]:
            return None
        return {
            "map_filename": map_filename,
            "inliers": int(num_inliers),
            "outliers": int(num_outliers),
            "time": match_time,
            "pred_lat": positioning_result["pred_lat"],
            "pred_lon": positioning_result["pred_lon"],
            "error_meters": positioning_result["error_meters"],
        }

    def _compute_positioning(
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
            ransac_successful: Whether RANSAC succeeded.
            homography: The estimated homography matrix.
            query_row: Query metadata.
            map_row: Map metadata.
            query_shape: Query image shape.
            map_shape: Map image shape.

        Returns:
            Dict[str, Any]: Dictionary containing positioning results.
        """
        result = {
            "pred_lat": None,
            "pred_lon": None,
            "error_meters": float("inf"),
            "success": False,
        }
        if not ransac_successful or homography is None:
            return result
        if (
            self._calculate_location_and_error is None
            or self._calculate_predicted_gps is None
            or self._haversine_distance is None
        ):
            return result
        h_q, w_q = query_shape[0], query_shape[1]
        h_m, w_m = map_shape[0], map_shape[1]
        norm_center = self._calculate_location_and_error(
            query_row.to_dict(),
            map_row.to_dict(),
            (h_q, w_q),
            (h_m, w_m),
            homography,
        )
        if norm_center is None:
            return result
        pred_lat, pred_lon = self._calculate_predicted_gps(
            map_row.to_dict(), norm_center, map_shape
        )
        gt_lat = query_row.get("Latitude")
        gt_lon = query_row.get("Longitude")
        if pred_lat is not None and gt_lat is not None and gt_lon is not None:
            error_meters = self._haversine_distance(
                float(gt_lat), float(gt_lon), float(pred_lat), float(pred_lon)
            )
            if error_meters != float("inf"):
                result["pred_lat"] = pred_lat
                result["pred_lon"] = pred_lon
                result["error_meters"] = error_meters
                result["success"] = True
        return result

    def _is_better_match(
        self, new_match: Dict[str, Any], current_result: QueryResult
    ) -> bool:
        """Compares a new match against the current best to determine if it's superior.

        Args:
            new_match: The new match result dictionary.
            current_result: The current best QueryResult.

        Returns:
            bool: True if the new match is better, False otherwise.
        """
        if not current_result.success:
            return True
        if new_match["inliers"] > current_result.inliers:
            return True
        if (
            new_match["inliers"] == current_result.inliers
            and new_match["error_meters"] < current_result.error_meters
        ):
            return True
        return False

    def _save_visualization(
        self,
        results_dir: Path,
        query_path: Path,
        map_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers_mask: np.ndarray,
        homography: Optional[np.ndarray] = None,
    ) -> None:
        """Generates and saves match visualization imagery.

        Args:
            results_dir: Directory to save visualization.
            query_path: Path to query image.
            map_path: Path to map image.
            mkpts0: Keypoints in query image.
            mkpts1: Keypoints in map image.
            inliers_mask: Boolean mask of inliers.
            homography: Optional homography matrix.
        """
        output_prefix = f"{query_path.stem}_vs_{map_path.stem}"
        output_path = results_dir / f"{output_prefix}_match.png"
        if hasattr(self.pipeline, "visualize_matches") and self.pipeline:
            try:
                self.pipeline.visualize_matches(
                    query_path,
                    map_path,
                    mkpts0,
                    mkpts1,
                    inliers_mask,
                    output_path,
                    homography=homography,
                )
            except Exception as e:
                print(f"  WARNING: Visualization failed: {e}")

    def _save_results(self, results: List[QueryResult]) -> None:
        """Saves the cumulative results and overall statistics for the current run.

        Args:
            results: List of QueryResult objects to save.
        """
        if not results or not self.output_dir:
            print("No results to save.")
            return
        summary_df = self._create_summary_dataframe(results)
        csv_path = self.output_dir / "positioning_results.csv"
        try:
            summary_df.to_csv(csv_path, index=False, float_format="%.7f")
            print(f"\nSummary saved: {csv_path}")
        except Exception as e:
            print(f"ERROR saving summary: {e}")
        self._save_statistics(summary_df, results)

    def _create_summary_dataframe(self, results: List[QueryResult]) -> pd.DataFrame:
        """Converts result objects into a clean pandas DataFrame.

        Args:
            results: List of QueryResult objects.

        Returns:
            pd.DataFrame: Summary DataFrame.
        """
        summary_data = []
        for r in results:
            summary_data.append(
                {
                    "Query Image": r.query_filename,
                    "Best Map Match": r.best_map_filename,
                    "Inliers": r.inliers,
                    "Outliers": r.outliers,
                    "Best Match Time (s)": r.time,
                    "GT Latitude": r.gt_latitude,
                    "GT Longitude": r.gt_longitude,
                    "Pred Latitude": r.predicted_latitude,
                    "Pred Longitude": r.predicted_longitude,
                    "Error (m)": r.error_meters,
                    "Positioning Success": r.success,
                }
            )
        return pd.DataFrame(summary_data)

    def _save_statistics(
        self, summary_df: pd.DataFrame, results: List[QueryResult]
    ) -> None:
        """Computes and writes global performance statistics.

        Args:
            summary_df: Summary DataFrame.
            results: List of QueryResult objects.
        """
        successful = summary_df[summary_df["Positioning Success"] == True]
        num_queries = len(self.query_df) if self.query_df is not None else 0
        num_processed = len(results)
        num_successful = len(successful)
        success_rate = (
            (float(num_successful) / num_processed * 100) if num_processed > 0 else 0.0
        )
        if num_successful > 0:
            avg_error = float(successful["Error (m)"].mean())
            median_error = float(successful["Error (m)"].median())
            avg_inliers = float(successful["Inliers"].mean())
            median_inliers = float(successful["Inliers"].median())
            avg_time = float(successful["Best Match Time (s)"].mean())
        else:
            avg_error = median_error = 0.0
            avg_inliers = median_inliers = avg_time = 0.0
        self._print_statistics(
            num_queries,
            num_processed,
            num_successful,
            success_rate,
            avg_error,
            median_error,
            avg_inliers,
            avg_time,
        )

    def _print_statistics(
        self,
        num_queries: int,
        num_processed: int,
        num_successful: int,
        success_rate: float,
        avg_error: float,
        median_error: float,
        avg_inliers: float,
        avg_time: float,
    ) -> None:
        """Displays statistics on the standard output.

        Args:
            num_queries: Total number of queries.
            num_processed: Number of processed queries.
            num_successful: Number of successful positioning.
            success_rate: Success rate percentage.
            avg_error: Average error in meters.
            median_error: Median error in meters.
            avg_inliers: Average number of inliers.
            avg_time: Average processing time.
        """
        print("\n" + "=" * 60)
        print("  POSITIONING STATISTICS")
        print("=" * 60)
        print(f"  Total Queries:    {num_queries}")
        print(f"  Processed:        {num_processed}")
        print(f"  Successful:       {num_successful}")
        print(f"  Success Rate:     {success_rate:.2f}%")
        if num_successful > 0:
            print("\n  Positioning Accuracy (successful only):")
            print(f"    Avg Error:      {avg_error:.2f} m")
            print(f"    Median Error:   {median_error:.2f} m")
            print(f"    Avg Inliers:    {avg_inliers:.1f}")
            print(f"    Avg Time:       {avg_time:.3f} s")
        print("=" * 60)

    def _write_statistics_file(
        self,
        num_queries: int,
        num_processed: int,
        num_successful: int,
        success_rate: float,
        avg_error: float,
        median_error: float,
        avg_inliers: float,
        median_inliers: float,
        avg_time: float,
    ) -> None:
        """Writes statistics to a positioning_stats.txt file.

        Args:
            num_queries: Total number of queries.
            num_processed: Number of processed queries.
            num_successful: Number of successful positioning.
            success_rate: Success rate percentage.
            avg_error: Average error in meters.
            median_error: Median error in meters.
            avg_inliers: Average number of inliers.
            median_inliers: Median number of inliers.
            avg_time: Average processing time.
        """
        if not self.output_dir:
            return
        stats_path = self.output_dir / "positioning_stats.txt"
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("=" * 50 + "\n")
                f.write("POSITIONING STATISTICS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Matcher: {self.config.matcher_type.upper()}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    f"Preprocessing: "
                    f"{self.config.preprocessing.get('enabled', False)}\n"
                )
                f.write(f"\n{'-' * 50}\n\n")
                f.write(f"Total Queries: {num_queries}\n")
                f.write(f"Queries Processed: {num_processed}\n")
                f.write(f"Successful Positionings: {num_successful}\n")
                f.write(f"Success Rate: {success_rate:.2f}%\n")
                if num_successful > 0:
                    f.write(f"\n{'-' * 50}\n\n")
                    f.write("Statistics (Successful Positionings):\n")
                    f.write(f"  Average Error: {avg_error:.2f} m\n")
                    f.write(f"  Median Error: {median_error:.2f} m\n")
                    f.write(f"  Average Inliers: {avg_inliers:.1f}\n")
                    f.write(f"  Median Inliers: {median_inliers:.1f}\n")
                    f.write(f"  Average Time: {avg_time:.3f} s\n")
            print(f"Statistics saved: {stats_path}")
        except Exception as e:
            print(f"ERROR saving statistics: {e}")
