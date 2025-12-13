#!/usr/bin/env python3
"""
Satellite Visual Localization Benchmark

This script provides a comprehensive benchmarking framework for evaluating
feature matching algorithms on drone-to-satellite visual localization tasks.

Usage:
    python benchmark.py --config config.yaml
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

# Add source directory to path
_src_path = Path(__file__).resolve().parent / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import pipeline modules
try:
    from lightgluePipeline import LightGluePipeline
    from supergluePipeline import SuperGluePipeline
    from gimPipeline import GimPipeline
    from loftrPipeline import LoFTRPipeline
except ImportError as e:
    print(f"ERROR: Failed to import pipeline modules: {e}")
    sys.exit(1)

# Import utility modules
PREPROCESSING_AVAILABLE = False
try:
    from utils.preprocessing import QueryProcessor, CameraModel
    from utils.helpers import (
        haversine_distance,
        calculate_predicted_gps,
        calculate_location_and_error,
    )
    PREPROCESSING_AVAILABLE = True
    print("Successfully imported utility modules.")
except ImportError as e:
    print(f"WARNING: Failed to import utility modules: {e}")
    QueryProcessor = None
    CameraModel = None
    haversine_distance = None
    calculate_predicted_gps = None
    calculate_location_and_error = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryResult:
    """Container for per-query benchmark results."""
    query_filename: str
    best_map_filename: Optional[str] = None
    inliers: int = -1
    outliers: int = -1
    time: float = 0.0
    gt_latitude: Optional[float] = None
    gt_longitude: Optional[float] = None
    predicted_latitude: Optional[float] = None
    predicted_longitude: Optional[float] = None
    error_meters: float = float('inf')
    success: bool = False


@dataclass
class BenchmarkConfig:
    """Parsed benchmark configuration."""
    matcher_type: str
    device: str
    data_paths: Dict[str, str]
    preprocessing: Dict[str, Any]
    camera_model: Optional[Dict[str, Any]]
    matcher_weights: Dict[str, Any]
    matcher_params: Dict[str, Any]
    ransac_params: Dict[str, Any]
    benchmark_params: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'BenchmarkConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            matcher_type=config.get('matcher_type', 'lightglue'),
            device=config.get('device', 'cuda'),
            data_paths=config.get('data_paths', {}),
            preprocessing=config.get('preprocessing', {'enabled': False}),
            camera_model=config.get('camera_model'),
            matcher_weights=config.get('matcher_weights', {}),
            matcher_params=config.get('matcher_params', {}),
            ransac_params=config.get('ransac_params', {}),
            benchmark_params=config.get('benchmark_params', {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline initialization."""
        return {
            'matcher_type': self.matcher_type,
            'device': self.device,
            'data_paths': self.data_paths,
            'preprocessing': self.preprocessing,
            'camera_model': self.camera_model,
            'matcher_weights': self.matcher_weights,
            'matcher_params': self.matcher_params,
            'ransac_params': self.ransac_params,
            'benchmark_params': self.benchmark_params,
        }


# =============================================================================
# Pipeline Factory
# =============================================================================

class PipelineFactory:
    """Factory for creating matcher pipeline instances."""

    PIPELINES = {
        'lightglue': LightGluePipeline,
        'superglue': SuperGluePipeline,
        'gim': GimPipeline,
        'loftr': LoFTRPipeline,
    }

    @classmethod
    def create(cls, config: BenchmarkConfig):
        """
        Create a matcher pipeline based on configuration.

        Args:
            config: Benchmark configuration.

        Returns:
            Initialized pipeline instance.

        Raises:
            ValueError: If matcher type is not supported.
        """
        matcher_type = config.matcher_type.lower()

        if matcher_type not in cls.PIPELINES:
            supported = ', '.join(cls.PIPELINES.keys())
            raise ValueError(f"Unsupported matcher: '{matcher_type}'. "
                           f"Supported: {supported}")

        pipeline_class = cls.PIPELINES[matcher_type]
        return pipeline_class(config.to_dict())


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Main benchmark runner for visual localization evaluation.

    This class orchestrates the benchmark process including:
    - Loading and validating configuration
    - Initializing preprocessing and matching pipelines
    - Processing query-map pairs
    - Computing localization metrics
    - Saving results and statistics
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config: Parsed benchmark configuration.
        """
        self.config = config
        self.pipeline = None
        self.preprocessor = None
        self.query_df = None
        self.map_df = None
        self.output_dir = None

    def run(self) -> None:
        """Execute the benchmark."""
        print("\n" + "=" * 60)
        print("  SATELLITE VISUAL LOCALIZATION BENCHMARK")
        print("=" * 60 + "\n")

        # Setup
        self._validate_paths()
        self._setup_output_directory()
        self._initialize_preprocessor()
        self._initialize_pipeline()
        self._load_metadata()

        # Run benchmark
        results = self._process_queries()

        # Save results
        self._save_results(results)

        print("\n" + "=" * 60)
        print("  BENCHMARK COMPLETE")
        print("=" * 60 + "\n")

    def _validate_paths(self) -> None:
        """Validate required paths exist in configuration."""
        paths = self.config.data_paths
        required = ['query_dir', 'map_dir', 'output_dir', 'query_metadata', 'map_metadata']

        missing = [p for p in required if not paths.get(p)]
        if missing:
            raise ValueError(f"Missing required paths in config: {missing}")

    def _setup_output_directory(self) -> None:
        """Create timestamped output directory."""
        base_dir = Path(self.config.data_paths['output_dir'])
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        preprocess_status = "preprocessed" if self.config.preprocessing.get('enabled') else "original"

        # Build directory name with model-specific suffix
        matcher_type = self.config.matcher_type
        suffix = ""

        if matcher_type == 'gim':
            model_type = self.config.matcher_weights.get('gim_model_type', 'unknown')
            suffix = f"_{model_type}"
        elif matcher_type == 'loftr':
            weights_path = self.config.matcher_weights.get('loftr_weights_path', 'unknown.ckpt')
            suffix = f"_{Path(weights_path).stem}"

        dir_name = f"{matcher_type}{suffix}_{preprocess_status}_{timestamp}"
        self.output_dir = base_dir / dir_name
        self.output_dir.mkdir(exist_ok=True)

        print(f"Output directory: {self.output_dir}")

    def _initialize_preprocessor(self) -> None:
        """Initialize query image preprocessor if enabled."""
        preprocess_config = self.config.preprocessing

        if not preprocess_config.get('enabled', False):
            print("Preprocessing: Disabled")
            return

        if not PREPROCESSING_AVAILABLE:
            raise RuntimeError("Preprocessing enabled but modules failed to import.")

        print("Initializing preprocessor...")

        # Setup camera model if required
        camera_model = None
        if self.config.camera_model and CameraModel:
            try:
                valid_keys = set(CameraModel.__annotations__.keys())
                cam_params = {k: v for k, v in self.config.camera_model.items()
                             if k in valid_keys}
                camera_model = CameraModel(**cam_params)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize camera model: {e}")

        elif 'warp' in preprocess_config.get('steps', []):
            raise ValueError("Warp step requires camera_model configuration.")

        # Initialize preprocessor
        try:
            self.preprocessor = QueryProcessor(
                processings=preprocess_config.get('steps', []),
                resize_target=preprocess_config.get('resize_target'),
                camera_model=camera_model,
                target_gimbal_yaw=preprocess_config.get('target_gimbal_yaw', 0.0),
                target_gimbal_pitch=preprocess_config.get('target_gimbal_pitch', -90.0),
                target_gimbal_roll=preprocess_config.get('target_gimbal_roll', 0.0),
            )
            print(f"  Steps: {self.preprocessor.processings or 'None'}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize preprocessor: {e}")

    def _initialize_pipeline(self) -> None:
        """Initialize the matcher pipeline."""
        matcher_type = self.config.matcher_type.upper()
        print(f"\nInitializing matcher: {matcher_type}")

        try:
            self.pipeline = PipelineFactory.create(self.config)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize pipeline: {e}")

    def _load_metadata(self) -> None:
        """Load and validate query and map metadata."""
        print("\nLoading metadata...")

        paths = self.config.data_paths

        try:
            self.query_df = pd.read_csv(paths['query_metadata'], skipinitialspace=True)
            self.map_df = pd.read_csv(paths['map_metadata'], skipinitialspace=True)

            # Strip whitespace from column names
            self.query_df.columns = self.query_df.columns.str.strip()
            self.map_df.columns = self.map_df.columns.str.strip()

            print(f"  Loaded {len(self.query_df)} queries, {len(self.map_df)} maps")

        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

        # Validate required columns
        required_query = ['Filename', 'Latitude', 'Longitude']
        required_map = ['Filename', 'Top_left_lat', 'Top_left_lon',
                       'Bottom_right_lat', 'Bottom_right_long']

        # Add warp columns if needed
        if (self.config.preprocessing.get('enabled') and
            'warp' in self.config.preprocessing.get('steps', [])):
            required_query.extend(['Gimball_Yaw', 'Gimball_Pitch',
                                  'Gimball_Roll', 'Flight_Yaw'])

        missing_query = [c for c in required_query if c not in self.query_df.columns]
        missing_map = [c for c in required_map if c not in self.map_df.columns]

        if missing_query:
            raise ValueError(f"Query metadata missing columns: {missing_query}")
        if missing_map:
            raise ValueError(f"Map metadata missing columns: {missing_map}")

    def _process_queries(self) -> List[QueryResult]:
        """Process all query images and return results."""
        results = []
        temp_dir = self.output_dir / "processed_queries"
        temp_dir.mkdir(exist_ok=True)

        paths = self.config.data_paths
        min_inliers = self.config.benchmark_params.get('min_inliers_for_success', 10)
        save_viz = self.config.benchmark_params.get('save_visualization', False)

        print("\n" + "-" * 60)
        print("Starting benchmark loop...")
        print("-" * 60)

        total_start = time.time()

        for idx, query_row in self.query_df.iterrows():
            result = self._process_single_query(
                query_row, idx, temp_dir, min_inliers, save_viz
            )
            results.append(result)

        total_time = time.time() - total_start
        print(f"\nTotal processing time: {total_time:.2f} seconds")

        return results

    def _process_single_query(
        self,
        query_row: pd.Series,
        idx: int,
        temp_dir: Path,
        min_inliers: int,
        save_viz: bool
    ) -> QueryResult:
        """Process a single query image against all maps."""
        query_filename = query_row['Filename']
        query_path = Path(self.config.data_paths['query_dir']) / query_filename

        print(f"\n--- Query: {query_filename} ({idx + 1}/{len(self.query_df)}) ---")

        # Initialize result
        result = QueryResult(
            query_filename=query_filename,
            gt_latitude=query_row.get('Latitude'),
            gt_longitude=query_row.get('Longitude'),
        )

        # Validate query exists
        if not query_path.is_file():
            print(f"  WARNING: Query image not found, skipping")
            return result

        # Preprocess query if needed
        query_for_match, query_shape = self._preprocess_query(
            query_path, query_row, temp_dir
        )

        if query_shape is None:
            print(f"  WARNING: Invalid query shape, skipping")
            return result

        # Create query results directory
        query_results_dir = self.output_dir / Path(query_filename).stem
        query_results_dir.mkdir(exist_ok=True)

        # Process against all maps
        for _, map_row in self.map_df.iterrows():
            match_result = self._match_query_to_map(
                query_for_match, query_shape, query_row,
                map_row, query_results_dir, min_inliers, save_viz
            )

            if match_result is not None:
                # Update best match if this is better
                if self._is_better_match(match_result, result):
                    result.best_map_filename = match_result['map_filename']
                    result.inliers = match_result['inliers']
                    result.outliers = match_result['outliers']
                    result.time = match_result['time']
                    result.predicted_latitude = match_result['pred_lat']
                    result.predicted_longitude = match_result['pred_lon']
                    result.error_meters = match_result['error_meters']
                    result.success = True

        # Print summary
        if result.success:
            print(f"  Best: {result.best_map_filename} "
                  f"({result.inliers} inliers, {result.error_meters:.2f}m error)")
        else:
            print("  No successful localization")

        return result

    def _preprocess_query(
        self,
        query_path: Path,
        query_row: pd.Series,
        temp_dir: Path
    ) -> Tuple[Path, Optional[Tuple[int, ...]]]:
        """Preprocess a query image if preprocessor is configured."""
        if self.preprocessor is None:
            # No preprocessing - just get shape
            img = cv2.imread(str(query_path))
            shape = img.shape if img is not None else None
            return query_path, shape

        # Load and preprocess
        img_original = cv2.imread(str(query_path))
        if img_original is None:
            return query_path, None

        processed = self.preprocessor(img_original, query_row.to_dict())
        shape = processed.shape

        # Check if preprocessing changed the image
        if (processed.shape == img_original.shape and
            np.array_equal(processed, img_original)):
            return query_path, shape

        # Save processed image
        processed_name = f"{Path(query_path.name).stem}_processed{Path(query_path.name).suffix}"
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
        save_viz: bool
    ) -> Optional[Dict[str, Any]]:
        """Match query against a single map and compute localization."""
        map_filename = map_row['Filename']
        map_path = Path(self.config.data_paths['map_dir']) / map_filename

        if not map_path.is_file():
            return None

        map_img = cv2.imread(str(map_path))
        if map_img is None:
            return None

        map_shape = map_img.shape

        # Run matching
        try:
            match_results = self.pipeline.match(query_path, map_path)
        except Exception as e:
            print(f"  ERROR matching with {map_filename}: {e}")
            return None

        # Extract results
        match_time = match_results.get('time', 0)
        homography = match_results.get('homography')
        inliers_mask = match_results.get('inliers')
        mkpts0 = match_results.get('mkpts0', np.array([]))
        mkpts1 = match_results.get('mkpts1', np.array([]))

        num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0
        num_total = len(mkpts0)
        num_outliers = num_total - num_inliers

        ransac_successful = (match_results.get('success', False) and
                            num_inliers >= min_inliers)

        # Compute localization if successful
        pred_lat, pred_lon = None, None
        error_meters = float('inf')
        localization_success = False

        if ransac_successful and homography is not None:
            norm_center = calculate_location_and_error(
                query_row.to_dict(), map_row.to_dict(),
                query_shape, map_shape, homography
            )

            if norm_center is not None:
                pred_lat, pred_lon = calculate_predicted_gps(
                    map_row.to_dict(), norm_center
                )

                gt_lat = query_row.get('Latitude')
                gt_lon = query_row.get('Longitude')

                if pred_lat is not None and gt_lat is not None:
                    error_meters = haversine_distance(
                        gt_lat, gt_lon, pred_lat, pred_lon
                    )
                    if error_meters != float('inf'):
                        localization_success = True

        # Save detailed results
        self._save_match_results(
            results_dir, query_path.name, map_filename,
            match_time, num_total, num_inliers, num_outliers,
            ransac_successful, localization_success,
            query_row, pred_lat, pred_lon, error_meters, homography
        )

        # Save visualization
        if save_viz and ransac_successful:
            self._save_visualization(
                results_dir, query_path, map_path,
                mkpts0, mkpts1, inliers_mask
            )

        if not localization_success:
            return None

        return {
            'map_filename': map_filename,
            'inliers': num_inliers,
            'outliers': num_outliers,
            'time': match_time,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'error_meters': error_meters,
        }

    def _is_better_match(
        self,
        new_match: Dict[str, Any],
        current_result: QueryResult
    ) -> bool:
        """Determine if new match is better than current best."""
        if not current_result.success:
            return True
        if new_match['inliers'] > current_result.inliers:
            return True
        if (new_match['inliers'] == current_result.inliers and
            new_match['error_meters'] < current_result.error_meters):
            return True
        return False

    def _save_match_results(
        self,
        results_dir: Path,
        query_name: str,
        map_name: str,
        match_time: float,
        num_total: int,
        num_inliers: int,
        num_outliers: int,
        ransac_successful: bool,
        localization_success: bool,
        query_row: pd.Series,
        pred_lat: Optional[float],
        pred_lon: Optional[float],
        error_meters: float,
        homography: Optional[np.ndarray]
    ) -> None:
        """Save detailed match results to text file."""
        output_prefix = f"{Path(query_name).stem}_vs_{Path(map_name).stem}"
        output_path = results_dir / f"{output_prefix}_results.txt"

        # Build matcher name
        matcher_name = self.config.matcher_type.upper()
        if self.config.matcher_type == 'gim':
            model_type = self.config.matcher_weights.get('gim_model_type', 'unknown')
            matcher_name += f" ({model_type})"
        elif self.config.matcher_type == 'loftr':
            weights_name = Path(self.config.matcher_weights.get(
                'loftr_weights_path', 'N/A')).name
            matcher_name += f" ({weights_name})"

        try:
            with open(output_path, 'w') as f:
                f.write(f"{'=' * 50}\n")
                f.write(f"Match Results: {query_name} vs {map_name}\n")
                f.write(f"{'=' * 50}\n\n")

                f.write(f"Matcher: {matcher_name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Preprocessing: {self.config.preprocessing.get('enabled', False)}\n")
                f.write(f"\n{'-' * 50}\n")

                f.write("MATCHING & RANSAC\n")
                f.write(f"  Time: {match_time:.4f} s\n")
                f.write(f"  Putative Matches: {num_total}\n")
                f.write(f"  Inliers: {num_inliers}\n")
                f.write(f"  Outliers: {num_outliers}\n")
                f.write(f"  RANSAC Success: {ransac_successful}\n")
                f.write(f"\n{'-' * 50}\n")

                f.write("LOCALIZATION\n")
                gt_lat = query_row.get('Latitude')
                gt_lon = query_row.get('Longitude')
                f.write(f"  Ground Truth: {gt_lat:.7f}, {gt_lon:.7f}\n"
                       if gt_lat else "  Ground Truth: N/A\n")
                f.write(f"  Predicted: {pred_lat:.7f}, {pred_lon:.7f}\n"
                       if localization_success else "  Predicted: N/A\n")
                f.write(f"  Error: {error_meters:.3f} m\n"
                       if localization_success else "  Error: N/A\n")
                f.write(f"  Success: {localization_success}\n")
                f.write(f"\n{'-' * 50}\n")

                f.write("HOMOGRAPHY (Query -> Map)\n")
                f.write(f"{homography}\n")

        except Exception as e:
            print(f"  WARNING: Failed to save results: {e}")

    def _save_visualization(
        self,
        results_dir: Path,
        query_path: Path,
        map_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers_mask: np.ndarray
    ) -> None:
        """Save match visualization image."""
        output_prefix = f"{query_path.stem}_vs_{map_path.stem}"
        output_path = results_dir / f"{output_prefix}_match.png"

        if hasattr(self.pipeline, 'visualize_matches'):
            try:
                self.pipeline.visualize_matches(
                    query_path, map_path,
                    mkpts0, mkpts1, inliers_mask,
                    output_path
                )
            except Exception as e:
                print(f"  WARNING: Visualization failed: {e}")

    def _save_results(self, results: List[QueryResult]) -> None:
        """Save benchmark summary and statistics."""
        if not results:
            print("No results to save.")
            return

        # Create summary DataFrame
        summary_data = []
        for r in results:
            summary_data.append({
                'Query Image': r.query_filename,
                'Best Map Match': r.best_map_filename,
                'Inliers': r.inliers,
                'Outliers': r.outliers,
                'Best Match Time (s)': r.time,
                'GT Latitude': r.gt_latitude,
                'GT Longitude': r.gt_longitude,
                'Pred Latitude': r.predicted_latitude,
                'Pred Longitude': r.predicted_longitude,
                'Error (m)': r.error_meters,
                'Localization Success': r.success,
            })

        summary_df = pd.DataFrame(summary_data)

        # Save CSV
        csv_path = self.output_dir / "benchmark_summary.csv"
        try:
            summary_df.to_csv(csv_path, index=False, float_format='%.7f')
            print(f"\nSummary saved: {csv_path}")
        except Exception as e:
            print(f"ERROR saving summary: {e}")

        # Compute and save statistics
        self._save_statistics(summary_df, results)

    def _save_statistics(
        self,
        summary_df: pd.DataFrame,
        results: List[QueryResult]
    ) -> None:
        """Compute and save overall benchmark statistics."""
        successful = summary_df[summary_df['Localization Success'] == True]

        num_queries = len(self.query_df)
        num_processed = len(results)
        num_successful = len(successful)
        success_rate = (num_successful / num_processed * 100) if num_processed > 0 else 0

        # Compute metrics for successful localizations
        if num_successful > 0:
            avg_error = successful['Error (m)'].mean()
            median_error = successful['Error (m)'].median()
            avg_inliers = successful['Inliers'].mean()
            median_inliers = successful['Inliers'].median()
            avg_time = successful['Best Match Time (s)'].mean()
        else:
            avg_error = median_error = avg_inliers = median_inliers = avg_time = float('nan')

        # Print statistics
        print("\n" + "=" * 60)
        print("  BENCHMARK STATISTICS")
        print("=" * 60)
        print(f"  Total Queries:    {num_queries}")
        print(f"  Processed:        {num_processed}")
        print(f"  Successful:       {num_successful}")
        print(f"  Success Rate:     {success_rate:.2f}%")

        if num_successful > 0:
            print(f"\n  Localization Accuracy (successful only):")
            print(f"    Avg Error:      {avg_error:.2f} m")
            print(f"    Median Error:   {median_error:.2f} m")
            print(f"    Avg Inliers:    {avg_inliers:.1f}")
            print(f"    Avg Time:       {avg_time:.3f} s")

        print("=" * 60)

        # Save to file
        stats_path = self.output_dir / "benchmark_stats.txt"
        try:
            with open(stats_path, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("BENCHMARK STATISTICS\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Matcher: {self.config.matcher_type.upper()}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Preprocessing: {self.config.preprocessing.get('enabled', False)}\n")
                f.write(f"\n{'-' * 50}\n\n")

                f.write(f"Total Queries: {num_queries}\n")
                f.write(f"Queries Processed: {num_processed}\n")
                f.write(f"Successful Localizations: {num_successful}\n")
                f.write(f"Success Rate: {success_rate:.2f}%\n")

                if num_successful > 0:
                    f.write(f"\n{'-' * 50}\n\n")
                    f.write("Statistics (Successful Localizations):\n")
                    f.write(f"  Average Error: {avg_error:.2f} m\n")
                    f.write(f"  Median Error: {median_error:.2f} m\n")
                    f.write(f"  Average Inliers: {avg_inliers:.1f}\n")
                    f.write(f"  Median Inliers: {median_inliers:.1f}\n")
                    f.write(f"  Average Time: {avg_time:.3f} s\n")

            print(f"Statistics saved: {stats_path}")

        except Exception as e:
            print(f"ERROR saving statistics: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run Satellite Visual Localization Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --config config.yaml
  python benchmark.py --config my_custom_config.yaml
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    try:
        config = BenchmarkConfig.from_yaml(str(config_path))
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # Validate preprocessing dependencies
    if config.preprocessing.get('enabled', False) and not PREPROCESSING_AVAILABLE:
        print("ERROR: Preprocessing enabled but required modules failed to import.")
        sys.exit(1)

    # Validate core helper functions
    if not all([haversine_distance, calculate_predicted_gps, calculate_location_and_error]):
        print("ERROR: Core helper functions failed to import.")
        sys.exit(1)

    # Run benchmark
    try:
        runner = BenchmarkRunner(config)
        runner.run()
    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()