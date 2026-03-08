"""Statistics and result management for visual positioning.

This module provides classes for managing positioning results, saving them to CSV,
and computing performance statistics.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.models.config import QueryResult

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ResultManager:
    """Manages the lifecycle of positioning results and statistics.

    Attributes:
        output_dir (Path): The root experiment directory.
        assets_dir (Path): The directory for summary assets.

    """

    def __init__(self, output_dir: Path, assets_dir: Path):
        """Initializes the manager with output directories.

        Args:
            output_dir: The root experiment directory.
            assets_dir: The directory for summary assets.

        Returns:
            None.

        """
        self.output_dir = output_dir
        self.assets_dir = assets_dir

    def save_results(self, results: List[QueryResult], query_df_len: int) -> None:
        """Saves cumulative results and prints overall statistics.

        Args:
            results (List[QueryResult]): List of results for each processed query.
            query_df_len (int): Original number of queries in the dataset.

        Returns:
            None.

        """
        if not results:
            _logger.info("No results to save.")
            return

        df = self._create_summary_df(results)
        csv_path = self.assets_dir / "positioning_results.csv"

        try:
            df.to_csv(csv_path, index=False, float_format="%.7f")
            _logger.info(f"\nSummary saved: {csv_path}")
        except Exception as e:
            _logger.info(f"ERROR saving summary CSV: {e}")

        self.print_statistics(df, results, query_df_len)

    def _create_summary_df(self, results: List[QueryResult]) -> pd.DataFrame:
        """Converts result objects into a pandas DataFrame.

        Args:
            results (List[QueryResult]): List of query results.

        Returns:
            pd.DataFrame: Tabular dataframe representation of query results.

        """
        data = []
        for r in results:
            data.append(
                {
                    "Query Image": r.query_filename,
                    "Best Map Match": r.best_map_filename,
                    "Inliers": r.inliers,
                    "Outliers": r.outliers,
                    "Best Match Time (s)": r.time,
                    "Matcher Time / Frame (s)": r.matcher_time_frame_s,
                    "Query Features": r.query_features,
                    "Map Features": r.map_features,
                    "Matched Features": r.matched_features,
                    "GT Latitude": r.gt_latitude,
                    "GT Longitude": r.gt_longitude,
                    "Pred Latitude": r.predicted_latitude,
                    "Pred Longitude": r.predicted_longitude,
                    "Error (m)": np.nan if not r.success else r.error_meters,
                    "Positioning Success": r.success,
                    "Search Radius (m)": r.search_radius_m,
                    "Candidate Maps": r.candidate_maps,
                    "Evaluated Maps": r.evaluated_maps,
                    "Failure Reason": r.failure_reason,
                }
            )
        return pd.DataFrame(data)

    def print_statistics(
        self, df: pd.DataFrame, results: List[QueryResult], total_queries: int
    ) -> None:
        """Computes and displays global performance statistics.

        Args:
            df (pd.DataFrame): Summary DataFrame of results.
            results (List[QueryResult]): List of result objects.
            total_queries (int): Total number of queries available.

        Returns:
            None.

        """
        successful = df[df["Positioning Success"]]
        num_processed = len(results)
        num_successful = len(successful)
        rate = (num_successful / num_processed * 100) if num_processed > 0 else 0.0

        _logger.info("\nPositioning Statistics")
        _logger.info(f"Total Queries: {total_queries}")
        _logger.info(f"Processed: {num_processed}")
        _logger.info(f"Successful: {num_successful}")
        _logger.info(f"Success Rate: {rate:.2f}%")
        if num_processed > 0 and "Matcher Time / Frame (s)" in df.columns:
            _logger.info(
                "Avg Matcher Time / Frame: "
                f"{df['Matcher Time / Frame (s)'].mean():.3f} s"
            )
        if num_processed > 0:
            _logger.info(
                f"Avg Query Features / Frame: {df['Query Features'].mean():.1f}"
            )
            _logger.info(f"Avg Map Features / Frame: {df['Map Features'].mean():.1f}")
            _logger.info(
                f"Avg Matched Features / Frame: {df['Matched Features'].mean():.1f}"
            )

        if num_successful > 0:
            errs = successful["Error (m)"].astype(float)
            _logger.info(f"Avg Error: {errs.mean():.2f} m")
            _logger.info(f"Median Error: {np.median(errs.tolist()):.2f} m")
            _logger.info(f"P90 Error: {np.percentile(errs, 90):.2f} m")
            _logger.info(f"Max Error: {errs.max():.2f} m")
            _logger.info(f"Avg Inliers: {successful['Inliers'].mean():.1f}")
            _logger.info(
                f"Avg Best Match Time: {df['Best Match Time (s)'].mean():.3f} s"
            )
        if "Search Radius (m)" in df.columns:
            radius_counts = (
                df["Search Radius (m)"]
                .dropna()
                .astype(float)
                .value_counts()
                .sort_index()
            )
            if not radius_counts.empty:
                radius_text = ", ".join(
                    f"{int(radius)}m: {int(count)}"
                    for radius, count in radius_counts.items()
                )
                _logger.info(f"Radius Usage: {radius_text}")
