"""Statistics and result management for visual positioning.

This module provides classes for managing positioning results, saving them to CSV,
and computing performance statistics.
"""

from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from src.models.config import QueryResult


class ResultManager:
    """Manages the lifecycle of positioning results and statistics."""

    def __init__(self, output_dir: Path, assets_dir: Path):
        """Initializes the manager with output directories.

        Args:
            output_dir: The root experiment directory.
            assets_dir: The directory for summary assets.
        """
        self.output_dir = output_dir
        self.assets_dir = assets_dir

    def save_results(self, results: List[QueryResult], query_df_len: int) -> None:
        """Saves cumulative results and prints overall statistics.

        Args:
            results: List of results for each processed query.
            query_df_len: Original number of queries in the dataset.
        """
        if not results:
            print("No results to save.")
            return

        df = self._create_summary_df(results)
        csv_path = self.assets_dir / "positioning_results.csv"
        
        try:
            df.to_csv(csv_path, index=False, float_format="%.7f")
            print(f"\nSummary saved: {csv_path}")
        except Exception as e:
            print(f"ERROR saving summary CSV: {e}")

        self.print_statistics(df, results, query_df_len)

    def _create_summary_df(self, results: List[QueryResult]) -> pd.DataFrame:
        """Converts result objects into a pandas DataFrame."""
        data = []
        for r in results:
            data.append({
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
            })
        return pd.DataFrame(data)

    def print_statistics(self, df: pd.DataFrame, results: List[QueryResult], total_queries: int) -> None:
        """Computes and displays global performance statistics.

        Args:
            df: Summary DataFrame of results.
            results: List of result objects.
            total_queries: Total number of queries available.
        """
        successful = df[df["Positioning Success"]]
        num_processed = len(results)
        num_successful = len(successful)
        rate = (num_successful / num_processed * 100) if num_processed > 0 else 0.0

        print("\nPositioning Statistics")
        print(f"Total Queries: {total_queries}")
        print(f"Processed: {num_processed}")
        print(f"Successful: {num_successful}")
        print(f"Success Rate: {rate:.2f}%")

        if num_successful > 0:
            errs = successful["Error (m)"].astype(float)
            print(f"Avg Error: {errs.mean():.2f} m")
            print(f"Median Error: {np.median(errs.tolist()):.2f} m")
            print(f"Avg Inliers: {successful['Inliers'].mean():.1f}")
            print(f"Avg Time: {df['Best Match Time (s)'].mean():.3f} s")
