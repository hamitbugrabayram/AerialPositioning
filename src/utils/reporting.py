"""Reporting and visualization utilities for visual positioning experiments.

This module provides classes for generating summary metrics and Markdown reports
from experiment results.
"""

import glob
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ReportGenerator:
    """Handles result summarization and report generation."""

    def __init__(self, project_root: Path, results_root: Path):
        """Initializes the ReportGenerator.

        Args:
            project_root: The root directory of the project.
            results_root: The root directory containing experiment results.
        """
        self.project_root = project_root
        self.results_root = results_root
        self.summary_md = project_root / "results_report.md"

    def get_experiment_metrics(self, output_dir: str) -> Optional[Dict[str, Any]]:
        """Parses positioning results and computes summary metrics."""
        patterns = ["positioning_results.csv"]
        results_files = []
        for p in patterns:
            results_files.extend(glob.glob(os.path.join(output_dir, "**", p), recursive=True))

        if not results_files:
            return None

        latest_csv = max(results_files, key=os.path.getmtime)
        df = pd.read_csv(latest_csv)

        success_col = "Positioning Success"
        if success_col not in df.columns:
            return None

        success_df = df[df[success_col]]

        metrics = {
            "Total": len(df),
            "Success%": f"{(len(success_df) * 100 / len(df)):.1f}%",
        }

        if not success_df.empty:
            metrics.update(
                {
                    "AvgErr": f"{success_df['Error (m)'].mean():.2f}m",
                    "AvgInliers": f"{success_df['Inliers'].mean():.1f}",
                    "AvgTime": f"{df['Best Match Time (s)'].mean():.3f}s",
                }
            )
        return metrics

    def generate_summary(self, region_ids: Optional[List[int]] = None) -> None:
        """Summarizes all results into a Markdown report.

        Args:
            region_ids: Optional list of region IDs to filter.
        """
        summary_data = []
        md_lines = [
            "# Aerial Positioning via Satellite Imagery - Results Report\n",
            f"Generated on: {time.ctime()}\n",
            "## Results Summary\n",
            "| Region | Provider | Zoom | Success% | Avg Error | Avg Inliers | Experiment Folder |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        if not self.results_root.exists():
            print("No results directory found.")
            return

        folders = sorted(glob.glob(str(self.results_root / "*")))
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            folder_path = Path(folder)

            try:
                parts = folder_path.name.split("_")
                rid = None
                provider = "unknown"
                zoom = "N/A"

                for i, p in enumerate(parts):
                    if p == "zoom" and i > 0:
                        rid = int(parts[i - 1])
                        if i + 1 < len(parts):
                            zoom = parts[i + 1]
                        if i + 2 < len(parts):
                            provider = parts[i + 2]
                        break

                if region_ids and (rid not in region_ids):
                    continue
            except Exception:
                rid, zoom, provider = "N/A", "N/A", "N/A"

            metrics = self.get_experiment_metrics(folder)
            if metrics:
                item = {
                    "RegionID": rid,
                    "Provider": provider,
                    "Zoom": zoom,
                    "Experiment": folder_path.name,
                }
                item.update(metrics)
                summary_data.append(item)

                md_lines.append(
                    f"| {rid} | {provider} | {zoom} | {metrics['Success%']} | "
                    f"{metrics.get('AvgErr', 'N/A')} | {metrics.get('AvgInliers', '0')} | "
                    f"`{folder_path.name}` |"
                )

        if summary_data:
            df = pd.DataFrame(summary_data)
            display_cols = [
                "RegionID",
                "Provider",
                "Zoom",
                "Success%",
                "AvgErr",
                "AvgInliers",
            ]
            print("\n" + df[display_cols].to_string(index=False))
        else:
            print("No valid results found to summarize.")
