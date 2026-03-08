"""Reporting and visualization utilities for visual positioning experiments."""

from __future__ import annotations


import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import yaml

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ReportGenerator:
    """Handles result summarization and report generation.

    Attributes:
        project_root (Path): Repository root directory.
        results_root (Path): Root directory that contains experiment outputs.
        summary_md (Path): Path to the generated markdown report.

    """

    def __init__(self, project_root: Path, results_root: Path):
        """Initializes report output paths.

        Args:
            project_root: Repository root directory.
            results_root: Root directory that contains experiment outputs.

        """
        self.project_root = project_root
        self.results_root = results_root
        self.summary_md = results_root / "report.md"

    def _parse_experiment_name(self, folder_name: str) -> Dict[str, Any]:
        """Extracts region/provider/zoom hints from experiment folder name.

        Args:
            folder_name: Experiment folder basename.

        Returns:
            Dictionary containing `region_id`, `zoom`, and `provider`.

        """
        region_id: Optional[int] = None
        zoom: Optional[int] = None
        provider = "unknown"
        parts = folder_name.split("_")
        for idx, part in enumerate(parts):
            if part == "zoom" and idx > 0:
                try:
                    region_id = int(parts[idx - 1])
                except ValueError:
                    region_id = None
                if idx + 1 < len(parts):
                    try:
                        zoom = int(parts[idx + 1])
                    except ValueError:
                        zoom = None
                if idx + 2 < len(parts):
                    provider = parts[idx + 2].lower()
                break
        return {"region_id": region_id, "zoom": zoom, "provider": provider}

    def _read_config_metadata(self, exp_dir: Path) -> Dict[str, Any]:
        """Reads experiment config if available.

        Args:
            exp_dir: Experiment directory path.

        Returns:
            Parsed config fields used in reporting.

        """
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file) or {}
            return {
                "matcher_type": str(cfg.get("matcher_type", "unknown")),
                "provider_cfg": str(
                    (cfg.get("tile_provider") or {}).get("name", "unknown")
                ),
                "device": str(cfg.get("device", "unknown")),
            }
        except Exception:
            return {}

    def _find_results_csv(self, exp_dir: Path) -> Optional[Path]:
        """Finds the canonical results CSV for an experiment.

        Args:
            exp_dir: Experiment directory path.

        Returns:
            Path to `positioning_results.csv` or `None` if not found.

        """
        candidate = exp_dir / "positioning_results.csv"
        if candidate.exists():
            return candidate
        legacy_candidate = exp_dir / "output" / "positioning_results.csv"
        if legacy_candidate.exists():
            return legacy_candidate
        fallback = sorted(exp_dir.rglob("positioning_results.csv"))
        return fallback[-1] if fallback else None

    def _discover_experiment_dirs(self, search_root: Path) -> List[Path]:
        """Discovers experiment directories under a results root.

        Args:
            search_root: Results root or results subfolder to scan.

        Returns:
            Sorted list of experiment directories that contain a
            ``positioning_results.csv`` file.

        """
        if not search_root.exists():
            return []

        experiment_dirs = set()
        for csv_path in search_root.rglob("positioning_results.csv"):
            exp_dir = csv_path.parent
            if exp_dir.name == "output":
                exp_dir = exp_dir.parent
            experiment_dirs.add(exp_dir.resolve())

        return sorted(experiment_dirs)

    def _numeric_series(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Returns a numeric Series for a DataFrame column.

        Missing columns yield an empty float series.
        """
        if column not in df.columns:
            return pd.Series(dtype=float)
        series = cast(pd.Series, pd.to_numeric(df[column], errors="coerce"))
        return series.dropna()

    def get_experiment_metrics(
        self,
        exp_dir: Path,
        display_root: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parses positioning results and computes summary metrics.

        Args:
            exp_dir: Experiment directory path.
            display_root: Root used to format relative experiment names.

        Returns:
            Aggregated experiment metrics dictionary, or `None` if no valid
            results file exists.

        """
        csv_path = self._find_results_csv(exp_dir)
        if csv_path is None:
            return None

        df = pd.read_csv(csv_path)
        if "Positioning Success" not in df.columns:
            return None

        info = self._parse_experiment_name(exp_dir.name)
        cfg_info = self._read_config_metadata(exp_dir)
        success_mask = cast(
            pd.Series, df["Positioning Success"].fillna(False).astype(bool)
        )
        success_df = cast(pd.DataFrame, df[success_mask])

        total = int(len(df))
        success = int(len(success_df))
        failed = total - success
        success_rate = (100.0 * success / total) if total > 0 else 0.0

        errors = self._numeric_series(success_df, "Error (m)")
        inliers = self._numeric_series(success_df, "Inliers")
        candidates = self._numeric_series(df, "Candidate Maps")
        evaluated = self._numeric_series(df, "Evaluated Maps")
        matched_features = self._numeric_series(df, "Match Count")

        radius_counts: Dict[str, int] = {}
        if "Search Radius (m)" in df.columns:
            radius_series = (
                self._numeric_series(df, "Search Radius (m)")
                .astype(int)
                .value_counts()
                .sort_index()
            )
            radius_counts = {
                f"{int(float(cast(Any, radius)))}m": int(count)
                for radius, count in radius_series.items()
            }

        top_failures: Dict[str, int] = {}
        if "Failure Reason" in df.columns:
            fail_series = (
                df.loc[~success_mask, "Failure Reason"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .head(5)
            )
            top_failures = {
                str(reason): int(count) for reason, count in fail_series.items()
            }

        def safe_stat(value: float) -> float:
            """Normalizes NaN numeric outputs for report serialization.

            Args:
                value: Input float value.

            Returns:
                The same value when finite, otherwise `nan`.

            """
            return value if not math.isnan(value) else float("nan")

        experiment_label = exp_dir.name
        if display_root is not None:
            try:
                experiment_label = str(exp_dir.relative_to(display_root))
            except ValueError:
                experiment_label = exp_dir.name

        return {
            "Experiment": experiment_label,
            "RegionID": info["region_id"],
            "Provider": cfg_info.get("provider_cfg", info["provider"]),
            "Zoom": info["zoom"],
            "Matcher": cfg_info.get("matcher_type", "unknown"),
            "Device": cfg_info.get("device", "unknown"),
            "ResultsCSV": str(csv_path.relative_to(self.project_root)),
            "TotalFrames": total,
            "SuccessfulFrames": success,
            "FailedFrames": failed,
            "SuccessRatePct": success_rate,
            "MeanErrorM": safe_stat(float(cast(Any, errors.mean())))
            if not errors.empty
            else float("nan"),
            "MedianErrorM": safe_stat(float(cast(Any, errors.median())))
            if not errors.empty
            else float("nan"),
            "StdErrorM": safe_stat(float(cast(Any, errors.std(ddof=1))))
            if len(errors) > 1
            else float("nan"),
            "P90ErrorM": safe_stat(float(cast(Any, errors.quantile(0.9))))
            if not errors.empty
            else float("nan"),
            "MaxErrorM": safe_stat(float(cast(Any, errors.max())))
            if not errors.empty
            else float("nan"),
            "MeanInliers": safe_stat(float(cast(Any, inliers.mean())))
            if not inliers.empty
            else float("nan"),
            "MeanCandidateMaps": safe_stat(float(cast(Any, candidates.mean())))
            if not candidates.empty
            else float("nan"),
            "MeanEvaluatedMaps": safe_stat(float(cast(Any, evaluated.mean())))
            if not evaluated.empty
            else float("nan"),
            "MeanMatchCount": safe_stat(float(cast(Any, matched_features.mean())))
            if not matched_features.empty
            else float("nan"),
            "RadiusUsage": radius_counts,
            "TopFailureReasons": top_failures,
        }

    def _fmt(self, value: Any, digits: int = 2, suffix: str = "") -> str:
        """Formats numeric values with sane fallbacks.

        Args:
            value: Value to be formatted.
            digits: Decimal precision for float formatting.
            suffix: Optional suffix appended to formatted value.

        Returns:
            Human-readable string representation.

        """
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if math.isnan(value):
                return "N/A"
            return f"{value:.{digits}f}{suffix}"
        return f"{value}{suffix}"

    def _build_markdown(self, metrics_list: List[Dict[str, Any]]) -> List[str]:
        """Builds markdown report lines from computed metrics.

        Args:
            metrics_list: Per-experiment metrics collection.

        Returns:
            List of markdown lines.

        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: List[str] = [
            "# Aerial Positioning Results Report",
            "",
            f"Generated: {now_utc}",
            "",
        ]

        summary_df = pd.DataFrame(metrics_list).sort_values(
            by=["RegionID", "Provider", "Zoom"],
            na_position="last",
        )

        total_frames = int(cast(Any, summary_df["TotalFrames"]).sum())
        total_success = int(cast(Any, summary_df["SuccessfulFrames"]).sum())
        overall_rate = (
            (100.0 * total_success / total_frames) if total_frames > 0 else 0.0
        )

        lines.extend(
            [
                "## Global Summary",
                "",
                f"- Experiments: {len(summary_df)}",
                f"- Total Frames: {total_frames}",
                f"- Successful Frames: {total_success}",
                f"- Overall Success Rate: {overall_rate:.2f}%",
                "",
                "## Experiment Table",
                "",
                "| Region | Provider | Zoom | Matcher | Frames | Success | Rate | Mean Err (m) | Median Err (m) | Std Err (m) | P90 Err (m) | Mean Inliers |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        for row in summary_df.to_dict(orient="records"):
            lines.append(
                f"| {row.get('RegionID', 'N/A')} | {row.get('Provider', 'N/A')} | "
                f"{row.get('Zoom', 'N/A')} | {row.get('Matcher', 'N/A')} | "
                f"{row.get('TotalFrames', 0)} | {row.get('SuccessfulFrames', 0)} | "
                f"{self._fmt(row.get('SuccessRatePct'), 2, '%')} | "
                f"{self._fmt(row.get('MeanErrorM'), 2)} | {self._fmt(row.get('MedianErrorM'), 2)} | "
                f"{self._fmt(row.get('StdErrorM'), 2)} | {self._fmt(row.get('P90ErrorM'), 2)} | "
                f"{self._fmt(row.get('MeanInliers'), 1)} |"
            )

        lines.extend(["", "## Aggregate By Provider", ""])
        provider_agg = (
            summary_df.groupby("Provider", dropna=False)
            .agg(
                Experiments=("Experiment", "count"),
                TotalFrames=("TotalFrames", "sum"),
                SuccessfulFrames=("SuccessfulFrames", "sum"),
                MeanErrorM=("MeanErrorM", "mean"),
                MedianErrorM=("MedianErrorM", "mean"),
                StdErrorM=("StdErrorM", "mean"),
                P90ErrorM=("P90ErrorM", "mean"),
                MeanInliers=("MeanInliers", "mean"),
            )
            .reset_index()
        )
        provider_agg["SuccessRatePct"] = (
            provider_agg["SuccessfulFrames"] / provider_agg["TotalFrames"] * 100.0
        ).fillna(0.0)

        lines.extend(
            [
                "| Provider | Experiments | Frames | Success | Rate | Mean Err (m) | Median Err (m) | Std Err (m) | P90 Err (m) | Mean Inliers |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for row in provider_agg.to_dict(orient="records"):
            lines.append(
                f"| {row.get('Provider', 'N/A')} | {int(row.get('Experiments', 0))} | "
                f"{int(row.get('TotalFrames', 0))} | {int(row.get('SuccessfulFrames', 0))} | "
                f"{self._fmt(float(row.get('SuccessRatePct', 0.0)), 2, '%')} | "
                f"{self._fmt(float(row.get('MeanErrorM', float('nan'))), 2)} | "
                f"{self._fmt(float(row.get('MedianErrorM', float('nan'))), 2)} | "
                f"{self._fmt(float(row.get('StdErrorM', float('nan'))), 2)} | "
                f"{self._fmt(float(row.get('P90ErrorM', float('nan'))), 2)} | "
                f"{self._fmt(float(row.get('MeanInliers', float('nan'))), 1)} |"
            )

        lines.extend(["", "## Per Experiment Details", ""])
        for row in summary_df.to_dict(orient="records"):
            lines.extend(
                [
                    f"### `{row['Experiment']}`",
                    "",
                    f"- Region: {row.get('RegionID', 'N/A')}",
                    f"- Provider: {row.get('Provider', 'N/A')}",
                    f"- Zoom: {row.get('Zoom', 'N/A')}",
                    f"- Matcher: {row.get('Matcher', 'N/A')}",
                    f"- Device: {row.get('Device', 'N/A')}",
                    f"- Frames: {row.get('TotalFrames', 0)}",
                    f"- Success: {row.get('SuccessfulFrames', 0)} / {row.get('TotalFrames', 0)} ({self._fmt(row.get('SuccessRatePct'), 2, '%')})",
                    f"- Error Mean/Median/Std/P90/Max (m): {self._fmt(row.get('MeanErrorM'), 2)} / {self._fmt(row.get('MedianErrorM'), 2)} / {self._fmt(row.get('StdErrorM'), 2)} / {self._fmt(row.get('P90ErrorM'), 2)} / {self._fmt(row.get('MaxErrorM'), 2)}",
                    f"- Mean Inliers: {self._fmt(row.get('MeanInliers'), 1)}",
                    f"- Mean Candidate Maps: {self._fmt(row.get('MeanCandidateMaps'), 1)}",
                    f"- Mean Evaluated Maps: {self._fmt(row.get('MeanEvaluatedMaps'), 1)}",
                    f"- Results CSV: `{row.get('ResultsCSV', 'N/A')}`",
                ]
            )

            radius_usage = row.get("RadiusUsage", {})
            if isinstance(radius_usage, dict) and radius_usage:
                radius_text = ", ".join(
                    f"{key}: {value}" for key, value in radius_usage.items()
                )
                lines.append(f"- Radius Usage: {radius_text}")

            top_failures = row.get("TopFailureReasons", {})
            if isinstance(top_failures, dict) and top_failures:
                fail_text = ", ".join(
                    f"{key}: {value}" for key, value in top_failures.items()
                )
                lines.append(f"- Top Failure Reasons: {fail_text}")

            lines.append("")

        return lines

    def generate_summary(
        self,
        region_ids: Optional[List[int]] = None,
        search_roots: Optional[List[Path]] = None,
    ) -> None:
        """Summarizes all results into markdown and CSV reports.

        Args:
            region_ids: Optional region filter list. If provided, only matching
                regions are included in report outputs.
            search_roots: Optional results roots or subfolders to scan.

        """
        if not self.results_root.exists() and not search_roots:
            _logger.info("No results directory found.")
            return

        roots = search_roots or [self.results_root]
        missing_roots = [root for root in roots if not root.exists()]
        if missing_roots:
            missing_text = ", ".join(str(root) for root in missing_roots)
            raise FileNotFoundError(f"Results directory not found: {missing_text}")

        summary_path = self.summary_md
        display_root = self.results_root
        if len(roots) == 1:
            summary_path = roots[0] / "report.md"
            display_root = roots[0]

        metrics_list: List[Dict[str, Any]] = []
        seen_dirs = set()
        for root in roots:
            for exp_dir in self._discover_experiment_dirs(root):
                if exp_dir in seen_dirs:
                    continue
                seen_dirs.add(exp_dir)
                metrics = self.get_experiment_metrics(
                    exp_dir, display_root=display_root
                )
                if metrics is None:
                    continue
                region_id = metrics.get("RegionID")
                if region_ids and region_id not in region_ids:
                    continue
                metrics_list.append(metrics)

        if not metrics_list:
            _logger.info("No valid results found to summarize.")
            return

        summary_df = pd.DataFrame(metrics_list).sort_values(
            by=["RegionID", "Provider", "Zoom"],
            na_position="last",
        )
        md_lines = self._build_markdown(metrics_list)
        summary_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

        display_cols = [
            "RegionID",
            "Provider",
            "Zoom",
            "Matcher",
            "TotalFrames",
            "SuccessfulFrames",
            "SuccessRatePct",
            "MeanErrorM",
            "MedianErrorM",
            "StdErrorM",
            "P90ErrorM",
            "MeanInliers",
        ]
        _logger.info("\n" + summary_df[display_cols].to_string(index=False))
        _logger.info(f"\nMarkdown report saved: {summary_path}")
