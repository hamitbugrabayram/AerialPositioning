"""Visual positioning and visualization module.

This module handles the primary visual positioning pipeline, including
displacement-based prediction and path visualization for GNSS-free
coordinate estimation.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.runner import PositioningRunner
from src.models.config import PositioningConfig, QueryResult


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
        """Initializes the Evaluator with configuration.

        Args:
            config: PositioningConfig instance with all settings.
        """
        super().__init__(config)
        self.full_query_df = None
        self.sampled_query_df = None
        self.frames_dir = None
        self.sample_interval = 30

    def run_trajectory(self) -> None:
        """Executes the visual positioning pipeline with sampling and prediction.

        Raises:
            RuntimeError: If required helper functions are not available.
        """
        print("\n[Visual Positioning System]")
        if not self._load_helpers():
            raise RuntimeError("Required helper functions not available.")

        self._validate_paths()
        self._setup_output_directory()

        if self.output_dir:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)

        self._initialize_preprocessor()
        self._initialize_pipeline()
        self._load_metadata()

        if self.query_df is not None:
            self.query_df = self.query_df.sort_values(by="Filename").reset_index(
                drop=True
            )
            self.full_query_df = self.query_df.copy()

            self.sampled_query_df = self.query_df.iloc[
                :: self.sample_interval
            ].reset_index(drop=True)
            print(
                f"Processing path at {self.sample_interval}-frame intervals. "
                f"Total checkpoints: {len(self.sampled_query_df)}"
            )

        results = self._process_eval_queries()
        self._save_results(results)
        self._generate_trajectory_plot(results)
        print("\n[Positioning Complete]")

    def _process_eval_queries(self) -> List[QueryResult]:
        """Processes images with GT-based displacement prediction.

        Returns:
            List of QueryResult objects for each processed checkpoint.
        """
        results = []
        if self.sampled_query_df is None or self.full_query_df is None:
            return results

        save_processed = self.config.preprocessing.get("save_processed", False)
        temp_dir = None
        if save_processed and self.output_dir:
            temp_dir = self.output_dir / "processed_queries"
        if temp_dir:
            temp_dir.mkdir(exist_ok=True)

        min_inliers = self.config.positioning_params.get("min_inliers_for_success", 10)
        save_viz = self.config.positioning_params.get("save_visualization", False)

        last_match_lat = self.sampled_query_df.iloc[0]["Latitude"]
        last_match_lon = self.sampled_query_df.iloc[0]["Longitude"]

        total_start = time.time()
        for idx, query_row in self.sampled_query_df.iterrows():
            filename = query_row["Filename"]
            print(
                f"\n[{int(idx) + 1}/{len(self.sampled_query_df)}] "
                f"Positioning: {filename}"
            )

            if idx == 0:
                search_lat, search_lon = last_match_lat, last_match_lon
                radius = 1000.0
            else:
                prev_filename = self.sampled_query_df.iloc[int(idx) - 1]["Filename"]
                curr_gt = self.full_query_df[
                    self.full_query_df["Filename"] == filename
                ].iloc[0]
                prev_gt = self.full_query_df[
                    self.full_query_df["Filename"] == prev_filename
                ].iloc[0]

                d_lat = curr_gt["Latitude"] - prev_gt["Latitude"]
                d_lon = curr_gt["Longitude"] - prev_gt["Longitude"]

                search_lat = last_match_lat + d_lat
                search_lon = last_match_lon + d_lon
                radius = 1000.0
                print(f"  Displacement Prediction: dLat {d_lat:.6f}, dLon {d_lon:.6f}")

            result = self._process_single_query_with_custom_radius(
                query_row,
                int(idx),
                temp_dir,
                min_inliers,
                save_viz,
                search_lat,
                search_lon,
                radius,
            )

            if result.success:
                print(
                    f"  Match SUCCESS! Inliers: {result.inliers}, "
                    f"Error: {result.error_meters:.2f}m"
                )
                last_match_lat = result.predicted_latitude
                last_match_lon = result.predicted_longitude
            else:
                print("  Match FAILED. Using predicted center for next window.")
                last_match_lat, last_match_lon = search_lat, search_lon
                result.success = False

            results.append(result)
            self._generate_trajectory_plot(results, frame_idx=int(idx))

        total_time = time.time() - total_start
        print(f"\nTotal Processing Time: {total_time:.2f}s")
        return results

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
            query_row: Pandas Series containing query metadata.
            idx: Index of the current query.
            temp_dir: Optional directory for saving processed images.
            min_inliers: Minimum inliers required for success.
            save_viz: Whether to save visualization outputs.
            ref_lat: Reference latitude for search center.
            ref_lon: Reference longitude for search center.
            radius: Search radius in meters.

        Returns:
            QueryResult with positioning results.
        """
        query_filename = str(query_row["Filename"])
        query_path = Path(self.config.data_paths["query_dir"]) / query_filename

        result = QueryResult(
            query_filename=query_filename,
            gt_latitude=query_row.get("Latitude"),
            gt_longitude=query_row.get("Longitude"),
        )

        if not query_path.is_file():
            return result

        query_for_match, query_shape = self._preprocess_query(
            query_path, query_row, temp_dir
        )

        if query_shape is None or self.map_df is None or self.output_dir is None:
            return result

        relevant_maps = self._filter_maps_by_custom_ref(
            ref_lat, ref_lon, self.map_df, radius
        )
        print(f"  Candidates in {radius}m: {len(relevant_maps)} tiles")

        query_results_dir = self.output_dir / Path(query_filename).stem
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

        return result

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
            map_df: DataFrame containing map tile metadata.
            radius: Maximum distance in meters.

        Returns:
            Filtered DataFrame with tiles within radius.
        """
        if self._haversine_distance is None:
            return map_df

        relevant_indices = []
        for idx, map_row in map_df.iterrows():
            m_lat = (
                float(map_row["Top_left_lat"]) + float(map_row["Bottom_right_lat"])
            ) / 2
            m_lon = (
                float(map_row["Top_left_lon"]) + float(map_row["Bottom_right_long"])
            ) / 2
            dist = self._haversine_distance(float(lat), float(lon), m_lat, m_lon)
            if dist <= radius:
                relevant_indices.append(idx)

        return map_df.loc[relevant_indices]

    def _generate_trajectory_plot(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Generates trajectory visualization plots.

        Args:
            results: List of QueryResult objects to visualize.
            frame_idx: Optional frame index for animation frames.
        """
        if not results or not self.output_dir:
            return

        self._save_static_plot(results, frame_idx)
        self._save_map_overlay(results, frame_idx)

    def _save_static_plot(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Saves a matplotlib static plot of the trajectory.

        Args:
            results: List of QueryResult objects to plot.
            frame_idx: Optional frame index for animation frames.
        """
        pred_lats = [r.predicted_latitude for r in results if r.success]
        pred_lons = [r.predicted_longitude for r in results if r.success]

        plt.figure(figsize=(12, 10))

        if self.full_query_df is not None:
            if frame_idx is not None:
                current_filename = results[-1].query_filename
                progress_end_idx = self.full_query_df[
                    self.full_query_df["Filename"] == current_filename
                ].index[0]
                gt_slice = self.full_query_df.iloc[: progress_end_idx + 1]
                plt.plot(
                    gt_slice["Longitude"],
                    gt_slice["Latitude"],
                    color="#FFA500",
                    linewidth=4,
                    label="GT Path",
                    alpha=0.8,
                    zorder=1,
                )
            else:
                plt.plot(
                    self.full_query_df["Longitude"],
                    self.full_query_df["Latitude"],
                    color="#FFA500",
                    linewidth=4,
                    label="Full GT Path",
                    alpha=0.8,
                    zorder=1,
                )

        if pred_lats:
            plt.scatter(
                pred_lons,
                pred_lats,
                c="blue",
                s=500,
                edgecolors="white",
                linewidths=3,
                label="Estimated",
                zorder=5,
            )

        plt.title(f"Visual Positioning - Frame: {len(results)}")
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="datalim")

        if frame_idx is not None and self.frames_dir:
            plt.savefig(self.frames_dir / f"plot_{frame_idx:04d}.png", dpi=100)
        else:
            plt.savefig(self.output_dir / "visual_positioning_plot.png", dpi=300)
        plt.close()

    def _save_map_overlay(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Saves a map overlay visualization with trajectory.

        Args:
            results: List of QueryResult objects to overlay.
            frame_idx: Optional frame index for animation frames.
        """
        if self.map_df is None or not self.output_dir:
            return

        try:
            min_lat = self.map_df["Bottom_right_lat"].min()
            max_lat = self.map_df["Top_left_lat"].max()
            min_lon = self.map_df["Top_left_lon"].min()
            max_lon = self.map_df["Bottom_right_long"].max()
            level = int(self.map_df.iloc[0]["Level"])

            from src.utils.tile_system import TileSystem

            x1, y1 = TileSystem.latlong_to_pixel_xy(max_lat, min_lon, level)
            x2, y2 = TileSystem.latlong_to_pixel_xy(min_lat, max_lon, level)
            total_w, total_h = x2 - x1, y2 - y1

            max_dim = 4096
            scale = min(1.0, max_dim / max(total_w, total_h))
            total_w, total_h = int(total_w * scale), int(total_h * scale)

            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            map_dir = Path(self.config.data_paths["map_dir"])

            for _, m_row in self.map_df.iterrows():
                m_path = map_dir / m_row["Filename"]
                if not m_path.exists():
                    continue
                tile_img = cv2.imread(str(m_path))
                if tile_img is None:
                    continue
                tx, ty = TileSystem.latlong_to_pixel_xy(
                    m_row["Top_left_lat"], m_row["Top_left_lon"], level
                )
                lx, ly = int((tx - x1) * scale), int((ty - y1) * scale)
                res_tw = int(tile_img.shape[1] * scale)
                res_th = int(tile_img.shape[0] * scale)
                tile_res = cv2.resize(tile_img, (res_tw, res_th))
                h_slice = min(res_th, total_h - ly)
                w_slice = min(res_tw, total_w - lx)
                if h_slice > 0 and w_slice > 0:
                    canvas[ly : ly + h_slice, lx : lx + w_slice] = tile_res[
                        :h_slice, :w_slice
                    ]

            def get_px(lat: float, lon: float) -> Tuple[int, int]:
                px, py = TileSystem.latlong_to_pixel_xy(lat, lon, level)
                return int((px - x1) * scale), int((py - y1) * scale)

            if self.full_query_df is not None:
                if frame_idx is not None:
                    current_filename = results[-1].query_filename
                    progress_end_idx = self.full_query_df[
                        self.full_query_df["Filename"] == current_filename
                    ].index[0]
                    gt_display_df = self.full_query_df.iloc[: progress_end_idx + 1]
                else:
                    gt_display_df = self.full_query_df

                pts = []
                for i in range(len(gt_display_df)):
                    p = get_px(
                        gt_display_df.iloc[i]["Latitude"],
                        gt_display_df.iloc[i]["Longitude"],
                    )
                    pts.append(p)

                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i + 1], (0, 165, 255), 8)

            for i, r in enumerate(results):
                p_gt = get_px(r.gt_latitude, r.gt_longitude)
                if not r.success:
                    cv2.drawMarker(
                        canvas, p_gt, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 35, 6
                    )
                    continue
                p_pred = get_px(r.predicted_latitude, r.predicted_longitude)
                cv2.line(canvas, p_pred, p_gt, (0, 0, 255), 8)
                cv2.circle(canvas, p_pred, 45, (255, 255, 255), -1)
                cv2.circle(canvas, p_pred, 35, (255, 0, 0), -1)

            provider = self.config.tile_provider.get("name", "Unknown").upper()

            cv2.putText(
                canvas,
                f"Map Provider: {provider}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (255, 255, 255),
                6,
            )
            cv2.putText(
                canvas,
                f"Zoom Level: {level}",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                4,
            )
            cv2.putText(
                canvas,
                "Orange: Ground Truth Path",
                (40, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 165, 255),
                4,
            )
            cv2.putText(
                canvas,
                "Blue: Visual Positioning",
                (40, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 0, 0),
                4,
            )
            cv2.putText(
                canvas,
                "Red Line: Precision Error",
                (40, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 0, 255),
                4,
            )

            if frame_idx is not None and self.frames_dir:
                cv2.imwrite(
                    str(self.frames_dir / f"overlay_{frame_idx:04d}.png"), canvas
                )
            else:
                cv2.imwrite(
                    str(self.output_dir / "visual_positioning_overlay.png"), canvas
                )
        except Exception as e:
            print(f"WARNING: Overlay failed: {e}")
