"""Plotting and trajectory visualization utilities.

This module provides the TrajectoryVisualizer class for generating
trajectory plots, map overlays, and error distributions.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.config import PositioningConfig, QueryResult
from src.utils.tile_system import TileSystem


class TrajectoryVisualizer:
    """Handles trajectory visualization for the positioning system."""

    def __init__(self, config: PositioningConfig):
        """Initializes the visualizer with configuration."""
        self.config = config
        self.map_df: Optional[pd.DataFrame] = None
        self.full_query_df: Optional[pd.DataFrame] = None
        self.output_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
        self.assets_dir: Optional[Path] = None

    def set_context(
        self,
        map_df: Optional[pd.DataFrame] = None,
        full_query_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        frames_dir: Optional[Path] = None,
        assets_dir: Optional[Path] = None,
    ) -> None:
        """Updates the context for visualization.

        Args:
            map_df: Metadata for satellite tiles.
            full_query_df: Complete query metadata.
            output_dir: Root experiment directory.
            frames_dir: Directory for frame-by-frame plots.
            assets_dir: Directory for summary assets.
        """
        if map_df is not None:
            self.map_df = map_df
        if full_query_df is not None:
            self.full_query_df = full_query_df
        if output_dir is not None:
            self.output_dir = output_dir
        if frames_dir is not None:
            self.frames_dir = frames_dir
        if assets_dir is not None:
            self.assets_dir = assets_dir

    def generate_trajectory_plot(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Generates trajectory visualization plots.

        Args:
            results: List of query results to plot.
            frame_idx: Optional index for frame-by-frame saving.
        """
        if not results or not self.output_dir:
            return

        try:
            self.save_static_plot(results, frame_idx)
            self.save_map_overlay(results, frame_idx)
        except Exception as e:
            print(f"  WARNING: Plot generation failed: {e}")

    def save_static_plot(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Saves a matplotlib static plot of the trajectory.

        Args:
            results: List of query results to plot.
            frame_idx: Optional index for frame-by-frame saving.
        """
        plt.figure(figsize=(12, 10))

        pred_lats, pred_lons = self._extract_predicted_path(results)

        if self.full_query_df is not None:
            self._plot_gt_path(results, frame_idx)

        if len(pred_lats) > 1:
            plt.plot(pred_lons, pred_lats, color="#0066FF", linewidth=3,
                     label="Pred Path", alpha=0.8, zorder=2)

        self._plot_error_lines(results)

        if pred_lats:
            plt.scatter(pred_lons, pred_lats, c="#0066FF", s=40,
                        edgecolors="white", linewidths=1, label="Estimated", zorder=5)

        self._plot_failed_markers(results)

        plt.title(f"Visual Positioning - Frame: {len(results)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="best")
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.grid(True, alpha=0.3)

        self._finalize_static_plot(frame_idx)

    def _extract_predicted_path(self, results: List[QueryResult]) -> Tuple[List[float], List[float]]:
        """Extracts successful predicted coordinates from results."""
        lats, lons = [], []
        for r in results:
            if r.success and r.predicted_latitude is not None and r.predicted_longitude is not None:
                lats.append(float(r.predicted_latitude))
                lons.append(float(r.predicted_longitude))
        return lats, lons

    def _plot_gt_path(self, results: List[QueryResult], frame_idx: Optional[int]) -> None:
        """Plots the ground truth path on the current figure."""
        if self.full_query_df is None:
            return
        if frame_idx is not None:
            current_filename = results[-1].query_filename
            indices = self.full_query_df[self.full_query_df["Filename"] == current_filename].index
            if not indices.empty:
                progress_end_idx = int(cast(Any, indices[0]))
                gt_slice = self.full_query_df.iloc[: progress_end_idx + 1]
                plt.plot(gt_slice["Longitude"], gt_slice["Latitude"], color="#FFA500",
                         linewidth=4, label="GT Path", alpha=0.8, zorder=1)
        else:
            plt.plot(self.full_query_df["Longitude"], self.full_query_df["Latitude"],
                     color="#FFA500", linewidth=4, label="Full GT Path", alpha=0.8, zorder=1)

    def _plot_error_lines(self, results: List[QueryResult]) -> None:
        """Plots error lines between GT and predicted positions."""
        drawn = False
        for r in results:
            if (r.success and r.predicted_latitude is not None and 
                r.predicted_longitude is not None and r.gt_latitude is not None and 
                r.gt_longitude is not None):
                plt.plot([r.gt_longitude, r.predicted_longitude],
                         [r.gt_latitude, r.predicted_latitude],
                         color="red", linewidth=1.5, alpha=0.6, zorder=3,
                         label="Error" if not drawn else None)
                drawn = True

    def _plot_failed_markers(self, results: List[QueryResult]) -> None:
        """Plots markers for failed positioning attempts."""
        lats, lons = [], []
        for r in results:
            if not r.success and r.gt_latitude is not None and r.gt_longitude is not None:
                lats.append(float(r.gt_latitude))
                lons.append(float(r.gt_longitude))
        if lats:
            plt.scatter(lons, lats, c="red", s=60, marker="x",
                        linewidths=1.5, label="Failed", zorder=4)

    def _finalize_static_plot(self, frame_idx: Optional[int]) -> None:
        """Saves and closes the current static plot."""
        if frame_idx is not None and self.frames_dir:
            plt.savefig(self.frames_dir / f"plot_{frame_idx:04d}.png", dpi=100)
        elif self.assets_dir:
            plt.savefig(self.assets_dir / "visual_positioning_plot.png", dpi=300)
        plt.close()

    def save_map_overlay(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Saves a map overlay visualization with trajectory paths."""
        if self.map_df is None or not self.output_dir:
            return

        try:
            canvas, scale, origin, level = self._prepare_map_canvas()
            if canvas is None:
                return

            self._draw_gt_overlay(canvas, results, frame_idx, scale, origin, level)
            self._draw_pred_overlay(canvas, results, scale, origin, level)
            self._draw_markers_and_errors(canvas, results, scale, origin, level)
            self._add_overlay_legend(canvas, level)

            self._save_overlay_canvas(canvas, frame_idx)
        except Exception as e:
            print(f"WARNING: Overlay failed: {e}")

    def _prepare_map_canvas(self) -> Tuple[Optional[np.ndarray], float, Tuple[int, int], int]:
        """Creates the base map canvas from satellite tiles."""
        if self.map_df is None:
            return None, 0.0, (0, 0), 0
        min_tx = int(cast(Any, self.map_df["TileX"]).min())
        max_tx = int(cast(Any, self.map_df["TileX"]).max())
        min_ty = int(cast(Any, self.map_df["TileY"]).min())
        max_ty = int(cast(Any, self.map_df["TileY"]).max())
        level = int(cast(Any, self.map_df.iloc[0])["Level"])

        full_w, full_h = (max_tx - min_tx + 1) * 256, (max_ty - min_ty + 1) * 256
        scale = min(1.0, 4096 / float(max(full_w, full_h))) if max(full_w, full_h) > 0 else 1.0
        tw, th = int(full_w * scale), int(full_h * scale)

        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        map_dir = Path(self.config.data_paths["map_dir"])

        for _, m_row in self.map_df.iterrows():
            tile_img = cv2.imread(str(map_dir / str(m_row["Filename"])))
            if tile_img is None:
                continue
            tx, ty = int(m_row["TileX"]), int(m_row["TileY"])
            x1, y1 = int((tx - min_tx) * 256 * scale), int((ty - min_ty) * 256 * scale)
            x2, y2 = int((tx - min_tx + 1) * 256 * scale), int((ty - min_ty + 1) * 256 * scale)
            if x2 - x1 > 0 and y2 - y1 > 0:
                tile_res = cv2.resize(tile_img, (x2 - x1, y2 - y1))
                h_fit, w_fit = min(y2 - y1, th - y1), min(x2 - x1, tw - x1)
                canvas[y1:y1 + h_fit, x1:x1 + w_fit] = tile_res[:h_fit, :w_fit]

        origin = TileSystem.tile_xy_to_pixel_xy(min_tx, min_ty)
        return canvas, scale, origin, level

    def _get_px(self, lat: float, lon: float, scale: float, origin: Tuple[int, int], level: int) -> Tuple[int, int]:
        """Converts lat/lon to canvas pixel coordinates."""
        px, py = TileSystem.latlong_to_pixel_xy(float(lat), float(lon), level)
        return int((px - origin[0]) * scale), int((py - origin[1]) * scale)

    def _draw_gt_overlay(self, canvas, results, frame_idx, scale, origin, level):
        """Draws ground truth path on the overlay."""
        if self.full_query_df is None:
            return
        df = self.full_query_df
        if frame_idx is not None:
            indices = df[df["Filename"] == results[-1].query_filename].index
            if not indices.empty:
                df = df.iloc[: int(indices[0]) + 1]

        pts = []
        for i in range(len(df)):
            pts.append(self._get_px(float(df.iloc[i]["Latitude"]), float(df.iloc[i]["Longitude"]), scale, origin, level))
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], (0, 165, 255), 8)

    def _draw_pred_overlay(self, canvas, results, scale, origin, level):
        """Draws predicted path on the overlay."""
        pts = []
        for r in results:
            if r.success and r.predicted_latitude is not None and r.predicted_longitude is not None:
                pts.append(self._get_px(float(r.predicted_latitude), float(r.predicted_longitude), scale, origin, level))
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], (255, 102, 0), 6)

    def _draw_markers_and_errors(self, canvas, results, scale, origin, level):
        """Draws markers and error lines on the overlay."""
        for r in results:
            if r.gt_latitude is None or r.gt_longitude is None:
                continue
            p_gt = self._get_px(float(r.gt_latitude), float(r.gt_longitude), scale, origin, level)
            if not r.success or r.predicted_latitude is None or r.predicted_longitude is None:
                cv2.drawMarker(canvas, p_gt, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 25, 4)
                continue
            p_pred = self._get_px(float(r.predicted_latitude), float(r.predicted_longitude), scale, origin, level)
            cv2.line(canvas, p_pred, p_gt, (0, 0, 255), 3)
            cv2.circle(canvas, p_pred, 18, (255, 255, 255), -1)
            cv2.circle(canvas, p_pred, 14, (255, 102, 0), -1)

    def _add_overlay_legend(self, canvas: np.ndarray, level: int) -> None:
        """Adds text legend to the map overlay."""
        provider = str(self.config.tile_provider.get("name", "Unknown")).upper()
        texts = [
            (f"Map Provider: {provider}", (255, 255, 255)),
            (f"Zoom Level: {level}", (255, 255, 255)),
            ("Ground Truth Path", (0, 165, 255)),
            ("Predicted Path", (255, 102, 0)),
            ("Error Line", (0, 0, 255)),
            ("Failed Match (X)", (0, 0, 255))
        ]
        for i, (text, color) in enumerate(texts):
            self._draw_text(canvas, text, (40, 100 + i * 90), 2.5 if i == 0 else 1.8, color, 6 if i == 0 else 4)

    def _draw_text(self, img, text, pos, scale, color, thickness):
        """Draws text with shadow for better visibility."""
        cv2.putText(img, text, (pos[0] + 4, pos[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _save_overlay_canvas(self, canvas: np.ndarray, frame_idx: Optional[int]) -> None:
        """Saves the final overlay canvas to disk."""
        if frame_idx is not None and self.frames_dir:
            cv2.imwrite(str(self.frames_dir / f"overlay_{frame_idx:04d}.png"), canvas)
        elif self.assets_dir:
            cv2.imwrite(str(self.assets_dir / "visual_positioning_overlay.png"), canvas)
