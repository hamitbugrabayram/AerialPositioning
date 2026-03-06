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

from src.utils.logger import get_logger
_logger = get_logger(__name__)


class TrajectoryVisualizer:
    """Handles trajectory visualization for the positioning system.

    Attributes:
        config (PositioningConfig): The positioning configuration.
        map_df (Optional[pd.DataFrame]): Map tile metadata dataframe.
        full_query_df (Optional[pd.DataFrame]): Complete query metadata dataframe.
        output_dir (Optional[Path]): Root experiment output directory.
        frames_dir (Optional[Path]): Directory for saving frame plots.
        assets_dir (Optional[Path]): Directory for saving summary plots.

    """

    _OVERLAY_MAX_SIDE = 4096
    """Maximum pixel length of the overlay image's longest side."""

    _LEGEND_OPACITY = 0.95
    """Opacity of the legend background box (0 = transparent, 1 = opaque)."""

    def __init__(self, config: PositioningConfig) -> None:
        """Initializes the visualizer with configuration.

        Args:
            config: The positioning configuration.

        """
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
            _logger.info(f"  WARNING: Plot generation failed: {e}")

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
            plt.plot(
                pred_lons,
                pred_lats,
                color="#0066FF",
                linewidth=3,
                label="Pred Path",
                alpha=0.8,
                zorder=2,
            )

        self._plot_error_lines(results)

        if pred_lats:
            plt.scatter(
                pred_lons,
                pred_lats,
                c="#0066FF",
                s=40,
                edgecolors="white",
                linewidths=1,
                label="Estimated",
                zorder=5,
            )

        self._plot_failed_markers(results)

        plt.title(f"Visual Positioning - Frame: {len(results)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="best")
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.grid(True, alpha=0.3)

        self._finalize_static_plot(frame_idx)

    def _extract_predicted_path(
        self, results: List[QueryResult]
    ) -> Tuple[List[float], List[float]]:
        """Extracts successful predicted coordinates from results."""
        lats, lons = [], []
        for r in results:
            if (
                r.success
                and r.predicted_latitude is not None
                and r.predicted_longitude is not None
            ):
                lats.append(float(r.predicted_latitude))
                lons.append(float(r.predicted_longitude))
        return lats, lons

    def _plot_gt_path(
        self, results: List[QueryResult], frame_idx: Optional[int]
    ) -> None:
        """Plots the ground truth path on the current figure."""
        if self.full_query_df is None:
            return
        if frame_idx is not None:
            current_filename = results[-1].query_filename
            indices = self.full_query_df[
                self.full_query_df["Filename"] == current_filename
            ].index
            if not indices.empty:
                progress_end_idx = int(cast(Any, indices[0]))
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

    def _plot_error_lines(self, results: List[QueryResult]) -> None:
        """Plots error lines between GT and predicted positions."""
        drawn = False
        for r in results:
            if (
                r.success
                and r.predicted_latitude is not None
                and r.predicted_longitude is not None
                and r.gt_latitude is not None
                and r.gt_longitude is not None
            ):
                plt.plot(
                    [r.gt_longitude, r.predicted_longitude],
                    [r.gt_latitude, r.predicted_latitude],
                    color="red",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=3,
                    label="Error" if not drawn else None,
                )
                drawn = True

    def _plot_failed_markers(self, results: List[QueryResult]) -> None:
        """Plots markers for failed positioning attempts."""
        lats, lons = [], []
        for r in results:
            if (
                not r.success
                and r.gt_latitude is not None
                and r.gt_longitude is not None
            ):
                lats.append(float(r.gt_latitude))
                lons.append(float(r.gt_longitude))
        if lats:
            plt.scatter(
                lons,
                lats,
                c="red",
                s=60,
                marker="x",
                linewidths=1.5,
                label="Failed",
                zorder=4,
            )

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
            canvas, scale, origin, level = self._prepare_map_canvas(
                results, frame_idx
            )
            if canvas is None:
                return

            self._draw_gt_overlay(canvas, results, frame_idx, scale, origin, level)
            self._draw_pred_overlay(canvas, results, scale, origin, level)
            self._draw_markers_and_errors(canvas, results, scale, origin, level)
            self._add_overlay_legend(canvas, results, scale, origin, level)

            self._save_overlay_canvas(canvas, frame_idx)
        except Exception as e:
            _logger.info(f"WARNING: Overlay failed: {e}")

    def _collect_trajectory_coords(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """Collects every lat/lon that the overlay needs to show.

        Gathers coordinates from the full ground-truth path and from
        predicted / GT positions stored in *results*.

        Returns:
            ``(lats, lons)`` lists.

        """
        lats: List[float] = []
        lons: List[float] = []

        if self.full_query_df is not None:
            df = self.full_query_df
            if frame_idx is not None and results:
                indices = df[
                    df["Filename"] == results[-1].query_filename
                ].index
                if not indices.empty:
                    df = df.iloc[: int(cast(Any, indices[0])) + 1]
            for i in range(len(df)):
                lats.append(float(df.iloc[i]["Latitude"]))
                lons.append(float(df.iloc[i]["Longitude"]))

        for r in results:
            if r.gt_latitude is not None and r.gt_longitude is not None:
                lats.append(float(r.gt_latitude))
                lons.append(float(r.gt_longitude))
            if (
                r.success
                and r.predicted_latitude is not None
                and r.predicted_longitude is not None
            ):
                lats.append(float(r.predicted_latitude))
                lons.append(float(r.predicted_longitude))

        return lats, lons

    @staticmethod
    def _find_safe_tile_rect(
        tile_index: dict,
        t_left: int,
        t_right: int,
        t_top: int,
        t_bottom: int,
    ) -> Tuple[int, int, int, int]:
        """Trims a tile rectangle until every cell has a tile.

        Iteratively removes the edge (left / right / top / bottom)
        that contains the most missing tiles until the remaining
        rectangle is fully covered.

        Returns:
            ``(t_left, t_right, t_top, t_bottom)`` of the safe rect.

        """
        max_iter = (t_right - t_left) + (t_bottom - t_top) + 2
        for _ in range(max_iter):
            if t_left > t_right or t_top > t_bottom:
                break
            missing_left = sum(
                1 for ty in range(t_top, t_bottom + 1)
                if (t_left, ty) not in tile_index
            )
            missing_right = sum(
                1 for ty in range(t_top, t_bottom + 1)
                if (t_right, ty) not in tile_index
            )
            missing_top = sum(
                1 for tx in range(t_left, t_right + 1)
                if (tx, t_top) not in tile_index
            )
            missing_bottom = sum(
                1 for tx in range(t_left, t_right + 1)
                if (tx, t_bottom) not in tile_index
            )
            worst = max(missing_left, missing_right, missing_top, missing_bottom)
            if worst == 0:
                break
            if missing_left == worst:
                t_left += 1
            elif missing_right == worst:
                t_right -= 1
            elif missing_top == worst:
                t_top += 1
            else:
                t_bottom -= 1
        return t_left, t_right, t_top, t_bottom

    def _prepare_map_canvas(
        self,
        results: List[QueryResult],
        frame_idx: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], float, Tuple[int, int], int]:
        """Creates a map canvas covering the full available tile grid.

        The canvas dimensions preserve the aspect ratio of the tile
        coverage with the longest side scaled to
        ``_OVERLAY_MAX_SIDE``.  Only tiles that actually exist are
        rendered — no black gaps and no repeated edge tiles.
        """
        if self.map_df is None:
            return None, 0.0, (0, 0), 0

        level = int(cast(Any, self.map_df.iloc[0])["Level"])

        map_dir = Path(self.config.data_paths["map_dir"])
        provider = str(self.config.tile_provider.get("name", "google"))

        tile_index: dict = {}
        for _, m_row in self.map_df.iterrows():
            tx, ty = int(m_row["TileX"]), int(m_row["TileY"])
            tile_index[(tx, ty)] = str(m_row["Filename"])

        min_tx = int(cast(Any, self.map_df["TileX"]).min())
        max_tx = int(cast(Any, self.map_df["TileX"]).max())
        min_ty = int(cast(Any, self.map_df["TileY"]).min())
        max_ty = int(cast(Any, self.map_df["TileY"]).max())

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if (tx, ty) not in tile_index:
                    fname = f"tile_{provider}_{level}_{tx}_{ty}.jpg"
                    if (map_dir / fname).exists():
                        tile_index[(tx, ty)] = fname

        min_tx, max_tx, min_ty, max_ty = self._find_safe_tile_rect(
            tile_index, min_tx, max_tx, min_ty, max_ty
        )

        cov_px_l = min_tx * 256
        cov_px_t = min_ty * 256
        cov_px_r = (max_tx + 1) * 256
        cov_px_b = (max_ty + 1) * 256
        cov_w = cov_px_r - cov_px_l
        cov_h = cov_px_b - cov_px_t

        if cov_w <= 0 or cov_h <= 0:
            return None, 0.0, (0, 0), 0

        longest = max(cov_w, cov_h)
        scale = float(self._OVERLAY_MAX_SIDE) / float(longest)
        canvas_w = round(cov_w * scale)
        canvas_h = round(cov_h * scale)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                fname = tile_index.get((tx, ty))
                if fname is None:
                    continue
                tile_img = cv2.imread(str(map_dir / fname))
                if tile_img is None:
                    continue

                gx1, gy1 = tx * 256, ty * 256
                gx2, gy2 = gx1 + 256, gy1 + 256

                cx1 = round((gx1 - cov_px_l) * scale)
                cy1 = round((gy1 - cov_px_t) * scale)
                cx2 = round((gx2 - cov_px_l) * scale)
                cy2 = round((gy2 - cov_px_t) * scale)

                tw, th = cx2 - cx1, cy2 - cy1
                if tw <= 0 or th <= 0:
                    continue

                dcx1 = max(0, cx1)
                dcy1 = max(0, cy1)
                dcx2 = min(canvas_w, cx2)
                dcy2 = min(canvas_h, cy2)
                if dcx2 <= dcx1 or dcy2 <= dcy1:
                    continue

                tile_res = cv2.resize(tile_img, (tw, th))
                sx1, sy1 = dcx1 - cx1, dcy1 - cy1
                w_fit, h_fit = dcx2 - dcx1, dcy2 - dcy1
                canvas[dcy1:dcy2, dcx1:dcx2] = (
                    tile_res[sy1 : sy1 + h_fit, sx1 : sx1 + w_fit]
                )

        origin = (cov_px_l, cov_px_t)
        return canvas, scale, origin, level

    def _get_px(
        self, lat: float, lon: float, scale: float, origin: Tuple[int, int], level: int
    ) -> Tuple[int, int]:
        """Converts lat/lon to canvas pixel coordinates.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            scale (float): Map scale factor.
            origin (Tuple[int, int]): Origin pixel coordinates of the map.
            level (int): Map zoom level.

        Returns:
            Tuple[int, int]: The pixel coordinates (x, y) on the canvas.

        """
        px, py = TileSystem.latlong_to_pixel_xy(float(lat), float(lon), level)
        return int((px - origin[0]) * scale), int((py - origin[1]) * scale)

    def _draw_gt_overlay(
        self,
        canvas: np.ndarray,
        results: List[QueryResult],
        frame_idx: Optional[int],
        scale: float,
        origin: Tuple[int, int],
        level: int,
    ) -> None:
        """Draws ground truth path on the overlay.

        Args:
            canvas (np.ndarray): The map canvas image array.
            results (List[QueryResult]): The list of query results.
            frame_idx (Optional[int]): Current frame index.
            scale (float): Map scale.
            origin (Tuple[int, int]): Origin pixel.
            level (int): Zoom level.

        """
        if self.full_query_df is None:
            return
        df = self.full_query_df
        if frame_idx is not None:
            indices = df[df["Filename"] == results[-1].query_filename].index
            if not indices.empty:
                df = df.iloc[: int(cast(Any, indices[0])) + 1]

        pts = []
        for i in range(len(df)):
            pts.append(
                self._get_px(
                    float(df.iloc[i]["Latitude"]),
                    float(df.iloc[i]["Longitude"]),
                    scale,
                    origin,
                    level,
                )
            )
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], (0, 0, 0), 12)
            cv2.line(canvas, pts[i], pts[i + 1], (0, 165, 255), 7)

    def _draw_pred_overlay(
        self,
        canvas: np.ndarray,
        results: List[QueryResult],
        scale: float,
        origin: Tuple[int, int],
        level: int,
    ) -> None:
        """Draws predicted path on the overlay.

        Args:
            canvas (np.ndarray): The map canvas image array.
            results (List[QueryResult]): The list of query results.
            scale (float): Map scale.
            origin (Tuple[int, int]): Origin pixel.
            level (int): Zoom level.

        """
        pts = []
        for r in results:
            if (
                r.success
                and r.predicted_latitude is not None
                and r.predicted_longitude is not None
            ):
                pts.append(
                    self._get_px(
                        float(r.predicted_latitude),
                        float(r.predicted_longitude),
                        scale,
                        origin,
                        level,
                    )
                )
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], (0, 0, 0), 10)
            cv2.line(canvas, pts[i], pts[i + 1], (255, 0, 0), 5)

    def _draw_markers_and_errors(
        self,
        canvas: np.ndarray,
        results: List[QueryResult],
        scale: float,
        origin: Tuple[int, int],
        level: int,
    ) -> None:
        """Draws markers and error lines on the overlay.

        Args:
            canvas (np.ndarray): The map canvas image array.
            results (List[QueryResult]): The list of query results.
            scale (float): Map scale.
            origin (Tuple[int, int]): Origin pixel.
            level (int): Zoom level.

        """
        error_color = (0, 0, 255)
        failed_color = (0, 0, 255)
        for r in results:
            if r.gt_latitude is None or r.gt_longitude is None:
                continue
            p_gt = self._get_px(
                float(r.gt_latitude), float(r.gt_longitude), scale, origin, level
            )
            if (
                not r.success
                or r.predicted_latitude is None
                or r.predicted_longitude is None
            ):
                cv2.drawMarker(
                    canvas, p_gt, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 7
                )
                cv2.drawMarker(
                    canvas, p_gt, failed_color, cv2.MARKER_TILTED_CROSS, 24, 5
                )
                continue
            p_pred = self._get_px(
                float(r.predicted_latitude),
                float(r.predicted_longitude),
                scale,
                origin,
                level,
            )
            cv2.line(canvas, p_pred, p_gt, (255, 255, 255), 7)
            cv2.line(canvas, p_pred, p_gt, error_color, 4)
            cv2.circle(canvas, p_pred, 18, (255, 255, 255), -1)
            cv2.circle(canvas, p_pred, 13, (255, 0, 0), -1)

    def _add_overlay_legend(
        self,
        canvas: np.ndarray,
        results: List[QueryResult],
        scale: float,
        origin: Tuple[int, int],
        level: int,
    ) -> None:
        """Adds a compact legend in a semi-transparent box.

        The box is placed near the top-left corner with a small margin
        so it is unlikely to obscure the main content. Only the region
        identifier and tile provider are shown.
        """
        h, w = canvas.shape[:2]
        s = max(h, w) / 4096.0

        output_dir = Path(self.config.data_paths.get("output_dir", ""))
        exp_name = output_dir.name
        if "_zoom_" in exp_name:
            region_id = exp_name.split("_zoom_")[0].replace("_", " ")
        else:
            region_id = exp_name.replace("_", " ")

        provider = str(self.config.tile_provider.get("name", "Unknown")).upper()

        lines = [
            f"Region: {region_id}",
            f"Provider: {provider} z{level}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0 * s
        thickness = max(1, int(5 * s))
        line_gap = int(26 * s)
        pad_x = int(32 * s)
        pad_y = int(26 * s)

        sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
        text_ws = [sz[0][0] for sz in sizes]
        text_hs = [sz[0][1] for sz in sizes]

        box_w = max(text_ws) + 2 * pad_x
        box_h = sum(text_hs) + (len(lines) - 1) * line_gap + 2 * pad_y

        margin = int(24 * s)
        box_x = margin
        box_y = margin

        bx1 = max(0, box_x)
        by1 = max(0, box_y)
        bx2 = min(w, box_x + box_w)
        by2 = min(h, box_y + box_h)
        if bx2 > bx1 and by2 > by1:
            roi = canvas[by1:by2, bx1:bx2].copy()
            dark = np.zeros_like(roi)
            dark[:] = (30, 30, 30)
            alpha = self._LEGEND_OPACITY
            cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, roi)
            canvas[by1:by2, bx1:bx2] = roi

        y_cursor = box_y + pad_y + text_hs[0]
        for i, text in enumerate(lines):
            pos = (box_x + pad_x, y_cursor)
            cv2.putText(
                canvas, text, pos, font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )
            if i < len(lines) - 1:
                y_cursor += text_hs[i + 1] + line_gap

    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draws text with shadow for better visibility.

        Args:
            img (np.ndarray): Image array to draw on.
            text (str): The text to draw.
            pos (Tuple[int, int]): Position (x, y) for the text.
            scale (float): Font scale.
            color (Tuple[int, int, int]): Font color in BGR format.
            thickness (int): Font thickness.

        """
        cv2.putText(
            img,
            text,
            (pos[0] + 4, pos[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
        )
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _save_overlay_canvas(
        self, canvas: np.ndarray, frame_idx: Optional[int]
    ) -> None:
        """Saves the final overlay canvas to disk."""
        if frame_idx is not None and self.frames_dir:
            cv2.imwrite(str(self.frames_dir / f"overlay_{frame_idx:04d}.png"), canvas)
        elif self.assets_dir:
            cv2.imwrite(str(self.assets_dir / "visual_positioning_overlay.png"), canvas)
