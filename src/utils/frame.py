"""Composite frame renderer for per-query diagnostic visualisation.

Produces a single composite image per evaluation frame, suitable for
assembly into diagnostic videos.  Each frame contains:

* **Top-left**    -- Frame info (status, error, radius) and a 2D
  trajectory mini-map showing predicted positions so far.
* **Bottom-left** -- The preprocessed drone (query) image.
* **Right**       -- Satellite tile mosaic with homography polygon.
  Inlier match lines are drawn across panels, connecting query
  keypoints (bottom-left) to their satellite correspondences (right).

Camera smoothing is applied between frames so the satellite view pans
and zooms smoothly, making the output watchable as a video.

Activated when ``pair_logging.enabled`` is *True* in the positioning
config. Frames are written to ``<output_dir>/frames/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.models.config import PositioningConfig, QueryResult
from src.utils.tile_system import TileSystem

from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class FrameData:
    """Per-frame bundle consumed by :class:`CompositeFrameRenderer`.

    Attributes:
        frame_idx: Sequential frame number in the evaluation.
        query_filename: Basename of the query image file.
        query_image_path: Absolute path to the original drone image.
        gt_lat: Ground-truth latitude of the query.
        gt_lon: Ground-truth longitude of the query.
        search_center_lat: Latitude of the adaptive-search centre.
        search_center_lon: Longitude of the adaptive-search centre.
        search_radius_m: Current adaptive search radius in metres.
        success: Whether the frame was successfully matched.
        predicted_lat: Predicted latitude (``None`` on failure).
        predicted_lon: Predicted longitude (``None`` on failure).
        error_meters: Positioning error in metres (``None`` on failure).
        inliers: RANSAC inlier count for the best match.
        homography: 3x3 homography matrix (``None`` on failure).
        effective_map_metadata: Georeferencing dict of the matched
            tile or composite (``None`` on failure).
        query_shape: Shape tuple of the preprocessed query image.
        map_shape: Shape tuple of the matched map image.
        preprocessed_image_path: Absolute path to the preprocessed
            drone image.  Falls back to ``query_image_path`` when
            ``None``.
        mkpts0: Matched keypoints in the query image (Nx2 float).
        mkpts1: Matched keypoints in the map image (Nx2 float).
        inliers_mask: Boolean RANSAC inlier mask aligned with
            ``mkpts0`` / ``mkpts1``.
        map_match_path: Path to the map image that was actually fed
            to the matcher (may be a context composite).

    """

    frame_idx: int
    query_filename: str
    query_image_path: str
    gt_lat: float
    gt_lon: float
    search_center_lat: float
    search_center_lon: float
    search_radius_m: float
    success: bool
    predicted_lat: Optional[float] = None
    predicted_lon: Optional[float] = None
    error_meters: Optional[float] = None
    inliers: int = 0
    homography: Optional[np.ndarray] = None
    effective_map_metadata: Optional[Dict[str, Any]] = None
    query_shape: Optional[Tuple[int, ...]] = None
    map_shape: Optional[Tuple[int, ...]] = None
    preprocessed_image_path: Optional[str] = None
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None
    inliers_mask: Optional[np.ndarray] = None
    map_match_path: Optional[str] = None


class CompositeFrameRenderer:
    """Renders composite visualisation frames for video analysis.

    Layout (1920 x 1080)::

        +------------------+---------------------------+
        | Info + Trajectory |                           |
        | (560 x 380)      |   Satellite mosaic        |
        +------------------+   with homography polygon  |
        |                  |   (1360 x 1080)            |
        | Query image      |                           |
        | (preprocessed)   |   Match lines connect     |
        | (560 x 700)      |   query <-> satellite     |
        +------------------+---------------------------+

    The satellite panel tracks a virtual camera that smoothly
    interpolates toward the target position using an exponential
    moving average.  On successful matches the camera converges
    faster (``ALPHA_SUCCESS``) so the homography polygon stays
    visible; on failures it drifts slowly (``ALPHA_FAIL``) in the
    direction of the search centre.

    Attributes:
        FRAME_W: Total frame width in pixels.
        FRAME_H: Total frame height in pixels.
        LEFT_W: Width of the left column.
        RIGHT_W: Width of the satellite panel.
        TOP_LEFT_H: Height of the info + trajectory panel.
        BOT_LEFT_H: Height of the query image panel.
        ALPHA_SUCCESS: EMA factor when the frame matched.
        ALPHA_FAIL: EMA factor when the frame failed.
        MIN_VIEW_RADIUS_M: Minimum visible radius in metres.
        VIEW_RADIUS_FACTOR: Multiplier applied to the search radius
            to determine the satellite viewport extent.
        MAX_DRAWN_MATCHES: Cap on inlier lines drawn to avoid clutter.
        JPEG_QUALITY: Output JPEG compression quality.

    """

    FRAME_W = 1920
    FRAME_H = 1080
    LEFT_W = 560
    RIGHT_W = 1360
    TOP_LEFT_H = 380
    BOT_LEFT_H = 700

    ALPHA_SUCCESS = 0.65
    ALPHA_FAIL = 0.30

    MIN_VIEW_RADIUS_M = 400.0
    VIEW_RADIUS_FACTOR = 1.4

    MAX_DRAWN_MATCHES = 50

    COL_PRED_OK = (0, 180, 0)
    COL_PRED_FAIL = (0, 0, 220)
    COL_CURRENT = (0, 80, 255)
    COL_POLYGON = (0, 255, 255)
    COL_POLYGON_FILL = (0, 255, 255)
    COL_POLYGON_CENTER = (0, 0, 255)
    COL_SEARCH = (0, 180, 255)
    COL_MATCH_LINE = (80, 220, 80)
    COL_MATCH_PT = (0, 255, 0)
    COL_PANEL_BG = 30
    COL_INFO_BG = 245

    JPEG_QUALITY = 88

    def __init__(
        self,
        config: PositioningConfig,
        map_df: pd.DataFrame,
        full_query_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Initialises the renderer.

        Args:
            config: Global positioning configuration.
            map_df: Map tile metadata dataframe (must contain a
                ``Level`` column).
            full_query_df: Complete query dataframe with ``Latitude``
                and ``Longitude`` columns for the trajectory extent.
            output_dir: Parent directory under which the
                ``frames/`` sub-directory is created.

        """
        self.config = config
        self.map_df = map_df
        self.full_query_df = full_query_df
        self.map_dir = Path(config.data_paths["map_dir"])
        self.output_dir = output_dir / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.level = int(map_df.iloc[0]["Level"])
        self.provider = config.tile_provider.get("name", "google")

        self.cam_lat: Optional[float] = None
        self.cam_lon: Optional[float] = None
        self.cam_radius: Optional[float] = None

        gt_lats = full_query_df["Latitude"].values.astype(float)
        gt_lons = full_query_df["Longitude"].values.astype(float)
        margin = 0.08
        lat_rng = max(gt_lats.max() - gt_lats.min(), 1e-4)
        lon_rng = max(gt_lons.max() - gt_lons.min(), 1e-4)
        self._plot_min_lat = gt_lats.min() - lat_rng * margin
        self._plot_max_lat = gt_lats.max() + lat_rng * margin
        self._plot_min_lon = gt_lons.min() - lon_rng * margin
        self._plot_max_lon = gt_lons.max() + lon_rng * margin

        self._frame_count = 0

    def render_frame(
        self,
        frame_data: FrameData,
        all_results: List[QueryResult],
    ) -> Optional[Path]:
        """Renders and saves one composite frame.

        The pipeline builds three panels independently, composites
        them, then draws cross-panel match lines on the final image.

        Args:
            frame_data: Rich per-frame data (positions, homography …).
            all_results: Accumulated results for the trajectory.

        Returns:
            Path to the saved JPEG, or ``None`` on failure.

        """
        try:
            self._update_camera(frame_data)

            sat_panel, g2p = self._render_satellite_panel()
            self._draw_search_circle(sat_panel, g2p, frame_data)
            if frame_data.success and frame_data.homography is not None:
                self._draw_homography_polygon(sat_panel, g2p, frame_data)

            traj_panel = self._render_trajectory_panel(
                all_results, frame_data,
            )
            query_panel, q_scale, q_xo, q_yo = self._render_query_panel(
                frame_data,
            )

            frame = self._composite(traj_panel, query_panel, sat_panel)
            self._draw_match_lines(
                frame, frame_data, q_scale, q_xo, q_yo, g2p,
            )

            out_path = self.output_dir / f"frame_{frame_data.frame_idx:04d}.jpg"
            cv2.imwrite(
                str(out_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
            )
            self._frame_count += 1
            return out_path

        except Exception as e:
            _logger.warning(
                f"Composite frame {frame_data.frame_idx} failed: {e}"
            )
            return None

    def _update_camera(self, fd: FrameData) -> None:
        """Smoothly moves the virtual camera toward the target.

        On the first invocation the camera snaps directly to the
        target.  Subsequent calls apply an exponential moving average
        whose factor depends on whether the frame matched.

        Args:
            fd: Current frame data providing the target position and
                search radius.

        """
        if fd.success and fd.predicted_lat is not None:
            target_lat = fd.predicted_lat
            target_lon = fd.predicted_lon or fd.search_center_lon
        else:
            target_lat = fd.search_center_lat
            target_lon = fd.search_center_lon

        target_radius = max(
            self.MIN_VIEW_RADIUS_M,
            fd.search_radius_m * self.VIEW_RADIUS_FACTOR,
        )

        if self.cam_lat is None:
            self.cam_lat = target_lat
            self.cam_lon = target_lon
            self.cam_radius = target_radius
            return

        alpha = self.ALPHA_SUCCESS if fd.success else self.ALPHA_FAIL
        self.cam_lat = alpha * target_lat + (1.0 - alpha) * self.cam_lat
        self.cam_lon = alpha * target_lon + (1.0 - alpha) * self.cam_lon
        self.cam_radius = alpha * target_radius + (1.0 - alpha) * self.cam_radius

    def _render_satellite_panel(
        self,
    ) -> Tuple[np.ndarray, Callable[[float, float], Tuple[int, int]]]:
        """Loads satellite tiles for the current camera view.

        Converts the camera centre and radius into a Mercator-pixel
        viewport, determines the enclosing tile range, stitches tiles
        into a canvas, crops to the viewport, and scales to the panel
        dimensions.

        Returns:
            Tuple of ``(panel_image, global_to_panel)`` where
            *global_to_panel* is a callable that maps Mercator
            global-pixel coordinates ``(gpx, gpy)`` to panel-pixel
            coordinates ``(px, py)``.

        """
        pw, ph = self.RIGHT_W, self.FRAME_H
        cam_lat = float(self.cam_lat)
        cam_lon = float(self.cam_lon)
        cam_radius = float(self.cam_radius)

        gres = TileSystem.ground_resolution(cam_lat, self.level)
        if gres <= 0:
            gres = 1.0
        radius_px = cam_radius / gres
        half_h = radius_px
        half_w = radius_px * (pw / ph)

        cx, cy = TileSystem.latlong_to_pixel_xy(cam_lat, cam_lon, self.level)
        vp_l, vp_t = cx - half_w, cy - half_h
        vp_r, vp_b = cx + half_w, cy + half_h

        nw_tx, nw_ty = TileSystem.pixel_xy_to_tile_xy(int(vp_l), int(vp_t))
        se_tx, se_ty = TileSystem.pixel_xy_to_tile_xy(int(vp_r), int(vp_b))

        origin_gpx = nw_tx * 256
        origin_gpy = nw_ty * 256
        n_tx = se_tx - nw_tx + 1
        n_ty = se_ty - nw_ty + 1
        canvas_w = n_tx * 256
        canvas_h = n_ty * 256
        canvas = np.full((canvas_h, canvas_w, 3), 200, dtype=np.uint8)

        for ty in range(nw_ty, se_ty + 1):
            for tx in range(nw_tx, se_tx + 1):
                tile_path = (
                    self.map_dir
                    / f"tile_{self.provider}_{self.level}_{tx}_{ty}.jpg"
                )
                if not tile_path.exists():
                    continue
                img = cv2.imread(str(tile_path))
                if img is None:
                    continue
                dy = (ty - nw_ty) * 256
                dx = (tx - nw_tx) * 256
                th, tw = img.shape[:2]
                canvas[dy : dy + th, dx : dx + tw] = img

        crop_x = max(0, int(vp_l - origin_gpx))
        crop_y = max(0, int(vp_t - origin_gpy))
        crop_w = max(1, min(int(vp_r - vp_l), canvas_w - crop_x))
        crop_h = max(1, min(int(vp_b - vp_t), canvas_h - crop_y))

        cropped = canvas[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        panel = cv2.resize(cropped, (pw, ph), interpolation=cv2.INTER_AREA)

        abs_cx = float(origin_gpx + crop_x)
        abs_cy = float(origin_gpy + crop_y)
        cw_f = float(crop_w)
        ch_f = float(crop_h)
        pw_f = float(pw)
        ph_f = float(ph)

        def global_to_panel(gpx: float, gpy: float) -> Tuple[int, int]:
            """Converts Mercator global-pixel to panel-pixel coords."""
            return (
                int((gpx - abs_cx) / cw_f * pw_f),
                int((gpy - abs_cy) / ch_f * ph_f),
            )

        return panel, global_to_panel

    def _draw_search_circle(
        self,
        panel: np.ndarray,
        g2p: Callable,
        fd: FrameData,
    ) -> None:
        """Draws the adaptive search radius as a translucent circle.

        Args:
            panel: Satellite panel image to draw on (mutated).
            g2p: Coordinate mapping from
                :meth:`_render_satellite_panel`.
            fd: Current frame data.

        """
        sc_gpx, sc_gpy = TileSystem.latlong_to_pixel_xy(
            fd.search_center_lat, fd.search_center_lon, self.level,
        )
        center = g2p(sc_gpx, sc_gpy)

        gres = TileSystem.ground_resolution(fd.search_center_lat, self.level)
        if gres <= 0:
            return
        radius_global_px = fd.search_radius_m / gres

        cam_gres = TileSystem.ground_resolution(
            float(self.cam_lat), self.level,
        )
        vp_h_gpx = (
            2.0 * float(self.cam_radius) / cam_gres
            if cam_gres > 0
            else 1.0
        )
        scale = self.FRAME_H / vp_h_gpx if vp_h_gpx > 0 else 1.0
        r_px = int(radius_global_px * scale)

        if r_px < 3:
            return

        overlay = panel.copy()
        cv2.circle(overlay, center, r_px, self.COL_SEARCH, -1)
        cv2.addWeighted(overlay, 0.07, panel, 0.93, 0, panel)
        cv2.circle(panel, center, r_px, self.COL_SEARCH, 1, cv2.LINE_AA)

    def _draw_homography_polygon(
        self,
        panel: np.ndarray,
        g2p: Callable,
        fd: FrameData,
    ) -> None:
        """Projects the drone image footprint onto the satellite panel.

        Uses the homography matrix to transform the four query-image
        corners into the matched tile's pixel space, converts to
        global Mercator pixels via the tile's georeferencing metadata,
        and finally maps to panel pixels.

        Args:
            panel: Satellite panel image to draw on (mutated).
            g2p: Coordinate mapping from
                :meth:`_render_satellite_panel`.
            fd: Current frame data (must have ``homography``,
                ``effective_map_metadata``, and ``query_shape``).

        """
        homography = fd.homography
        meta = fd.effective_map_metadata
        q_shape = fd.query_shape
        if homography is None or meta is None or q_shape is None:
            return

        h_q, w_q = q_shape[:2]
        corners = np.array(
            [[0, 0], [w_q, 0], [w_q, h_q], [0, h_q]], dtype=np.float32,
        ).reshape(-1, 1, 2)

        try:
            corners_map = cv2.perspectiveTransform(corners, homography)
        except cv2.error:
            return
        if corners_map is None:
            return

        tl_lat = float(meta["Top_left_lat"])
        tl_lon = float(meta["Top_left_lon"])
        nw_gpx, nw_gpy = TileSystem.latlong_to_pixel_xy(
            tl_lat, tl_lon, self.level,
        )

        poly_pts = []
        for pt in corners_map.reshape(-1, 2):
            px, py = g2p(nw_gpx + pt[0], nw_gpy + pt[1])
            poly_pts.append([px, py])
        poly_pts = np.array(poly_pts, dtype=np.int32)

        limit = max(self.RIGHT_W, self.FRAME_H) * 3
        if np.any(np.abs(poly_pts) > limit):
            return

        overlay = panel.copy()
        cv2.fillPoly(overlay, [poly_pts], self.COL_POLYGON_FILL)
        cv2.addWeighted(overlay, 0.18, panel, 0.82, 0, panel)
        cv2.polylines(
            panel, [poly_pts], True, self.COL_POLYGON, 2, cv2.LINE_AA,
        )

        cq = np.array([[[w_q / 2.0, h_q / 2.0]]], dtype=np.float32)
        try:
            cm = cv2.perspectiveTransform(cq, homography)
        except cv2.error:
            return
        if cm is not None:
            pt = cm[0, 0]
            px, py = g2p(nw_gpx + pt[0], nw_gpy + pt[1])
            if abs(px) < limit and abs(py) < limit:
                cv2.circle(
                    panel, (px, py), 7, self.COL_POLYGON_CENTER, -1, cv2.LINE_AA,
                )

    def _render_trajectory_panel(
        self,
        all_results: List[QueryResult],
        fd: FrameData,
    ) -> np.ndarray:
        """Draws the info block and 2D trajectory mini-map.

        The top portion displays frame status, positioning error,
        inlier count, and search radius.  The remainder is a compact
        trajectory plot showing green dots for successful predictions
        (connected by lines) and red dots for failures.

        Args:
            all_results: Accumulated query results so far.
            fd: Current frame data.

        Returns:
            BGR image array of the info + trajectory panel.

        """
        pw, ph = self.LEFT_W, self.TOP_LEFT_H
        panel = np.full((ph, pw, 3), self.COL_INFO_BG, dtype=np.uint8)

        info_h = 70
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        grey = (160, 160, 160)

        cv2.rectangle(panel, (0, 0), (pw, info_h), (35, 35, 35), -1)

        tag = "MATCH" if fd.success else "FAIL"
        tag_col = (0, 220, 0) if fd.success else (0, 0, 255)
        cv2.putText(
            panel, f"#{fd.frame_idx:04d}", (10, 22),
            font, 0.50, white, 1, cv2.LINE_AA,
        )
        cv2.putText(
            panel, tag, (120, 22),
            font, 0.50, tag_col, 1, cv2.LINE_AA,
        )

        if fd.success and fd.error_meters is not None:
            cv2.putText(
                panel, f"Error: {fd.error_meters:.1f} m", (10, 44),
                font, 0.42, white, 1, cv2.LINE_AA,
            )
            cv2.putText(
                panel, f"Inliers: {fd.inliers}", (250, 44),
                font, 0.42, white, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                panel, "No match", (10, 44),
                font, 0.42, grey, 1, cv2.LINE_AA,
            )

        cv2.putText(
            panel, f"Radius: {fd.search_radius_m:.0f} m", (10, 64),
            font, 0.40, grey, 1, cv2.LINE_AA,
        )
        total = len(all_results)
        ok_n = sum(1 for r in all_results if r.success)
        cv2.putText(
            panel, f"Success: {ok_n}/{total}", (250, 64),
            font, 0.40, grey, 1, cv2.LINE_AA,
        )

        plot_y0 = info_h + 6
        plot_m = 12
        plot_w = pw - 2 * plot_m
        plot_h = ph - plot_y0 - plot_m

        cv2.rectangle(
            panel,
            (plot_m - 1, plot_y0 - 1),
            (plot_m + plot_w + 1, plot_y0 + plot_h + 1),
            (210, 210, 210), 1,
        )

        lat_rng = self._plot_max_lat - self._plot_min_lat
        lon_rng = self._plot_max_lon - self._plot_min_lon

        sx = plot_w / lon_rng if lon_rng > 0 else 1.0
        sy = plot_h / lat_rng if lat_rng > 0 else 1.0
        s = min(sx, sy)
        used_w = lon_rng * s
        used_h = lat_rng * s
        ox = plot_m + (plot_w - used_w) / 2.0
        oy = plot_y0 + (plot_h - used_h) / 2.0

        def ll2px(lat: float, lon: float) -> Tuple[int, int]:
            """Converts lat/lon to panel pixel coordinates."""
            return (
                int((lon - self._plot_min_lon) * s + ox),
                int((self._plot_max_lat - lat) * s + oy),
            )

        prev_pred_px: Optional[Tuple[int, int]] = None
        for r in all_results:
            if (
                r.success
                and r.predicted_latitude is not None
                and r.predicted_longitude is not None
            ):
                cur_px = ll2px(r.predicted_latitude, r.predicted_longitude)
                if prev_pred_px is not None:
                    cv2.line(
                        panel, prev_pred_px, cur_px,
                        self.COL_PRED_OK, 1, cv2.LINE_AA,
                    )
                cv2.circle(
                    panel, cur_px, 2, self.COL_PRED_OK, -1, cv2.LINE_AA,
                )
                prev_pred_px = cur_px
            elif r.gt_latitude is not None and r.gt_longitude is not None:
                cv2.circle(
                    panel,
                    ll2px(r.gt_latitude, r.gt_longitude),
                    2, self.COL_PRED_FAIL, -1, cv2.LINE_AA,
                )

        if fd.success and fd.predicted_lat is not None and fd.predicted_lon is not None:
            cv2.circle(
                panel, ll2px(fd.predicted_lat, fd.predicted_lon),
                4, self.COL_CURRENT, 1, cv2.LINE_AA,
            )
        elif fd.gt_lat is not None and fd.gt_lon is not None:
            cv2.circle(
                panel, ll2px(fd.gt_lat, fd.gt_lon),
                4, self.COL_CURRENT, 1, cv2.LINE_AA,
            )

        return panel

    def _render_query_panel(
        self,
        fd: FrameData,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Loads and letterboxes the preprocessed query image.

        Preserves the full query image by fitting it inside the panel while
        keeping aspect ratio. Returns the panel together with the scale and
        offset so :meth:`_draw_match_lines` can map
        keypoint coordinates back to composite-frame pixels.

        Args:
            fd: Current frame data.

        Returns:
            Tuple of ``(panel, scale, x_offset, y_offset)``.

        """
        pw, ph = self.LEFT_W, self.BOT_LEFT_H
        panel = np.full((ph, pw, 3), self.COL_PANEL_BG, dtype=np.uint8)

        q_path = fd.preprocessed_image_path or fd.query_image_path
        img = cv2.imread(q_path)
        if img is None:
            cv2.putText(
                panel, "Image not found", (10, ph // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
            )
            return panel, 1.0, 0, 0

        h, w = img.shape[:2]
        scale = min(pw / w, ph / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        xo = (pw - nw) // 2
        yo = (ph - nh) // 2
        panel[yo : yo + nh, xo : xo + nw] = resized

        return panel, scale, xo, yo

    def _draw_match_lines(
        self,
        frame: np.ndarray,
        fd: FrameData,
        q_scale: float,
        q_xo: int,
        q_yo: int,
        g2p: Callable,
    ) -> None:
        """Draws inlier match lines across the query and satellite panels.

        Each line connects a keypoint in the bottom-left query panel
        to its corresponding location on the right satellite panel.
        Lines are drawn directly on the final composite image so they
        can cross panel boundaries.

        When there are more inliers than ``MAX_DRAWN_MATCHES``, a
        uniformly spaced subset is drawn to keep the image readable.

        Args:
            frame: The full composite image (mutated in-place).
            fd: Current frame data with keypoints and metadata.
            q_scale: Scale factor applied when letterboxing the query.
            q_xo: Horizontal offset of the letterboxed query image.
            q_yo: Vertical offset of the letterboxed query image.
            g2p: Mercator-to-satellite-panel coordinate mapping.

        """
        if not fd.success:
            return
        if fd.mkpts0 is None or fd.mkpts1 is None:
            return
        if len(fd.mkpts0) == 0 or fd.effective_map_metadata is None:
            return

        meta = fd.effective_map_metadata
        tl_lat = float(meta["Top_left_lat"])
        tl_lon = float(meta["Top_left_lon"])
        nw_gpx, nw_gpy = TileSystem.latlong_to_pixel_xy(
            tl_lat, tl_lon, self.level,
        )

        mask = fd.inliers_mask
        inlier_indices = [
            i for i in range(len(fd.mkpts0))
            if mask is not None and i < len(mask) and bool(mask[i])
        ]
        if not inlier_indices:
            return

        if len(inlier_indices) > self.MAX_DRAWN_MATCHES:
            step = len(inlier_indices) / self.MAX_DRAWN_MATCHES
            inlier_indices = [
                inlier_indices[int(j * step)]
                for j in range(self.MAX_DRAWN_MATCHES)
            ]

        limit = max(self.FRAME_W, self.FRAME_H) * 2

        for i in inlier_indices:
            qx = int(fd.mkpts0[i][0] * q_scale) + q_xo
            qy = int(fd.mkpts0[i][1] * q_scale) + q_yo + self.TOP_LEFT_H

            if qx < 0 or qx >= self.LEFT_W:
                continue
            if qy < self.TOP_LEFT_H or qy >= self.FRAME_H:
                continue

            gpx = nw_gpx + fd.mkpts1[i][0]
            gpy = nw_gpy + fd.mkpts1[i][1]
            sx, sy = g2p(gpx, gpy)
            sx += self.LEFT_W

            if abs(sx) > limit or abs(sy) > limit:
                continue

            cv2.line(
                frame, (qx, qy), (sx, sy),
                self.COL_MATCH_LINE, 1, cv2.LINE_AA,
            )
            cv2.circle(
                frame, (qx, qy), 3, self.COL_MATCH_PT, -1, cv2.LINE_AA,
            )
            cv2.circle(
                frame, (sx, sy), 3, self.COL_MATCH_PT, -1, cv2.LINE_AA,
            )

    def _composite(
        self,
        traj_panel: np.ndarray,
        query_panel: np.ndarray,
        sat_panel: np.ndarray,
    ) -> np.ndarray:
        """Assembles the three panels into a single frame.

        Places the info + trajectory panel at top-left, the query
        image at bottom-left, and the satellite mosaic on the right.

        Args:
            traj_panel: Info + trajectory panel.
            query_panel: Preprocessed query image panel.
            sat_panel: Satellite mosaic panel.

        Returns:
            Full composite frame as a BGR image array.

        """
        frame = np.full(
            (self.FRAME_H, self.FRAME_W, 3),
            self.COL_PANEL_BG,
            dtype=np.uint8,
        )
        frame[0 : self.TOP_LEFT_H, 0 : self.LEFT_W] = traj_panel
        frame[self.TOP_LEFT_H : self.FRAME_H, 0 : self.LEFT_W] = query_panel
        frame[0 : self.FRAME_H, self.LEFT_W : self.FRAME_W] = sat_panel

        cv2.line(
            frame, (self.LEFT_W, 0), (self.LEFT_W, self.FRAME_H),
            (60, 60, 60), 5,
        )
        cv2.line(
            frame, (0, self.TOP_LEFT_H), (self.LEFT_W, self.TOP_LEFT_H),
            (60, 60, 60), 5,
        )
        return frame
