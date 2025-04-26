"""Visualization utilities for feature match display.

This module provides functions for creating visual representations
of feature matches between image pairs using OpenCV.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def create_match_visualization(
    image0_path: Union[str, Path],
    image1_path: Union[str, Path],
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inliers_mask: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Feature Matches",
    line_color_inlier: Tuple[int, int, int] = (0, 255, 0),
    point_color_inlier: Tuple[int, int, int] = (0, 255, 0),
    line_color_outlier: Optional[Tuple[int, int, int]] = (150, 150, 150),
    show_outliers: bool = False,
    point_size: int = 2,
    line_thickness: int = 1,
    text_info: Optional[List[str]] = None,
    target_height: Optional[int] = 600,
    homography: Optional[np.ndarray] = None,
) -> bool:
    """Create and save a side-by-side visualization of feature matches.

    Args:
        image0_path: Path to the first image (query).
        image1_path: Path to the second image (map).
        mkpts0: Keypoints in the first image (N, 2).
        mkpts1: Keypoints in the second image (N, 2).
        inliers_mask: Boolean mask of inliers (N,).
        output_path: Path to save the visualization.
        title: Title of the visualization (unused in current impl, but kept for API).
        line_color_inlier: RGB color for inlier lines.
        point_color_inlier: RGB color for inlier points.
        line_color_outlier: RGB color for outlier lines.
        show_outliers: Whether to draw outlier matches.
        point_size: Radius of keypoints.
        line_thickness: Thickness of lines.
        text_info: Optional list of text lines to add (unused).
        target_height: Height to resize images to for visualization.
        homography: Optional 3x3 homography matrix from image0 to image1.

    Returns:
        True if successful, False otherwise.
    """
    try:
        img0 = cv2.imread(str(image0_path), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)

        if img0 is None or img1 is None:
            print(f"Error: Could not read images: {image0_path}, {image1_path}")
            return False

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        if h0 <= 0 or w0 <= 0 or h1 <= 0 or w1 <= 0:
            print("Error: Invalid image dimensions for visualization.")
            return False

        h_target = target_height if target_height and target_height > 0 else max(h0, h1)

        scale0 = h_target / h0
        scale1 = h_target / h1

        w0_new = max(1, int(round(w0 * scale0)))
        w1_new = max(1, int(round(w1 * scale1)))

        img0_resized = cv2.resize(
            img0, (w0_new, h_target), interpolation=cv2.INTER_LINEAR
        )
        img1_resized = cv2.resize(
            img1, (w1_new, h_target), interpolation=cv2.INTER_LINEAR
        )

        mkpts0_scaled = mkpts0 * scale0
        mkpts1_scaled = mkpts1 * scale1

        w_total = w0_new + w1_new
        canvas = 255 * np.ones((h_target, w_total, 3), dtype=np.uint8)
        canvas[:h_target, :w0_new] = img0_resized
        canvas[:h_target, w0_new:w_total] = img1_resized

        # Save clean image (without lines) for inspection
        clean_output_path = Path(str(output_path).replace(".png", "_clean.png"))
        cv2.imwrite(str(clean_output_path), canvas)

        # Draw homography polygon if available
        if homography is not None:
            # Image 0 corners
            corners_0 = np.array(
                [[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float32
            ).reshape(-1, 1, 2)

            # Transform to Image 1 coordinates
            corners_1 = cv2.perspectiveTransform(corners_0, homography)

            if corners_1 is not None:
                # Scale to visualization size
                corners_1_scaled = corners_1 * scale1

                # Shift to right side of canvas
                corners_1_vis = corners_1_scaled + np.array([w0_new, 0])

                # Draw polygon
                cv2.polylines(
                    canvas, [np.int32(corners_1_vis)], True, (255, 0, 0), 2, cv2.LINE_AA
                )

                # Draw center point of projected drone image
                # Calculate center of the projected polygon
                # It is better to project the center point directly

                query_h, query_w = h0, w0
                query_center = np.array(
                    [[[query_w / 2.0, query_h / 2.0]]], dtype=np.float32
                )

                projected_center = cv2.perspectiveTransform(query_center, homography)

                if projected_center is not None:
                    center_pt = projected_center[0, 0]
                    # Scale to visualization
                    center_pt_scaled = center_pt * scale1
                    # Shift
                    center_pt_vis = center_pt_scaled + np.array([w0_new, 0])

                    center_pt_vis_int = tuple(np.round(center_pt_vis).astype(int))

                    # Draw crosshair
                    # Blue color (BGR)
                    color = (255, 0, 0)
                    size = 10
                    cv2.line(
                        canvas,
                        (center_pt_vis_int[0] - size, center_pt_vis_int[1]),
                        (center_pt_vis_int[0] + size, center_pt_vis_int[1]),
                        color,
                        2,
                    )
                    cv2.line(
                        canvas,
                        (center_pt_vis_int[0], center_pt_vis_int[1] - size),
                        (center_pt_vis_int[0], center_pt_vis_int[1] + size),
                        color,
                        2,
                    )
                    cv2.circle(canvas, center_pt_vis_int, 3, color, -1)

        if len(inliers_mask) != len(mkpts0):
            inliers_mask = np.zeros(len(mkpts0), dtype=bool)
        else:
            inliers_mask = inliers_mask.astype(bool)

        pts0 = np.round(mkpts0_scaled).astype(int)
        pts1 = np.round(mkpts1_scaled).astype(int)

        if show_outliers and line_color_outlier is not None:
            outlier_mask = ~inliers_mask
            pts0_out = pts0[outlier_mask]
            pts1_out = pts1[outlier_mask]

            for i in range(len(pts0_out)):
                pt0 = tuple(pts0_out[i])
                pt1 = tuple(pts1_out[i] + np.array([w0_new, 0]))
                cv2.line(
                    canvas,
                    pt0,
                    pt1,
                    line_color_outlier,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )

        pts0_in = pts0[inliers_mask]
        pts1_in = pts1[inliers_mask]

        for i in range(len(pts0_in)):
            pt0 = tuple(pts0_in[i])
            pt1 = tuple(pts1_in[i] + np.array([w0_new, 0]))

            cv2.line(
                canvas,
                pt0,
                pt1,
                line_color_inlier,
                line_thickness,
                lineType=cv2.LINE_AA,
            )

            if point_size > 0:
                cv2.circle(
                    canvas,
                    pt0,
                    point_size,
                    point_color_inlier,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    canvas,
                    pt1,
                    point_size,
                    point_color_inlier,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path_obj), canvas)
        if not success:
            print(f"Error: Failed to save visualization to {output_path_obj}")
            return False

        return True

    except Exception as e:
        print(f"Error: Visualization creation failed - {e}")
        return False
