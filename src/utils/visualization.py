"""
Visualization utilities for feature match display.

This module provides functions for creating visual representations
of feature matches between image pairs.
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
    target_height: Optional[int] = 600
) -> bool:
    """
    Create and save a side-by-side visualization of feature matches.

    Images are resized to a common height while preserving aspect ratios.
    Inlier matches are drawn as lines connecting corresponding keypoints.

    Args:
        image0_path: Path to the first (query) image.
        image1_path: Path to the second (map) image.
        mkpts0: Matched keypoints in image 0 (N x 2) in original coordinates.
        mkpts1: Matched keypoints in image 1 (N x 2) in original coordinates.
        inliers_mask: Boolean mask indicating inliers (N,).
        output_path: Path to save the visualization image.
        title: Title for the visualization (unused, for API compatibility).
        line_color_inlier: BGR color tuple for inlier match lines.
        point_color_inlier: BGR color tuple for inlier keypoint circles.
        line_color_outlier: BGR color tuple for outlier matches.
        show_outliers: Whether to draw outlier matches.
        point_size: Radius of keypoint circles (0 to disable).
        line_thickness: Thickness of match lines.
        text_info: List of text strings (unused, for API compatibility).
        target_height: Target height for the output visualization.
            If None, uses the maximum input height.

    Returns:
        True if visualization was saved successfully, False otherwise.
    """
    try:
        # Load images
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

        # Determine target height
        h_target = target_height if target_height and target_height > 0 else max(h0, h1)

        # Calculate scaling factors
        scale0 = h_target / h0
        scale1 = h_target / h1

        w0_new = max(1, int(round(w0 * scale0)))
        w1_new = max(1, int(round(w1 * scale1)))

        # Resize images
        interpolation = cv2.INTER_AREA if (scale0 < 1.0 or scale1 < 1.0) else cv2.INTER_LINEAR
        img0_resized = cv2.resize(img0, (w0_new, h_target), interpolation=interpolation)
        img1_resized = cv2.resize(img1, (w1_new, h_target), interpolation=interpolation)

        # Scale keypoints
        mkpts0_scaled = mkpts0 * scale0
        mkpts1_scaled = mkpts1 * scale1

        # Create output canvas
        w_total = w0_new + w1_new
        canvas = 255 * np.ones((h_target, w_total, 3), dtype=np.uint8)
        canvas[:h_target, :w0_new] = img0_resized
        canvas[:h_target, w0_new:w_total] = img1_resized

        # Validate inliers mask
        if len(inliers_mask) != len(mkpts0):
            print(f"Warning: Inlier mask length ({len(inliers_mask)}) differs "
                  f"from keypoint count ({len(mkpts0)}).")
            inliers_mask = np.zeros(len(mkpts0), dtype=bool)
        else:
            inliers_mask = inliers_mask.astype(bool)

        # Convert to integer pixel coordinates
        pts0 = np.round(mkpts0_scaled).astype(int)
        pts1 = np.round(mkpts1_scaled).astype(int)

        # Draw outliers first (if enabled)
        if show_outliers and line_color_outlier is not None:
            outlier_mask = ~inliers_mask
            pts0_out = pts0[outlier_mask]
            pts1_out = pts1[outlier_mask]

            for i in range(len(pts0_out)):
                pt0 = tuple(pts0_out[i])
                pt1 = tuple(pts1_out[i] + np.array([w0_new, 0]))
                cv2.line(canvas, pt0, pt1, line_color_outlier, line_thickness, lineType=cv2.LINE_AA)

        # Draw inliers
        pts0_in = pts0[inliers_mask]
        pts1_in = pts1[inliers_mask]

        for i in range(len(pts0_in)):
            pt0 = tuple(pts0_in[i])
            pt1 = tuple(pts1_in[i] + np.array([w0_new, 0]))

            cv2.line(canvas, pt0, pt1, line_color_inlier, line_thickness, lineType=cv2.LINE_AA)

            if point_size > 0:
                cv2.circle(canvas, pt0, point_size, point_color_inlier, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, pt1, point_size, point_color_inlier, -1, lineType=cv2.LINE_AA)

        # Save output
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path_obj), canvas)
        if not success:
            print(f"Error: Failed to save visualization to {output_path_obj}")
            return False

        return True

    except FileNotFoundError as e:
        print(f"Error: Image file not found - {e}")
        return False
    except cv2.error as e:
        print(f"Error: OpenCV error during visualization - {e}")
        return False
    except Exception as e:
        print(f"Error: Visualization creation failed - {e}")
        import traceback
        traceback.print_exc()
        return False