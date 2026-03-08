"""Base classes for feature matching algorithms.

This module defines the abstract base class and result data structure
for all matching engines in the system. Contains shared functionality
to eliminate code duplication across matcher implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger

_logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()


@dataclass
class MatchResult:
    """Standardized matching result structure.

    Attributes:
        mkpts0: Matching keypoints in image 0 with shape (N, 2).
        mkpts1: Matching keypoints in image 1 with shape (N, 2).
        inliers: Inlier mask with shape (N,) indicating reliable matches.
        homography: Estimated 3x3 Homography matrix.
        time_taken: Total execution time in seconds.
        success: Whether a valid homography was found.
        mconf: Match confidence scores for each point pair.

    """

    mkpts0: np.ndarray
    mkpts1: np.ndarray
    inliers: np.ndarray
    homography: Optional[np.ndarray]
    time_taken: float
    success: bool
    mconf: Optional[np.ndarray] = None


class BaseMatcher(ABC):
    """Abstract base class for all feature matching engines.

    Provides shared functionality including RANSAC homography estimation,
    result creation, and match visualization. Subclasses must implement
    the `name` property and `match` method.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the matcher.
        device (str): Compute device (cuda or cpu).
        ransac_params (Dict[str, Any]): RANSAC algorithm parameters.
        cv2_method (int): OpenCV RANSAC method constant.
        ransac_method (str): Name of RANSAC method.
        ransac_thresh (float): RANSAC reprojection threshold.
        ransac_conf (float): RANSAC confidence level.
        ransac_iter (int): Maximum RANSAC iterations.

    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the matcher with common RANSAC settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing matcher and
                RANSAC parameters.

        """
        self.config = config
        requested_device = str(config["device"]).lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            _logger.info(
                "WARNING: CUDA requested but not available. Falling back to CPU."
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        self.ransac_params = config["ransac_params"]
        self.ransac_method = self.ransac_params["method"]
        self.ransac_thresh = self.ransac_params["reproj_threshold"]
        self.ransac_conf = self.ransac_params["confidence"]
        self.ransac_iter = self.ransac_params["max_iter"]

        if self.ransac_method == "USAC_MAGSAC":
            self.cv2_method = cv2.USAC_MAGSAC
        else:
            self.cv2_method = cv2.RANSAC

    @property
    @abstractmethod
    def name(self) -> str:
        """The identifying name of the matcher."""

    @abstractmethod
    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Matches features between two images.

        Args:
            image0_path (Union[str, Path]): Path to the first image (query).
            image1_path (Union[str, Path]): Path to the second image (map).

        Returns:
            Dict[str, Any]: Dictionary containing match results with keys:
                - mkpts0: Matched keypoints in image 0
                - mkpts1: Matched keypoints in image 1
                - inliers: Boolean inlier mask
                - homography: Estimated homography matrix
                - time: Processing time in seconds
                - success: Whether matching succeeded

        """

    def _create_empty_result(self) -> Dict[str, Any]:
        """Creates a default empty result dictionary for failed matches.

        Returns:
            Dict[str, Any]: Dictionary with empty arrays and failure status.

        """
        return {
            "mkpts0": np.array([]),
            "mkpts1": np.array([]),
            "inliers": np.array([]),
            "homography": None,
            "time": 0.0,
            "success": False,
            "mconf": np.array([]),
            "matched_features": 0,
        }

    def _set_feature_counts(
        self,
        results: Dict[str, Any],
        matched_features: int,
    ) -> None:
        """Stores normalized match counts in the result dict."""
        results["matched_features"] = max(0, int(matched_features))

    def _update_result_with_homography(
        self,
        results: Dict[str, Any],
        homography: Optional[np.ndarray],
        inlier_mask: np.ndarray,
    ) -> None:
        """Updates result dictionary with homography estimation results.

        Args:
            results (Dict[str, Any]): Result dictionary to update in-place.
            homography (Optional[np.ndarray]): Estimated homography matrix or None.
            inlier_mask (np.ndarray): Boolean mask of inlier matches.

        """
        if homography is not None:
            results["homography"] = homography
            results["inliers"] = inlier_mask
            results["success"] = True

    def estimate_homography(
        self, mkpts0: np.ndarray, mkpts1: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimates the homography matrix using RANSAC.

        Args:
            mkpts0 (np.ndarray): Keypoints in image 0 with shape (N, 2).
            mkpts1 (np.ndarray): Keypoints in image 1 with shape (N, 2).

        Returns:
            Tuple[Optional[np.ndarray], np.ndarray]: Tuple containing:
                - 3x3 homography matrix or None if estimation failed
                - Boolean inlier mask array

        """
        if len(mkpts0) < 4:
            return None, np.zeros(len(mkpts0), dtype=bool)

        try:
            homography, mask = cv2.findHomography(
                mkpts0,
                mkpts1,
                method=self.cv2_method,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=self.ransac_iter,
                confidence=self.ransac_conf,
            )

            if mask is None:
                mask = np.zeros(len(mkpts0), dtype=bool)
            else:
                mask = mask.ravel().astype(bool)

            return homography, mask

        except cv2.error:
            return None, np.zeros(len(mkpts0), dtype=bool)

    def visualize_matches(
        self,
        image0_path: Union[str, Path],
        image1_path: Union[str, Path],
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Union[str, Path],
        title: str = "Matches",
        homography: Optional[np.ndarray] = None,
    ) -> bool:
        """Creates and saves a visualization of feature matches.

        Args:
            image0_path (Union[str, Path]): Path to the query image.
            image1_path (Union[str, Path]): Path to the map image.
            mkpts0 (np.ndarray): Matched keypoints in query image with shape (N, 2).
            mkpts1 (np.ndarray): Matched keypoints in map image with shape (N, 2).
            inliers (np.ndarray): Boolean mask indicating inlier matches.
            output_path (Union[str, Path]): Destination path for the visualization image.
            title (str): Title for the visualization (unused, kept for API compatibility).
            homography (Optional[np.ndarray]): Optional homography matrix to overlay on visualization.

        Returns:
            bool: True if visualization was saved successfully, False otherwise.

        """
        try:
            from src.utils.visualization import create_match_visualization
        except ImportError:
            _logger.info("Visualization module unavailable.")
            return False

        num_inliers = int(np.sum(inliers)) if len(inliers) > 0 else 0
        num_total = len(mkpts0)
        text_info = [self.name, f"Matches: {num_inliers} / {num_total}"]

        try:
            return create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title=f"{self.name} Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600,
                homography=homography,
            )
        except Exception as e:
            _logger.info(f"ERROR during visualization: {e}")
            return False
