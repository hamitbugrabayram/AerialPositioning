"""
Abstract base class for feature matching pipelines.

This module defines the interface that all matcher pipelines must implement,
ensuring consistent behavior across different matching algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class MatchResult:
    """
    Container for feature matching results.

    Attributes:
        mkpts0: Matched keypoints in image 0 (N x 2).
        mkpts1: Matched keypoints in image 1 (N x 2).
        inliers: Boolean mask indicating inliers after RANSAC (N,).
        homography: 3x3 homography matrix (query -> map), or None.
        time: Execution time in seconds.
        success: Whether matching was successful.
        mconf: Match confidence scores (N,), optional.
    """

    mkpts0: np.ndarray = field(default_factory=lambda: np.array([]))
    mkpts1: np.ndarray = field(default_factory=lambda: np.array([]))
    inliers: np.ndarray = field(default_factory=lambda: np.array([]))
    homography: Optional[np.ndarray] = None
    time: float = 0.0
    success: bool = False
    mconf: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'mkpts0': self.mkpts0,
            'mkpts1': self.mkpts1,
            'inliers': self.inliers,
            'homography': self.homography,
            'time': self.time,
            'success': self.success,
            'mconf': self.mconf,
        }


class BaseMatcher(ABC):
    """
    Abstract base class for feature matching pipelines.

    All matcher implementations should inherit from this class and
    implement the required abstract methods.

    Attributes:
        config: Configuration dictionary.
        device: Computation device (cuda/cpu).
        ransac_thresh: RANSAC reprojection threshold in pixels.
        ransac_conf: RANSAC confidence level.
        ransac_max_iter: Maximum RANSAC iterations.
        ransac_method: OpenCV RANSAC method constant.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the base matcher.

        Args:
            config: Configuration dictionary containing device and RANSAC parameters.
        """
        self.config = config
        self.device = config.get('device', 'cpu')

        # RANSAC configuration
        ransac_params = config.get('ransac_params', {})
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)

        # Determine RANSAC method
        ransac_method_name = ransac_params.get('method', 'RANSAC')
        if ransac_method_name == 'USAC_MAGSAC' and hasattr(cv2, 'USAC_MAGSAC'):
            self.ransac_method = cv2.USAC_MAGSAC
        else:
            self.ransac_method = cv2.RANSAC

    @abstractmethod
    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
        """
        Match features between two images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match results with keys:
                - mkpts0: Matched keypoints in image 0
                - mkpts1: Matched keypoints in image 1
                - inliers: Boolean inlier mask
                - homography: 3x3 transformation matrix
                - time: Execution time
                - success: Whether matching succeeded
        """
        pass

    @abstractmethod
    def visualize_matches(
        self,
        image0_path: Path,
        image1_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Path
    ) -> None:
        """
        Save a visualization of the matches.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.
            mkpts0: Matched keypoints in image 0.
            mkpts1: Matched keypoints in image 1.
            inliers: Boolean inlier mask.
            output_path: Path to save the visualization.
        """
        pass

    def estimate_homography(
        self,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray
    ) -> tuple:
        """
        Estimate homography matrix using RANSAC.

        Args:
            mkpts0: Keypoints from image 0 (N x 2).
            mkpts1: Corresponding keypoints from image 1 (N x 2).

        Returns:
            Tuple of (homography_matrix, inlier_mask) or (None, None) on failure.
        """
        if len(mkpts0) < 4:
            return None, None

        try:
            H, inlier_mask = cv2.findHomography(
                mkpts0, mkpts1,
                method=self.ransac_method,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.ransac_conf,
                maxIters=self.ransac_max_iter
            )

            if H is None or inlier_mask is None:
                return None, None

            return H, inlier_mask.ravel().astype(bool)

        except cv2.error as e:
            print(f"Error during homography estimation: {e}")
            return None, None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the matcher name for display purposes."""
        pass

    def get_display_name(self) -> str:
        """Get formatted display name including variant information."""
        return self.name
