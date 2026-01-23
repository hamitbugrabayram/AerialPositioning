"""Base classes for feature matching algorithms.

This module defines the abstract base class and result data structure
for all matching engines in the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

@dataclass
class MatchResult:
    """Standardized matching result structure.

    Attributes:
        mkpts0: Matching keypoints in image 0 (N, 2).
        mkpts1: Matching keypoints in image 1 (N, 2).
        inliers: Inlier mask (N,) indicating reliable matches.
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
    """Abstract base class for all feature matching engines."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the matcher with common RANSAC settings.

        Args:
            config: Configuration dictionary containing matcher and RANSAC parameters.
        """
        self.config = config
        self.device = config.get("device", "cuda")

        self.ransac_params = config.get("ransac_params", {})
        self.ransac_method = self.ransac_params.get("method", "RANSAC")
        self.ransac_thresh = self.ransac_params.get("reproj_threshold", 8.0)
        self.ransac_conf = self.ransac_params.get("confidence", 0.999)
        self.ransac_iter = self.ransac_params.get("max_iter", 10000)

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
            image0_path: Path to the first image (query).
            image1_path: Path to the second image (map).

        Returns:
            A dictionary containing match results.
        """

    @abstractmethod
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
        """Creates a visual representation of the matches.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the map image.
            mkpts0: Matched points in the query image.
            mkpts1: Matched points in the map image.
            inliers: Boolean mask of inlier matches.
            output_path: Destination path for the visualization image.
            title: Title for the visualization plot.
            homography: Optional homography matrix to overlay.

        Returns:
            True if visualization was successful, False otherwise.
        """

    def _create_empty_result(self) -> Dict[str, Any]:
        """Creates a default empty result dictionary for failed matches.

        Returns:
            A dictionary with empty arrays and failure status.
        """
        return {
            "mkpts0": np.array([]),
            "mkpts1": np.array([]),
            "inliers": np.array([]),
            "homography": None,
            "time": 0.0,
            "success": False,
            "mconf": np.array([]),
        }

    def estimate_homography(
        self, mkpts0: np.ndarray, mkpts1: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimates the homography matrix using RANSAC.

        Args:
            mkpts0: Keypoints in image 0 (N, 2).
            mkpts1: Keypoints in image 1 (N, 2).

        Returns:
            A tuple of (3x3 homography matrix or None, boolean inlier mask).
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
