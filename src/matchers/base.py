"""Base matcher implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()


@dataclass
class MatchResult:
    """Standardized matching result.

    Attributes:
        mkpts0: Matching keypoints in image 0 (N, 2).
        mkpts1: Matching keypoints in image 1 (N, 2).
        inliers: Inlier mask (N,) or indices.
        homography: 3x3 Homography matrix.
        time_taken: Execution time in seconds.
        success: Whether matching was successful.
        mconf: Match confidence scores.
    """

    mkpts0: np.ndarray
    mkpts1: np.ndarray
    inliers: np.ndarray
    homography: Optional[np.ndarray]
    time_taken: float
    success: bool
    mconf: Optional[np.ndarray] = None


class BaseMatcher(ABC):
    """Abstract base class for all matchers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the matcher.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = config.get('device', 'cuda')

        self.ransac_params = config.get('ransac_params', {})
        self.ransac_method = self.ransac_params.get('method', 'RANSAC')
        self.ransac_thresh = self.ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = self.ransac_params.get('confidence', 0.999)
        self.ransac_iter = self.ransac_params.get('max_iter', 10000)

        if self.ransac_method == 'USAC_MAGSAC':
            self.cv2_method = cv2.USAC_MAGSAC
        else:
            self.cv2_method = cv2.RANSAC

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the matcher."""
        pass

    @abstractmethod
    def match(
        self,
        image0_path: Union[str, Path],
        image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Match two images.

        Args:
            image0_path: Path to the first image (query).
            image1_path: Path to the second image (map).

        Returns:
            Dictionary containing match results (mkpts0, mkpts1, inliers, etc.)
            or MatchResult.to_dict().
        """
        pass

    @abstractmethod
    def visualize_matches(
        self,
        image0_path: Union[str, Path],
        image1_path: Union[str, Path],
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Union[str, Path],
        title: str = "Matches"
    ) -> bool:
        """Visualize matches.

        Args:
            image0_path: Path to query image.
            image1_path: Path to map image.
            mkpts0: Keypoints in query image.
            mkpts1: Keypoints in map image.
            inliers: Inlier mask.
            output_path: Path to save visualization.
            title: Plot title.

        Returns:
            True if successful, False otherwise.
        """
        pass

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result dictionary.

        Returns:
            Dictionary with empty numpy arrays and None values.
        """
        return {
            'mkpts0': np.array([]),
            'mkpts1': np.array([]),
            'inliers': np.array([]),
            'homography': None,
            'time': 0.0,
            'success': False,
            'mconf': np.array([])
        }

    def estimate_homography(
        self,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimate homography matrix using OpenCV RANSAC.

        Args:
            mkpts0: Keypoints in image 0 (N, 2).
            mkpts1: Keypoints in image 1 (N, 2).

        Returns:
            Tuple of (Homography matrix 3x3, inlier mask).
        """
        if len(mkpts0) < 4:
            return None, np.zeros(len(mkpts0), dtype=bool)

        try:
            H, mask = cv2.findHomography(
                mkpts0, mkpts1,
                method=self.cv2_method,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=self.ransac_iter,
                confidence=self.ransac_conf
            )

            if mask is None:
                mask = np.zeros(len(mkpts0), dtype=bool)
            else:
                mask = mask.ravel().astype(bool)

            return H, mask

        except cv2.error:
            return None, np.zeros(len(mkpts0), dtype=bool)
