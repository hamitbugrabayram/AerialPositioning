"""SuperGlue feature matching pipeline.

This module implements the SuperGlue matcher using SuperPoint features.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .base import BaseMatcher
from .lightglue import LightGluePipeline


class SuperGluePipeline(LightGluePipeline):
    """Feature matching pipeline using SuperGlue.

    Inherits from LightGluePipeline but uses SuperGlue weights/config.
    Maintains compatibility with existing LightGlue/SuperPoint infrastructure.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the SuperGlue pipeline.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.name_override = "SuperGlue"

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return self.name_override

    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
        """Match features between two images using SuperGlue logic.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match results.
        """
        return super().match(image0_path, image1_path)

    def visualize_matches(
        self,
        image0_path: Path,
        image1_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Path,
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
            True if successful.
        """
        return super().visualize_matches(
            image0_path, image1_path, mkpts0, mkpts1, inliers, output_path
        )
