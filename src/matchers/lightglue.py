"""LightGlue feature matching pipeline.

This module implements the LightGlue matcher with SuperPoint or DISK
feature extraction for image matching and localization.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import BaseMatcher

_lightglue_path = Path(__file__).parent.parent.parent / "matchers/LightGlue"
if str(_lightglue_path) not in sys.path:
    sys.path.insert(0, str(_lightglue_path))

try:
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, rbd
except ImportError as e:
    raise ImportError(f"Failed to import LightGlue components: {e}")


class LightGluePipeline(BaseMatcher):
    """Feature matching pipeline using LightGlue.

    LightGlue is a lightweight and accurate feature matcher that works
    with various local feature descriptors (SuperPoint, DISK).

    Attributes:
        extractor: Feature extractor (SuperPoint or DISK).
        matcher: LightGlue matcher instance.
    """

    SUPPORTED_FEATURES = ("superpoint", "disk")

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the LightGlue pipeline.

        Args:
            config: Configuration dictionary with keys:
                - device: Computation device ('cuda' or 'cpu')
                - matcher_weights: Dict with 'lightglue_features' key
                - matcher_params.lightglue: Dict with extractor settings
                - ransac_params: Dict with RANSAC settings
        """
        super().__init__(config)
        self._device = torch.device(self.device)

        weights_config = config.get("matcher_weights", {})
        matcher_params = config.get("matcher_params", {}).get("lightglue", {})

        self._feature_type = weights_config.get("lightglue_features", "superpoint")
        max_keypoints = matcher_params.get("extractor_max_keypoints", 2048)

        if self._feature_type not in self.SUPPORTED_FEATURES:
            raise ValueError(
                f"Unsupported feature type: {self._feature_type}. "
                f"Supported: {self.SUPPORTED_FEATURES}"
            )

        print(f"Initializing LightGlue with {self._feature_type} features...")

        if self._feature_type == "superpoint":
            self.extractor = (
                SuperPoint(max_num_keypoints=max_keypoints).eval().to(self._device)
            )
        else:
            self.extractor = (
                DISK(max_num_keypoints=max_keypoints).eval().to(self._device)
            )

        self.matcher = LightGlue(features=self._feature_type).eval().to(self._device)

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"LightGlue ({self._feature_type})"

    def _preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess an image for LightGlue.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed image tensor, or None on error.
        """
        try:
            image = load_image(image_path)
            return image.to(self._device)
        except Exception as e:
            print(f"Error loading image {image_path.name}: {e}")
            return None

    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
        """Match features between two images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match results.
        """
        start_time = time.time()
        results = self._create_empty_result()

        try:
            image0 = self._preprocess_image(image0_path)
            image1 = self._preprocess_image(image1_path)

            if image0 is None or image1 is None:
                results["time"] = time.time() - start_time
                return results

            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)

            matches01 = self.matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
            matches = matches01["matches"]

            mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
            mkpts1 = kpts1[matches[..., 1]].cpu().numpy()

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1

            H, inlier_mask = self.estimate_homography(mkpts0, mkpts1)

            if H is not None:
                results["homography"] = H
                results["inliers"] = inlier_mask
                results["success"] = True

        except Exception as e:
            print(f"ERROR during LightGlue matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results

    def visualize_matches(
        self,
        image0_path: Path,
        image1_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Path,
    ) -> bool:
        """Save a visualization of the feature matches.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference image.
            mkpts0: Matched keypoints in image 0.
            mkpts1: Matched keypoints in image 1.
            inliers: Boolean inlier mask.
            output_path: Path to save the visualization.

        Returns:
            True if visualization was saved successfully.
        """
        try:
            from src.utils.visualization import create_match_visualization
        except ImportError:
            print("Visualization module unavailable.")
            return False

        num_inliers = np.sum(inliers) if len(inliers) > 0 else 0
        num_total = len(mkpts0)

        text_info = [
            self.name,
            f"Matches: {num_inliers} / {num_total}",
        ]

        try:
            return create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title="LightGlue Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600,
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            return False
