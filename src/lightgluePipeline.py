"""
LightGlue feature matching pipeline.

This module implements the LightGlue matcher with SuperPoint or DISK
feature extraction for image matching and localization.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

# Add LightGlue to path
_lightglue_path = Path(__file__).parent.parent / 'matchers/LightGlue'
if str(_lightglue_path) not in sys.path:
    sys.path.append(str(_lightglue_path))

try:
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, rbd
except ImportError as e:
    print(f"ERROR: Failed to import LightGlue components: {e}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("WARNING: Could not import visualization module")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False


class LightGluePipeline:
    """
    Feature matching pipeline using LightGlue.

    LightGlue is a lightweight and accurate feature matcher that works
    with various local feature descriptors (SuperPoint, DISK).

    Attributes:
        config: Configuration dictionary.
        device: Torch device for computation.
        extractor: Feature extractor (SuperPoint or DISK).
        matcher: LightGlue matcher instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LightGlue pipeline.

        Args:
            config: Configuration dictionary with keys:
                - device: Computation device ('cuda' or 'cpu')
                - matcher_weights: Dict with 'lightglue_features' key
                - matcher_params.lightglue: Dict with extractor settings
                - ransac_params: Dict with RANSAC settings
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Get configuration parameters
        weights_config = config.get('matcher_weights', {})
        matcher_params = config.get('matcher_params', {}).get('lightglue', {})
        ransac_params = config.get('ransac_params', {})

        # Feature extractor configuration
        feature_type = weights_config.get('lightglue_features', 'superpoint')
        max_keypoints = matcher_params.get('extractor_max_keypoints', 2048)

        print(f"Initializing LightGlue with {feature_type} features...")

        # Initialize feature extractor
        if feature_type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        elif feature_type == 'disk':
            self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self.device)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        self._feature_type = feature_type

        # Initialize matcher
        self.matcher = LightGlue(features=feature_type).eval().to(self.device)

        # RANSAC parameters
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method = cv2.RANSAC

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"LightGlue ({self._feature_type})"

    def _preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """
        Load and preprocess an image for LightGlue.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed image tensor, or None on error.
        """
        try:
            image = load_image(image_path)
            return image.to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path.name}: {e}")
            return None

    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
        """
        Match features between two images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing:
                - mkpts0, mkpts1: Matched keypoint arrays (N x 2)
                - inliers: Boolean inlier mask (N,)
                - homography: 3x3 transformation matrix or None
                - time: Execution time in seconds
                - success: Whether matching succeeded
        """
        start_time = time.time()

        results = {
            'mkpts0': np.array([]),
            'mkpts1': np.array([]),
            'inliers': np.array([]),
            'homography': None,
            'time': 0.0,
            'success': False
        }

        try:
            # Load and preprocess images
            image0 = self._preprocess_image(image0_path)
            image1 = self._preprocess_image(image1_path)

            if image0 is None or image1 is None:
                results['time'] = time.time() - start_time
                return results

            # Extract features
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)

            # Match features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})

            # Remove batch dimension
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # Get matched keypoints
            kpts0, kpts1 = feats0['keypoints'], feats1['keypoints']
            matches = matches01['matches']

            mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
            mkpts1 = kpts1[matches[..., 1]].cpu().numpy()

            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1

            # Check minimum matches for RANSAC
            if len(mkpts0) < 4:
                print(f"  Warning: Only {len(mkpts0)} matches found. Skipping RANSAC.")
                results['time'] = time.time() - start_time
                return results

            # Estimate homography with RANSAC
            H, inlier_mask = cv2.findHomography(
                mkpts0, mkpts1,
                method=self.ransac_method,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.ransac_conf,
                maxIters=self.ransac_max_iter
            )

            if H is None or inlier_mask is None:
                print("  Warning: RANSAC failed to find homography.")
                results['time'] = time.time() - start_time
                return results

            results['homography'] = H
            results['inliers'] = inlier_mask.ravel().astype(bool)
            results['success'] = True

        except Exception as e:
            print(f"ERROR during LightGlue matching: {e}")
            import traceback
            traceback.print_exc()

        finally:
            results['time'] = time.time() - start_time

        return results

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
        Save a visualization of the feature matches.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference image.
            mkpts0: Matched keypoints in image 0.
            mkpts1: Matched keypoints in image 1.
            inliers: Boolean inlier mask.
            output_path: Path to save the visualization.
        """
        num_inliers = np.sum(inliers)
        num_total = len(mkpts0)

        text_info = [
            self.name,
            f'Matches: {num_inliers} / {num_total}',
        ]

        try:
            create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title="LightGlue Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")