"""
SuperGlue feature matching pipeline.

This module implements the SuperGlue matcher with SuperPoint feature
extraction for image matching and localization.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# Add SuperGlue to path
_superglue_path = Path(__file__).parent.parent / 'matchers/SuperGluePretrainedNetwork'
if str(_superglue_path) not in sys.path:
    sys.path.append(str(_superglue_path))

try:
    from models.matching import Matching
    from models.utils import frame2tensor
except ImportError as e:
    print(f"ERROR: Failed to import SuperGlue components: {e}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("WARNING: Could not import visualization module")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False


class SuperGluePipeline:
    """
    Feature matching pipeline using SuperGlue.

    SuperGlue is a graph neural network-based feature matcher that uses
    attention mechanisms to match SuperPoint features.

    Attributes:
        config: Configuration dictionary.
        device: Torch device for computation.
        matching: SuperGlue matching model.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SuperGlue pipeline.

        Args:
            config: Configuration dictionary with keys:
                - device: Computation device ('cuda' or 'cpu')
                - matcher_weights: Dict with 'superglue_weights' key
                - matcher_params.superglue: Dict with matcher settings
                - ransac_params: Dict with RANSAC settings
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Get configuration parameters
        weights_config = config.get('matcher_weights', {})
        matcher_params = config.get('matcher_params', {}).get('superglue', {})
        ransac_params = config.get('ransac_params', {})

        # Build SuperGlue configuration
        superglue_config = {
            'superpoint': {
                'nms_radius': matcher_params.get('superpoint_nms_radius', 3),
                'keypoint_threshold': matcher_params.get('superpoint_keypoint_threshold', 0.005),
                'max_keypoints': matcher_params.get('superpoint_max_keypoints', 2048)
            },
            'superglue': {
                'weights': weights_config.get('superglue_weights', 'outdoor'),
                'sinkhorn_iterations': matcher_params.get('superglue_sinkhorn_iterations', 20),
                'match_threshold': matcher_params.get('superglue_match_threshold', 0.2),
            }
        }

        self._weights_type = weights_config.get('superglue_weights', 'outdoor')

        print(f"Initializing SuperGlue with '{self._weights_type}' weights...")
        self.matching = Matching(superglue_config).eval().to(self.device)

        # RANSAC parameters
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method = cv2.RANSAC

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"SuperGlue ({self._weights_type})"

    def _preprocess_image(
        self,
        image_path: Path
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        """
        Load and preprocess an image for SuperGlue.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (image_tensor, shape) or (None, None) on error.
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path.name}")

            image_tensor = frame2tensor(image, self.device)
            return image_tensor, image.shape[:2]

        except Exception as e:
            print(f"Error loading image {image_path.name}: {e}")
            return None, None

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
                - mconf: Match confidence scores (N,)
        """
        start_time = time.time()

        results = {
            'mkpts0': np.array([]),
            'mkpts1': np.array([]),
            'inliers': np.array([]),
            'homography': None,
            'time': 0.0,
            'success': False,
            'mconf': np.array([])
        }

        try:
            # Load and preprocess images
            image0, shape0 = self._preprocess_image(image0_path)
            image1, shape1 = self._preprocess_image(image1_path)

            if image0 is None or image1 is None:
                results['time'] = time.time() - start_time
                return results

            # Run matching
            with torch.no_grad():
                pred = self.matching({'image0': image0, 'image1': image1})

            # Convert to numpy
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

            # Extract matched keypoints
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1
            results['mconf'] = mconf

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
            print(f"ERROR during SuperGlue matching: {e}")
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
                title="SuperGlue Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")