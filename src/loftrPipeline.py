"""
LoFTR feature matching pipeline.

This module implements the LoFTR (Detector-Free Local Feature Matching)
matcher for dense feature matching between image pairs.
"""

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# Add LoFTR to path
_loftr_path = Path(__file__).resolve().parent.parent / 'matchers/LoFTR'
if str(_loftr_path) not in sys.path:
    if _loftr_path.exists():
        sys.path.insert(0, str(_loftr_path))
        _loftr_src_path = _loftr_path / 'src'
        if _loftr_src_path.exists() and str(_loftr_src_path) not in sys.path:
            sys.path.insert(0, str(_loftr_src_path))
    else:
        print(f"ERROR: LoFTR directory not found: {_loftr_path}")
        sys.exit(1)

try:
    from loftr import LoFTR as LoFTRModel
    from loftr.utils.cvpr_ds_config import default_cfg as loftr_default_cfg
except ImportError as e:
    print(f"ERROR: Failed to import LoFTR components: {e}")
    print("Ensure LoFTR submodule is present in 'matchers/LoFTR'")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("WARNING: Could not import visualization module")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False


class LoFTRPipeline:
    """
    Feature matching pipeline using LoFTR.

    LoFTR (Detector-Free Local Feature Matching with Transformers) is a
    dense matching method that doesn't require explicit keypoint detection.

    Attributes:
        config: Configuration dictionary.
        device: Torch device for computation.
        model: LoFTR model instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LoFTR pipeline.

        Args:
            config: Configuration dictionary with keys:
                - device: Computation device ('cuda' or 'cpu')
                - matcher_weights: Dict with 'loftr_weights_path' key
                - matcher_params.loftr: Dict with model settings
                - ransac_params: Dict with RANSAC settings
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Get configuration parameters
        weights_config = config.get('matcher_weights', {})
        self.loftr_params = config.get('matcher_params', {}).get('loftr', {})
        ransac_params = config.get('ransac_params', {})

        # Validate weights path
        weights_path = weights_config.get('loftr_weights_path')
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"LoFTR weights not found: {weights_path}")

        self._weights_name = Path(weights_path).stem

        # Build model configuration
        model_config = deepcopy(loftr_default_cfg)

        if 'temp_bug_fix' in self.loftr_params:
            model_config['coarse']['temp_bug_fix'] = self.loftr_params['temp_bug_fix']
        if 'match_thr' in self.loftr_params:
            model_config['match_coarse']['thr'] = self.loftr_params['match_thr']

        print(f"Initializing LoFTR with weights: {weights_path}")

        # Initialize model
        self.model = LoFTRModel(config=model_config)

        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading LoFTR checkpoint: {e}")
            raise

        self.model = self.model.eval().to(self.device)
        print("LoFTR model initialized successfully.")

        # RANSAC parameters
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)

        ransac_method_name = ransac_params.get('method', 'RANSAC')
        if ransac_method_name == 'USAC_MAGSAC' and hasattr(cv2, 'USAC_MAGSAC'):
            self.ransac_method = cv2.USAC_MAGSAC
            print("Using RANSAC method: USAC_MAGSAC")
        else:
            self.ransac_method = cv2.RANSAC

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"LoFTR ({self._weights_name})"

    def _preprocess_image(
        self,
        image_path: Path
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Load and preprocess an image for LoFTR.

        Converts to grayscale, resizes ensuring divisibility by 8,
        and creates the input tensor.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (tensor, scale_factors, original_size) or (None, None, None).
        """
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image_path.name}")

            h_orig, w_orig = image_bgr.shape[:2]
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # Calculate target dimensions
            resize_opt = self.loftr_params.get('resize')
            target_w, target_h = w_orig, h_orig

            if isinstance(resize_opt, (list, tuple)):
                if len(resize_opt) == 1 and resize_opt[0] > 0:
                    scale = resize_opt[0] / max(w_orig, h_orig)
                    target_w = int(round(w_orig * scale))
                    target_h = int(round(h_orig * scale))
                elif len(resize_opt) == 2:
                    target_w, target_h = int(resize_opt[0]), int(resize_opt[1])
            elif isinstance(resize_opt, int) and resize_opt > 0:
                scale = resize_opt / max(w_orig, h_orig)
                target_w = int(round(w_orig * scale))
                target_h = int(round(h_orig * scale))

            # Ensure dimensions are divisible by 8 (LoFTR requirement)
            w_resized = (target_w // 8) * 8
            h_resized = (target_h // 8) * 8

            if w_resized == 0 or h_resized == 0:
                print(f"Warning: Resize resulted in zero dimensions. Using original.")
                w_resized = (w_orig // 8) * 8
                h_resized = (h_orig // 8) * 8
                if w_resized == 0 or h_resized == 0:
                    raise ValueError(f"Image too small for LoFTR: {image_path.name}")

            # Resize image
            interpolation = (cv2.INTER_AREA if w_resized * h_resized < w_orig * h_orig
                           else cv2.INTER_LINEAR)
            image_resized = cv2.resize(image_gray, (w_resized, h_resized),
                                       interpolation=interpolation)

            # Calculate scale factors
            scale_w = w_orig / w_resized if w_resized > 0 else 1.0
            scale_h = h_orig / h_resized if h_resized > 0 else 1.0

            # Convert to tensor [B, C, H, W]
            image_tensor = torch.from_numpy(image_resized).float()[None, None] / 255.0

            return (
                image_tensor.to(self.device),
                np.array([scale_w, scale_h]),
                (w_orig, h_orig)
            )

        except Exception as e:
            print(f"Error preprocessing image {image_path.name}: {e}")
            return None, None, None

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
            # Preprocess images
            img0, scale0, orig_size0 = self._preprocess_image(image0_path)
            img1, scale1, orig_size1 = self._preprocess_image(image1_path)

            if img0 is None or img1 is None:
                results['time'] = time.time() - start_time
                return results

            # Run LoFTR matching
            batch = {'image0': img0, 'image1': img1}

            with torch.no_grad():
                self.model(batch)

            # Extract matches (in resized coordinates)
            mkpts0_loftr = batch['mkpts0_f'].cpu().numpy()
            mkpts1_loftr = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            # Scale back to original coordinates
            mkpts0 = mkpts0_loftr * scale0
            mkpts1 = mkpts1_loftr * scale1

            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1
            results['mconf'] = mconf

            # Check minimum matches
            if len(mkpts0) < 4:
                print(f"  Warning: Only {len(mkpts0)} matches found. Skipping RANSAC.")
                results['time'] = time.time() - start_time
                return results

            # Estimate homography
            H, inlier_mask = cv2.findHomography(
                mkpts0, mkpts1,
                method=self.ransac_method,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.ransac_conf,
                maxIters=self.ransac_max_iter
            )

            if H is not None and inlier_mask is not None:
                results['homography'] = H
                results['inliers'] = inlier_mask.ravel().astype(bool)
                results['success'] = True
            else:
                print("  Warning: RANSAC failed to find homography.")

        except Exception as e:
            print(f"ERROR during LoFTR matching: {e}")
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
        num_inliers = np.sum(inliers) if inliers is not None else 0
        num_total = len(mkpts0)

        text_info = [
            'LoFTR',
            f'Matches: {num_inliers} / {num_total}'
        ]

        try:
            create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title="LoFTR Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")