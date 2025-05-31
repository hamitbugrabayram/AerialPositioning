import torch
import cv2
import numpy as np
import time
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union, List
from copy import deepcopy

# Add LoFTR submodule path
loftr_repo_path = Path(__file__).resolve().parent.parent / 'matchers/LoFTR'
if str(loftr_repo_path) not in sys.path:
    if loftr_repo_path.exists():
        sys.path.insert(0, str(loftr_repo_path))
        # Also add LoFTR's own src to path if necessary for its internal imports
        loftr_src_path = loftr_repo_path / 'src'
        if loftr_src_path.exists() and str(loftr_src_path) not in sys.path:
            sys.path.insert(0, str(loftr_src_path))
    else:
        print(f"ERROR: LoFTR directory not found at expected path: {loftr_repo_path}")
        sys.exit(1)

try:
    from loftr import LoFTR as LoFTRModel
    from loftr.utils.cvpr_ds_config import default_cfg as loftr_default_cfg
except ImportError as e:
    print(f"ERROR: Failed importing LoFTR components: {e}")
    print(f"Ensure LoFTR submodule is present in 'matchers/LoFTR' and its dependencies are installed.")
    print(f"Current sys.path includes: {sys.path}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("ERROR: Could not import create_match_visualization from utils.visualization")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False

class LoFTRPipeline:
    """Pipeline for feature matching using LoFTR."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        loftr_weights_config = config.get('matcher_weights', {})
        self.loftr_params = config.get('matcher_params', {}).get('loftr', {})
        ransac_params = config.get('ransac_params', {})

        weights_path = loftr_weights_config.get('loftr_weights_path')
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"LoFTR weights file missing or not specified: {weights_path}")

        # Initialize LoFTR model
        # Start with LoFTR's default config and allow overrides from main config.yaml
        model_config = deepcopy(loftr_default_cfg)

        # Override specific LoFTR config items if present in pipeline's config.yaml
        # Example: temp_bug_fix or coarse matching threshold
        if 'temp_bug_fix' in self.loftr_params:
            model_config['coarse']['temp_bug_fix'] = self.loftr_params['temp_bug_fix']
        if 'match_thr' in self.loftr_params:
            model_config['match_coarse']['thr'] = self.loftr_params['match_thr']
        # Add any other LoFTR specific config overrides here if needed

        print(f"Initializing LoFTR with weights: {weights_path}")
        self.model = LoFTRModel(config=model_config)
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            # LoFTR checkpoints might be nested under 'state_dict' (from PyTorch Lightning)
            state_dict = ckpt.get('state_dict', ckpt)
            # LoFTR's LoFTRModel.load_state_dict handles stripping 'matcher.' prefix
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading LoFTR checkpoint from {weights_path}: {e}")
            raise
        
        self.model = self.model.eval().to(self.device)
        print("LoFTR model initialized successfully.")

        # RANSAC params
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method_str = ransac_params.get('method', 'RANSAC')
        if self.ransac_method_str == 'USAC_MAGSAC' and hasattr(cv2, 'USAC_MAGSAC'):
            self.ransac_method = cv2.USAC_MAGSAC
            print("Using RANSAC Method: USAC_MAGSAC for LoFTR")
        else:
            self.ransac_method = cv2.RANSAC
            if self.ransac_method_str != 'RANSAC':
                 print(f"Warning: RANSAC method '{self.ransac_method_str}' not found for LoFTR, using default RANSAC.")


    def _preprocess_image(self, image_path: Path) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Reads, converts to grayscale, resizes (ensuring divisibility by 8),
        and converts an image to a tensor for LoFTR.

        Returns:
            Tuple of (tensor, scale_factor_wh, original_wh) or (None, None, None) on error.
            scale_factor_wh: [scale_w, scale_h] to map processed coords back to original.
        """
        try:
            img_bgr_orig = cv2.imread(str(image_path))
            if img_bgr_orig is None:
                raise ValueError(f"Could not read image: {image_path.name}")

            h_orig, w_orig = img_bgr_orig.shape[:2]
            img_gray_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2GRAY)

            resize_opt = self.loftr_params.get('resize')
            target_w, target_h = w_orig, h_orig

            if isinstance(resize_opt, (list, tuple)):
                if len(resize_opt) == 1 and resize_opt[0] > 0: # Max dimension
                    scale = resize_opt[0] / max(w_orig, h_orig)
                    target_w, target_h = int(round(w_orig * scale)), int(round(h_orig * scale))
                elif len(resize_opt) == 2: # Exact [W, H]
                    target_w, target_h = int(resize_opt[0]), int(resize_opt[1])
            elif isinstance(resize_opt, int) and resize_opt > 0: # Max dimension (if single int)
                 scale = resize_opt / max(w_orig, h_orig)
                 target_w, target_h = int(round(w_orig * scale)), int(round(h_orig * scale))


            # Ensure dimensions are divisible by 8 (LoFTR requirement)
            w_resized = (target_w // 8) * 8
            h_resized = (target_h // 8) * 8

            if w_resized == 0 or h_resized == 0:
                print(f"Warning: Resize for LoFTR resulted in zero dimension ({w_resized}x{h_resized}) for {image_path.name}. Using original size adjusted for divisibility by 8.")
                w_resized = (w_orig // 8) * 8
                h_resized = (h_orig // 8) * 8
                if w_resized == 0 or h_resized == 0: # Still zero, e.g. very small original image
                    raise ValueError(f"Cannot resize image {image_path.name} to be divisible by 8, dimensions too small.")
            
            img_gray_resized = cv2.resize(img_gray_orig, (w_resized, h_resized), interpolation=cv2.INTER_AREA if (w_resized*h_resized < w_orig*h_orig) else cv2.INTER_LINEAR)
            
            scale_to_original_w = w_orig / w_resized if w_resized > 0 else 1.0
            scale_to_original_h = h_orig / h_resized if h_resized > 0 else 1.0
            
            img_tensor = torch.from_numpy(img_gray_resized).float()[None, None] / 255.0 # Batch, Channel, H, W

            return img_tensor.to(self.device), np.array([scale_to_original_w, scale_to_original_h]), (w_orig, h_orig)

        except Exception as e:
            print(f"Error preprocessing image {image_path.name} for LoFTR: {e}")
            return None, None, None

    def match(self, image0_path: Path, image1_path: Path) -> Dict:
        start_time = time.time()
        results = {
            'mkpts0': np.array([]), 'mkpts1': np.array([]), 'inliers': np.array([]),
            'homography': None, 'time': 0.0, 'success': False, 'mconf': np.array([])
        }
        try:
            img0_tensor, scale0_wh, orig_size0_wh = self._preprocess_image(image0_path)
            img1_tensor, scale1_wh, orig_size1_wh = self._preprocess_image(image1_path)

            if img0_tensor is None or img1_tensor is None:
                results['time'] = time.time() - start_time
                return results

            batch = {'image0': img0_tensor, 'image1': img1_tensor}
            
            with torch.no_grad():
                self.model(batch) # LoFTR updates batch_data in-place

            # mkpts are in the coordinate system of the (potentially resized) input images to LoFTR
            mkpts0_loftr = batch['mkpts0_f'].cpu().numpy()
            mkpts1_loftr = batch['mkpts1_f'].cpu().numpy()
            mconf_loftr = batch['mconf'].cpu().numpy()

            # Scale keypoints back to original image dimensions
            # scale_wh is [scale_w, scale_h]
            # mkpts are [x, y]
            mkpts0_orig = mkpts0_loftr * scale0_wh 
            mkpts1_orig = mkpts1_loftr * scale1_wh

            results['mkpts0'] = mkpts0_orig
            results['mkpts1'] = mkpts1_orig
            results['mconf'] = mconf_loftr

            if len(mkpts0_orig) < 4:
                print(f"  Warning: LoFTR found only {len(mkpts0_orig)} initial matches. Skipping RANSAC.")
                results['time'] = time.time() - start_time
                return results
            
            H, inlier_mask = cv2.findHomography(
                mkpts0_orig, mkpts1_orig,
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
                print(f"  Warning: RANSAC failed to find Homography for LoFTR.")


        except Exception as e:
            print(f"ERROR during LoFTR matching ({image0_path.name} vs {image1_path.name}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            results['time'] = time.time() - start_time
            return results

    def visualize_matches(self, image0_path, image1_path, mkpts0, mkpts1, inliers, output_path):
        """Saves a visualization of the matches using the standardized function."""
        num_inliers = np.sum(inliers) if inliers is not None else 0
        num_total = len(mkpts0)
        text = [
            f'LoFTR',
            f'Matches: {num_inliers} / {num_total}'
        ]
        try:
            create_match_visualization(
                image0_path=image0_path, image1_path=image1_path, mkpts0=mkpts0, mkpts1=mkpts1,
                inliers_mask=inliers, output_path=output_path, title="LoFTR Matches", text_info=text,
                show_outliers=False, target_height=600)
        except Exception as e:
            print(f"ERROR during LoFTR visualization call: {e}")