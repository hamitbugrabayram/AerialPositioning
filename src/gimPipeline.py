"""
GIM (Generalized Image Matching) feature matching pipeline.

This module implements the GIM framework which supports multiple matching
backends including DKM, RoMa, LoFTR, and LightGlue variants.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Add GIM to path
_gim_path = Path(__file__).resolve().parent.parent / 'matchers/gim'
if str(_gim_path) not in sys.path:
    if _gim_path.exists():
        sys.path.append(str(_gim_path))
    else:
        print(f"WARNING: GIM directory not found: {_gim_path}")
        _gim_path_alt = Path(__file__).resolve().parent.parent.parent / 'gim'
        if _gim_path_alt.exists() and str(_gim_path_alt) not in sys.path:
            sys.path.append(str(_gim_path_alt))

try:
    from tools import get_padding_size
    from networks.roma.roma import RoMa
    from networks.loftr.loftr import LoFTR
    from networks.loftr.misc import lower_config
    from networks.loftr.config import get_cfg_defaults
    from networks.dkm.models.model_zoo.DKMv3 import DKMv3
    from networks.lightglue.superpoint import SuperPoint
    from networks.lightglue.models.matchers.lightglue import LightGlue
except ImportError as e:
    print(f"ERROR: Failed to import GIM components: {e}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("WARNING: Could not import visualization module")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False


def preprocess_for_gim(
    image: np.ndarray,
    grayscale: bool = False,
    resize_max: Optional[int] = None,
    dfactor: int = 8
) -> Optional[Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]]:
    """
    Preprocess an image for GIM models.

    Handles resizing, color conversion, normalization, and ensures
    dimensions are divisible by dfactor.

    Args:
        image: Input image as numpy array (HxW or HxWxC).
        grayscale: If True, convert to grayscale.
        resize_max: Maximum dimension for resizing (None to skip).
        dfactor: Ensure dimensions are divisible by this factor.

    Returns:
        Tuple of (tensor, scale_to_original, original_size_wh) or
        (None, None, None) on error.
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        image = image.astype(np.float32, copy=False)

        if image.ndim < 2 or image.ndim > 3:
            raise ValueError(f"Invalid image dimensions: {image.ndim}")

        original_shape = image.shape
        original_size_wh = (original_shape[1], original_shape[0])

        # Resize if needed
        if resize_max and resize_max > 0:
            h, w = original_shape[:2]
            if max(h, w) > 0:
                scale = resize_max / max(h, w)
                if scale < 1.0:
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    if new_w > 0 and new_h > 0:
                        image = cv2.resize(image, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)

        # Color conversion and tensor creation
        if grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.ndim != 2:
                raise ValueError(f"Expected 2D grayscale, got {image.ndim}D")
            image_tensor = torch.from_numpy(image[None] / 255.0).float()
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.ndim != 3:
                raise ValueError(f"Expected 3D RGB, got {image.ndim}D")
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1)) / 255.0).float()

        # Ensure divisibility by dfactor
        h, w = image_tensor.shape[-2:]
        target_h = int(h // dfactor * dfactor)
        target_w = int(w // dfactor * dfactor)

        if target_h > 0 and target_w > 0 and (target_h != h or target_w != w):
            try:
                image_tensor = TF.resize(image_tensor, size=(target_h, target_w),
                                        antialias=True)
            except TypeError:
                image_tensor = TF.resize(image_tensor, size=(target_h, target_w))

        processed_shape_wh = (image_tensor.shape[-1], image_tensor.shape[-2])
        if processed_shape_wh[0] <= 0 or processed_shape_wh[1] <= 0:
            raise ValueError(f"Invalid processed shape: {processed_shape_wh}")

        # Calculate scale factors
        if original_size_wh[0] == 0 or original_size_wh[1] == 0:
            scale_to_original = np.array([1.0, 1.0])
        else:
            scale_to_original = (np.array(original_size_wh, dtype=float) /
                               np.array(processed_shape_wh, dtype=float))

        return image_tensor, scale_to_original, original_size_wh

    except Exception as e:
        print(f"ERROR during GIM preprocessing: {e}")
        return None, None, None


class GimPipeline:
    """
    Feature matching pipeline using GIM (Generalized Image Matching).

    GIM provides a unified interface for multiple matching architectures
    including DKM, RoMa, LoFTR, and LightGlue.

    Attributes:
        config: Configuration dictionary.
        device: Torch device for computation.
        model_type: Selected GIM variant.
        model: The matcher model instance.
        detector: Feature detector (for LightGlue variant only).
    """

    # Supported model types
    SUPPORTED_MODELS = {'dkm', 'roma', 'loftr', 'lightglue'}

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the GIM pipeline.

        Args:
            config: Configuration dictionary with keys:
                - device: Computation device ('cuda' or 'cpu')
                - matcher_weights: Dict with model type and weights path
                - matcher_params.gim: Dict with model-specific settings
                - ransac_params: Dict with RANSAC settings

        Raises:
            FileNotFoundError: If weights file is not found.
            ValueError: If unsupported model type is specified.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Get configuration
        weights_config = config.get('matcher_weights', {})
        self.model_type = weights_config.get('gim_model_type', 'dkm').lower()
        self.weights_path = weights_config.get('gim_weights_path')
        self.gim_params = config.get('matcher_params', {}).get('gim', {})
        ransac_params = config.get('ransac_params', {})

        # Validate
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported GIM model: {self.model_type}. "
                           f"Supported: {self.SUPPORTED_MODELS}")

        if not self.weights_path or not Path(self.weights_path).is_file():
            raise FileNotFoundError(f"GIM weights not found: {self.weights_path}")

        print(f"Initializing GIM with model type: {self.model_type}")
        self._load_model()

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
        return f"GIM ({self.model_type.upper()})"

    def _load_model(self) -> None:
        """Load the specified GIM model and weights."""
        self.model = None
        self.detector = None

        # Load checkpoint
        try:
            state_dict = torch.load(self.weights_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if not isinstance(state_dict, dict):
                raise TypeError("Loaded state_dict is not a dictionary.")
        except Exception as e:
            print(f"ERROR loading checkpoint {self.weights_path}: {e}")
            raise

        # Initialize model based on type
        try:
            if self.model_type == 'dkm':
                self._init_dkm(state_dict)
            elif self.model_type == 'roma':
                self._init_roma(state_dict)
            elif self.model_type == 'loftr':
                self._init_loftr(state_dict)
            elif self.model_type == 'lightglue':
                self._init_lightglue(state_dict)

            if self.model:
                self.model = self.model.eval().to(self.device)

            print(f"GIM model '{self.model_type}' initialized successfully.")

        except Exception as e:
            print(f"ERROR initializing GIM model '{self.model_type}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_dkm(self, state_dict: Dict) -> None:
        """Initialize DKM model."""
        h = self.gim_params.get('dkm_h', 672)
        w = self.gim_params.get('dkm_w', 896)
        self.model = DKMv3(weights=None, h=h, w=w)

        clean_sd = {k.replace('model.', '', 1): v for k, v in state_dict.items()
                   if 'encoder.net.fc' not in k}
        load_info = self.model.load_state_dict(clean_sd, strict=False)

        if load_info.missing_keys or load_info.unexpected_keys:
            print(f"DKM load info: Missing={len(load_info.missing_keys)}, "
                  f"Unexpected={len(load_info.unexpected_keys)}")

    def _init_roma(self, state_dict: Dict) -> None:
        """Initialize RoMa model."""
        img_size = self.gim_params.get('roma_img_size', 672)
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
            raise ValueError("RoMa img_size must be int or [H, W]")

        self.model = RoMa(img_size=img_size)
        clean_sd = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_sd)

    def _init_loftr(self, state_dict: Dict) -> None:
        """Initialize LoFTR model."""
        loftr_config = lower_config(get_cfg_defaults())['loftr']
        self.model = LoFTR(loftr_config)
        self.model.load_state_dict(state_dict)

    def _init_lightglue(self, state_dict: Dict) -> None:
        """Initialize LightGlue detector and matcher."""
        # Initialize detector
        max_keypoints = self.gim_params.get('gim_lightglue_max_keypoints', 2048)
        self.detector = SuperPoint({
            'max_num_keypoints': max_keypoints,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False
        })

        detector_sd = {k.replace('superpoint.', '', 1): v
                      for k, v in state_dict.items()
                      if k.startswith('superpoint.')}
        self.detector.load_state_dict(detector_sd, strict=False)
        self.detector = self.detector.eval().to(self.device)
        print("GIM-LightGlue detector loaded.")

        # Initialize matcher
        filter_threshold = self.gim_params.get('gim_lightglue_filter_threshold', 0.1)
        self.model = LightGlue({
            'filter_threshold': filter_threshold,
            'flash': False,
            'checkpointed': True
        })

        matcher_sd = {k.replace('model.', '', 1): v
                     for k, v in state_dict.items()
                     if k.startswith('model.')}
        self.model.load_state_dict(matcher_sd, strict=False)
        print("GIM-LightGlue matcher loaded.")

    def _read_and_preprocess(
        self,
        image_path: Path,
        grayscale: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Read and preprocess an image file.

        Args:
            image_path: Path to the image.
            grayscale: Whether to convert to grayscale.

        Returns:
            Tuple of (tensor, scale, original_size) or (None, None, None).
        """
        try:
            mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(str(image_path), mode)
            if image is None:
                raise ValueError(f'Cannot read image {image_path.name}.')

            # Convert color space
            if not grayscale and image.ndim == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            elif not grayscale and image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif grayscale and image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            tensor, scale, orig_size = preprocess_for_gim(
                image,
                grayscale=grayscale,
                resize_max=self.gim_params.get('resize_max'),
                dfactor=self.gim_params.get('dfactor', 8)
            )

            if tensor is None:
                raise ValueError("Preprocessing failed.")

            return tensor.to(self.device)[None], scale, orig_size

        except Exception as e:
            print(f"Error preprocessing {image_path.name}: {e}")
            return None, None, None

    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
        """
        Match features between two images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match results.
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
            # Determine if grayscale is needed
            use_grayscale = self.model_type in ['loftr', 'lightglue']

            # Preprocess images
            image0, scale0, orig_size0 = self._read_and_preprocess(
                image0_path, grayscale=use_grayscale)
            image1, scale1, orig_size1 = self._read_and_preprocess(
                image1_path, grayscale=use_grayscale)

            if image0 is None or image1 is None:
                results['time'] = time.time() - start_time
                return results

            # Prepare data
            data = {
                'image0': image0, 'image1': image1,
                'scale0': scale0, 'scale1': scale1,
                'hw0_i': image0.shape[-2:], 'hw1_i': image1.shape[-2:],
                'hw0_o': (orig_size0[1], orig_size0[0]),
                'hw1_o': (orig_size1[1], orig_size1[0])
            }
            if use_grayscale:
                data['gray0'] = image0
                data['gray1'] = image1

            # Run matching
            kpts0, kpts1, mconf = self._run_matching(data)

            # Scale to original coordinates
            mkpts0 = kpts0.detach().cpu().numpy() * scale0
            mkpts1 = kpts1.detach().cpu().numpy() * scale1
            mconf_np = mconf.cpu().numpy()

            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1
            results['mconf'] = mconf_np

            # RANSAC
            if len(mkpts0) >= 4:
                H, inlier_mask = cv2.findHomography(
                    mkpts0, mkpts1, self.ransac_method,
                    self.ransac_thresh,
                    confidence=self.ransac_conf,
                    maxIters=self.ransac_max_iter
                )

                if H is not None and inlier_mask is not None:
                    results['homography'] = H
                    results['inliers'] = inlier_mask.ravel().astype(bool)
                    results['success'] = True

        except Exception as e:
            print(f"ERROR during GIM ({self.model_type}) matching: {e}")
            import traceback
            traceback.print_exc()

        finally:
            results['time'] = time.time() - start_time

        return results

    def _run_matching(
        self,
        data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the model-specific matching logic.

        Args:
            data: Dictionary containing preprocessed images and metadata.

        Returns:
            Tuple of (kpts0, kpts1, confidence) tensors.
        """
        empty_kpts = torch.empty((0, 2), device=self.device)
        empty_conf = torch.empty((0,), device=self.device)

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type in ['dkm', 'roma']:
                return self._match_dense(data)
            elif self.model_type == 'loftr':
                return self._match_loftr(data)
            elif self.model_type == 'lightglue':
                return self._match_lightglue(data)

        return empty_kpts, empty_kpts, empty_conf

    def _match_dense(
        self,
        data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using DKM or RoMa (dense matching)."""
        image0, image1 = data['image0'], data['image1']

        if self.model_type == 'dkm':
            target_h = self.gim_params.get('dkm_h', 672)
            target_w = self.gim_params.get('dkm_w', 896)
        else:
            img_size = self.gim_params.get('roma_img_size', 672)
            target_h = target_w = img_size if isinstance(img_size, int) else img_size[0]

        # Apply padding
        ow0, oh0, pl0, pr0, pt0, pb0 = get_padding_size(image0, target_w, target_h)
        ow1, oh1, pl1, pr1, pt1, pb1 = get_padding_size(image1, target_w, target_h)

        img0_pad = torch.nn.functional.pad(image0, (pl0, pr0, pt0, pb0))
        img1_pad = torch.nn.functional.pad(image1, (pl1, pr1, pt1, pb1))

        # Get dense matches
        dense_matches, dense_certainty = self.model.match(img0_pad, img1_pad)
        sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)

        # Convert to pixel coordinates
        h0p, w0p = img0_pad.shape[-2:]
        h1p, w1p = img1_pad.shape[-2:]

        kpts0_pad = torch.stack((
            w0p * (sparse_matches[:, 0] + 1) / 2,
            h0p * (sparse_matches[:, 1] + 1) / 2
        ), dim=-1)

        kpts1_pad = torch.stack((
            w1p * (sparse_matches[:, 2] + 1) / 2,
            h1p * (sparse_matches[:, 3] + 1) / 2
        ), dim=-1)

        # Remove padding offset
        kpts0 = kpts0_pad - kpts0_pad.new_tensor([pl0, pt0])
        kpts1 = kpts1_pad - kpts1_pad.new_tensor([pl1, pt1])

        # Filter valid points
        mask = ((kpts0[:, 0] >= 0) & (kpts0[:, 0] < ow0) &
                (kpts0[:, 1] >= 0) & (kpts0[:, 1] < oh0) &
                (kpts1[:, 0] >= 0) & (kpts1[:, 0] < ow1) &
                (kpts1[:, 1] >= 0) & (kpts1[:, 1] < oh1))

        return kpts0[mask], kpts1[mask], mconf[mask]

    def _match_loftr(
        self,
        data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using LoFTR."""
        self.model(data)

        kpts0 = data.get('mkpts0_f')
        kpts1 = data.get('mkpts1_f')

        if kpts0 is None or kpts1 is None:
            raise KeyError("LoFTR missing keypoints in output")

        mconf = data.get('mconf', torch.ones(len(kpts0), device=self.device))
        return kpts0, kpts1, mconf

    def _match_lightglue(
        self,
        data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using LightGlue."""
        empty_kpts = torch.empty((0, 2), device=self.device)
        empty_conf = torch.empty((0,), device=self.device)

        if self.detector is None or self.model is None:
            raise RuntimeError("LightGlue modules not loaded.")

        # Detect keypoints
        pred = {}
        pred.update({k + '0': v for k, v in
                    self.detector({"image": data["gray0"]}).items()})
        pred.update({k + '1': v for k, v in
                    self.detector({"image": data["gray1"]}).items()})

        # Match
        size0 = (data['image0'].shape[-1], data['image0'].shape[-2])
        size1 = (data['image1'].shape[-1], data['image1'].shape[-2])

        matcher_input = {
            **pred,
            'image_size0': torch.tensor([size0], device=self.device),
            'image_size1': torch.tensor([size1], device=self.device)
        }
        pred.update(self.model(matcher_input))

        # Extract matches
        kpts0_det = pred.get('keypoints0')
        kpts1_det = pred.get('keypoints1')
        matches = pred.get('matches0')
        mconf_raw = pred.get('matching_scores0')

        if not all(v is not None for v in [kpts0_det, kpts1_det, matches, mconf_raw]):
            return empty_kpts, empty_kpts, empty_conf

        # Handle batch dimension
        if kpts0_det.ndim > 2:
            kpts0_det = kpts0_det[0]
            kpts1_det = kpts1_det[0]
            matches = matches[0]
            mconf_raw = mconf_raw[0]

        # Process matches
        if matches.ndim == 1:
            valid = matches > -1
            idx0 = torch.where(valid)[0]
            idx1 = matches[valid].long()

            if idx0.numel() > 0:
                valid_idx = (idx0 < len(kpts0_det)) & (idx1 < len(kpts1_det))
                idx0, idx1 = idx0[valid_idx], idx1[valid_idx]

                if idx0.numel() > 0:
                    kpts0 = kpts0_det[idx0]
                    kpts1 = kpts1_det[idx1]
                    mconf = mconf_raw[idx0] if len(mconf_raw) == len(valid) else torch.ones_like(idx0, dtype=torch.float)
                    return kpts0, kpts1, mconf

        elif matches.ndim == 2 and matches.shape[1] == 2:
            if matches.numel() > 0:
                idx0, idx1 = matches[:, 0].long(), matches[:, 1].long()
                valid_idx = (idx0 < len(kpts0_det)) & (idx1 < len(kpts1_det))
                idx0, idx1 = idx0[valid_idx], idx1[valid_idx]

                if idx0.numel() > 0:
                    kpts0 = kpts0_det[idx0]
                    kpts1 = kpts1_det[idx1]
                    mconf = mconf_raw[valid_idx] if len(mconf_raw) == len(matches) else torch.ones_like(idx0, dtype=torch.float)
                    return kpts0, kpts1, mconf

        return empty_kpts, empty_kpts, empty_conf

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
            self.name,
            f'Matches: {num_inliers} / {num_total}'
        ]

        try:
            success = create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title="GIM Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600
            )
            if not success:
                print("Warning: Failed to create GIM visualization.")
        except Exception as e:
            print(f"ERROR during visualization: {e}")