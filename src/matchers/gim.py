"""GIM (Generalized Image Matching) feature matching pipeline.

This module implements the GIM framework which supports multiple matching
backends including DKM, LoFTR, and LightGlue variants.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseMatcher

_gim_path = Path(__file__).resolve().parent.parent.parent / "matchers/gim"
if str(_gim_path) not in sys.path:
    if _gim_path.exists():
        sys.path.insert(0, str(_gim_path))

try:
    from tools import get_padding_size
except ImportError as e:
    # This might fail if tools is not in path, but usually it is fine.
    # We will handle network imports lazily.
    pass


def preprocess_for_gim(
    image: np.ndarray,
    grayscale: bool = False,
    resize_max: Optional[int] = None,
    dfactor: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """Preprocess an image for GIM models.

    Args:
        image: Input image as numpy array (HxW or HxWxC).
        grayscale: If True, convert to grayscale.
        resize_max: Maximum dimension for resizing.
        dfactor: Ensure dimensions are divisible by this factor.
        device: Torch device to use for processing.

    Returns:
        Tuple of (tensor, scale_to_original, original_size_wh).
    """
    if image is None or image.size == 0:
        return None, None, None

    try:
        original_shape = image.shape
        original_size_wh = (original_shape[1], original_shape[0])
        h, w = original_shape[:2]

        new_w, new_h = w, h
        if resize_max and resize_max > 0:
            scale = resize_max / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(round(w * scale)), int(round(h * scale))

        if grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (new_w, new_h) != (w, h) and new_w > 0 and new_h > 0:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cur_h, cur_w = image.shape[:2]
        target_h = int(cur_h // dfactor * dfactor)
        target_w = int(cur_w // dfactor * dfactor)

        if target_h > 0 and target_w > 0 and (target_h != cur_h or target_w != cur_w):
            image = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

        if grayscale:
            image_tensor = torch.from_numpy(image[None].astype(np.float32) / 255.0)
        else:
            image_tensor = torch.from_numpy(
                image.transpose((2, 0, 1)).astype(np.float32) / 255.0
            )

        processed_wh = (image_tensor.shape[-1], image_tensor.shape[-2])
        scale_to_original = np.array(original_size_wh, dtype=float) / np.array(
            processed_wh, dtype=float
        )

        if device:
            image_tensor = image_tensor.to(device)

        return image_tensor, scale_to_original, original_size_wh

    except Exception as e:
        print(f"ERROR during GIM preprocessing: {e}")
        return None, None, None


class GimPipeline(BaseMatcher):
    """Feature matching pipeline using GIM (Generalized Image Matching).

    GIM provides a unified interface for DKM, LoFTR, and LightGlue.

    Attributes:
        model_type: Selected GIM variant.
        model: The matcher model instance.
        detector: Feature detector (for LightGlue variant only).
    """

    SUPPORTED_MODELS = {"dkm", "loftr", "lightglue"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the GIM pipeline.

        Args:
            config: Configuration dictionary.

        Raises:
            FileNotFoundError: If weights file is not found.
            ValueError: If unsupported model type is specified.
        """
        super().__init__(config)
        self._device = torch.device(self.device)

        weights_config = config.get("matcher_weights", {})
        self.model_type = weights_config.get("gim_model_type", "dkm").lower()
        self.weights_path = weights_config.get("gim_weights_path")
        self.gim_params = config.get("matcher_params", {}).get("gim", {})

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported GIM model: {self.model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        if not self.weights_path or not Path(self.weights_path).is_file():
            raise FileNotFoundError(f"GIM weights not found: {self.weights_path}")

        print(f"Initializing GIM with model type: {self.model_type}")
        self._load_model()

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"GIM ({self.model_type.upper()})"

    def _load_model(self) -> None:
        """Load the specified GIM model and weights."""
        self.model = None
        self.detector = None

        state_dict = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if self.model_type == "dkm":
            self._init_dkm(state_dict)
        elif self.model_type == "loftr":
            self._init_loftr(state_dict)
        elif self.model_type == "lightglue":
            self._init_lightglue(state_dict)

        if self.model:
            self.model = self.model.eval().to(self._device)

        print(f"GIM model '{self.model_type}' initialized successfully.")

    def _init_dkm(self, state_dict: Dict) -> None:
        """Initialize DKM model."""
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3

        h = self.gim_params.get("dkm_h", 672)
        w = self.gim_params.get("dkm_w", 896)
        self.model = DKMv3(weights=None, h=h, w=w)

        clean_sd = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if "encoder.net.fc" not in k
        }
        self.model.load_state_dict(clean_sd, strict=False)

    def _init_loftr(self, state_dict: Dict) -> None:
        """Initialize LoFTR model."""
        from networks.loftr.loftr import LoFTR
        from networks.loftr.misc import lower_config
        from networks.loftr.config import get_cfg_defaults

        loftr_config = lower_config(get_cfg_defaults())["loftr"]
        self.model = LoFTR(loftr_config)
        self.model.load_state_dict(state_dict)

    def _init_lightglue(self, state_dict: Dict) -> None:
        """Initialize LightGlue detector and matcher."""
        from networks.lightglue.superpoint import SuperPoint
        from networks.lightglue.models.matchers.lightglue import LightGlue

        max_keypoints = self.gim_params.get("gim_lightglue_max_keypoints", 2048)

        self.detector = SuperPoint(
            {
                "max_num_keypoints": max_keypoints,
                "force_num_keypoints": True,
                "detection_threshold": 0.0,
                "nms_radius": 3,
                "trainable": False,
            }
        )

        detector_sd = {
            k.replace("superpoint.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("superpoint.")
        }
        self.detector.load_state_dict(detector_sd, strict=False)
        self.detector = self.detector.eval().to(self._device)

        filter_threshold = self.gim_params.get("gim_lightglue_filter_threshold", 0.1)
        self.model = LightGlue(
            {"filter_threshold": filter_threshold, "flash": False, "checkpointed": True}
        )

        matcher_sd = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(matcher_sd, strict=False)

    def _read_and_preprocess(
        self, image_path: Path, grayscale: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Read and preprocess an image file.

        Args:
            image_path: Path to the image file.
            grayscale: Whether to convert to grayscale.

        Returns:
            Tuple of (tensor, scale_factors, original_dimensions).
        """
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), mode)

        if image is None:
            print(f"Cannot read image: {image_path.name}")
            return None, None, None

        if not grayscale and image.ndim == 3:
            image = image[:, :, ::-1]

        tensor, scale, orig_size = preprocess_for_gim(
            image,
            grayscale=grayscale,
            resize_max=self.gim_params.get("resize_max"),
            dfactor=self.gim_params.get("dfactor", 8),
            device=self._device,
        )

        if tensor is None:
            return None, None, None

        return tensor.to(self._device)[None], scale, orig_size

    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
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
            # Check if model type requires grayscale
            # LoFTR usually works on grayscale, but some weights/configs might be RGB
            # The error 'weight of size [64, 3, 7, 7], expected input ... to have 3 channels'
            # indicates this model expects RGB.
            use_grayscale = self.model_type in ["lightglue"]

            image0, scale0, orig_size0 = self._read_and_preprocess(
                Path(image0_path), grayscale=use_grayscale
            )
            image1, scale1, orig_size1 = self._read_and_preprocess(
                Path(image1_path), grayscale=use_grayscale
            )

            if image0 is None or image1 is None:
                results["time"] = time.time() - start_time
                return results

            # Convert scales to tensor for LoFTR internal use (coarse_matching.py indexes them)
            # scale0 is (2,), we make it (1, 2) for batch size 1
            scale0_t = torch.from_numpy(scale0).float().to(self._device).unsqueeze(0)
            scale1_t = torch.from_numpy(scale1).float().to(self._device).unsqueeze(0)

            data = {
                "image0": image0,
                "image1": image1,
                "scale0": scale0_t,
                "scale1": scale1_t,
                "hw0_i": image0.shape[-2:],
                "hw1_i": image1.shape[-2:],
                "hw0_o": (orig_size0[1], orig_size0[0]),
                "hw1_o": (orig_size1[1], orig_size1[0]),
            }
            if use_grayscale:
                data["gray0"] = image0
                data["gray1"] = image1

            # Ensure color keys are present as some models expect them
            data["color0"] = image0
            data["color1"] = image1

            kpts0, kpts1, mconf = self._run_matching(data)

            # Ensure kpts are on CPU
            if isinstance(kpts0, torch.Tensor):
                kpts0_np = kpts0.detach().cpu().numpy()
            else:
                kpts0_np = kpts0

            if isinstance(kpts1, torch.Tensor):
                kpts1_np = kpts1.detach().cpu().numpy()
            else:
                kpts1_np = kpts1

            # Ensure scale is numpy (it should be)
            if isinstance(scale0, torch.Tensor):
                scale0 = scale0.detach().cpu().numpy()
            if isinstance(scale1, torch.Tensor):
                scale1 = scale1.detach().cpu().numpy()

            mkpts0 = kpts0_np * scale0
            mkpts1 = kpts1_np * scale1

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1
            results["mconf"] = mconf.cpu().numpy()

            H, inlier_mask = self.estimate_homography(mkpts0, mkpts1)

            if H is not None:
                results["homography"] = H
                results["inliers"] = inlier_mask
                results["success"] = True

        except Exception as e:
            print(f"ERROR during GIM ({self.model_type}) matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results

    def _run_matching(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the model-specific matching logic."""
        empty = (
            torch.empty((0, 2), device=self._device),
            torch.empty((0, 2), device=self._device),
            torch.empty((0,), device=self._device),
        )

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type == "dkm":
                return self._match_dkm(data)
            elif self.model_type == "loftr":
                return self._match_loftr(data)
            elif self.model_type == "lightglue":
                return self._match_lightglue(data)

        return empty

    def _match_dkm(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using DKM (dense matching)."""
        image0, image1 = data["image0"], data["image1"]

        target_h = self.gim_params.get("dkm_h", 672)
        target_w = self.gim_params.get("dkm_w", 896)

        ow0, oh0, pl0, pr0, pt0, pb0 = get_padding_size(image0, target_w, target_h)
        ow1, oh1, pl1, pr1, pt1, pb1 = get_padding_size(image1, target_w, target_h)

        img0_pad = torch.nn.functional.pad(image0, (pl0, pr0, pt0, pb0))
        img1_pad = torch.nn.functional.pad(image1, (pl1, pr1, pt1, pb1))

        dense_matches, dense_certainty = self.model.match(img0_pad, img1_pad)
        sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)

        h0p, w0p = img0_pad.shape[-2:]
        h1p, w1p = img1_pad.shape[-2:]

        kpts0_pad = torch.stack(
            (
                w0p * (sparse_matches[:, 0] + 1) / 2,
                h0p * (sparse_matches[:, 1] + 1) / 2,
            ),
            dim=-1,
        )

        kpts1_pad = torch.stack(
            (
                w1p * (sparse_matches[:, 2] + 1) / 2,
                h1p * (sparse_matches[:, 3] + 1) / 2,
            ),
            dim=-1,
        )

        kpts0 = kpts0_pad - kpts0_pad.new_tensor([pl0, pt0])
        kpts1 = kpts1_pad - kpts1_pad.new_tensor([pl1, pt1])

        mask = (
            (kpts0[:, 0] >= 0)
            & (kpts0[:, 0] < ow0)
            & (kpts0[:, 1] >= 0)
            & (kpts0[:, 1] < oh0)
            & (kpts1[:, 0] >= 0)
            & (kpts1[:, 0] < ow1)
            & (kpts1[:, 1] >= 0)
            & (kpts1[:, 1] < oh1)
        )

        return kpts0[mask], kpts1[mask], mconf[mask]

    def _match_loftr(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using LoFTR."""
        self.model(data)

        kpts0 = data.get("mkpts0_f")
        kpts1 = data.get("mkpts1_f")
        mconf = data.get("mconf", torch.ones(len(kpts0), device=self._device))

        return kpts0, kpts1, mconf

    def _match_lightglue(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match using LightGlue."""
        empty = (
            torch.empty((0, 2), device=self._device),
            torch.empty((0, 2), device=self._device),
            torch.empty((0,), device=self._device),
        )

        pred = {}
        pred.update(
            {k + "0": v for k, v in self.detector({"image": data["gray0"]}).items()}
        )
        pred.update(
            {k + "1": v for k, v in self.detector({"image": data["gray1"]}).items()}
        )

        size0 = (data["image0"].shape[-1], data["image0"].shape[-2])
        size1 = (data["image1"].shape[-1], data["image1"].shape[-2])

        matcher_input = {
            **pred,
            "image_size0": torch.tensor([size0], device=self._device),
            "image_size1": torch.tensor([size1], device=self._device),
        }
        pred.update(self.model(matcher_input))

        kpts0_det = pred.get("keypoints0")
        kpts1_det = pred.get("keypoints1")
        matches = pred.get("matches0")
        mconf_raw = pred.get("matching_scores0")

        if any(v is None for v in [kpts0_det, kpts1_det, matches, mconf_raw]):
            return empty

        if kpts0_det.ndim > 2:
            kpts0_det, kpts1_det = kpts0_det[0], kpts1_det[0]
            matches, mconf_raw = matches[0], mconf_raw[0]

        if matches.ndim == 1:
            valid = matches > -1
            idx0 = torch.where(valid)[0]
            idx1 = matches[valid].long()

            if idx0.numel() > 0:
                valid_idx = (idx0 < len(kpts0_det)) & (idx1 < len(kpts1_det))
                idx0, idx1 = idx0[valid_idx], idx1[valid_idx]

                if idx0.numel() > 0:
                    return (
                        kpts0_det[idx0],
                        kpts1_det[idx1],
                        mconf_raw[idx0]
                        if len(mconf_raw) == len(valid)
                        else torch.ones_like(idx0, dtype=torch.float),
                    )

        return empty

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
        """Save a visualization of the feature matches.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference image.
            mkpts0: Matched keypoints in image 0.
            mkpts1: Matched keypoints in image 1.
            inliers: Boolean inlier mask.
            output_path: Path to save the visualization.
            title: Plot title.
            homography: Optional homography matrix to visualize projection.

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
                title="GIM Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600,
                homography=homography,
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            return False
