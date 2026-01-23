"""LoFTR feature matching pipeline implementation.

This module implements the LoFTR (Detector-Free Local Feature Matching)
matcher for dense feature matching between image pairs.
"""

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .base import BaseMatcher

_LOFTR_PATH = Path(__file__).resolve().parent.parent.parent / "matchers/LoFTR"
if str(_LOFTR_PATH) not in sys.path:
    if _LOFTR_PATH.exists():
        sys.path.insert(0, str(_LOFTR_PATH))
        _loftr_src_path = _LOFTR_PATH / "src"
        if _loftr_src_path.exists() and str(_loftr_src_path) not in sys.path:
            sys.path.insert(0, str(_loftr_src_path))

try:
    from loftr import LoFTR as LoFTRModel
    from loftr.utils.cvpr_ds_config import default_cfg as loftr_default_cfg
except ImportError as e:
    raise ImportError(f"Failed to import LoFTR components: {e}") from e

class LoFTRPipeline(BaseMatcher):
    """Feature matching pipeline using the LoFTR algorithm.

    LoFTR (Detector-Free Local Feature Matching with Transformers) is a
    dense matching method that doesn't require explicit keypoint detection.

    Attributes:
        model (LoFTRModel): The initialized LoFTR model.
        loftr_params (Dict[str, Any]): Configuration specific to LoFTR.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the LoFTR pipeline with configured weights.

        Args:
            config: Configuration dictionary containing matcher parameters.
        """
        super().__init__(config)
        self._device = torch.device(self.device)

        weights_config = config.get("matcher_weights", {})
        self.loftr_params = config.get("matcher_params", {}).get("loftr", {})

        weights_path = weights_config.get("loftr_weights_path")
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"LoFTR weights not found: {weights_path}")

        self._weights_name = Path(weights_path).stem

        model_config = deepcopy(loftr_default_cfg)

        if "temp_bug_fix" in self.loftr_params:
            model_config["coarse"]["temp_bug_fix"] = self.loftr_params["temp_bug_fix"]
        if "match_thr" in self.loftr_params:
            model_config["match_coarse"]["thr"] = self.loftr_params["match_thr"]

        print(f"Initializing LoFTR with weights: {weights_path}")

        self.model = LoFTRModel(config=model_config)
        self._load_weights(weights_path)
        self.model = self.model.eval().to(self._device)

        print("LoFTR model initialized successfully.")

    def _load_weights(self, weights_path: str) -> None:
        """Loads model weights from a checkpoint file."""
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Error loading LoFTR checkpoint: {e}") from e

    @property
    def name(self) -> str:
        """Returns the identifying name of the matcher."""
        return f"LoFTR ({self._weights_name})"

    def _preprocess_image(
        self, image_path: Path
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Loads and prepares an image for LoFTR dense matching."""
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image_path.name}")

            h_orig, w_orig = image_bgr.shape[:2]

            target_w, target_h = self._calculate_target_size(w_orig, h_orig)

            w_resized = (target_w // 8) * 8
            h_resized = (target_h // 8) * 8

            if w_resized == 0 or h_resized == 0:
                w_resized = (w_orig // 8) * 8
                h_resized = (h_orig // 8) * 8
                if w_resized == 0 or h_resized == 0:
                    raise ValueError(f"Image too small for LoFTR: {image_path.name}")

            scale_w = w_orig / w_resized if w_resized > 0 else 1.0
            scale_h = h_orig / h_resized if h_resized > 0 else 1.0

            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            interpolation = (
                cv2.INTER_AREA
                if w_resized * h_resized < w_orig * h_orig
                else cv2.INTER_LINEAR
            )
            image_resized = cv2.resize(
                image_gray, (w_resized, h_resized), interpolation=interpolation
            )

            image_tensor = torch.from_numpy(image_resized).float()[None, None] / 255.0

            return (
                image_tensor.to(self._device),
                np.array([scale_w, scale_h]),
                (w_orig, h_orig),
            )

        except Exception as e:
            print(f"Error preprocessing image {image_path.name}: {e}")
            return None, None, None

    def _calculate_target_size(self, w_orig: int, h_orig: int) -> Tuple[int, int]:
        """Calculates target dimensions based on resize configurations."""
        resize_opt = self.loftr_params.get("resize")
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

        return target_w, target_h

    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Performs dense feature matching between two images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match coordinates and success status.
        """
        start_time = time.time()
        results = self._create_empty_result()

        try:
            img0, scale0, _ = self._preprocess_image(Path(image0_path))
            img1, scale1, _ = self._preprocess_image(Path(image1_path))

            if img0 is None or img1 is None:
                results["time"] = time.time() - start_time
                return results

            batch = {"image0": img0, "image1": img1}

            with torch.no_grad():
                self.model(batch)

            mkpts0_loftr = batch["mkpts0_f"].cpu().numpy()
            mkpts1_loftr = batch["mkpts1_f"].cpu().numpy()
            mconf = batch["mconf"].cpu().numpy()

            mkpts0 = mkpts0_loftr * scale0
            mkpts1 = mkpts1_loftr * scale1

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1
            results["mconf"] = mconf

            homography, inlier_mask = self.estimate_homography(mkpts0, mkpts1)

            if homography is not None:
                results["homography"] = homography
                results["inliers"] = inlier_mask
                results["success"] = True

        except Exception as e:
            print(f"ERROR during LoFTR matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results

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
        """Saves a visualization image of the dense match result."""
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
                title="LoFTR Matches",
                text_info=text_info,
                show_outliers=False,
                target_height=600,
                homography=homography,
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            return False
