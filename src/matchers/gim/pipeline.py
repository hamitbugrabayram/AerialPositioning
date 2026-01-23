"""GIM feature matching pipeline using Strategy Pattern.

This module implements the GIM framework which supports multiple matching
backends including DKM, LoFTR, and LightGlue variants via strategy pattern.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..base import BaseMatcher
from .strategies import DkmStrategy, GimStrategy, LightGlueStrategy, LoftrStrategy

_GIM_PATH = Path(__file__).resolve().parent.parent.parent.parent / "matchers/gim"
if str(_GIM_PATH) not in sys.path:
    if _GIM_PATH.exists():
        sys.path.insert(0, str(_GIM_PATH))


def preprocess_for_gim(
    image: np.ndarray,
    grayscale: bool = False,
    resize_max: Optional[int] = None,
    dfactor: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """Preprocesses an image for GIM models.

    Args:
        image: Input image as numpy array.
        grayscale: If True, convert to grayscale.
        resize_max: Maximum dimension for resizing.
        dfactor: Factor for ensuring dimensions are divisible.
        device: Torch device for processing.

    Returns:
        Tuple of (normalized_tensor, scale_to_original, original_size_wh).
    """
    if image is None or image.size == 0:
        return None, None, None

    try:
        original_shape = image.shape
        original_size_wh = (original_shape[1], original_shape[0])
        height, width = original_shape[:2]

        new_w, new_h = width, height
        if resize_max and resize_max > 0:
            scale = resize_max / max(height, width)
            if scale < 1.0:
                new_w, new_h = int(round(width * scale)), int(round(height * scale))

        if grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (new_w, new_h) != (width, height) and new_w > 0 and new_h > 0:
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

    Uses Strategy Pattern to support DKM, LoFTR, and LightGlue backends.

    Attributes:
        model_type: Selected GIM variant ('dkm', 'loftr', 'lightglue').
        strategy: The active matching strategy instance.
    """

    SUPPORTED_MODELS = {"dkm", "loftr", "lightglue"}
    STRATEGY_MAP = {
        "dkm": DkmStrategy,
        "loftr": LoftrStrategy,
        "lightglue": LightGlueStrategy,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the GIM pipeline with the selected strategy.

        Args:
            config: Configuration dictionary.

        Raises:
            FileNotFoundError: If weights file is missing.
            ValueError: If an unsupported model type is specified.
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

        self.strategy: GimStrategy = self._create_strategy()

    @property
    def name(self) -> str:
        """Returns the identifying name of the matcher."""
        return f"GIM ({self.model_type.upper()})"

    def _create_strategy(self) -> GimStrategy:
        """Creates the appropriate strategy based on model type."""
        state_dict = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        strategy_class = self.STRATEGY_MAP[self.model_type]
        return strategy_class(self._device, self.gim_params, state_dict)

    def _read_and_preprocess(
        self, image_path: Path, grayscale: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Reads and prepares an image file for matching."""
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
        """Matches features between query and map images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the satellite map tile.

        Returns:
            Dictionary containing match coordinates and success status.
        """
        start_time = time.time()
        results = self._create_empty_result()

        try:
            use_grayscale = self.strategy.requires_grayscale

            image0, scale0, orig_size0 = self._read_and_preprocess(
                Path(image0_path), grayscale=use_grayscale
            )
            image1, scale1, orig_size1 = self._read_and_preprocess(
                Path(image1_path), grayscale=use_grayscale
            )

            if (
                image0 is None
                or image1 is None
                or orig_size0 is None
                or orig_size1 is None
            ):
                results["time"] = time.time() - start_time
                return results

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

            data["color0"] = image0
            data["color1"] = image1

            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpts0, kpts1, mconf = self.strategy.match(data)

            if isinstance(kpts0, torch.Tensor):
                kpts0_np = kpts0.detach().cpu().numpy()
            else:
                kpts0_np = kpts0

            if isinstance(kpts1, torch.Tensor):
                kpts1_np = kpts1.detach().cpu().numpy()
            else:
                kpts1_np = kpts1

            mkpts0 = kpts0_np * scale0
            mkpts1 = kpts1_np * scale1

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1
            results["mconf"] = mconf.cpu().numpy()

            homography, inlier_mask = self.estimate_homography(mkpts0, mkpts1)
            self._update_result_with_homography(results, homography, inlier_mask)

        except Exception as e:
            print(f"ERROR during GIM ({self.model_type}) matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results
