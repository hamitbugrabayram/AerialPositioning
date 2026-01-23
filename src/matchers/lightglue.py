"""LightGlue feature matching pipeline implementation.

This module implements the LightGlue matcher with SuperPoint or DISK
feature extraction for image matching and positioning.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .base import BaseMatcher

_LIGHTGLUE_PATH = Path(__file__).parent.parent.parent / "matchers/LightGlue"
if str(_LIGHTGLUE_PATH) not in sys.path:
    sys.path.insert(0, str(_LIGHTGLUE_PATH))

try:
    from lightglue import DISK, LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError as e:
    raise ImportError(f"Failed to import LightGlue components: {e}") from e


class LightGluePipeline(BaseMatcher):
    """Feature matching pipeline using the LightGlue framework.

    LightGlue is a lightweight and accurate feature matcher that works
    with various local feature descriptors like SuperPoint and DISK.

    Attributes:
        extractor (Any): The feature extractor instance (SuperPoint or DISK).
        matcher (Any): The LightGlue matcher instance.
    """

    SUPPORTED_FEATURES = ("superpoint", "disk")

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the LightGlue pipeline with configured extractors.

        Args:
            config: Configuration dictionary containing matcher parameters.
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
        """Returns the identifying name of the matcher."""
        return f"LightGlue ({self._feature_type})"

    def _preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Loads and prepares an image for extraction."""
        try:
            image = load_image(image_path)
            return image.to(self._device)
        except Exception as e:
            print(f"Error loading image {image_path.name}: {e}")
            return None

    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Matches features between query and reference images.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the satellite map tile.

        Returns:
            Dictionary containing match results.
        """
        start_time = time.time()
        results = self._create_empty_result()

        try:
            image0 = self._preprocess_image(Path(image0_path))
            image1 = self._preprocess_image(Path(image1_path))

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

            homography, inlier_mask = self.estimate_homography(mkpts0, mkpts1)
            self._update_result_with_homography(results, homography, inlier_mask)

        except Exception as e:
            print(f"ERROR during LightGlue matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results
