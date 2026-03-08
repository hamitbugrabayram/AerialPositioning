"""ORB feature matching pipeline implementation.

This module implements a classical matcher based on ORB keypoints and
brute-force Hamming matching with deterministic filtering rules.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import BaseMatcher

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class OrbPipeline(BaseMatcher):
    """Classical matcher using ORB features and BF matching."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the classical feature matcher."""
        super().__init__(config)

        params = config["matcher_params"]["orb"]
        self.max_features = max(128, int(params["max_features"]))
        self.scale_factor = max(1.01, float(params["scale_factor"]))
        self.nlevels = max(1, int(params["nlevels"]))
        self.edge_threshold = max(5, int(params["edge_threshold"]))
        self.patch_size = max(5, int(params["patch_size"]))
        self.fast_threshold = max(0, int(params["fast_threshold"]))
        self.ratio_test = min(max(float(params["ratio_test"]), 0.1), 0.99)
        self.max_matches = max(0, int(params["max_matches"]))
        self.resize_max = max(0, int(params["resize_max"]))
        self.use_clahe = bool(params["use_clahe"])
        self.min_descriptor_matches = max(4, int(params["min_descriptor_matches"]))

        self.detector = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=self.scale_factor,
            nlevels=self.nlevels,
            edgeThreshold=self.edge_threshold,
            patchSize=self.patch_size,
            fastThreshold=self.fast_threshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if self.use_clahe
            else None
        )

    @property
    def name(self) -> str:
        """Returns the identifying name of the matcher."""
        return "ORB"

    def _load_image(
        self,
        image_path: Path,
    ) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
        """Loads and normalizes an image for handcrafted matching."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            _logger.info(f"Error loading image {image_path.name}")
            return None, (1.0, 1.0)

        original_h, original_w = image.shape[:2]
        processed = image
        if self.resize_max > 0 and max(original_h, original_w) > self.resize_max:
            scale = self.resize_max / float(max(original_h, original_w))
            new_w = max(1, int(round(original_w * scale)))
            new_h = max(1, int(round(original_h * scale)))
            processed = cv2.resize(
                processed, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        if self.clahe is not None:
            processed = self.clahe.apply(processed)

        scale_x = original_w / float(processed.shape[1])
        scale_y = original_h / float(processed.shape[0])
        return processed, (scale_x, scale_y)

    def _filter_matches(self, knn_matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
        """Applies deterministic filtering rules to descriptor matches."""
        ratio_kept: List[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ratio_test * second.distance:
                ratio_kept.append(first)

        ratio_kept.sort(key=lambda match: match.distance)

        unique_matches: List[cv2.DMatch] = []
        used_query = set()
        used_train = set()
        for match in ratio_kept:
            if match.queryIdx in used_query or match.trainIdx in used_train:
                continue
            unique_matches.append(match)
            used_query.add(match.queryIdx)
            used_train.add(match.trainIdx)
            if self.max_matches > 0 and len(unique_matches) >= self.max_matches:
                break

        return unique_matches

    def _keypoints_to_array(
        self,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        index_attr: str,
        scale: Tuple[float, float],
    ) -> np.ndarray:
        """Converts OpenCV keypoints referenced by matches to arrays."""
        points = np.array(
            [keypoints[getattr(match, index_attr)].pt for match in matches],
            dtype=np.float32,
        )
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        points[:, 0] *= scale[0]
        points[:, 1] *= scale[1]
        return points

    def match(
        self,
        image0_path: Union[str, Path],
        image1_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Matches features between query and reference images."""
        start_time = time.time()
        results = self._create_empty_result()

        try:
            image0, scale0 = self._load_image(Path(image0_path))
            image1, scale1 = self._load_image(Path(image1_path))

            if image0 is None or image1 is None:
                return results

            keypoints0, descriptors0 = self.detector.detectAndCompute(image0, None)
            keypoints1, descriptors1 = self.detector.detectAndCompute(image1, None)

            if (
                descriptors0 is None
                or descriptors1 is None
                or len(keypoints0) < 4
                or len(keypoints1) < 4
            ):
                return results

            knn_matches = self.matcher.knnMatch(descriptors0, descriptors1, k=2)
            filtered_matches = self._filter_matches(knn_matches)
            self._set_feature_counts(
                results,
                len(keypoints0),
                len(keypoints1),
                len(filtered_matches),
            )
            if len(filtered_matches) < self.min_descriptor_matches:
                return results

            mkpts0 = self._keypoints_to_array(
                keypoints0,
                filtered_matches,
                "queryIdx",
                scale0,
            )
            mkpts1 = self._keypoints_to_array(
                keypoints1,
                filtered_matches,
                "trainIdx",
                scale1,
            )

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1
            results["mconf"] = np.array(
                [
                    max(0.0, 1.0 - (match.distance / 256.0))
                    for match in filtered_matches
                ],
                dtype=np.float32,
            )

            homography, inlier_mask = self.estimate_homography(mkpts0, mkpts1)
            self._update_result_with_homography(results, homography, inlier_mask)

        except Exception as e:
            _logger.info(f"ERROR during ORB matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results
