"""MINIMA feature matching pipeline.

This module implements the MINIMA matcher supporting multiple methods:
xoftr, sp_lg, loftr for cross-modal and multi-modal image matching.
"""

import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMatcher

_minima_path = Path(__file__).resolve().parent.parent.parent / "matchers/MINIMA"


class MinimaPipeline(BaseMatcher):
    """Feature matching pipeline using MINIMA.

    MINIMA supports multiple matching methods for cross-modal matching:
    - xoftr: XoFTR-based matcher
    - sp_lg: SuperPoint + LightGlue matcher
    - loftr: LoFTR-based matcher

    Attributes:
        method: Selected matching method.
        matcher: MINIMA matcher callable.
    """

    SUPPORTED_METHODS = ["xoftr", "sp_lg", "loftr"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the MINIMA pipeline.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

        weights_config = config.get("matcher_weights", {})
        self.minima_params = config.get("matcher_params", {}).get("minima", {})

        self.method = weights_config.get("minima_method", "xoftr").lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported MINIMA method: '{self.method}'. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        weights_dir = weights_config.get(
            "minima_weights_dir", "matchers/MINIMA/weights"
        )
        weights_dir = Path(weights_dir)
        if not weights_dir.is_absolute():
            weights_dir = Path(__file__).resolve().parent.parent.parent / weights_dir

        args = self._build_method_args(weights_config, weights_dir)

        print(f"Initializing MINIMA with method: {self.method}")
        print(f"  Weights directory: {weights_dir}")

        self._load_matcher(args)
        print(f"MINIMA ({self.method}) initialized successfully.")

    def _load_matcher(self, args: Namespace) -> None:
        """Load the MINIMA matcher with proper path handling.

        Args:
            args: Method-specific arguments.
        """
        original_dir = os.getcwd()
        original_path = sys.path.copy()

        try:
            os.chdir(str(_minima_path))

            if not hasattr(np, "float"):
                np.float = np.float64  # type: ignore

            sys.path = [p for p in sys.path if "SatelliteLocalization/src" not in p]

            saved_modules = {}
            for mod_name in list(sys.modules.keys()):
                if mod_name == "src" or mod_name.startswith("src."):
                    saved_modules[mod_name] = sys.modules.pop(mod_name)

            minima_paths = [
                str(_minima_path),
                str(_minima_path / "third_party"),
                str(_minima_path / "third_party" / "RoMa"),
            ]
            for p in reversed(minima_paths):
                if p not in sys.path:
                    sys.path.insert(0, p)

            from load_model import load_xoftr, load_loftr, load_sp_lg

            if self.method == "xoftr":
                matcher = load_xoftr(args)
            elif self.method == "loftr":
                matcher = load_loftr(args, test_orginal_megadepth=False)
            elif self.method == "sp_lg":
                matcher = load_sp_lg(args, test_orginal_megadepth=False)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.matcher = matcher.from_paths

        finally:
            os.chdir(original_dir)
            sys.path = original_path
            sys.modules.update(saved_modules)

    def _build_method_args(
        self, weights_config: Dict[str, Any], weights_dir: Path
    ) -> Namespace:
        """Build method-specific arguments for MINIMA.

        Args:
            weights_config: Weights configuration dictionary.
            weights_dir: Path to weights directory.

        Returns:
            Namespace with method-specific arguments.
        """
        args = Namespace()

        if self.method == "xoftr":
            ckpt_name = weights_config.get("minima_xoftr_ckpt", "minima_xoftr.ckpt")
            args.ckpt = str(weights_dir / ckpt_name)
            args.match_threshold = self.minima_params.get("match_threshold", 0.3)
            args.fine_threshold = self.minima_params.get("fine_threshold", 0.1)

        elif self.method == "loftr":
            ckpt_name = weights_config.get("minima_loftr_ckpt", "minima_loftr.ckpt")
            args.ckpt = str(weights_dir / ckpt_name)
            args.thr = self.minima_params.get("loftr_threshold", 0.2)

        elif self.method == "sp_lg":
            ckpt_name = weights_config.get("minima_sp_lg_ckpt", "minima_lightglue.pth")
            args.ckpt = str(weights_dir / ckpt_name)

        return args

    @property
    def name(self) -> str:
        """Return the matcher display name."""
        return f"MINIMA ({self.method})"

    def match(self, image0_path: Path, image1_path: Path) -> Dict[str, Any]:
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
            match_result = self.matcher(str(image0_path), str(image1_path))

            mkpts0 = match_result.get("mkpts0", np.array([]))
            mkpts1 = match_result.get("mkpts1", np.array([]))
            mconf = match_result.get("mconf", np.array([]))

            if hasattr(mkpts0, "cpu"):
                mkpts0 = mkpts0.cpu().numpy()
            if hasattr(mkpts1, "cpu"):
                mkpts1 = mkpts1.cpu().numpy()
            if hasattr(mconf, "cpu"):
                mconf = mconf.cpu().numpy()

            results["mkpts0"] = mkpts0
            results["mkpts1"] = mkpts1
            results["mconf"] = mconf

            H, inlier_mask = self.estimate_homography(mkpts0, mkpts1)

            if H is not None:
                results["homography"] = H
                results["inliers"] = inlier_mask
                results["success"] = True

        except Exception as e:
            print(f"ERROR during MINIMA matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results

    def visualize_matches(
        self,
        image0_path: Path,
        image1_path: Path,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        inliers: np.ndarray,
        output_path: Path,
        homography: Optional[np.ndarray] = None,
    ) -> bool:
        """Save a visualization of the feature matches.

        Args:
            image0_path: Path to query image.
            image1_path: Path to map image.
            mkpts0: Keypoints in query image.
            mkpts1: Keypoints in map image.
            inliers: Inlier mask.
            output_path: Path to save visualization.
            homography: Optional homography matrix.

        Returns:
            True if visualization saved, False otherwise.
        """
        try:
            from src.utils.visualization import create_match_visualization
        except ImportError:
            print("Visualization module unavailable.")
            return False

        num_inliers = np.sum(inliers) if len(inliers) > 0 else 0

        try:
            return create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title=f"MINIMA ({self.method}) Matches",
                text_info=[self.name, f"Matches: {num_inliers} / {len(mkpts0)}"],
                show_outliers=False,
                target_height=600,
                homography=homography,
            )
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            return False
