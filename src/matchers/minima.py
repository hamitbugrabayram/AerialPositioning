"""MINIMA feature matching pipeline implementation.

This module implements the MINIMA matcher supporting multiple methods:
xoftr, sp_lg, loftr for cross-modal and multi-modal image matching.
"""

import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .base import BaseMatcher

_MINIMA_PATH = Path(__file__).resolve().parent.parent.parent / "matchers/MINIMA"


class MinimaPipeline(BaseMatcher):
    """Feature matching pipeline using the MINIMA framework.

    MINIMA supports multiple matching methods for cross-modal matching:
    - xoftr: XoFTR-based matcher
    - sp_lg: SuperPoint + LightGlue matcher
    - loftr: LoFTR-based matcher

    Attributes:
        method (str): Selected matching method.
        matcher (Any): The loaded MINIMA matcher callable.
    """

    SUPPORTED_METHODS = ["xoftr", "sp_lg", "loftr"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the MINIMA pipeline with the chosen method.

        Args:
            config: Configuration dictionary containing matcher parameters.
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

        weights_dir_raw = weights_config.get(
            "minima_weights_dir", "matchers/MINIMA/weights"
        )
        weights_dir = Path(weights_dir_raw)
        if not weights_dir.is_absolute():
            weights_dir = Path(__file__).resolve().parent.parent.parent / weights_dir

        method_args = self._build_method_args(weights_config, weights_dir)

        self.matcher = None
        self._load_matcher(method_args)

    def _load_matcher(self, args: Namespace) -> None:
        """Loads the MINIMA matcher and handles internal path dependencies."""
        original_dir = os.getcwd()
        original_path = sys.path.copy()

        try:
            os.chdir(str(_MINIMA_PATH))

            if not hasattr(np, "float"):
                np.float = np.float64

            sys.path = [p for p in sys.path if "AerialPositioning/src" not in p]

            saved_modules = {}
            for mod_name in list(sys.modules.keys()):
                if mod_name == "src" or mod_name.startswith("src."):
                    saved_modules[mod_name] = sys.modules.pop(mod_name)

            minima_paths = [
                str(_MINIMA_PATH),
                str(_MINIMA_PATH / "third_party"),
                str(_MINIMA_PATH / "third_party" / "RoMa"),
            ]
            for path_str in reversed(minima_paths):
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)

            from load_model import load_loftr, load_sp_lg, load_xoftr

            if self.method == "xoftr":
                loaded_model = load_xoftr(args)
            elif self.method == "loftr":
                loaded_model = load_loftr(args, test_orginal_megadepth=False)
            elif self.method == "sp_lg":
                loaded_model = load_sp_lg(args, test_orginal_megadepth=False)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.matcher = loaded_model.from_paths

        finally:
            os.chdir(original_dir)
            sys.path = original_path
            sys.modules.update(saved_modules)

    def _build_method_args(
        self, weights_config: Dict[str, Any], weights_dir: Path
    ) -> Namespace:
        """Constructs the Namespace expected by MINIMA loading functions."""
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
        """Returns the identifying name of the matcher."""
        return f"MINIMA ({self.method})"

    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Matches features between two images using the MINIMA engine.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference image.

        Returns:
            Dictionary containing match results.
        """
        start_time = time.time()
        results = self._create_empty_result()

        if self.matcher is None:
            results["time"] = time.time() - start_time
            return results

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

            homography, inlier_mask = self.estimate_homography(mkpts0, mkpts1)
            self._update_result_with_homography(results, homography, inlier_mask)

        except Exception as e:
            print(f"ERROR during MINIMA matching: {e}")

        finally:
            results["time"] = time.time() - start_time

        return results
