"""LightGlue matching strategy for GIM framework."""

from typing import Any, Dict, Tuple

import torch

from .base import GimStrategy


class LightGlueStrategy(GimStrategy):
    """LightGlue strategy implementation with SuperPoint detector.

    Attributes:
        detector: SuperPoint keypoint detector instance.
    """

    def __init__(
        self, device: torch.device, params: Dict[str, Any], state_dict: Dict[str, Any]
    ) -> None:
        """Initializes the LightGlue strategy with detector."""
        self.detector: Any = None
        super().__init__(device, params, state_dict)

    @property
    def name(self) -> str:
        """Returns the strategy name."""
        return "LightGlue"

    @property
    def requires_grayscale(self) -> bool:
        """LightGlue requires grayscale input images."""
        return True

    def _load_model(self, state_dict: Dict[str, Any]) -> None:
        """Loads the LightGlue model and SuperPoint detector."""
        from networks.lightglue.models.matchers.lightglue import LightGlue
        from networks.lightglue.superpoint import SuperPoint

        max_keypoints = self.params.get("gim_lightglue_max_keypoints", 2048)

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
        self.detector = self.detector.eval().to(self.device)

        filter_threshold = self.params.get("gim_lightglue_filter_threshold", 0.1)
        self.model = LightGlue(
            {"filter_threshold": filter_threshold, "flash": False, "checkpointed": True}
        )

        matcher_sd = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(matcher_sd, strict=False)
        self.model = self.model.eval().to(self.device)

    def match(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs LightGlue matching on the input data."""
        empty = (
            torch.empty((0, 2), device=self.device),
            torch.empty((0, 2), device=self.device),
            torch.empty((0,), device=self.device),
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
            "image_size0": torch.tensor([size0], device=self.device),
            "image_size1": torch.tensor([size1], device=self.device),
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
