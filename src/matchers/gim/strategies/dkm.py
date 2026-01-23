"""DKM matching strategy for GIM framework."""

from typing import Any, Dict, Tuple

import torch

from .base import GimStrategy


class DkmStrategy(GimStrategy):
    """DKM (Dense Kernelized Matching) strategy implementation.

    Attributes:
        target_h: Target height for image processing.
        target_w: Target width for image processing.
    """

    def __init__(
        self, device: torch.device, params: Dict[str, Any], state_dict: Dict[str, Any]
    ) -> None:
        """Initializes the DKM strategy.

        Args:
            device: Torch device for computation.
            params: Configuration parameters from gim_params.
            state_dict: Pretrained weights dictionary.
        """
        self.target_h = params.get("dkm_h", 672)
        self.target_w = params.get("dkm_w", 896)
        super().__init__(device, params, state_dict)

    @property
    def name(self) -> str:
        """Returns the strategy name."""
        return "DKM"

    def _load_model(self, state_dict: Dict[str, Any]) -> None:
        """Loads the DKM model from state dict."""
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3

        self.model = DKMv3(weights=None, h=self.target_h, w=self.target_w)

        clean_sd = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if "encoder.net.fc" not in k
        }
        self.model.load_state_dict(clean_sd, strict=False)
        self.model = self.model.eval().to(self.device)

    def match(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs DKM matching on the input data."""
        try:
            from tools import get_padding_size
        except ImportError:
            raise ImportError("GIM tools module not available for DKM matching")

        image0, image1 = data["image0"], data["image1"]

        ow0, oh0, pl0, pr0, pt0, pb0 = get_padding_size(
            image0, self.target_w, self.target_h
        )
        ow1, oh1, pl1, pr1, pt1, pb1 = get_padding_size(
            image1, self.target_w, self.target_h
        )

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
