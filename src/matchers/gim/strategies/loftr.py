"""LoFTR matching strategy for GIM framework."""

from typing import Any, Dict, Tuple

import torch

from .base import GimStrategy


class LoftrStrategy(GimStrategy):
    """LoFTR (Local Feature Transformer) strategy implementation."""

    @property
    def name(self) -> str:
        """Returns the strategy name."""
        return "LoFTR"

    def _load_model(self, state_dict: Dict[str, Any]) -> None:
        """Loads the LoFTR model from state dict."""
        from networks.loftr.config import get_cfg_defaults
        from networks.loftr.loftr import LoFTR
        from networks.loftr.misc import lower_config

        loftr_config = lower_config(get_cfg_defaults())["loftr"]
        self.model = LoFTR(loftr_config)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval().to(self.device)

    def match(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs LoFTR matching on the input data."""
        self.model(data)

        kpts0 = data.get("mkpts0_f")
        kpts1 = data.get("mkpts1_f")
        mconf = data.get("mconf", torch.ones(len(kpts0), device=self.device))

        return kpts0, kpts1, mconf
