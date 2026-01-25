"""Base strategy class for GIM matching backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class GimStrategy(ABC):
    """Abstract base class for GIM matching strategies.

    Each strategy implements a specific matching backend (DKM, LoFTR, LightGlue).

    Attributes:
        device: Torch device for computation.
        params: Configuration parameters for the strategy.
        model: The loaded model instance.
    """

    def __init__(
        self, device: torch.device, params: Dict[str, Any], state_dict: Dict[str, Any]
    ) -> None:
        """Initializes the strategy.

        Args:
            device: Torch device for computation.
            params: Configuration parameters from gim_params.
            state_dict: Pretrained weights dictionary.
        """
        self.device = device
        self.params = params
        self.model: Any = None
        self._load_model(state_dict)

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the strategy name for identification."""

    @property
    def requires_grayscale(self) -> bool:
        """Whether this strategy requires grayscale input images."""
        return False

    @abstractmethod
    def _load_model(self, state_dict: Dict[str, Any]) -> None:
        """Loads and initializes the model from state dict.

        Args:
            state_dict: Pretrained weights dictionary.
        """

    @abstractmethod
    def match(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs feature matching on preprocessed data.

        Args:
            data: Dictionary containing preprocessed images and metadata.

        Returns:
            Tuple of (keypoints0, keypoints1, confidence_scores).
        """
