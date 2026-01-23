"""SuperGlue feature matching pipeline implementation.

This module implements the SuperGlue matcher using SuperPoint features.
It inherits from LightGluePipeline to reuse consistent preprocessing and
postprocessing logic.
"""

from pathlib import Path
from typing import Any, Dict, Union

from .lightglue import LightGluePipeline


class SuperGluePipeline(LightGluePipeline):
    """Feature matching pipeline using the SuperGlue algorithm.

    Inherits from LightGluePipeline but is specialized for SuperGlue weights.
    Maintains compatibility with existing matching infrastructure.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the SuperGlue pipeline.

        Args:
            config: Configuration dictionary containing matcher parameters.
        """
        super().__init__(config)
        self.name_override = "SuperGlue"

    @property
    def name(self) -> str:
        """Returns the identifying name of the matcher."""
        return self.name_override

    def match(
        self, image0_path: Union[str, Path], image1_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Matches features between two images using SuperGlue.

        Args:
            image0_path: Path to the query image.
            image1_path: Path to the reference/map image.

        Returns:
            Dictionary containing match results.
        """
        return super().match(image0_path, image1_path)
