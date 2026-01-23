"""Abstract base class for satellite tile providers.

This module defines the interface that all tile providers must implement
for downloading satellite imagery tiles.
"""

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
import requests


class BaseTileProvider(ABC):
    """Abstract base class for satellite tile providers.

    Subclasses must implement the name property and get_tile_url method
    to define provider-specific URL construction.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the provider name."""
        pass

    @abstractmethod
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Constructs the tile URL for the given coordinates."""
        pass

    def download_tile(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """Downloads a tile and returns it as a numpy array.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            Image as numpy array (BGR), or None if download failed.

        Raises:
            RuntimeError: If the download fails with a non-200 status code.
            requests.RequestException: If the network request fails.
        """
        url = self.get_tile_url(x, y, z)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download tile from {self.name}. "
                f"Status: {response.status_code}, URL: {url}"
            )

        image_data = np.asarray(bytearray(response.content), dtype="uint8")
        decoded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if decoded_image is None:
            raise RuntimeError(f"Failed to decode image content from {url}")

        return decoded_image
