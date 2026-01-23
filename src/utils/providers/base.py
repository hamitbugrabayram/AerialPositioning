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
        """Returns the provider name.

        Returns:
            String identifier for the provider.
        """
        pass

    @abstractmethod
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Constructs the tile URL for the given coordinates.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            URL string for the tile image.
        """
        pass

    def download_tile(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """Downloads a tile and returns it as a numpy array.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            Image as numpy array (BGR), or None if download failed.
        """
        url = self.get_tile_url(x, y, z)
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/119.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                image = np.asarray(bytearray(response.content), dtype="uint8")
                return cv2.imdecode(image, cv2.IMREAD_COLOR)
            return None
        except Exception as e:
            print(f"Error downloading tile from {self.name}: {e}")
            return None
