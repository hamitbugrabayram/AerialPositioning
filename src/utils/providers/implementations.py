"""Concrete implementations of satellite tile providers.

This module provides implementations for downloading tiles from
ESRI and Google satellite imagery services.
"""

from .base import BaseTileProvider


class ESRIProvider(BaseTileProvider):
    """ESRI World Imagery tile provider.

    Uses the ArcGIS World Imagery MapServer for satellite tiles.
    """

    @property
    def name(self) -> str:
        """Returns the provider name.

        Returns:
            Provider identifier string.

        """
        return "esri"

    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Constructs the ESRI tile URL.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            URL string for the tile.

        """
        return (
            f"https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )


class GoogleProvider(BaseTileProvider):
    """Google Maps satellite tile provider.

    Uses Google Maps satellite imagery layer.
    """

    @property
    def name(self) -> str:
        """Returns the provider name.

        Returns:
            Provider identifier string.

        """
        return "google"

    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Constructs the Google tile URL.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            URL string for the tile.

        """
        return f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
