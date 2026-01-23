"""Concrete implementations of satellite tile providers.

This module provides implementations for downloading tiles from
ESRI, Google, and Bing satellite imagery services.
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


class BingLegacyProvider(BaseTileProvider):
    """Bing Maps satellite tile provider.

    Uses Bing Virtual Earth tiles with QuadKey addressing.
    """

    @property
    def name(self) -> str:
        """Returns the provider name.

        Returns:
            Provider identifier string.
        """
        return "bing"

    def _get_quad_key(self, x: int, y: int, z: int) -> str:
        """Converts tile coordinates to Bing QuadKey format.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            QuadKey string for the tile.
        """
        quad_key = ""
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quad_key += str(digit)
        return quad_key

    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Constructs the Bing tile URL.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            z: Zoom level.

        Returns:
            URL string for the tile.
        """
        quad_key = self._get_quad_key(x, y, z)
        subdomain = (x + y) % 4
        return (
            f"https://ecn.t{subdomain}.tiles.virtualearth.net/tiles/"
            f"a{quad_key}.jpeg?g=14000"
        )
