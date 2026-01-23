"""Tile provider module for satellite imagery retrieval.

This module provides a unified interface for downloading satellite imagery
from various providers including ESRI, Google, and Bing.
"""

from .base import BaseTileProvider
from .implementations import BingLegacyProvider, ESRIProvider, GoogleProvider


def get_provider(name: str) -> BaseTileProvider:
    """Returns a tile provider instance by name.

    Args:
        name: Provider name ('esri', 'google', or 'bing').

    Returns:
        Tile provider instance. Defaults to ESRI if name not found.
    """
    providers = {
        "esri": ESRIProvider(),
        "google": GoogleProvider(),
        "bing": BingLegacyProvider(),
    }
    return providers.get(name.lower(), ESRIProvider())


__all__ = [
    "BaseTileProvider",
    "ESRIProvider",
    "GoogleProvider",
    "BingLegacyProvider",
    "get_provider",
]
