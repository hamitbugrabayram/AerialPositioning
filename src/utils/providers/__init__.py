"""Tile provider module for satellite imagery retrieval.

This module provides a unified interface for downloading satellite imagery
from ESRI and Google providers.
"""

from .base import BaseTileProvider
from .implementations import ESRIProvider, GoogleProvider


def get_provider(name: str) -> BaseTileProvider:
    """Returns a tile provider instance by name.

    Args:
        name: Provider name ('esri' or 'google').

    Returns:
        Tile provider instance.

    Raises:
        ValueError: If provider is not supported.

    """
    providers = {
        "esri": ESRIProvider(),
        "google": GoogleProvider(),
    }
    provider = providers.get(name.lower())
    if provider is None:
        supported = ", ".join(sorted(providers.keys()))
        raise ValueError(
            f"Unsupported tile provider '{name}'. Supported providers: {supported}"
        )
    return provider


__all__ = [
    "BaseTileProvider",
    "ESRIProvider",
    "GoogleProvider",
    "get_provider",
]
