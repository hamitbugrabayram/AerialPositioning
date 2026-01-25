"""Feature matching pipelines for satellite visual positioning.

This module provides matcher implementations for visual positioning:

Classes:
    BaseMatcher: Abstract base class defining the matcher interface.
    MatchResult: Standardized matching result structure.
    LightGluePipeline: LightGlue-based sparse feature matching.
    SuperGluePipeline: SuperGlue-based sparse feature matching.
    LoFTRPipeline: LoFTR-based dense feature matching.
    GimPipeline: GIM (Generalized Image Matching) framework.
    MinimaPipeline: MINIMA cross-modal matching framework.
"""

from src.matchers.base import BaseMatcher, MatchResult

__all__ = [
    "BaseMatcher",
    "MatchResult",
    "LightGluePipeline",
    "SuperGluePipeline",
    "LoFTRPipeline",
    "GimPipeline",
    "MinimaPipeline",
]


def __getattr__(name: str):
    """Lazy import of pipeline classes to avoid loading all dependencies.

    This allows the package to be imported without requiring all matcher
    dependencies to be installed. Pipeline classes are loaded on first access.

    Args:
        name: Name of the attribute to retrieve.

    Returns:
        The requested pipeline class.

    Raises:
        AttributeError: If the requested attribute is not found.
    """
    if name == "LightGluePipeline":
        from src.matchers.lightglue import LightGluePipeline

        return LightGluePipeline
    elif name == "SuperGluePipeline":
        from src.matchers.superglue import SuperGluePipeline

        return SuperGluePipeline
    elif name == "LoFTRPipeline":
        from src.matchers.loftr import LoFTRPipeline

        return LoFTRPipeline
    elif name == "GimPipeline":
        from src.matchers.gim import GimPipeline

        return GimPipeline
    elif name == "MinimaPipeline":
        from src.matchers.minima import MinimaPipeline

        return MinimaPipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
