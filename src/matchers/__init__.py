"""Feature matching pipelines for satellite localization.

This module provides matcher implementations:
- BaseMatcher: Abstract base class for all matchers
- LightGluePipeline: LightGlue-based matching
- SuperGluePipeline: SuperGlue-based matching
- LoFTRPipeline: LoFTR-based matching
- GimPipeline: GIM-based matching
- MinimaPipeline: MINIMA-based matching
"""

from src.matchers.base import BaseMatcher, MatchResult

__all__ = [
    'BaseMatcher',
    'MatchResult',
    'LightGluePipeline',
    'SuperGluePipeline',
    'LoFTRPipeline',
    'GimPipeline',
    'MinimaPipeline',
]

def __getattr__(name: str):
    """Lazy import of pipeline classes to avoid loading all dependencies."""
    if name == 'LightGluePipeline':
        from src.matchers.lightglue import LightGluePipeline
        return LightGluePipeline
    elif name == 'SuperGluePipeline':
        from src.matchers.superglue import SuperGluePipeline
        return SuperGluePipeline
    elif name == 'LoFTRPipeline':
        from src.matchers.loftr import LoFTRPipeline
        return LoFTRPipeline
    elif name == 'GimPipeline':
        from src.matchers.gim import GimPipeline
        return GimPipeline
    elif name == 'MinimaPipeline':
        from src.matchers.minima import MinimaPipeline
        return MinimaPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
