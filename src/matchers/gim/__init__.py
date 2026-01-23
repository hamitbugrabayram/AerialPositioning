"""GIM feature matching module with Strategy Pattern.

This module provides the GimPipeline class for feature matching
using DKM, LoFTR, or LightGlue backends.
"""

from .pipeline import GimPipeline, preprocess_for_gim

__all__ = ["GimPipeline", "preprocess_for_gim"]
