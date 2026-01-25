"""Data models for satellite visual positioning.

This module provides configuration and result data classes:

Classes:
    PositioningConfig: Configuration container for all positioning settings
        including data paths, matcher configuration, and processing parameters.
    QueryResult: Per-query result container for positioning outputs
        including predicted coordinates, error metrics, and timing information.
"""

from src.models.config import PositioningConfig, QueryResult

__all__ = ["PositioningConfig", "QueryResult"]
