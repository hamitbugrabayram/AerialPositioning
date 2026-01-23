"""Core module for satellite visual positioning.

This module provides the main components for running visual positioning:

Classes:
    PipelineFactory: Factory for creating matcher pipelines based on configuration.
    PositioningRunner: Main orchestration class for the positioning process.
    Evaluator: Extended runner for trajectory evaluation with displacement prediction.
"""

from src.core.factory import PipelineFactory
from src.core.runner import PositioningRunner

__all__ = ["PipelineFactory", "PositioningRunner"]
