"""
Core module for satellite localization.
This module provides the main components for running localization:
- PipelineFactory: Factory for creating matcher pipelines
- LocalizationRunner: Main orchestration class
"""
from src.core.factory import PipelineFactory
from src.core.runner import LocalizationRunner
__all__ = ['PipelineFactory', 'LocalizationRunner']
