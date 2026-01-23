"""
Data models for satellite localization.
This module provides configuration and result data classes:
- LocalizationConfig: Configuration container
- QueryResult: Per-query result container
"""
from src.models.config import LocalizationConfig, QueryResult
__all__ = ['LocalizationConfig', 'QueryResult']
