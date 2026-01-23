"""Pipeline factory for creating matcher instances.

This module provides a factory pattern for instantiating different
feature matching pipelines based on configuration settings.
"""

import importlib
from typing import Any, Dict, List, Tuple

from src.models.config import PositioningConfig


class PipelineFactory:
    """Factory for creating matcher pipeline instances.

    This class provides a centralized way to create matcher pipelines
    based on the configuration. It supports lazy loading of pipeline
    classes to avoid import errors when certain matchers are not available.

    Attributes:
        PIPELINE_REGISTRY: Mapping of matcher type to module and class names.
    """

    PIPELINE_REGISTRY: Dict[str, Tuple[str, str]] = {
        "lightglue": ("src.matchers.lightglue", "LightGluePipeline"),
        "superglue": ("src.matchers.superglue", "SuperGluePipeline"),
        "gim": ("src.matchers.gim", "GimPipeline"),
        "loftr": ("src.matchers.loftr", "LoFTRPipeline"),
        "minima": ("src.matchers.minima", "MinimaPipeline"),
    }

    @classmethod
    def get_supported_matchers(cls) -> List[str]:
        """Gets the list of supported matcher types.

        Returns:
            List[str]: List of supported matcher type strings.
        """
        return list(cls.PIPELINE_REGISTRY.keys())

    @classmethod
    def create(cls, config: PositioningConfig) -> Any:
        """Creates a matcher pipeline based on the provided configuration.

        Args:
            config: Positioning configuration object containing matcher settings.

        Returns:
            Any: An initialized pipeline instance ready for matching operations.

        Raises:
            ValueError: If the matcher type is not supported.
            ImportError: If the pipeline module or class cannot be loaded.
        """
        matcher_type = config.matcher_type.lower()

        if matcher_type not in cls.PIPELINE_REGISTRY:
            supported = ", ".join(cls.get_supported_matchers())
            raise ValueError(
                f"Unsupported matcher: '{matcher_type}'. Supported: {supported}"
            )

        module_name, class_name = cls.PIPELINE_REGISTRY[matcher_type]

        try:
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_name}: {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(f"Class {class_name} not found in {module_name}") from e

        return pipeline_class(config.to_dict())
