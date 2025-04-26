"""Pipeline factory for creating matcher instances.

This module provides a factory pattern for instantiating different
feature matching pipelines based on configuration.
"""

from typing import Dict, List

from src.models.config import LocalizationConfig


class PipelineFactory:
    """Factory for creating matcher pipeline instances.

    This class provides a centralized way to create matcher pipelines
    based on the configuration. It supports lazy loading of pipeline
    classes to avoid import errors when certain matchers are not available.

    Attributes:
        PIPELINE_REGISTRY: Mapping of matcher type to module and class names.
    """

    PIPELINE_REGISTRY: Dict[str, tuple] = {
        'lightglue': ('src.matchers.lightglue', 'LightGluePipeline'),
        'superglue': ('src.matchers.superglue', 'SuperGluePipeline'),
        'gim': ('src.matchers.gim', 'GimPipeline'),
        'loftr': ('src.matchers.loftr', 'LoFTRPipeline'),
        'minima': ('src.matchers.minima', 'MinimaPipeline'),
    }

    @classmethod
    def get_supported_matchers(cls) -> List[str]:
        """Get list of supported matcher types.

        Returns:
            List of supported matcher type strings.
        """
        return list(cls.PIPELINE_REGISTRY.keys())

    @classmethod
    def create(cls, config: LocalizationConfig):
        """Create a matcher pipeline based on configuration.

        Args:
            config: Localization configuration.

        Returns:
            Initialized pipeline instance.

        Raises:
            ValueError: If matcher type is not supported.
            ImportError: If pipeline module cannot be imported.
        """
        matcher_type = config.matcher_type.lower()

        if matcher_type not in cls.PIPELINE_REGISTRY:
            supported = ', '.join(cls.get_supported_matchers())
            raise ValueError(
                f"Unsupported matcher: '{matcher_type}'. "
                f"Supported: {supported}"
            )

        module_name, class_name = cls.PIPELINE_REGISTRY[matcher_type]

        try:
            import importlib
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_name}: {e}"
            )
        except AttributeError:
            raise ImportError(
                f"Class {class_name} not found in {module_name}"
            )

        return pipeline_class(config.to_dict())
