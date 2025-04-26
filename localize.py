"""Satellite Visual Localization System.

This script provides an aerial localization system using satellite imagery.
It estimates the GPS position of an aerial vehicle by matching onboard camera images
against satellite map tiles using state-of-the-art feature matching algorithms.

Usage:
    python localize.py --config config.yaml
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core import LocalizationRunner
from src.models import LocalizationConfig


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Aerial Position Estimation using Satellite Imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python localize.py --config config.yaml
    python localize.py --config config.yaml --verbose
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the localization system.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return 1

    print(f"Loading configuration from: {config_path}")

    try:
        config = LocalizationConfig.from_yaml(str(config_path))

        runner = LocalizationRunner(config)
        runner.run()

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nERROR: Localization failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
