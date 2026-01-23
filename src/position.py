"""Aerial Positioning via Satellite Imagery.

This script provides a GNSS-free aerial positioning system using satellite imagery.
It estimates the geographic coordinates of an aerial vehicle by matching onboard
camera images against satellite map tiles using deep feature matching algorithms.

The system supports multiple feature matching backends including LightGlue, LoFTR,
SuperGlue, GIM, and MINIMA for robust cross-domain image matching.

Example:
    Run the positioning system with a configuration file::

        $ python -m src.position --config config.yaml
        $ python -m src.position --config config.yaml --verbose
        $ python -m src.position --config config.yaml --eval
"""

import argparse
import sys
import traceback
from pathlib import Path

from src.core import PositioningRunner
from src.core.evaluator import Evaluator
from src.models import PositioningConfig


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the positioning system.

    Returns:
        Parsed arguments namespace containing:
            - config: Path to configuration YAML file
            - verbose: Enable verbose output
            - eval: Run in evaluation mode
    """
    parser = argparse.ArgumentParser(
        description="Aerial Positioning via Satellite Imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.position --config config.yaml
    python -m src.position --config config.yaml --verbose
    python -m src.position --config config.yaml --eval
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run in evaluation mode for full flight track estimation",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the positioning system.

    Loads configuration from YAML file and executes either standard
    positioning or trajectory evaluation based on command line arguments.

    Returns:
        Exit code where 0 indicates success and 1 indicates failure.
    """
    args = parse_arguments()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    try:
        config = PositioningConfig.from_yaml(str(config_path))

        if args.eval:
            runner = Evaluator(config)
            runner.run_trajectory()
        else:
            runner = PositioningRunner(config)
            runner.run()

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1

    except Exception as e:
        print(f"\nERROR: Positioning failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
