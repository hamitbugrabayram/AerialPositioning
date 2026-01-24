"""Satellite Visual Localization System.

This script provides a GNSS-free aerial localization system using satellite imagery.
It estimates the geographic coordinates of an aerial vehicle by matching onboard
camera images against satellite map tiles using deep feature matching algorithms.

Usage:
    python localize.py --config config.yaml
"""
import argparse
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.core import LocalizationRunner
from src.core.consecutive_runner import ConsecutiveRunner
from src.models import LocalizationConfig

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments.
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Aerial Position Estimation using Satellite Imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python localize.py --config config.yaml
    python localize.py --config config.yaml --verbose
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
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--consecutive",
        "-t",
        action="store_true",
        help="Run in consecutive localization mode for continuous coordinate estimation",
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
        if args.consecutive:
            runner = ConsecutiveRunner(config)
            runner.run_trajectory()
        else:
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
if __name__ == "__main__":
    sys.exit(main())
