"""Runner script for satellite-aided visual positioning experiments.

This script manages the end-to-end pipeline:
1. "dataset-prepare": Prepare central dataset and maps (fetching satellite tiles, organizing central storage).
2. "dataset-eval": Execution of the visual positioning engine.
3. "eval-summary": Generate summary report.
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from src.utils.dataset_manager import DatasetManager
from src.utils.reporting import ReportGenerator


class ExperimentRunner:
    """Orchestrates satellite-aided UAV visual positioning experiments.

    Attributes:
        project_root: Absolute path to the project root directory.
        raw_dataset_root: Path to the raw UAV-VisLoc dataset.
        central_datasets_root: Path to the processed central datasets.
        results_root: Path to store experiment results.
        base_config_path: Path to the base configuration YAML file.
        dataset_manager: Manager for dataset-related operations.
        report_generator: Generator for summary reports and visualizations.
    """

    def __init__(self):
        """Initializes the ExperimentRunner with project paths and managers."""
        self.project_root = Path(__file__).resolve().parent
        self.raw_dataset_root = self.project_root / "_VisLoc_dataset"
        self.central_datasets_root = self.project_root / "datasets"
        self.results_root = self.project_root / "results"
        self.base_config_path = self.project_root / "config.yaml"

        self.dataset_manager = DatasetManager(
            self.project_root,
            self.raw_dataset_root,
            self.central_datasets_root,
            self.base_config_path,
        )
        self.report_generator = ReportGenerator(self.project_root, self.results_root)

    def _parse_region_ids(self, args_input: Optional[List[str]]) -> Optional[List[int]]:
        """Parses region ID arguments into a list of integers.

        Args:
            args_input: List of string inputs from argparse.

        Returns:
            List of integer region IDs or None if input is None.
        """
        if args_input is None:
            return None
        if not args_input or "all" in [str(x).lower() for x in args_input]:
            return list(range(1, 12))
        return [int(x) for x in args_input if x.isdigit()]

    def run_dataset_eval(self, index: int, zoom: int, provider: str) -> None:
        """Runs primary visual positioning for a specific region, zoom, and provider.

        Args:
            index: Region index.
            zoom: Zoom level for satellite tiles.
            provider: Satellite tile provider (e.g., "esri", "google").

        Raises:
            RuntimeError: If the positioning process fails.
        """
        region_id = f"{index:02d}"
        region_name = self.dataset_manager.get_region_name(region_id)
        dataset_dir = self.dataset_manager.get_dataset_dir(index)
        query_dir = dataset_dir / "query"
        map_dir = dataset_dir / "map" / provider / str(zoom)

        if not map_dir.exists():
            print(f"\nERROR: Map directory not found: {map_dir}")
            print(
                f"HINT: Run '--dataset-prepare {index} --zoom-levels {zoom} "
                f"--tile-provider {provider}' first."
            )
            sys.exit(1)

        exp_folder_name = f"{region_name}_{region_id}_zoom_{zoom}_{provider}"
        output_path = self.results_root / exp_folder_name

        try:
            config_path = self.dataset_manager.generate_region_config(
                output_path, query_dir, map_dir, provider
            )
        except Exception as e:
            print(f"\nERROR: Failed to generate configuration for {exp_folder_name}: {e}")
            sys.exit(1)

        print(f"Running: {exp_folder_name}")

        cmd = [
            sys.executable, "-m", "src.position",
            "--config", str(config_path),
            "--eval",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: Visual Positioning process failed for {exp_folder_name}")
            print(f"Details: Command '{' '.join(cmd)}' returned exit status {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR: An unexpected error occurred while starting the process: {e}")
            sys.exit(1)

    def main(self) -> None:
        """Main execution logic for the experiment runner."""
        parser = argparse.ArgumentParser(description="UAV-VisLoc Experiment Runner")
        parser.add_argument(
            "--dataset-prepare",
            nargs="*",
            metavar="ID",
            help="Step 1: Prepare central dataset and maps (all or specific IDs)",
        )
        parser.add_argument(
            "--dataset-eval",
            nargs="*",
            metavar="ID",
            help="Step 2: Run primary visual positioning",
        )
        parser.add_argument(
            "--eval-summary",
            nargs="*",
            metavar="ID",
            help="Step 3: Generate summary report and GIFs",
        )
        parser.add_argument(
            "--zoom-levels",
            type=int,
            nargs="+",
            help="Zoom levels (Mandatory for eval/prepare)",
        )
        parser.add_argument(
            "--tile-provider",
            type=str,
            nargs="+",
            choices=["esri", "google", "bing"],
            help="Map sources (Mandatory for eval/prepare)",
        )

        args = parser.parse_args()

        prep_ids = self._parse_region_ids(args.dataset_prepare)
        eval_ids = self._parse_region_ids(args.dataset_eval)
        summary_ids = self._parse_region_ids(args.eval_summary)

        if prep_ids or eval_ids:
            if not args.zoom_levels:
                print("ERROR: --zoom-levels is mandatory for preparation and evaluation.")
                sys.exit(1)
            if not args.tile_provider:
                print("ERROR: --tile-provider is mandatory for preparation and evaluation.")
                sys.exit(1)

        if prep_ids:
            for i in prep_ids:
                print(f"Preparing Region {i}")
                try:
                    self.dataset_manager.prepare_region_data(
                        i, args.zoom_levels, args.tile_provider
                    )
                except Exception as e:
                    print(f"Error during preparation for Region {i}: {e}")
                    sys.exit(1)

        if eval_ids:
            for i in eval_ids:
                for p in args.tile_provider:
                    for z in args.zoom_levels:
                        self.run_dataset_eval(i, z, p)

        if summary_ids is not None:
            try:
                self.report_generator.generate_summary(summary_ids)
            except Exception as e:
                print(f"Error: Summary failed: {e}")
                sys.exit(1)

        if not any([args.dataset_prepare, args.dataset_eval, args.eval_summary is not None]):
            parser.print_help()


if __name__ == "__main__":
    try:
        runner = ExperimentRunner()
        runner.main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nCRITICAL UNHANDLED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
