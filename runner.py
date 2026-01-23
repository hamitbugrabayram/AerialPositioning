"""Runner script for automating satellite-aided UAV positioning experiments.

This script manages the end-to-end pipeline for visual positioning experiments:
    1. Dataset preparation (fetching satellite tiles, organizing central storage).
    2. Execution of the visual positioning engine.
    3. Results management and automated report generation (Markdown + GIFs).

Example:
    Prepare datasets and run evaluation::

        $ python runner.py --dataset-prepare all --zoom-levels 18 --tile-provider esri
        $ python runner.py --dataset-eval all --zoom-levels 18 --tile-provider esri --sample-interval 30
        $ python runner.py --eval-summary all
"""

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.utils.tile_system import TileSystem

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "_VisLoc_dataset"
DATASETS_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
EVAL_RESULTS_ROOT = RESULTS_ROOT / "eval"
BASE_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SUMMARY_CSV = PROJECT_ROOT / "experiments_summary.csv"
SUMMARY_MD = PROJECT_ROOT / "results_report.md"


def load_base_config() -> Optional[Dict[str, Any]]:
    """Loads the base configuration template from the project root.

    Returns:
        Configuration dictionary if file exists, None otherwise.
    """
    if not os.path.isfile(str(BASE_CONFIG_PATH)):
        print(f"Base config not found at {BASE_CONFIG_PATH}.")
        return None

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Saves a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Target file path for the YAML output.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_region_name(index_str: str) -> str:
    """Retrieves a sanitized region name from the dataset mapping.

    Args:
        index_str: Zero-padded region index string (e.g., '01', '02').

    Returns:
        Sanitized region name with spaces and hyphens replaced by underscores.
    """
    csv_path = DATASET_ROOT / "satellite_ coordinates_range.csv"
    region_name = f"Region_{index_str}"

    if os.path.isfile(str(csv_path)):
        df = pd.read_csv(csv_path)
        map_name = f"satellite{index_str}.tif"
        row = df[df["mapname"] == map_name]
        if not row.empty:
            region_name = str(row.iloc[0]["region"])

    return region_name.replace(" ", "_").replace("-", "_")


def get_dataset_dir(index: int) -> Path:
    """Returns the central dataset directory for a region.

    Args:
        index: Integer region index.

    Returns:
        Path to the region's dataset directory.
    """
    region_id = f"{index:02d}"
    region_name = get_region_name(region_id)
    return DATASETS_ROOT / f"{region_id}_{region_name}"


def _get_telemetry_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely extracts a numeric value from a pandas row.

    Args:
        row: Pandas Series containing telemetry data.
        key: Column key to extract.
        default: Default value if key is missing or null.

    Returns:
        Extracted numeric value or default.
    """
    val = row.get(key)
    if pd.isna(val) or val is None:
        return float(default)
    return float(val)


def prepare_shared_dataset(index: int) -> Path:
    """Ensures query images and base metadata are in the central datasets folder.

    Copies drone images and prepares photo_metadata.csv from the raw dataset
    structure into the standardized central storage format.

    Args:
        index: Integer region index.

    Returns:
        Path to the prepared dataset directory.
    """
    region_id = f"{index:02d}"
    dataset_dir = get_dataset_dir(index)
    query_dir = dataset_dir / "query"

    region_dir = DATASET_ROOT / region_id
    drone_img_dir = region_dir / "drone"
    csv_path = region_dir / f"{region_id}.csv"

    if not (query_dir / "photo_metadata.csv").exists():
        os.makedirs(str(query_dir), exist_ok=True)
        print(f"Initializing central dataset for Region {region_id} at {dataset_dir}")

        if drone_img_dir.exists():
            for img in drone_img_dir.glob("*.[jJ][pP][gG]"):
                shutil.copy2(img, query_dir / img.name)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            query_metadata = []

            for _, row in df.iterrows():
                filename = str(row["filename"])
                query_metadata.append(
                    {
                        "Filename": filename,
                        "Latitude": _get_telemetry_value(row, "lat"),
                        "Longitude": _get_telemetry_value(row, "lon"),
                        "Altitude": _get_telemetry_value(row, "height"),
                        "Gimball_Roll": _get_telemetry_value(row, "Kappa"),
                        "Gimball_Pitch": -90.0 + _get_telemetry_value(row, "Omega"),
                        "Gimball_Yaw": 0.0,
                        "Flight_Roll": 0.0,
                        "Flight_Pitch": 0.0,
                        "Flight_Yaw": _get_telemetry_value(row, "Phi1"),
                    }
                )

            pd.DataFrame(query_metadata).to_csv(
                str(query_dir / "photo_metadata.csv"), index=False
            )

    return dataset_dir


def generate_region_config(
    output_path: Path,
    query_dir: Path,
    map_dir: Path,
    provider_name: Optional[str] = None,
) -> Path:
    """Generates a localized config.yaml inside the results folder.

    Creates a customized configuration file with paths pointing to the
    specific query and map directories for the experiment.

    Args:
        output_path: Directory where the config will be saved.
        query_dir: Path to query images directory.
        map_dir: Path to satellite map tiles directory.
        provider_name: Optional tile provider name override.

    Returns:
        Path to the generated configuration file.

    Raises:
        RuntimeError: If base configuration template is not found.
    """
    base_config = load_base_config()
    if base_config is None:
        raise RuntimeError("Base config not found")

    config = yaml.safe_load(yaml.dump(base_config))

    if provider_name:
        if "tile_provider" not in config:
            config["tile_provider"] = {}
        config["tile_provider"]["name"] = provider_name

    config["data_paths"] = {
        "query_dir": str(query_dir.resolve()),
        "map_dir": str(map_dir.resolve()),
        "output_dir": str(output_path.resolve()),
        "query_metadata": str((query_dir / "photo_metadata.csv").resolve()),
        "map_metadata": str((map_dir / "map.csv").resolve()),
    }

    try:
        sample_images = list(query_dir.glob("*.[jJ][pP][gG]"))
        if sample_images:
            img = cv2.imread(str(sample_images[0]))
            if img is not None:
                h, w = img.shape[:2]
                if "camera_model" not in config:
                    config["camera_model"] = {}
                config["camera_model"]["resolution_width"] = int(w)
                config["camera_model"]["resolution_height"] = int(h)
                if int(w) == 3000:
                    config["camera_model"]["focal_length"] = 4.5
                    config["camera_model"]["hfov_deg"] = 72.0
    except Exception:
        pass

    os.makedirs(str(output_path), exist_ok=True)
    config_file = output_path / "config.yaml"
    save_config(config, config_file)
    return config_file


def prepare_region_data(
    index: int,
    zoom_levels: List[int],
    provider_names: List[str],
) -> None:
    """Prepares map tiles for a region across multiple zooms and providers.

    Downloads and organizes satellite imagery tiles for the specified
    region at each combination of zoom level and provider.

    Args:
        index: Integer region index.
        zoom_levels: List of zoom levels to prepare tiles for.
        provider_names: List of tile provider names to use.
    """
    dataset_dir = prepare_shared_dataset(index)
    region_id = f"{index:02d}"
    region_name = get_region_name(region_id)
    query_dir = dataset_dir / "query"

    full_metadata_path = query_dir / "photo_metadata.csv"
    df = pd.read_csv(full_metadata_path)

    for provider in provider_names:
        for zoom in zoom_levels:
            map_dir = dataset_dir / "map" / provider / str(zoom)
            os.makedirs(str(map_dir), exist_ok=True)

            if not (map_dir / "map.csv").exists():
                lats = df["Latitude"].tolist()
                lons = df["Longitude"].tolist()
                margin = 0.01

                lat_max = max(lats) + margin
                lon_min = min(lons) - margin
                lat_min = min(lats) - margin
                lon_max = max(lons) + margin

                print(
                    f"  Fetching {provider.upper()} tiles for "
                    f"Region {region_id} Zoom {zoom}..."
                )

                tiles = TileSystem.retrieve_map_tiles(
                    lat_max,
                    lon_min,
                    lat_min,
                    lon_max,
                    zoom,
                    str(map_dir),
                    provider_name=provider,
                )

                if tiles:
                    pd.DataFrame(tiles).to_csv(str(map_dir / "map.csv"), index=False)
            else:
                print(f"  Maps already exist: {map_dir}")


def run_dataset_eval(
    index: int,
    zoom: int,
    provider: str,
    sample_interval: int = 30,
) -> None:
    """Runs primary visual positioning for a region, zoom, and provider.

    Generates a localized configuration and executes the positioning
    script in evaluation mode.

    Args:
        index: Integer region index.
        zoom: Zoom level for satellite tiles.
        provider: Tile provider name.
        sample_interval: Frame sampling interval for evaluation.
    """
    region_id = f"{index:02d}"
    region_name = get_region_name(region_id)
    dataset_dir = get_dataset_dir(index)
    query_dir = dataset_dir / "query"
    map_dir = dataset_dir / "map" / provider / str(zoom)

    if not map_dir.exists():
        print(f"Error: Map directory {map_dir} not found. Run --dataset-prepare first.")
        return

    exp_folder_name = f"{region_name}_{region_id}_zoom_{zoom}_{provider}"
    output_path = EVAL_RESULTS_ROOT / exp_folder_name

    config_path = generate_region_config(output_path, query_dir, map_dir, provider)

    print(
        f"Running Visual Positioning: {exp_folder_name} (Interval: {sample_interval})"
    )
    os.system(
        f"{sys.executable} -m src.estimate "
        f"--config {config_path} --eval --sample-interval {sample_interval}"
    )


def create_showcase_gif(experiment_dir: Path) -> Optional[Path]:
    """Generates an animated GIF from the frames in the experiment directory.

    Searches for overlay frames in the experiment's timestamped subfolder
    and combines them into an animated GIF.

    Args:
        experiment_dir: Path to the experiment results directory.

    Returns:
        Path to the generated GIF, or None if no frames found.
    """
    frames_dirs = list(experiment_dir.glob("**/frames"))
    if not frames_dirs:
        return None

    frames_dir = max(frames_dirs, key=os.path.getmtime)
    frame_files = sorted(list(frames_dir.glob("overlay_*.png")))
    if not frame_files:
        return None

    gif_path = experiment_dir / "positioning_showcase.gif"
    print(f"  Generating GIF: {gif_path.name} ({len(frame_files)} frames)")

    frames = []
    for f in frame_files:
        img = Image.open(f)
        if img.width > 1280:
            img = img.resize(
                (1280, int(1280 * img.height / img.width)), Image.Resampling.LANCZOS
            )
        frames.append(img)

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
            optimize=True,
        )
        return gif_path

    return None


def get_experiment_metrics(output_dir: str) -> Optional[Dict[str, Any]]:
    """Computes metrics from an output directory.

    Parses positioning results CSV and calculates summary statistics
    including success rate, average error, and timing information.

    Args:
        output_dir: Path to the experiment output directory.

    Returns:
        Metrics dictionary, or None if no results found.
    """
    results_files = glob.glob(
        os.path.join(output_dir, "**", "positioning_results.csv"), recursive=True
    )
    if not results_files:
        return None

    latest_csv = max(results_files, key=os.path.getmtime)
    df = pd.read_csv(latest_csv)
    success_df = df[df["Positioning Success"] == True]

    metrics = {
        "Total": len(df),
        "Success%": f"{(len(success_df) * 100 / len(df)):.1f}%",
    }

    if not success_df.empty:
        metrics.update(
            {
                "AvgErr": f"{success_df['Error (m)'].mean():.2f}m",
                "AvgInliers": f"{success_df['Inliers'].mean():.1f}",
                "AvgTime": f"{df['Best Match Time (s)'].mean():.3f}s",
            }
        )

    return metrics


def evaluate_all_results(region_ids: Optional[List[int]] = None) -> None:
    """Summarizes all results into a CSV and a Markdown report with GIFs.

    Scans the evaluation results directory, extracts metrics from each
    experiment, and generates summary reports.

    Args:
        region_ids: Optional list of region IDs to filter. If None, includes all.
    """
    summary_data = []
    md_lines = [
        "# Visual Positioning Experiment Report\n",
        f"Generated on: {time.ctime()}\n",
    ]

    if not EVAL_RESULTS_ROOT.exists():
        print("No evaluation results found.")
        return

    md_lines.append("## Results Summary\n")

    folders = sorted(glob.glob(str(EVAL_RESULTS_ROOT / "*")))
    if not folders:
        print("No evaluation folders found.")
        return

    md_lines.append(
        "| Region | Provider | Zoom | Success% | Avg Error | Avg Inliers | "
        "Experiment Folder |"
    )
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for folder in folders:
        if not os.path.isdir(folder):
            continue

        folder_path = Path(folder)

        try:
            parts = folder_path.name.split("_")
            rid = None
            provider = "unknown"
            zoom = "N/A"

            for i, p in enumerate(parts):
                if p == "zoom" and i > 0:
                    rid = int(parts[i - 1])
                    if i + 1 < len(parts):
                        zoom = parts[i + 1]
                    if i + 2 < len(parts):
                        provider = parts[i + 2]
                    break

            if region_ids and (rid not in region_ids):
                continue

        except Exception:
            rid = "N/A"
            zoom = "N/A"
            provider = "N/A"

        metrics = get_experiment_metrics(folder)
        if metrics:
            item = {
                "RegionID": rid,
                "Provider": provider,
                "Zoom": zoom,
                "Experiment": folder_path.name,
            }
            item.update(metrics)
            summary_data.append(item)

            md_lines.append(
                f"| {rid} | {provider} | {zoom} | {metrics['Success%']} | "
                f"{metrics.get('AvgErr', 'N/A')} | {metrics.get('AvgInliers', '0')} | "
                f"`{folder_path.name}` |"
            )

            gif_path = create_showcase_gif(folder_path)
            if gif_path:
                rel_gif = gif_path.relative_to(PROJECT_ROOT)
                md_lines.append(
                    f"\n> **Showcase for {folder_path.name}:**\n"
                    f"> ![{folder_path.name}]({rel_gif})\n"
                )

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(str(SUMMARY_CSV), index=False)

        with open(SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"\nReport generated: {SUMMARY_MD}")
        print(f"Summary saved to {SUMMARY_CSV}")

        display_cols = [
            "RegionID",
            "Provider",
            "Zoom",
            "Success%",
            "AvgErr",
            "AvgInliers",
        ]
        print(df[display_cols].to_string(index=False))
    else:
        print("No results found to evaluate.")


def _parse_region_ids(args_input: Optional[List[str]]) -> Optional[List[int]]:
    """Parses region ID arguments into a list of integers.

    Args:
        args_input: List of string arguments from command line.

    Returns:
        List of region IDs, or None if no valid input.
    """
    if args_input is None:
        return None
    if not args_input or "all" in [str(x).lower() for x in args_input]:
        return list(range(1, 12))
    return [int(x) for x in args_input if x.isdigit()]


def main() -> None:
    """Main entry point for the experiment runner.

    Parses command line arguments and executes the appropriate workflow
    stage (prepare, evaluate, or summarize).
    """
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
        help="Step 2: Run primary visual positioning "
        "(Mandatory: --tile-provider, --zoom-levels)",
    )
    parser.add_argument(
        "--eval-summary",
        nargs="*",
        metavar="ID",
        help="Step 3: Generate summary report and GIFs",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        help="Frame sampling interval for visual positioning",
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
    prep_ids = _parse_region_ids(args.dataset_prepare)
    eval_ids = _parse_region_ids(args.dataset_eval)
    summary_ids = _parse_region_ids(args.eval_summary)

    if prep_ids or eval_ids:
        if not args.zoom_levels:
            print("Error: --zoom-levels is mandatory for preparation and evaluation.")
            sys.exit(1)
        if not args.tile_provider:
            print("Error: --tile-provider is mandatory for preparation and evaluation.")
            sys.exit(1)

    if eval_ids and args.sample_interval is None:
        print("Error: --sample-interval is mandatory for evaluation.")
        sys.exit(1)

    if prep_ids:
        for i in prep_ids:
            print(f"\n--- Preparing Region {i} ---")
            prepare_region_data(i, args.zoom_levels, args.tile_provider)

    if eval_ids:
        for i in eval_ids:
            for p in args.tile_provider:
                for z in args.zoom_levels:
                    run_dataset_eval(i, z, p, args.sample_interval)

    if summary_ids is not None:
        evaluate_all_results(summary_ids)

    if not any(
        [args.dataset_prepare, args.dataset_eval, args.eval_summary is not None]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
