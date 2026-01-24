"""Runner script for automating satellite-aided UAV localization experiments.

This script manages the end-to-end pipeline:
1. Dataset preparation (sampling images, fetching satellite tiles).
2. Configuration generation for specific regions and zoom levels.
3. Execution of the localization engine.
4. Evaluation and compilation of results across all experiments.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

import time
from src.utils.satellite_retrieval import retrieve_map_tiles

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "_VisLoc_dataset"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
CONSECUTIVE_ROOT = EXPERIMENTS_ROOT / "consecutive"
BASE_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SUMMARY_CSV = PROJECT_ROOT / "experiments_summary.csv"


def load_base_config() -> Optional[Dict[str, Any]]:
    """Loads the base configuration template from the project root.

    Returns:
        The configuration dictionary or None if the file is missing.
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
        path: Destination path for the YAML file.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_region_name(index_str: str) -> str:
    """Retrieves the human-readable region name from the dataset mapping.

    Args:
        index_str: Two-digit string ID of the region.

    Returns:
        The region name or a default "Region_XX" string.
    """
    csv_path = DATASET_ROOT / "satellite_ coordinates_range.csv"
    if not os.path.isfile(str(csv_path)):
        return f"Region_{index_str}"

    df = pd.read_csv(csv_path)
    map_name = f"satellite{index_str}.tif"
    row = df[df["mapname"] == map_name]

    if not row.empty:
        return str(row.iloc[0]["region"])
    return f"Region_{index_str}"


def generate_region_config(
    exp_path: Path,
    query_dir: Path,
    map_dir: Path,
    output_dir: Path,
    region_drone_dir: Path,
) -> None:
    """Generates a localized config.yaml for a specific experiment.

    Adjusts camera parameters based on detected image resolution.

    Args:
        exp_path: Path to the experiment folder.
        query_dir: Path to sampled query images.
        map_dir: Path to downloaded satellite tiles.
        output_dir: Path where results should be saved.
        region_drone_dir: Path to the original drone images for resolution check.
    """
    base_config = load_base_config()
    if base_config is None:
        return

    config = yaml.safe_load(yaml.dump(base_config))
    config["data_paths"] = {
        "query_dir": str(query_dir.resolve()),
        "map_dir": str(map_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "query_metadata": str((query_dir / "photo_metadata.csv").resolve()),
        "map_metadata": str((map_dir / "map.csv").resolve()),
    }

    try:
        sample_images = list(region_drone_dir.glob("*.[jJ][pP][gG]"))
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
                    print(f"  Adjusted camera model for 3000x2000 resolution.")

                print(f"  Detected resolution: {w}x{h} for {exp_path.name}")
    except Exception as e:
        print(f"  Warning: Failed to auto-detect resolution for {exp_path.name}: {e}")

    save_config(config, exp_path / "config.yaml")
    print(f"Config generated for {exp_path.name}")


def _get_telemetry_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely extracts a numeric value from a pandas row.

    Args:
        row: The pandas Series containing data.
        key: The column name to extract.
        default: Default value if key is missing or NaN.

    Returns:
        Extracted float value.
    """
    val = row.get(key)
    if pd.isna(val) or val is None:
        return float(default)
    return float(val)


def prepare_region_data(index: int, max_queries: int, zoom_levels: List[int]) -> None:
    """Prepares data for a specific region by sampling queries and fetching tiles.

    Args:
        index: Numeric ID of the region (1 to 11).
        max_queries: Maximum number of query images to sample.
        zoom_levels: List of zoom levels to prepare.
    """
    region_id = f"{index:02d}"
    raw_name = get_region_name(region_id)
    region_name = raw_name.replace(" ", "_").replace("-", "_")
    region_dir = DATASET_ROOT / region_id
    csv_path = region_dir / f"{region_id}.csv"
    drone_img_dir = region_dir / "drone"

    if not os.path.isfile(str(csv_path)):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    num_samples = min(len(df), max_queries)
    subset = df.sample(n=num_samples, random_state=42).copy()

    for zoom in zoom_levels:
        exp_name = f"{region_name}_{region_id}_zoom_{zoom}"
        exp_path = EXPERIMENTS_ROOT / exp_name
        query_dir = exp_path / "data/query"
        map_dir = exp_path / "data/map"
        output_dir = exp_path / "data/output"

        os.makedirs(str(query_dir), exist_ok=True)
        os.makedirs(str(map_dir), exist_ok=True)
        os.makedirs(str(output_dir), exist_ok=True)

        target_metadata = query_dir / "photo_metadata.csv"
        if not os.path.isfile(str(target_metadata)):
            query_metadata = []
            for _, row in subset.iterrows():
                filename = str(row["filename"])
                src_img = drone_img_dir / filename
                dst_img = query_dir / filename

                if src_img.exists() and not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

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
            pd.DataFrame(query_metadata).to_csv(str(target_metadata), index=False)
            print(f"Query metadata created for {exp_name}")

        lats = subset["lat"].tolist()
        lons = subset["lon"].tolist()
        margin = 0.008
        lat_max, lon_min = max(lats) + margin, min(lons) - margin
        lat_min, lon_max = min(lats) - margin, max(lons) + margin

        tiles = retrieve_map_tiles(lat_max, lon_min, lat_min, lon_max, zoom, str(map_dir))
        if isinstance(tiles, list) and len(tiles) > 0:
            pd.DataFrame(tiles).to_csv(str(map_dir / "map.csv"), index=False)
            print(f"Map data prepared/verified for {exp_name}")

        generate_region_config(exp_path, query_dir, map_dir, output_dir, drone_img_dir)


def run_localization(index: int, zoom_levels: List[int]) -> None:
    """Runs the localization pipeline for all zooms in a region.

    Args:
        index: Numeric ID of the region (1 to 11).
        zoom_levels: List of zoom levels to run.
    """
    region_id = f"{index:02d}"
    region_name = get_region_name(region_id).replace(" ", "_").replace("-", "_")

    for zoom in zoom_levels:
        exp_name = f"{region_name}_{region_id}_zoom_{zoom}"
        exp_path = EXPERIMENTS_ROOT / exp_name
        config_path = exp_path / "config.yaml"

        if not os.path.isfile(str(config_path)):
            print(f"Config not found for {exp_name}, skipping.")
            continue

        print(f"Running localization for {exp_name}...")
        localize_script = PROJECT_ROOT / "localize.py"
        cmd = f"{sys.executable} {localize_script} --config {config_path}"
        os.system(cmd)


def prepare_consecutive_data(index: int, zoom: int) -> Optional[Path]:
    """Prepares all images for a region for consecutive localization testing.

    Args:
        index: Numeric ID of the region (1 to 11).
        zoom: Zoom level to prepare.

    Returns:
        Path to the generated experiment directory.
    """
    region_id = f"{index:02d}"
    raw_name = get_region_name(region_id)
    region_name = raw_name.replace(" ", "_").replace("-", "_")
    region_dir = DATASET_ROOT / region_id
    csv_path = region_dir / f"{region_id}.csv"
    drone_img_dir = region_dir / "drone"

    if not os.path.isfile(str(csv_path)):
        print(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="filename")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{region_name}_{region_id}_zoom_{zoom}_{timestamp}"
    exp_path = CONSECUTIVE_ROOT / exp_name
    query_dir = exp_path / "data/query"
    map_dir = exp_path / "data/map"
    output_dir = exp_path / "data/output"

    os.makedirs(str(query_dir), exist_ok=True)
    os.makedirs(str(map_dir), exist_ok=True)
    os.makedirs(str(output_dir), exist_ok=True)

    map_source_found = False
    
    print(f"Searching for existing maps in experiments/ for Region {region_id} Zoom {zoom}...")
    exp_matches = glob.glob(str(EXPERIMENTS_ROOT / f"*_{region_id}_zoom_{zoom}"))
    if exp_matches:
        source_map_dir = Path(exp_matches[0]) / "data/map"
        if (source_map_dir / "map.csv").exists():
            print(f"  Found maps in experiment: {exp_matches[0]}. Copying...")
            for item in source_map_dir.iterdir():
                if item.is_file(): shutil.copy2(item, map_dir / item.name)
            map_source_found = True

    if not map_source_found:
        cache_dir = DATASET_ROOT / "consecutive_maps" / f"region_{region_id}_zoom_{zoom}"
        if cache_dir.exists() and (cache_dir / "map.csv").exists():
            print(f"  Found maps in cache. Copying...")
            for item in cache_dir.iterdir():
                if item.is_file(): shutil.copy2(item, map_dir / item.name)
            map_source_found = True

    if not map_source_found:
        lats = df["lat"].tolist()
        lons = df["lon"].tolist()
        margin = 0.01
        lat_max, lon_min = max(lats) + margin, min(lons) - margin
        lat_min, lon_max = min(lats) - margin, max(lons) + margin

        print(f"  No existing maps found. Fetching from satellite source...")
        tiles = retrieve_map_tiles(lat_max, lon_min, lat_min, lon_max, zoom, str(map_dir))
        if isinstance(tiles, list) and len(tiles) > 0:
            pd.DataFrame(tiles).to_csv(str(map_dir / "map.csv"), index=False)

            cache_dir = DATASET_ROOT / "consecutive_maps" / f"region_{region_id}_zoom_{zoom}"
            os.makedirs(str(cache_dir), exist_ok=True)
            for item in map_dir.iterdir():
                if item.is_file(): shutil.copy2(item, cache_dir / item.name)
            print(f"  Map data downloaded and cached.")

    query_metadata = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        src_img = drone_img_dir / filename
        dst_img = query_dir / filename

        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

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

    pd.DataFrame(query_metadata).to_csv(str(query_dir / "photo_metadata.csv"), index=False)
    generate_region_config(exp_path, query_dir, map_dir, output_dir, drone_img_dir)
    return exp_path


def run_consecutive_test(index: int) -> None:
    """Executes the consecutive localization test for a region.

    Args:
        index: Numeric ID of the region (1 to 11).
    """
    exp_dirs = glob.glob(str(CONSECUTIVE_ROOT / "*"))
    if not exp_dirs:
        print("No prepared localization experiments found.")
        return

    region_id = f"{index:02d}"
    relevant_dirs = [d for d in exp_dirs if f"_{region_id}_zoom_" in d]
    if not relevant_dirs:
        print(f"No prepared localization data for region {region_id}")
        return

    latest_exp = max(relevant_dirs, key=os.path.getmtime)
    config_path = Path(latest_exp) / "config.yaml"

    print(f"Running consecutive localization test: {latest_exp}")
    
    localize_script = PROJECT_ROOT / "localize.py"
    cmd = f"{sys.executable} {localize_script} --config {config_path} --consecutive"
    os.system(cmd)


def get_experiment_metrics(output_dir: str) -> Optional[Dict[str, Any]]:
    """Computes summary metrics from the latest experiment output.

    Args:
        output_dir: Path to the experiment output directory.

    Returns:
        Dictionary of metrics or None if results are missing.
    """
    pattern = os.path.join(output_dir, "*", "localization_results.csv")
    results_files = glob.glob(pattern)

    if not results_files:
        return None

    latest_csv = max(results_files, key=os.path.getmtime)
    df = pd.read_csv(latest_csv)

    total_queries = len(df)

    success_mask = df["Localization Success"] == True
    success_df = df[success_mask].copy()
    num_success = len(success_df)
    success_rate = (float(num_success) * 100.0 / total_queries) if total_queries > 0 else 0.0

    if success_df.empty:
        return {
            "Total Queries": int(total_queries),
            "Success Rate (%)": f"{success_rate:.1f}%",
            "Avg Error (m)": "N/A",
            "Min Error (m)": "N/A",
            "Max Error (m)": "N/A",
            "Median Error (m)": "N/A",
            "Avg Inliers": 0.0,
            "Max Inliers": 0,
            "Min Inliers": 0,
            "Avg Match Time (s)": (
                f"{df['Best Match Time (s)'].mean():.3f}s" if not df.empty else "N/A"
            ),
        }

    errors = [float(x) for x in success_df["Error (m)"].tolist()]
    inliers = [float(x) for x in success_df["Inliers"].tolist()]
    times = [float(x) for x in success_df["Best Match Time (s)"].tolist()]

    return {
        "Total Queries": int(total_queries),
        "Success Rate (%)": f"{success_rate:.1f}%",
        "Avg Error (m)": f"{np.mean(errors):.2f}",
        "Min Error (m)": f"{np.min(errors):.2f}",
        "Max Error (m)": f"{np.max(errors):.2f}",
        "Median Error (m)": f"{np.median(errors):.2f}",
        "Avg Inliers": f"{np.mean(inliers):.1f}",
        "Max Inliers": int(np.max(inliers)),
        "Min Inliers": int(np.min(inliers)),
        "Avg Match Time (s)": f"{np.mean(times):.3f}",
    }


def evaluate_all_results() -> None:
    """Gathers results from all experiment folders and saves a summary CSV."""
    summary_list = []
    exp_dirs = glob.glob(os.path.join(str(EXPERIMENTS_ROOT), "*"))

    for exp_path_str in exp_dirs:
        exp_path = Path(exp_path_str)
        if not os.path.isdir(str(exp_path)):
            continue

        metrics = get_experiment_metrics(str(exp_path / "data/output"))
        if metrics:
            parts = exp_path.name.split("_")
            region = "_".join(parts[:-3])
            zoom = parts[-1]
            summary_item = {
                "Experiment": exp_path.name,
                "Region": region,
                "Zoom": zoom,
            }
            summary_item.update(metrics)
            summary_list.append(summary_item)

    if summary_list:
        df = pd.DataFrame(summary_list)
        df.to_csv(str(SUMMARY_CSV), index=False)
        print(f"\nEvaluation complete. Summary saved to {SUMMARY_CSV}")
        print(df.to_string(index=False))
    else:
        print("No results found to evaluate.")


def main() -> None:
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Runner for Satellite Localization")
    parser.add_argument(
        "--dataset-prepare", action="store_true", help="Step 1: Prep data and configs"
    )
    parser.add_argument("--run", action="store_true", help="Step 2: Run localization")
    parser.add_argument("--eval", action="store_true", help="Step 3: Evaluate results")
    parser.add_argument(
        "--consecutive-test",
        type=int,
        metavar="REGION_ID",
        help="Run consecutive localization test for a region (1-11)",
    )
    
    parser.add_argument("--max-queries", type=int, default=20, help="Max images to sample per region (Scenario 1)")
    parser.add_argument("--zoom-levels", type=int, nargs="+", default=[15, 16, 17, 18], help="Zoom levels to test (Scenario 1)")
    parser.add_argument("--consecutive-zoom", type=int, default=17, help="Zoom level for consecutive test (Scenario 2)")
    
    args = parser.parse_args()

    if not any([args.dataset_prepare, args.run, args.eval, args.consecutive_test]):
        parser.print_help()
        sys.exit(0)

    for i in range(1, 12):
        if args.dataset_prepare:
            print(f"\n{'='*20} Preparing Dataset: Region {i} {'='*20}")
            prepare_region_data(i, args.max_queries, args.zoom_levels)
        if args.run:
            print(f"\n{'='*20} Running Localization: Region {i} {'='*20}")
            run_localization(i, args.zoom_levels)

    if args.eval:
        print(f"\n{'='*20} Evaluating All Results {'='*20}")
        evaluate_all_results()

    if args.consecutive_test:
        region_id = args.consecutive_test
        print(f"\n{'='*20} Consecutive Localization Test: Region {region_id} {'='*20}")
        prepare_consecutive_data(region_id, args.consecutive_zoom)
        run_consecutive_test(region_id)


if __name__ == "__main__":
    main()
