"""Dataset management utilities for aerial visual positioning.

This module provides a DatasetManager class to handle the organization,
preparation, and configuration of drone and satellite imagery datasets.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import yaml

from src.utils.tile_system import TileSystem


class DatasetManager:
    """Manages central dataset storage and experiment configuration."""

    def __init__(
        self,
        project_root: Path,
        raw_dataset_root: Path,
        central_datasets_root: Path,
        base_config_path: Path,
    ):
        """Initializes the DatasetManager.

        Args:
            project_root: Root directory of the project.
            raw_dataset_root: Path to the raw aerial positioning dataset.
            central_datasets_root: Path where processed datasets are stored.
            base_config_path: Path to the base template config.yaml.
        """
        self.project_root = project_root
        self.raw_root = raw_dataset_root
        self.central_root = central_datasets_root
        self.base_config_path = base_config_path

    def load_base_config(self) -> Optional[Dict[str, Any]]:
        """Loads the base configuration template."""
        if not self.base_config_path.exists():
            return None
        with open(self.base_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_config(self, config: Dict[str, Any], path: Path) -> None:
        """Saves a configuration dictionary to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_region_name(self, index_str: str) -> str:
        """Retrieves a sanitized region name from the dataset mapping."""
        csv_path = self.raw_root / "satellite_ coordinates_range.csv"
        region_name = f"Region_{index_str}"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            map_name = f"satellite{index_str}.tif"
            row = df[df["mapname"] == map_name]
            if not row.empty:
                region_name = str(row.iloc[0]["region"])

        return region_name.replace(" ", "_").replace("-", "_")

    def get_dataset_dir(self, index: int) -> Path:
        """Returns the central dataset directory for a region."""
        region_id = f"{index:02d}"
        region_name = self.get_region_name(region_id)
        return self.central_root / f"{region_id}_{region_name}"

    def _get_telemetry_value(
        self, row: pd.Series, key: str, default: float = 0.0
    ) -> float:
        """Safely extracts a numeric value from a pandas row.

        Args:
            row: Pandas Series row.
            key: Column key to extract.
            default: Default value if missing or NaN.

        Returns:
            Extracted float value.
        """
        val = row.get(key)
        if pd.api.types.is_scalar(val):
            if val is None or pd.isna(val):
                return float(default)
            return float(val)
        return float(default)

    def prepare_shared_dataset(self, index: int) -> Path:
        """Ensures query images and base metadata are in the central datasets folder."""
        region_id = f"{index:02d}"
        dataset_dir = self.get_dataset_dir(index)
        query_dir = dataset_dir / "query"

        raw_region_dir = self.raw_root / region_id
        drone_img_dir = raw_region_dir / "drone"
        csv_path = raw_region_dir / f"{region_id}.csv"

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
                            "Latitude": self._get_telemetry_value(row, "lat"),
                            "Longitude": self._get_telemetry_value(row, "lon"),
                            "Altitude": self._get_telemetry_value(row, "height"),
                            "Gimball_Roll": self._get_telemetry_value(row, "Kappa"),
                            "Gimball_Pitch": -90.0 + self._get_telemetry_value(row, "Omega"),
                            "Gimball_Yaw": 0.0,
                            "Flight_Roll": 0.0,
                            "Flight_Pitch": 0.0,
                            "Flight_Yaw": self._get_telemetry_value(row, "Phi1"),
                        }
                    )

                pd.DataFrame(query_metadata).to_csv(
                    str(query_dir / "photo_metadata.csv"), index=False
                )

        return dataset_dir

    def generate_region_config(
        self,
        output_path: Path,
        query_dir: Path,
        map_dir: Path,
        provider_name: Optional[str] = None,
    ) -> Path:
        """Generates a positioning config.yaml inside the results folder."""
        base_config = self.load_base_config()
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

        sample_images = list(query_dir.glob("*.[jJ][pP][gG]"))
        if not sample_images:
            raise FileNotFoundError(f"No images found in {query_dir} to detect resolution.")

        img = cv2.imread(str(sample_images[0]))
        if img is None:
            raise RuntimeError(
                f"Failed to read sample image for resolution detection: {sample_images[0]}"
            )

        h, w = img.shape[:2]
        if "camera_model" not in config:
            config["camera_model"] = {}
        config["camera_model"]["resolution_width"] = int(w)
        config["camera_model"]["resolution_height"] = int(h)

        if int(w) == 3000:
            config["camera_model"]["focal_length"] = 4.5
            config["camera_model"]["hfov_deg"] = 72.0

        os.makedirs(str(output_path), exist_ok=True)
        config_file = output_path / "config.yaml"
        self.save_config(config, config_file)
        return config_file

    def prepare_region_data(
        self, index: int, zoom_levels: List[int], provider_names: List[str]
    ) -> None:
        """Prepares map tiles for a region across multiple zooms and providers."""
        dataset_dir = self.prepare_shared_dataset(index)
        region_id = f"{index:02d}"
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
