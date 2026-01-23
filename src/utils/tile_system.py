"""Bing Maps TileSystem utilities for geographic projections and tile retrieval.

This module implements methods used for Bing Maps tile system calculations
and provides a unified interface for downloading satellite imagery.

Reference:
    https://msdn.microsoft.com/en-us/library/bb259689.aspx
"""

import re
from itertools import chain
from math import atan, cos, exp, floor, log, pi, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.providers import get_provider


class TileSystem:
    """Implements methods for the Bing Maps tile system and satellite retrieval.

    This class provides static methods for coordinate conversions between
    geographic (lat/lon), pixel, tile, and quadkey representations.

    Attributes:
        EARTH_RADIUS: Radius of the Earth in meters (WGS-84).
        MIN_LAT: Minimum latitude supported by the projection.
        MAX_LAT: Maximum latitude supported by the projection.
        MIN_LON: Minimum longitude supported by the projection.
        MAX_LON: Maximum longitude supported by the projection.
        MAX_LEVEL: Maximum zoom level supported.
        DEFAULT_PROVIDER: Default satellite imagery provider.
    """

    EARTH_RADIUS = 6378137
    MIN_LAT, MAX_LAT = -85.05112878, 85.05112878
    MIN_LON, MAX_LON = -180.0, 180.0
    MAX_LEVEL = 23
    DEFAULT_PROVIDER = "esri"

    @staticmethod
    def clip(val: float, min_val: float, max_val: float) -> float:
        """Clips a number to be within specified minimum and maximum bounds.

        Args:
            val: The value to clip.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            The clipped value.
        """
        return min(max(val, min_val), max_val)

    @staticmethod
    def map_size(level: int) -> int:
        """Determines the map width and height (in pixels) at a specified level.

        Args:
            level: Zoom level (0-23).

        Returns:
            Map size in pixels (width = height).
        """
        return 256 << level

    @staticmethod
    def ground_resolution(lat: float, level: int) -> float:
        """Determines the ground resolution (meters per pixel) at a latitude.

        Args:
            lat: Latitude in degrees.
            level: Zoom level.

        Returns:
            Ground resolution in meters per pixel.
        """
        lat = TileSystem.clip(lat, TileSystem.MIN_LAT, TileSystem.MAX_LAT)
        return (
            cos(lat * pi / 180)
            * 2
            * pi
            * TileSystem.EARTH_RADIUS
            / TileSystem.map_size(level)
        )

    @staticmethod
    def map_scale(lat: float, level: int, screen_dpi: float) -> float:
        """Determines the map scale at a specified latitude and level.

        Args:
            lat: Latitude in degrees.
            level: Zoom level.
            screen_dpi: Screen resolution in dots per inch.

        Returns:
            Map scale denominator.
        """
        return TileSystem.ground_resolution(lat, level) * screen_dpi / 0.0254

    @staticmethod
    def latlong_to_pixel_xy(lat: float, lon: float, level: int) -> Tuple[int, int]:
        """Converts lat/long WGS-84 coordinates into pixel XY coordinates.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            level: Zoom level.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        lat = TileSystem.clip(lat, TileSystem.MIN_LAT, TileSystem.MAX_LAT)
        lon = TileSystem.clip(lon, TileSystem.MIN_LON, TileSystem.MAX_LON)
        x = (lon + 180) / 360
        sin_lat = sin(lat * pi / 180)
        y = 0.5 - log((1 + sin_lat) / (1 - sin_lat)) / (4 * pi)
        size = TileSystem.map_size(level)
        pixel_x = int(floor(TileSystem.clip(x * size + 0.5, 0, size - 1)))
        pixel_y = int(floor(TileSystem.clip(y * size + 0.5, 0, size - 1)))
        return pixel_x, pixel_y

    @staticmethod
    def pixel_xy_to_latlong(
        pixel_x: int,
        pixel_y: int,
        level: int,
    ) -> Tuple[float, float]:
        """Converts pixel XY coordinates at a specified level into lat/long.

        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.
            level: Zoom level.

        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        size = TileSystem.map_size(level)
        x = TileSystem.clip(pixel_x, 0, size - 1) / size - 0.5
        y = 0.5 - TileSystem.clip(pixel_y, 0, size - 1) / size
        lat = 90 - 360 * atan(exp(-y * 2 * pi)) / pi
        lon = 360 * x
        return lat, lon

    @staticmethod
    def pixel_xy_to_tile_xy(pixel_x: int, pixel_y: int) -> Tuple[int, int]:
        """Converts pixel XY coordinates into tile XY coordinates.

        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.

        Returns:
            Tuple of (tile_x, tile_y).
        """
        return int(floor(pixel_x / 256)), int(floor(pixel_y / 256))

    @staticmethod
    def tile_xy_to_pixel_xy(tile_x: int, tile_y: int) -> Tuple[int, int]:
        """Converts tile XY coordinates into upper-left pixel XY coordinates.

        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.

        Returns:
            Tuple of (pixel_x, pixel_y) for the tile's upper-left corner.
        """
        return tile_x * 256, tile_y * 256

    @staticmethod
    def tile_xy_to_quadkey(tile_x: int, tile_y: int, level: int) -> str:
        """Converts tile XY coordinates into a QuadKey.

        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.
            level: Zoom level.

        Returns:
            QuadKey string.
        """
        tile_x_bits = "{0:0{1}b}".format(tile_x, level)
        tile_y_bits = "{0:0{1}b}".format(tile_y, level)
        quadkey_binary = "".join(chain(*zip(tile_y_bits, tile_x_bits)))
        return "".join([str(int(num, 2)) for num in re.findall("..?", quadkey_binary)])

    @staticmethod
    def quadkey_to_tile_xy(quadkey: str) -> Tuple[int, int]:
        """Converts a QuadKey into tile XY coordinates.

        Args:
            quadkey: QuadKey string.

        Returns:
            Tuple of (tile_x, tile_y).
        """
        quadkey_binary = "".join(["{0:02b}".format(int(num)) for num in quadkey])
        tile_x = int(quadkey_binary[1::2], 2)
        tile_y = int(quadkey_binary[::2], 2)
        return tile_x, tile_y

    @staticmethod
    def get_tile_bounds(
        tile_x: int,
        tile_y: int,
        level: int,
    ) -> Tuple[float, float, float, float]:
        """Retrieves geodetic bounds for a specific tile.

        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.
            level: Zoom level.

        Returns:
            Tuple of (lat1, lon1, lat2, lon2) representing NW and SE corners.
        """
        pixel_x1, pixel_y1 = TileSystem.tile_xy_to_pixel_xy(tile_x, tile_y)
        pixel_x2, pixel_y2 = pixel_x1 + 256, pixel_y1 + 256
        lat1, lon1 = TileSystem.pixel_xy_to_latlong(pixel_x1, pixel_y1, level)
        lat2, lon2 = TileSystem.pixel_xy_to_latlong(pixel_x2, pixel_y2, level)
        return lat1, lon1, lat2, lon2

    @staticmethod
    def download_image(
        quad_key: str,
        provider_name: str = DEFAULT_PROVIDER,
    ) -> Optional[np.ndarray]:
        """Downloads a satellite tile image from the specified provider.

        Args:
            quad_key: QuadKey identifying the tile.
            provider_name: Name of the tile provider.

        Returns:
            Image as numpy array, or None if download failed.
        """
        tile_x, tile_y = TileSystem.quadkey_to_tile_xy(quad_key)
        level = len(quad_key)
        provider = get_provider(provider_name)
        return provider.download_tile(tile_x, tile_y, level)

    @staticmethod
    def download_tiles(
        upper_left_tile: Tuple[int, int],
        lower_right_tile: Tuple[int, int],
        level: int,
        output_dir: str,
        provider_name: str = DEFAULT_PROVIDER,
    ) -> List[Dict[str, Any]]:
        """Downloads a range of tiles and saves them to a directory.

        Args:
            upper_left_tile: Upper-left tile coordinates (x, y).
            lower_right_tile: Lower-right tile coordinates (x, y).
            level: Zoom level.
            output_dir: Directory to save tile images.
            provider_name: Name of the tile provider.

        Returns:
            List of metadata dictionaries for each downloaded tile.
        """
        tiles_metadata = []
        x_range = range(upper_left_tile[0], lower_right_tile[0] + 1)
        y_range = range(upper_left_tile[1], lower_right_tile[1] + 1)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        provider = get_provider(provider_name)

        print(
            f"Downloading {provider_name.upper()} tiles from "
            f"{upper_left_tile} to {lower_right_tile}..."
        )

        for y in y_range:
            for x in x_range:
                filename = f"tile_{provider_name}_{level}_{x}_{y}.jpg"
                file_path = output_dir_path / filename

                if not file_path.exists():
                    try:
                        image = provider.download_tile(x, y, level)
                        if image is not None:
                            if image.shape[:2] == (256, 256):
                                cv2.imwrite(str(file_path), image)
                            else:
                                image = cv2.resize(image, (256, 256))
                                cv2.imwrite(str(file_path), image)
                    except Exception as e:
                        print(f"CRITICAL: Failed to download tile {x},{y} at level {level}: {e}")
                        raise

                lat1, lon1, lat2, lon2 = TileSystem.get_tile_bounds(x, y, level)
                quad_key = TileSystem.tile_xy_to_quadkey(x, y, level)

                tiles_metadata.append(
                    {
                        "Filename": filename,
                        "Top_left_lat": lat1,
                        "Top_left_lon": lon1,
                        "Bottom_right_lat": lat2,
                        "Bottom_right_long": lon2,
                        "TileX": x,
                        "TileY": y,
                        "Level": level,
                        "QuadKey": quad_key,
                        "Provider": provider_name,
                    }
                )

        return tiles_metadata

    @staticmethod
    def retrieve_map_tiles(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        level: int,
        output_dir: str,
        provider_name: str = DEFAULT_PROVIDER,
    ) -> List[Dict[str, Any]]:
        """Retrieves and prepares all satellite tiles covering a geodetic box.

        Args:
            lat1: First corner latitude.
            lon1: First corner longitude.
            lat2: Second corner latitude.
            lon2: Second corner longitude.
            level: Zoom level.
            output_dir: Directory to save tile images.
            provider_name: Name of the tile provider.

        Returns:
            List of metadata dictionaries for each tile.
        """
        print(
            f"Retrieving {provider_name.upper()} tiles for bbox: "
            f"({lat1:.4f}, {lon1:.4f}) - ({lat2:.4f}, {lon2:.4f}) at Level {level}"
        )

        p1_x, p1_y = TileSystem.latlong_to_pixel_xy(lat1, lon1, level)
        p2_x, p2_y = TileSystem.latlong_to_pixel_xy(lat2, lon2, level)
        t1_x, t1_y = TileSystem.pixel_xy_to_tile_xy(p1_x, p1_y)
        t2_x, t2_y = TileSystem.pixel_xy_to_tile_xy(p2_x, p2_y)

        min_tx, max_tx = min(t1_x, t2_x), max(t1_x, t2_x)
        min_ty, max_ty = min(t1_y, t2_y), max(t1_y, t2_y)

        return TileSystem.download_tiles(
            (min_tx, min_ty),
            (max_tx, max_ty),
            level,
            output_dir,
            provider_name,
        )
