"""Bing Maps TileSystem utilities for geographic projections.
This module implements methods used for Bing Maps tile system calculations,
including Mercator projection, QuadKey generation, and coordinate conversions.
Reference: https://msdn.microsoft.com/en-us/library/bb259689.aspx
"""
import re
from itertools import chain
from math import atan, cos, exp, floor, log, pi, sin
from typing import Tuple

class TileSystem:
    """Implements static methods for the Bing Maps tile system.
    Attributes:
        EARTH_RADIUS (int): Radius of the Earth in meters (WGS-84).
        MIN_LAT (float): Minimum latitude supported by the projection.
        MAX_LAT (float): Maximum latitude supported by the projection.
        MIN_LON (float): Minimum longitude supported by the projection.
        MAX_LON (float): Maximum longitude supported by the projection.
        MAX_LEVEL (int): Maximum zoom level supported.
    """
    EARTH_RADIUS = 6378137
    MIN_LAT, MAX_LAT = -85.05112878, 85.05112878
    MIN_LON, MAX_LON = -180.0, 180.0
    MAX_LEVEL = 23
    @staticmethod
    def clip(val: float, min_val: float, max_val: float) -> float:
        """Clips a number to be within specified minimum and maximum bounds.
        Args:
            val: Value to be clipped.
            min_val: Minimum value bound.
            max_val: Maximum value bound.
        Returns:
            The clipped value.
        """
        return min(max(val, min_val), max_val)
    @staticmethod
    def map_size(level: int) -> int:
        """Determines the map width and height (in pixels) at a specified level.
        Args:
            level: Level of detail, from 1 to 23.
        Returns:
            The map width and height in pixels.
        """
        return 256 << level
    @staticmethod
    def ground_resolution(lat: float, level: int) -> float:
        """Determines the ground resolution (meters per pixel) at a latitude.
        Args:
            lat: Latitude in degrees.
            level: Level of detail, from 1 to 23.
        Returns:
            The ground resolution in meters per pixel.
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
            level: Level of detail, from 1 to 23.
            screen_dpi: Resolution of the screen in dots per inch.
        Returns:
            The map scale, expressed as the denominator N of the ratio 1:N.
        """
        return TileSystem.ground_resolution(lat, level) * screen_dpi / 0.0254
    @staticmethod
    def latlong_to_pixel_xy(lat: float, lon: float, level: int) -> Tuple[int, int]:
        """Converts lat/long WGS-84 coordinates into pixel XY coordinates.
        Args:
            lat: Latitude of the point in degrees.
            lon: Longitude of the point in degrees.
            level: Level of detail, from 1 to 23.
        Returns:
            A tuple of (pixel_x, pixel_y).
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
    def pixel_xy_to_latlong(pixel_x: int, pixel_y: int, level: int) -> Tuple[float, float]:
        """Converts pixel XY coordinates at a specified level into lat/long.
        Args:
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.
            level: Level of detail, from 1 to 23.
        Returns:
            A tuple of (latitude, longitude) in degrees.
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
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.
        Returns:
            A tuple of (tile_x, tile_y).
        """
        return int(floor(pixel_x / 256)), int(floor(pixel_y / 256))
    @staticmethod
    def tile_xy_to_pixel_xy(tile_x: int, tile_y: int) -> Tuple[int, int]:
        """Converts tile XY coordinates into upper-left pixel XY coordinates.
        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.
        Returns:
            A tuple of (pixel_x, pixel_y).
        """
        return tile_x * 256, tile_y * 256
    @staticmethod
    def tile_xy_to_quadkey(tile_x: int, tile_y: int, level: int) -> str:
        """Converts tile XY coordinates into a QuadKey.
        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.
            level: Level of detail.
        Returns:
            The QuadKey string.
        """
        tile_x_bits = "{0:0{1}b}".format(tile_x, level)
        tile_y_bits = "{0:0{1}b}".format(tile_y, level)
        quadkey_binary = "".join(chain(*zip(tile_y_bits, tile_x_bits)))
        return "".join([str(int(num, 2)) for num in re.findall("..?", quadkey_binary)])
    @staticmethod
    def quadkey_to_tile_xy(quadkey: str) -> Tuple[int, int]:
        """Converts a QuadKey into tile XY coordinates.
        Args:
            quadkey: QuadKey string of the tile.
        Returns:
            A tuple of (tile_x, tile_y).
        """
        quadkey_binary = "".join(["{0:02b}".format(int(num)) for num in quadkey])
        tile_x = int(quadkey_binary[1::2], 2)
        tile_y = int(quadkey_binary[::2], 2)
        return tile_x, tile_y
