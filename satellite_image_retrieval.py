# -*- coding: utf-8 -*-
"""
Modified from Aerial-Satellite-Imagery-Retrieval/main.py
"""

import math
import cv2
import numpy as np
import requests
import os
from pathlib import Path


def calculatePixelPosition(latitude, longitude, level):
    map_size = 256 * 2**level
    latitude = min(max(latitude, -85.05112878), 85.05112878)
    longitude = min(max(longitude, 0.0), 180.0)
    sin_latitude = math.sin(latitude * math.pi / 180)
    pixel_x = ((longitude + 180) / 360) * map_size
    pixel_y = (
        0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)
    ) * map_size
    pixel_x = min(max(pixel_x, 0), map_size - 1)
    pixel_y = min(max(pixel_y, 0), map_size - 1)
    return (int(pixel_x), int(pixel_y))


def calculateTilePosition(pixel_position):
    tile_x = math.floor(pixel_position[0] / 256.0)
    tile_y = math.floor(pixel_position[1] / 256.0)
    return (int(tile_x), int(tile_y))


def calculateQuadKey(tile_position, level):
    tile_x = tile_position[0]
    tile_y = tile_position[1]
    quad_key = ""
    i = level
    while i > 0:
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quad_key += str(digit)
        i -= 1
    return quad_key


def pixelXY_to_latlong(pixelX, pixelY, level):
    map_size = 256 * 2**level
    x = (pixelX / map_size) - 0.5
    y = 0.5 - (pixelY / map_size)

    latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    longitude = 360 * x

    return latitude, longitude


def get_tile_bounds(tile_x, tile_y, level):
    """
    Returns (top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon) for a given tile.
    """
    pixel_x1 = tile_x * 256
    pixel_y1 = tile_y * 256
    pixel_x2 = (tile_x + 1) * 256
    pixel_y2 = (tile_y + 1) * 256

    lat1, lon1 = pixelXY_to_latlong(pixel_x1, pixel_y1, level)
    lat2, lon2 = pixelXY_to_latlong(pixel_x2, pixel_y2, level)

    return lat1, lon1, lat2, lon2


def downloadImage(quad_key):
    url = "http://h0.ortho.tiles.virtualearth.net/tiles/a" + quad_key + ".jpeg?g=131"
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to download tile {quad_key}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading tile {quad_key}: {e}")
        return None


def download_tiles(upper_left_tile, lower_right_tile, level, output_dir):
    """
    Downloads individual tiles and saves them separately.
    Returns a list of dictionaries with tile metadata.
    """
    tiles_metadata = []
    x_range = range(upper_left_tile[0], lower_right_tile[0] + 1)
    y_range = range(upper_left_tile[1], lower_right_tile[1] + 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tiles from ({upper_left_tile}) to ({lower_right_tile})...")

    for y in y_range:
        for x in x_range:
            quad_key = calculateQuadKey((x, y), level)
            filename = f"tile_{level}_{x}_{y}.jpg"
            file_path = output_dir / filename

            if not file_path.exists():
                image = downloadImage(quad_key)
                if image is not None:
                    if image.shape == (256, 256, 3):
                        cv2.imwrite(str(file_path), image)
                    else:
                        print(
                            f"Warning: Tile {quad_key} has unexpected shape {image.shape}"
                        )
                        image = cv2.resize(image, (256, 256))
                        cv2.imwrite(str(file_path), image)
                else:
                    continue

            lat1, lon1, lat2, lon2 = get_tile_bounds(x, y, level)

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
                }
            )

    return tiles_metadata


def retrieve_map_tiles(lat1, lon1, lat2, lon2, level, output_dir):
    print(
        f"Retrieving map tiles for bbox: ({lat1:.4f}, {lon1:.4f}) - ({lat2:.4f}, {lon2:.4f}) at Level {level}"
    )

    p1 = calculatePixelPosition(lat1, lon1, level)
    p2 = calculatePixelPosition(lat2, lon2, level)

    t1 = calculateTilePosition(p1)
    t2 = calculateTilePosition(p2)

    min_tx = min(t1[0], t2[0])
    max_tx = max(t1[0], t2[0])
    min_ty = min(t1[1], t2[1])
    max_ty = max(t1[1], t2[1])

    return download_tiles((min_tx, min_ty), (max_tx, max_ty), level, output_dir)
