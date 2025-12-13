"""
Image preprocessing utilities for query image transformation.

This module provides tools for:
- Image resizing with aspect ratio preservation
- Camera intrinsics modeling
- Perspective warping for nadir view simulation
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def compute_resize_dimensions(
    width: int,
    height: int,
    resize_target: Union[int, List[int]]
) -> Tuple[int, int]:
    """
    Calculate new dimensions based on resize parameters.

    Args:
        width: Original image width.
        height: Original image height.
        resize_target: Target size specification:
            - int > 0: Maximum dimension constraint
            - [W, H]: Exact target dimensions
            - [S]: Maximum dimension constraint
            - -1 or invalid: Keep original size

    Returns:
        Tuple of (new_width, new_height).
    """
    if height <= 0 or width <= 0:
        return width, height

    max_dim = max(height, width)

    if isinstance(resize_target, int) and resize_target > 0:
        scale = resize_target / max_dim
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
    elif isinstance(resize_target, (list, tuple)) and len(resize_target) == 2:
        new_width, new_height = int(resize_target[0]), int(resize_target[1])
    elif isinstance(resize_target, (list, tuple)) and len(resize_target) == 1 and resize_target[0] > 0:
        scale = resize_target[0] / max_dim
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
    else:
        new_width, new_height = width, height

    # Ensure minimum dimension of 1
    return max(1, new_width), max(1, new_height)


@dataclass
class CameraModel:
    """
    Camera intrinsic parameters model.

    Calculates focal length in pixels from horizontal field of view (HFOV)
    and assumes principal point at image center.

    Attributes:
        focal_length: Focal length in millimeters.
        resolution_width: Sensor/image width in pixels.
        resolution_height: Sensor/image height in pixels.
        hfov_deg: Horizontal field of view in degrees.
    """

    focal_length: float
    resolution_width: int
    resolution_height: int
    hfov_deg: float

    # Computed fields
    hfov_rad: float = field(init=False)
    aspect_ratio: float = field(init=False)
    focal_length_px: float = field(init=False)
    principal_point_x: float = field(init=False)
    principal_point_y: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived camera parameters after initialization."""
        if self.resolution_width <= 0 or self.resolution_height <= 0 or self.hfov_deg <= 0:
            raise ValueError("Camera dimensions and HFOV must be positive.")

        self.hfov_rad = math.radians(self.hfov_deg)
        self.aspect_ratio = self.resolution_width / self.resolution_height

        tan_half_fov = math.tan(self.hfov_rad / 2.0)
        if abs(tan_half_fov) < 1e-9:
            raise ValueError("HFOV results in near-zero tan value.")

        self.focal_length_px = (self.resolution_width / 2.0) / tan_half_fov
        self.principal_point_x = self.resolution_width / 2.0
        self.principal_point_y = self.resolution_height / 2.0


def get_intrinsic_matrix(
    camera_model: CameraModel,
    scale: float = 1.0
) -> np.ndarray:
    """
    Construct the 3x3 camera intrinsics matrix K.

    Args:
        camera_model: CameraModel instance with camera parameters.
        scale: Optional scaling factor for focal length and principal point.

    Returns:
        3x3 numpy array representing the intrinsic matrix K.
    """
    if scale <= 0:
        scale = 1.0

    fx = camera_model.focal_length_px / scale
    fy = camera_model.focal_length_px / scale
    cx = camera_model.principal_point_x / scale
    cy = camera_model.principal_point_y / scale

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float
) -> np.ndarray:
    """
    Compute 3x3 rotation matrix from Euler angles.

    Uses 'xyz' extrinsic Euler angle convention (roll-pitch-yaw).

    Args:
        roll: Rotation about x-axis in degrees.
        pitch: Rotation about y-axis in degrees.
        yaw: Rotation about z-axis in degrees.

    Returns:
        3x3 rotation matrix as numpy array.
    """
    try:
        roll, pitch, yaw = float(roll), float(pitch), float(yaw)
        rotation = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
        return rotation.as_matrix().astype(np.float32)
    except (TypeError, ValueError) as e:
        print(f"Error calculating rotation matrix: {e}")
        return np.identity(3, dtype=np.float32)


class QueryPreprocessor:
    """
    Image preprocessing pipeline for query images.

    Applies configurable preprocessing steps (resize, warp) to transform
    drone/UAV images for improved matching with satellite imagery.

    Attributes:
        steps: List of processing step names to apply.
        resize_target: Target size for resize operation.
        camera_model: Camera model for perspective warping.
        target_gimbal_yaw: Target yaw angle in degrees.
        target_gimbal_pitch: Target pitch angle in degrees (-90 = nadir).
        target_gimbal_roll: Target roll angle in degrees.
    """

    def __init__(
        self,
        processings: Optional[List[str]] = None,
        resize_target: Optional[Union[int, List[int]]] = None,
        camera_model: Optional[CameraModel] = None,
        target_gimbal_yaw: float = 0.0,
        target_gimbal_pitch: float = -90.0,
        target_gimbal_roll: float = 0.0,
    ) -> None:
        """
        Initialize the query preprocessor.

        Args:
            processings: List of processing steps ('resize', 'warp').
            resize_target: Resize target specification.
            camera_model: Camera model (required for 'warp' step).
            target_gimbal_yaw: Target yaw for warping.
            target_gimbal_pitch: Target pitch for warping.
            target_gimbal_roll: Target roll for warping.
        """
        self.resize_target = resize_target
        self.camera_model = camera_model
        self.target_gimbal_yaw = target_gimbal_yaw
        self.target_gimbal_pitch = target_gimbal_pitch
        self.target_gimbal_roll = target_gimbal_roll
        self.processings = processings if processings else []

        # Map step names to methods
        self._step_handlers: Dict[str, Callable] = {
            "resize": self._apply_resize,
            "warp": self._apply_warp,
        }

    def __call__(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply preprocessing pipeline to an image.

        Args:
            image: Input image as numpy array (BGR or grayscale).
            metadata: Image metadata dictionary containing orientation angles.

        Returns:
            Processed image. Returns original on processing failure.
        """
        processed = image.copy()

        for step_name in self.processings:
            if step_name in self._step_handlers:
                try:
                    processed = self._step_handlers[step_name](processed, metadata)
                except Exception as e:
                    print(f"Warning: Preprocessing step '{step_name}' failed: {e}")
                    return image
            else:
                print(f"Warning: Unknown preprocessing step '{step_name}' skipped.")

        return processed

    def _apply_resize(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply resize transformation to image."""
        if self.resize_target is None:
            return image

        height, width = image.shape[:2]
        new_width, new_height = compute_resize_dimensions(width, height, self.resize_target)

        if (new_width, new_height) == (width, height):
            return image

        interpolation = cv2.INTER_AREA if (new_width * new_height < width * height) else cv2.INTER_LINEAR

        try:
            return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        except cv2.error as e:
            print(f"Error during resize: {e}")
            return image

    def _apply_warp(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply perspective warp transformation for nadir view simulation."""
        if self.camera_model is None:
            print("Warning: Camera model required for warping. Skipping.")
            return image

        height_orig, width_orig = image.shape[:2]
        if height_orig <= 0 or width_orig <= 0:
            print("Warning: Invalid image dimensions for warp.")
            return image

        # Extract orientation from metadata
        gimbal_yaw = float(metadata.get('Gimball_Yaw', 0.0))
        gimbal_pitch = float(metadata.get('Gimball_Pitch', -90.0))
        gimbal_roll = float(metadata.get('Gimball_Roll', 0.0))
        flight_yaw = float(metadata.get('Flight_Yaw', 0.0))

        # Compute current and target rotations
        current_yaw = gimbal_yaw + flight_yaw
        R_current = euler_to_rotation_matrix(gimbal_roll, gimbal_pitch, current_yaw)
        R_target = euler_to_rotation_matrix(
            self.target_gimbal_roll,
            self.target_gimbal_pitch,
            self.target_gimbal_yaw
        )

        # Compute homography from rotation
        scale_w = width_orig / self.camera_model.resolution_width
        scale_h = height_orig / self.camera_model.resolution_height
        K = get_intrinsic_matrix(self.camera_model, scale=1 / max(scale_w, scale_h))

        try:
            H = K @ R_target @ R_current.T @ np.linalg.inv(K)
            H = H.astype(np.float32)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix during homography calculation.")
            return image

        # Compute warped image bounds
        corners = np.array([
            [0, 0], [width_orig, 0],
            [width_orig, height_orig], [0, height_orig]
        ], dtype=np.float32).reshape(-1, 1, 2)

        try:
            warped_corners = cv2.perspectiveTransform(corners, H)
            if warped_corners is None:
                raise ValueError("perspectiveTransform returned None")
        except cv2.error as e:
            print(f"Warning: perspectiveTransform failed: {e}")
            return image

        x_min, y_min = np.min(warped_corners, axis=0).ravel()
        x_max, y_max = np.max(warped_corners, axis=0).ravel()

        width_new = int(round(x_max - x_min))
        height_new = int(round(y_max - y_min))

        # Validate output dimensions
        if width_new <= 0 or height_new <= 0:
            print(f"Warning: Invalid warped dimensions ({width_new}x{height_new}).")
            return image

        if width_new > width_orig * 10 or height_new > height_orig * 10:
            print(f"Warning: Warped dimensions too large ({width_new}x{height_new}).")
            return image

        # Apply translation to keep image in positive coordinates
        T = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)

        H_translated = T @ H

        try:
            return cv2.warpPerspective(
                image, H_translated, (width_new, height_new),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        except cv2.error as e:
            print(f"Error during warpPerspective: {e}")
            return image


# Backward compatibility aliases
process_resize = compute_resize_dimensions
get_intrinsics = get_intrinsic_matrix
rotation_matrix_from_angles = euler_to_rotation_matrix
QueryProcessor = QueryPreprocessor