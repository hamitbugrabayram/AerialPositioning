"""Image preprocessing utilities for aerial imagery.

This module provides tools for image resizing and perspective warping
to simulate nadir views. It strictly aligns oblique drone images to a
nadir (top-down), north-facing perspective to match satellite map tiles.

Attributes:
    _RESIZE_TARGET (int): The target length for the longest edge of the image.
    _TARGET_YAW (float): Geographic north orientation target (0.0).
    _TARGET_PITCH (float): Nadir (top-down) orientation target (-90.0).
    _TARGET_ROLL (float): Horizontal level target (0.0).

"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_RESIZE_TARGET = 1024
_TARGET_YAW = 0.0
_TARGET_PITCH = -90.0
_TARGET_ROLL = 0.0
_MAX_WARP_DISTORTION = 10


class PreprocessingError(Exception):
    """Custom exception for image preprocessing failures."""


@dataclass
class CameraModel:
    """Camera intrinsic parameters model.

    Attributes:
        focal_length (float): Focal length in millimeters.
        resolution_width (int): Sensor width in pixels.
        resolution_height (int): Sensor height in pixels.
        hfov_deg (float): Horizontal Field of View in degrees.
        hfov_rad (float): Horizontal Field of View in radians.
        aspect_ratio (float): Image aspect ratio (width / height).
        focal_length_px (float): Focal length in pixels.
        principal_point_x (float): Principal point X coordinate in pixels.
        principal_point_y (float): Principal point Y coordinate in pixels.

    """

    focal_length: float
    resolution_width: int
    resolution_height: int
    hfov_deg: float
    hfov_rad: float = field(init=False)
    aspect_ratio: float = field(init=False)
    focal_length_px: float = field(init=False)
    principal_point_x: float = field(init=False)
    principal_point_y: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculates derived camera parameters.

        Raises:
            ValueError: If camera dimensions or field of view are invalid.

        """
        if (
            self.resolution_width <= 0
            or self.resolution_height <= 0
            or self.hfov_deg <= 0
        ):
            raise ValueError("Camera dimensions and HFOV must be positive.")

        self.hfov_rad = math.radians(self.hfov_deg)
        self.aspect_ratio = self.resolution_width / self.resolution_height
        tan_half_fov = math.tan(self.hfov_rad / 2.0)

        if abs(tan_half_fov) < 1e-9:
            raise ValueError("HFOV results in near-zero tan value.")

        self.focal_length_px = (self.resolution_width / 2.0) / tan_half_fov
        self.principal_point_x = self.resolution_width / 2.0
        self.principal_point_y = self.resolution_height / 2.0


def _compute_resize_dimensions(
    width: int, height: int, target_max_dim: int
) -> Tuple[int, int]:
    """Calculates new dimensions maintaining aspect ratio.

    Args:
        width: Original image width.
        height: Original image height.
        target_max_dim: Desired size for the longest edge.

    Returns:
        Tuple of (new_width, new_height).

    """
    if height <= 0 or width <= 0:
        return width, height

    max_dim = max(height, width)
    scale = target_max_dim / float(max_dim)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return max(1, new_width), max(1, new_height)


def _get_intrinsic_matrix(
    camera_model: CameraModel,
    scale: float = 1.0,
    as_tensor: bool = False,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Constructs the 3x3 camera intrinsics matrix K.

    Args:
        camera_model: The camera parameters model.
        scale: Scale factor for the intrinsic matrix.
        as_tensor: Whether to return a PyTorch tensor instead of NumPy array.
        device: The PyTorch device to place the tensor on.

    Returns:
        The 3x3 intrinsic matrix.

    """
    if scale <= 0:
        scale = 1.0

    fx = camera_model.focal_length_px / scale
    fy = camera_model.focal_length_px / scale
    cx = camera_model.principal_point_x / scale
    cy = camera_model.principal_point_y / scale

    k_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    if as_tensor and device is not None and device.type == "cuda":
        return torch.from_numpy(k_matrix).to(device)

    return k_matrix


def _euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    as_tensor: bool = False,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Computes a 3x3 rotation matrix from Euler angles.

    Args:
        roll: Roll angle in degrees.
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.
        as_tensor: Whether to return a PyTorch tensor instead of NumPy array.
        device: The PyTorch device to place the tensor on.

    Returns:
        The 3x3 rotation matrix.

    """
    r_rad = math.radians(float(roll))
    p_rad = math.radians(float(pitch))
    y_rad = math.radians(float(yaw))

    cz = np.array(
        [
            math.cos(y_rad) * math.cos(p_rad),
            math.sin(y_rad) * math.cos(p_rad),
            math.sin(p_rad),
        ]
    )
    cx = np.array([math.sin(y_rad), -math.cos(y_rad), 0.0])
    cy = np.cross(cz, cx)

    rotation_matrix = np.stack([cx, cy, cz], axis=1).astype(np.float32)

    if abs(roll) > 1e-6:
        cos_r = math.cos(r_rad)
        sin_r = math.sin(r_rad)
        rr_matrix = np.array(
            [[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=np.float32
        )
        rotation_matrix = rotation_matrix @ rr_matrix

    if as_tensor and device is not None and device.type == "cuda":
        return torch.from_numpy(rotation_matrix).to(device)

    return rotation_matrix


class QueryPreprocessor:
    """Generic image preprocessing pipeline for satellite alignment.

    This pipeline unconditionally applies a fixed sequence of operations:
    1. Resizing to standard dimensions (`_RESIZE_TARGET`).
    2. Perspective warping to a nadir, north-oriented view (`_TARGET_PITCH`,
       `_TARGET_YAW`, `_TARGET_ROLL`).

    Attributes:
        camera_model (CameraModel): The camera intrinsics used for warping.
        device (torch.device): Compute device for tensor operations.
        use_gpu (bool): Flag indicating if CUDA is available and selected.

    """

    def __init__(
        self,
        camera_model: Optional[CameraModel] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes the generic preprocessor.

        Args:
            camera_model: Optional camera model; required for warping.
            device: Optional specific device string (e.g., "cuda:0").

        """
        self.camera_model = camera_model

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_gpu = self.device.type == "cuda"

    def __call__(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Applies the fixed resize and warp pipeline to the input image.

        Args:
            image: The raw input image array.
            metadata: Query metadata containing gimbal and flight angles.

        Returns:
            The processed, nadir-aligned image array.

        Raises:
            PreprocessingError: If any pipeline step fails.

        """
        try:
            resized = self._apply_resize(image)
            warped = self._apply_warp(resized, metadata)
            return warped
        except Exception as e:
            raise PreprocessingError(
                f"CRITICAL: Preprocessing pipeline failed: {e}"
            ) from e

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Converts an OpenCV BGR/RGB image to a normalized PyTorch tensor.

        Args:
            image: The NumPy image array.

        Returns:
            A normalized float tensor on the target device.

        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts a normalized PyTorch tensor back to an OpenCV BGR image.

        Args:
            tensor: The normalized PyTorch tensor.

        Returns:
            An 8-bit unsigned integer NumPy image array.

        """
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _apply_resize(self, image: np.ndarray) -> np.ndarray:
        """Resizes the image to the standard target size.

        Args:
            image: The input NumPy image.

        Returns:
            The resized NumPy image.

        Raises:
            PreprocessingError: If GPU-based resizing fails.

        """
        height, width = image.shape[:2]
        new_width, new_height = _compute_resize_dimensions(
            width, height, _RESIZE_TARGET
        )

        if (new_width, new_height) == (width, height):
            return image

        if self.use_gpu:
            try:
                tensor = self._to_tensor(image)
                is_upscale = new_width * new_height > width * height
                resized = F.interpolate(
                    tensor,
                    size=(new_height, new_width),
                    mode="bilinear" if is_upscale else "area",
                    align_corners=False if is_upscale else None,
                )
                return self._to_numpy(resized)
            except Exception as e:
                raise PreprocessingError(f"GPU resize failed: {e}") from e

        is_downscale = new_width * new_height < width * height
        interpolation = cv2.INTER_AREA if is_downscale else cv2.INTER_LINEAR
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    def _apply_warp(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Applies perspective warp to simulate a nadir, north-oriented view.

        Uses gimbal angles from metadata to compute the rotation correction.
        When explicit camera intrinsics are available (via ``CameraModel``),
        the intrinsic matrix is built from them.  Otherwise, a reasonable
        intrinsic matrix is estimated directly from the image dimensions
        (principal point at center, focal length = ``max(w, h)``).

        Args:
            image: The resized NumPy image.
            metadata: Metadata containing 'Gimball_Yaw', 'Gimball_Pitch',
                and 'Gimball_Roll'.

        Returns:
            The warped, top-down NumPy image.

        Raises:
            PreprocessingError: If dimensions are invalid or the
                homography computation fails.

        """
        height_orig, width_orig = image.shape[:2]
        if height_orig <= 0 or width_orig <= 0:
            raise PreprocessingError(
                f"Invalid image dimensions: {width_orig}x{height_orig}"
            )

        gimbal_yaw = float(metadata.get("Gimball_Yaw", 0.0))
        gimbal_pitch = float(metadata.get("Gimball_Pitch", -90.0))
        gimbal_roll = float(metadata.get("Gimball_Roll", 0.0))
        current_yaw = gimbal_yaw
        current_pitch = gimbal_pitch
        current_roll = gimbal_roll

        r_current = _euler_to_rotation_matrix(current_roll, current_pitch, current_yaw)
        r_target = _euler_to_rotation_matrix(_TARGET_ROLL, _TARGET_PITCH, _TARGET_YAW)

        if self.camera_model is not None:
            scale_w = width_orig / self.camera_model.resolution_width
            scale_h = height_orig / self.camera_model.resolution_height
            k_matrix = _get_intrinsic_matrix(
                self.camera_model, scale=1.0 / max(scale_w, scale_h)
            )
        else:
            fx = fy = float(max(width_orig, height_orig))
            cx = width_orig / 2.0
            cy = height_orig / 2.0
            k_matrix = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

        try:
            homography = k_matrix @ r_current.T @ r_target @ np.linalg.inv(k_matrix)
            homography = homography.astype(np.float32)
        except np.linalg.LinAlgError as e:
            raise PreprocessingError(
                f"Singular matrix during homography calculation: {e}"
            ) from e

        corners = np.array(
            [[0, 0], [width_orig, 0], [width_orig, height_orig], [0, height_orig]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        try:
            warped_corners = cv2.perspectiveTransform(corners, homography)
            if warped_corners is None:
                raise PreprocessingError("Perspective transformation returned None.")

            x_min, y_min = np.min(warped_corners, axis=0).ravel()
            x_max, y_max = np.max(warped_corners, axis=0).ravel()
            width_new = int(round(x_max - x_min))
            height_new = int(round(y_max - y_min))

            if width_new <= 0 or height_new <= 0:
                raise PreprocessingError(
                    f"Invalid warped dimensions: {width_new}x{height_new}"
                )

            if (
                width_new > width_orig * _MAX_WARP_DISTORTION
                or height_new > height_orig * _MAX_WARP_DISTORTION
            ):
                raise PreprocessingError(
                    f"Extreme distortion detected: {width_new}x{height_new} "
                    f"vs original {width_orig}x{height_orig}"
                )

            translation_matrix = np.array(
                [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32
            )
            homography_translated = translation_matrix @ homography

            return cv2.warpPerspective(
                image,
                homography_translated,
                (width_new, height_new),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        except Exception as e:
            raise PreprocessingError(f"Warp perspective operation failed: {e}") from e
