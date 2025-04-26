"""Image preprocessing utilities.

This module provides tools for:
- Image resizing with aspect ratio preservation
- Camera intrinsics modeling
- Perspective warping for nadir view simulation

Uses PyTorch and OpenCV for image processing.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()


def find_nearest_90_multiple(angle: float) -> float:
    """Find the nearest multiple of 90 degrees to the given angle.

    This is useful for adaptive yaw targeting where we want to minimize
    the rotation angle needed during perspective warping. By rounding to
    the nearest 90-degree multiple, we avoid extreme rotations that cause
    severe image distortion.

    Args:
        angle: Input angle in degrees (can be any value, will be normalized).

    Returns:
        Nearest multiple of 90 degrees in range [-180, 180).

    Examples:
        >>> find_nearest_90_multiple(45.0)
        0.0
        >>> find_nearest_90_multiple(-179.4)
        -180.0
        >>> find_nearest_90_multiple(95.0)
        90.0
    """
    normalized = ((angle + 180) % 360) - 180

    candidates = [-180.0, -90.0, 0.0, 90.0, 180.0]

    best_candidate = 0.0
    min_diff = float('inf')

    for candidate in candidates:
        diff = abs(normalized - candidate)
        if diff < min_diff:
            min_diff = diff
            best_candidate = candidate

    if best_candidate == 180.0:
        best_candidate = -180.0

    return best_candidate


def compute_resize_dimensions(
    width: int,
    height: int,
    resize_target: Union[int, List[int]]
) -> Tuple[int, int]:
    """Calculate new dimensions based on resize parameters.

    Args:
        width: Original image width.
        height: Original image height.
        resize_target: Target size specification.

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
    elif (isinstance(resize_target, (list, tuple)) and
          len(resize_target) == 1 and resize_target[0] > 0):
        scale = resize_target[0] / max_dim
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
    else:
        new_width, new_height = width, height

    return max(1, new_width), max(1, new_height)


@dataclass
class CameraModel:
    """Camera intrinsic parameters model.

    Attributes:
        focal_length: Focal length in millimeters.
        resolution_width: Sensor width in pixels.
        resolution_height: Sensor height in pixels.
        hfov_deg: Horizontal Field of View in degrees.
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
        """Calculate derived camera parameters."""
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
    scale: float = 1.0,
    as_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Construct the 3x3 camera intrinsics matrix K.

    Args:
        camera_model: CameraModel instance.
        scale: Scaling factor for focal length and principal point.
        as_tensor: Return as torch.Tensor on GPU if True.

    Returns:
        3x3 intrinsic matrix K.
    """
    if scale <= 0:
        scale = 1.0

    fx = camera_model.focal_length_px / scale
    fy = camera_model.focal_length_px / scale
    cx = camera_model.principal_point_x / scale
    cy = camera_model.principal_point_y / scale

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    if as_tensor and USE_GPU:
        return torch.from_numpy(K).to(DEVICE)
    return K


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    as_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Compute 3x3 rotation matrix from Euler angles.
    
    Convention:
    - World: East (X), North (Y), Up (Z)
    - Camera: Right (X), Down (Y), Forward (Z)
    - Yaw (phi): 0=East, 90=North (CCW from East)
    - Pitch (theta): 0=Horizontal, -90=Down
    - Roll (psi): 0=Level, CCW about optical axis
    """
    r_rad = math.radians(float(roll))
    p_rad = math.radians(float(pitch))
    y_rad = math.radians(float(yaw))

    # Optical axis (Z) direction
    # if p=-90, Z=[0,0,-1] (Down)
    # if p=0, y=0, Z=[1,0,0] (East)
    cz = np.array([
        math.cos(y_rad) * math.cos(p_rad),
        math.sin(y_rad) * math.cos(p_rad),
        math.sin(p_rad)
    ])

    # Right axis (X) direction (perpendicular to Z and Up)
    # if y=0, p=0, X=[0,-1,0] (South)
    # if y=90, p=-90, X=[1,0,0] (East)
    cx = np.array([
        math.sin(y_rad),
        -math.cos(y_rad),
        0.0
    ])
    
    # Down axis (Y) direction (Z x X)
    cy = np.cross(cz, cx)

    R = np.stack([cx, cy, cz], axis=1).astype(np.float32)

    # Apply Roll (rotation about Z axis)
    if abs(roll) > 1e-6:
        cos_r = math.cos(r_rad)
        sin_r = math.sin(r_rad)
        Rr = np.array([
            [cos_r, -sin_r, 0],
            [sin_r,  cos_r, 0],
            [0,      0,     1]
        ], dtype=np.float32)
        R = R @ Rr

    if as_tensor and USE_GPU:
        return torch.from_numpy(R).to(DEVICE)
    return R


class QueryPreprocessor:
    """Image preprocessing pipeline using PyTorch and OpenCV."""

    def __init__(
        self,
        processings: Optional[List[str]] = None,
        resize_target: Optional[Union[int, List[int]]] = None,
        camera_model: Optional[CameraModel] = None,
        target_gimbal_yaw: float = 0.0,
        target_gimbal_pitch: float = -90.0,
        target_gimbal_roll: float = 0.0,
        adaptive_yaw: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            processings: List of preprocessing steps to apply.
            resize_target: Target size for resize step.
            camera_model: Camera model for warp step.
            target_gimbal_yaw: Target yaw angle (used when adaptive_yaw=False).
            target_gimbal_pitch: Target pitch angle (typically -90 for nadir).
            target_gimbal_roll: Target roll angle (typically 0).
            adaptive_yaw: If True, automatically choose target yaw as nearest
                90-degree multiple to minimize rotation distortion.
            device: Compute device (cuda/cpu).
        """
        self.resize_target = resize_target
        self.camera_model = camera_model
        self.target_gimbal_yaw = target_gimbal_yaw
        self.target_gimbal_pitch = target_gimbal_pitch
        self.target_gimbal_roll = target_gimbal_roll
        self.adaptive_yaw = adaptive_yaw
        self.processings = processings if processings else []

        if device:
            self.device = torch.device(device)
        else:
            self.device = DEVICE

        self.use_gpu = self.device.type == 'cuda'

        self._step_handlers: Dict[str, Callable] = {
            "resize": self._apply_resize,
            "warp": self._apply_warp,
        }

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image (HWC, BGR, uint8) to GPU tensor (NCHW, RGB, float).

        Args:
            image: Input image as numpy array.

        Returns:
            Normalized tensor on GPU with batch dimension.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        return tensor.unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor (NCHW, RGB, float) to numpy image (HWC, BGR, uint8).

        Args:
            tensor: Input tensor from GPU.

        Returns:
            Denormalized image as numpy array (BGR).
        """
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def __call__(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply preprocessing pipeline to an image.

        Args:
            image: Input image.
            metadata: Image metadata for warping.

        Returns:
            Processed image.
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
                print(f"Warning: Unknown step '{step_name}' skipped.")

        return processed

    def _apply_resize(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply resize transformation using PyTorch F.interpolate."""
        if self.resize_target is None:
            return image

        height, width = image.shape[:2]
        new_width, new_height = compute_resize_dimensions(width, height, self.resize_target)

        if (new_width, new_height) == (width, height):
            return image

        if self.use_gpu:
            try:
                tensor = self._to_tensor(image)
                is_upscale = new_width * new_height > width * height
                resized = F.interpolate(
                    tensor,
                    size=(new_height, new_width),
                    mode='bilinear' if is_upscale else 'area',
                    align_corners=False if is_upscale else None
                )
                return self._to_numpy(resized)
            except Exception as e:
                print(f"GPU resize failed, using CPU: {e}")

        is_downscale = new_width * new_height < width * height
        interpolation = cv2.INTER_AREA if is_downscale else cv2.INTER_LINEAR
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    def _apply_warp(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Apply perspective warp transformation using OpenCV.

        The gimbal provides stabilized pitch and roll angles (world-relative),
        but yaw is relative to the drone body. Therefore:
        - Pitch and Roll: Use gimbal values directly (already stabilized)
        - Yaw: Combine gimbal_yaw + flight_yaw (gimbal yaw is drone-relative)

        When adaptive_yaw is enabled, the target yaw is automatically set to
        the nearest 90-degree multiple of the current yaw, minimizing rotation
        distortion.
        """
        if self.camera_model is None:
            print("Warning: Camera model required for warping.")
            return image

        height_orig, width_orig = image.shape[:2]
        if height_orig <= 0 or width_orig <= 0:
            return image

        gimbal_yaw = float(metadata.get('Gimball_Yaw', 0.0))
        gimbal_pitch = float(metadata.get('Gimball_Pitch', -90.0))
        gimbal_roll = float(metadata.get('Gimball_Roll', 0.0))
        flight_yaw = float(metadata.get('Flight_Yaw', 0.0))

        current_yaw = gimbal_yaw + flight_yaw
        current_pitch = gimbal_pitch
        current_roll = gimbal_roll

        if self.adaptive_yaw:
            target_yaw = find_nearest_90_multiple(current_yaw)
        else:
            target_yaw = self.target_gimbal_yaw

        R_current = euler_to_rotation_matrix(current_roll, current_pitch, current_yaw)
        R_target = euler_to_rotation_matrix(
            self.target_gimbal_roll,
            self.target_gimbal_pitch,
            target_yaw
        )

        scale_w = width_orig / self.camera_model.resolution_width
        scale_h = height_orig / self.camera_model.resolution_height
        K = get_intrinsic_matrix(self.camera_model, scale=1 / max(scale_w, scale_h))

        try:
            # Fix: Correct homography calculation (target to source)
            # x_source = K @ R_current.T @ R_target @ K_inv @ x_target
            H = K @ R_current.T @ R_target @ np.linalg.inv(K)
            H = H.astype(np.float32)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix during homography.")
            return image

        corners = np.array([
            [0, 0], [width_orig, 0],
            [width_orig, height_orig], [0, height_orig]
        ], dtype=np.float32).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, H)
        if warped_corners is None:
            return image

        x_min, y_min = np.min(warped_corners, axis=0).ravel()
        x_max, y_max = np.max(warped_corners, axis=0).ravel()

        width_new = int(round(x_max - x_min))
        height_new = int(round(y_max - y_min))

        if width_new <= 0 or height_new <= 0:
            return image
        if width_new > width_orig * 10 or height_new > height_orig * 10:
            return image

        T = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)

        H_translated = T @ H

        return cv2.warpPerspective(
            image, H_translated, (width_new, height_new),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )


process_resize = compute_resize_dimensions
get_intrinsics = get_intrinsic_matrix
rotation_matrix_from_angles = euler_to_rotation_matrix
QueryProcessor = QueryPreprocessor
