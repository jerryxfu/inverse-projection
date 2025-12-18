import cv2
import numpy as np
from typing import Tuple, List

"""
Camera space coordinate system:
- X: Right
- Y: Down
- Z: Forward
"""

# Configuration flags
PREVIEW_MODE = False


class CameraConfig:
    """Camera physical and intrinsic parameters"""
    HEIGHT_M = 0.2025  # Camera height above ground (meters)
    PITCH_DEG = 0  # Camera downward pitch (degrees)
    OFFSET_X_M = -0.2286  # Camera X offset in robot frame (forward)
    OFFSET_Y_M = HEIGHT_M  # Camera Y in world (up) if using Y-up
    OFFSET_Z_M = 0  # Camera Z offset (left/right) in robot frame

    # Baseline calibration resolution
    CALIB_WIDTH = 1280  # Baseline calibration image width (px)
    CALIB_HEIGHT = 960  # Baseline calibration image height (px)

    # Baseline intrinsic matrix for CALIB_WIDTH x CALIB_HEIGHT
    K_BASE = np.array([
        [1038.543 * 0.95, 0.000, 609.345],
        [0.000, 1037.537 *0.595, 469.070],
        [0.000, 0.000, 1.000]
    ], dtype=np.float32)

    SCALE_INTRINSICS = True  # Scale intrinsics to current frame size


class VisionConfig:
    """HSV thresholds and contour size limits"""
    # HSV thresholds in percentage or H half-range
    LOWER_HSV_PERCENT = np.array([37, 80, 110])
    UPPER_HSV_PERCENT = np.array([67, 255, 235])

    # Contour size thresholds (pixels)
    MIN_WIDTH = 15
    MIN_HEIGHT = 15
    MAX_WIDTH = 300
    MAX_HEIGHT = 300


def create_rotation_matrix_x(pitch_rad: float) -> np.ndarray:
    """Create rotation matrix for pitch around X-axis"""
    cos_a, sin_a = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ], dtype=np.float32)


def scale_intrinsics(K: np.ndarray, frame_w: int, frame_h: int,
                     calib_w: int, calib_h: int) -> np.ndarray:
    """Scale intrinsic matrix to match current frame resolution"""
    scale_x = frame_w / float(calib_w)
    scale_y = frame_h / float(calib_h)

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def convert_hsv_thresholds(lower_percent: np.ndarray,
                          upper_percent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert HSV percentage thresholds to OpenCV HSV ranges"""
    lower = np.array([
        int(lower_percent[0]),
        int(lower_percent[1]),
        int(lower_percent[2])
    ])
    upper = np.array([
        int(upper_percent[0]),
        int(upper_percent[1]),
        int(upper_percent[2])
    ])
    return lower, upper


def get_centroid(contour: np.ndarray) -> Tuple[int, int]:
    """Calculate centroid of contour, fallback to bbox center if degenerate"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
    return cx, cy


def filter_contours_by_size(contours: List, min_w: int, min_h: int,
                            max_w: int, max_h: int) -> List:
    """Filter contours based on bounding box size constraints"""
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_w < w < max_w and min_h < h < max_h:
            valid_contours.append(contour)
    return valid_contours


def draw_detection(image: np.ndarray, valid_contours: List,
                  contour: np.ndarray, cx: int, cy: int) -> None:
    """Draw detection visualization on image"""
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)


def pixel_to_ground_plane(u: int, v: int, K: np.ndarray,
                          camera_height: float, pitch_deg: float,
                          camera_x: float, camera_z: float) -> Tuple[float, float, bool]:
    """
    Back-project pixel coordinates to ground plane intersection

    Args:
        u, v: Pixel coordinates
        K: Camera intrinsic matrix
        camera_height: Camera height above ground (meters)
        pitch_deg: Camera pitch angle (degrees)
        camera_x, camera_z: Camera offsets in robot frame

    Returns:
        X_m, Z_m: World coordinates on ground plane
        success: Whether intersection was successful
    """
    # Back-project pixel to ray in camera frame
    pixel = np.array([[float(u)], [float(v)], [1.0]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    ray_camera = K_inv @ pixel  # Direction in camera frame (unnormalized)

    # Apply camera pitch rotation (positive pitch = looking down)
    R = create_rotation_matrix_x(np.deg2rad(pitch_deg))
    ray_camera = R @ ray_camera

    # Extract ray components (Camera frame: X=right, Y=down, Z=forward)
    dx = float(ray_camera[0, 0])
    dy = float(ray_camera[1, 0])  # Positive Y is down in camera frame
    dz = float(ray_camera[2, 0])  # Forward depth

    # Intersect with ground plane (Y = 0)
    # Ray equation: Y_world = camera_height - dy*t
    # At ground: 0 = camera_height - dy*t => t = camera_height / dy
    if dy <= 1e-6:  # Ray parallel to ground or pointing upward
        print(f"Ray not pointing down (dy={dy:.3f}), no ground intersection")
        return 0.0, 0.0, False

    t = camera_height / dy
    if t <= 0.0:  # Intersection behind camera
        print(f"Negative t={t:.3f}, no valid intersection")
        return 0.0, 0.0, False

    # Calculate world position
    X_m = camera_x + dx * t  # Right in camera → Side in world
    Z_m = camera_z + dz * t

    return X_m, Z_m, True


def runPipeline(image, llrobot):
    """
    Main vision pipeline called every frame by Limelight

    Returns:
        largestContour: Largest detected contour
        image: Annotated image with visualizations
        llpython: [hasTarget, u, v, X_m, Z_m, w, h, area]
    """
    # Convert HSV thresholds
    lower_hsv, upper_hsv = convert_hsv_thresholds(
        VisionConfig.LOWER_HSV_PERCENT,
        VisionConfig.UPPER_HSV_PERCENT
    )

    # Threshold image in HSV space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # Find contours
    contours, _ = cv2.findContours(
        img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize output
    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    if not contours:
        return largestContour, image, llpython

    # Filter contours by size
    valid_contours = filter_contours_by_size(
        contours,
        VisionConfig.MIN_WIDTH,
        VisionConfig.MIN_HEIGHT,
        VisionConfig.MAX_WIDTH,
        VisionConfig.MAX_HEIGHT
    )

    if not valid_contours:
        return largestContour, image, llpython

    # Get largest valid contour
    largestContour = max(valid_contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largestContour))

    # Get bounding box and centroid
    x, y, w, h = cv2.boundingRect(largestContour)
    u, v = get_centroid(largestContour)

    # Draw visualization
    draw_detection(image, valid_contours, largestContour, u, v)

    # Scale intrinsics to current frame resolution
    if CameraConfig.SCALE_INTRINSICS:
        frame_h, frame_w = image.shape[:2]
        K = scale_intrinsics(
            CameraConfig.K_BASE,
            frame_w, frame_h,
            CameraConfig.CALIB_WIDTH,
            CameraConfig.CALIB_HEIGHT
        )
    else:
        K = CameraConfig.K_BASE

    # Back-project to ground plane
    X_m, Z_m, success = pixel_to_ground_plane(
        u, v, K,
        CameraConfig.HEIGHT_M,
        CameraConfig.PITCH_DEG,
        CameraConfig.OFFSET_X_M,
        CameraConfig.OFFSET_Z_M
    )

    if success:
        # Pack results: [hasTarget, u, v, X_m, Z_m, w, h, area]
        llpython = [1, int(u), int(v), X_m, Z_m, int(w), int(h), area]
        print(f"Found intersection at World: (X={X_m:.2f}m, Z={Z_m:.2f}m), "
              f"Pixel: ({u}, {v}), size: ({w}x{h}), area: {area:.1f}")
    else:
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    return largestContour, image, llpython
