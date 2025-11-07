import cv2
import numpy as np

"""
Camera space
- X: Right
- Y: Down
- Z: Forward
"""
preview = False

# 0.2066749994
CAMERA_HEIGHT_M = 0.206675  # camera height above ground (meters)
CAMERA_PITCH_DEG = 0  # camera downward pitch (+ degrees)
CAMERA_X_M = 0.0  # camera X offset in robot frame (forward)
CAMERA_Y_M = CAMERA_HEIGHT_M  # camera Y in world (up) if using Y-up
CAMERA_Z_M = 0.0  # camera Z offset (left/right) in robot frame

# Intrinsics from baseline calibration at a known resolution
CALIB_WIDTH = 1280  # baseline calibration image width (px)
CALIB_HEIGHT = 960  # baseline calibration image height (px)
# Baseline intrinsic matrix for CALIB_WIDTH x CALIB_HEIGHT
K_BASE = np.array(
    [
        [1038.543 * 0.67, 0.000, 609.345],
        [0.000, 1037.537 * 0.67, 469.070],
        [0.000, 0.000, 1.000],
    ],
    dtype=np.float32,
)
SCALE_INTRINSICS = True  # scale intrinsics to current frame size

# HSV threshold [H,S,V] in percent or H half-range
LOWER_HSV_PERCENT = np.array([64, 40, 35]) if not preview else np.array([60, 50, 40])
UPPER_HSV_PERCENT = np.array([85, 98, 90]) if not preview else np.array([70, 70, 90])

# Contour size threshold
MIN_TARGET_WIDTH_PX = 8
MIN_TARGET_HEIGHT_PX = 8
MAX_TARGET_WIDTH_PX = 300
MAX_TARGET_HEIGHT_PX = 300


def _rotation_x(pitch_rad: float) -> np.ndarray:
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def scale_intrinsics(
    K: np.ndarray, frame_w: int, frame_h: int, calib_w: int, calib_h: int
) -> np.ndarray:
    scaled_x = frame_w / float(calib_w)
    scaled_y = frame_h / float(calib_h)

    K_scaled = K.copy()
    K_scaled[0, 0] *= scaled_x  # fx
    K_scaled[1, 1] *= scaled_y  # fy
    K_scaled[0, 2] *= scaled_x  # cx
    K_scaled[1, 2] *= scaled_y  # cy
    return K_scaled


# runPipeline() is called every frame by Limelight
def runPipeline(image, llrobot):
    # Convert HSV percentage thresholds to HSV ranges
    lower_target = np.array(
        [
            int(LOWER_HSV_PERCENT[0] / 2),
            int(LOWER_HSV_PERCENT[1] * 2.55),
            int(LOWER_HSV_PERCENT[2] * 2.55),
        ]
    )
    upper_target = np.array(
        [
            int(UPPER_HSV_PERCENT[0] / 2),
            int(UPPER_HSV_PERCENT[1] * 2.55),
            int(UPPER_HSV_PERCENT[2] * 2.55),
        ]
    )
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, lower_target, upper_target)

    # find contours
    contours, _ = cv2.findContours(
        img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    if len(contours) > 0:
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (
                w > MIN_TARGET_WIDTH_PX
                and h > MIN_TARGET_HEIGHT_PX
                and w < MAX_TARGET_WIDTH_PX
                and h < MAX_TARGET_HEIGHT_PX
            ):
                valid_contours.append(contour)

        if valid_contours:
            largestContour = max(valid_contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largestContour))

            # bbox
            x, y, w, h = cv2.boundingRect(largestContour)
            cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # centroid (fallback to bbox center if moment degenerates)
            M = cv2.moments(largestContour)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
            else:
                u = int(x + w / 2)
                v = int(y + h / 2)
            cv2.circle(image, (u, v), 5, (0, 0, 255), -1)

            # scale intrinsics
            if SCALE_INTRINSICS:
                frame_h, frame_w = image.shape[:2]
                K = scale_intrinsics(
                    K_BASE, frame_w, frame_h, CALIB_WIDTH, CALIB_HEIGHT
                )
            else:
                K = K_BASE

            # Back‑project pixel -> ray in camera frame
            # Camera frame: X right, Y down, Z forward
            pixel = np.array([[float(u)], [float(v)], [1.0]], dtype=np.float32)
            Kinv = np.linalg.inv(K)
            ray_camera = Kinv @ pixel  # direction in camera frame (unnormalized)

            # Apply camera pitch rotation (positive pitch = looking down)
            R = _rotation_x(np.deg2rad(CAMERA_PITCH_DEG))
            ray_camera = R @ ray_camera

            # In standard camera frame: X=right, Y=down, Z=forward
            # For ground plane intersection, we need Y component (down direction)
            dx = float(ray_camera[0, 0])
            dy = float(ray_camera[1, 0])  # positive Y is down in camera frame
            dz = float(ray_camera[2, 0])  # forward depth

            # print(f"Ray direction: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

            # Intersect with ground plane
            # Camera is at height CAMERA_HEIGHT_M above ground
            # We want to find where ray intersects Y=0 ground plane
            # Ray equation: Y_world = CAMERA_HEIGHT_M - dy*t (since camera Y points down)
            # At ground: 0 = CAMERA_HEIGHT_M - dy*t => t = CAMERA_HEIGHT_M / dy

            if dy > 1e-6:
                t = CAMERA_HEIGHT_M / dy
                if t > 0.0:
                    X_m = CAMERA_X_M + dx * t  # Right in camera → Side in world
                    Z_m = CAMERA_Z_M + dz * t

                    # Pack results: [hasTarget, u, v, X_m, Z_m, w, h, area]
                    llpython = [1, int(u), int(v), X_m, Z_m, int(w), int(h), area]
                    print(
                        f"Found intersection at World: (X={X_m:.2f}m, Z={Z_m:.2f}m), Pixel: ({u}, {v}), size: ({w}x{h}), area: {area:.1f}"
                    )
                else:
                    print(f"Negative t={t:.3f}, no valid intersection")
                    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
            else:
                # Ray parallel to ground or pointing upward
                print(f"Ray not pointing down (dy={dy:.3f}), no ground intersection")
                llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    return largestContour, image, llpython
