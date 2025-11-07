import cv2
import numpy as np

preview = True

# Tunable camera parameters
FX_FY = 600.0  # focal length (px)
CAMERA_HEIGHT_M = 21.59  # camera height (m)
CAMERA_PITCH_DEG = 0  # down tilt (+) (degrees)


def _rotation_x(pitch_rad: float) -> np.ndarray:
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float32)


# runPipeline() is called every frame by Limelight
def runPipeline(image, llrobot):
    # threshold
    lower_threshold = np.array([64, 40, 35]) if not preview else np.array([60, 50, 40])
    upper_threshold = np.array([85, 98, 90]) if not preview else np.array([70, 70, 90])
    lower_target = np.array([
        int(lower_threshold[0] / 2), int(lower_threshold[1] * 2.55), int(lower_threshold[2] * 2.55)
    ])
    upper_target = np.array([
        int(upper_threshold[0] / 2), int(upper_threshold[1] * 2.55), int(upper_threshold[2] * 2.55)
    ])
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, lower_target, upper_target)

    # find contours
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    if len(contours) > 0:
        min_width = 0  # px
        min_height = 0  # px
        max_width = 300  # px
        max_height = 300  # px
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width and h > min_height and w < max_width and h < max_height:
                valid_contours.append(contour)

        if valid_contours:
            print("Detected object")
            largestContour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largestContour))

            # bbox
            x, y, w, h = cv2.boundingRect(largestContour)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # centroid
            M = cv2.moments(largestContour)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
            else:
                # fallback to bbox center
                u = int(x + w / 2)
                v = int(y + h / 2)

            cv2.circle(image, (u, v), 5, (0, 0, 255), -1)

            # intrinsics matrix K
            K = np.array([
                [1038.543, 0.000, 609.345],
                [0.000, 1037.537, 469.070],
                [0.000, 0.000, 1.000]
            ], dtype=np.float32)

            # back‑project the pixel
            pixel = np.array([[float(u)], [float(v)], [1.0]], dtype=np.float32)
            Kinv = np.linalg.inv(K)
            ray = Kinv @ pixel  # direction in camera frame
            R = _rotation_x(np.deg2rad(CAMERA_PITCH_DEG))
            ray = R @ ray  # account for camera tilt

            # intersect with ground plane
            # Assumptions: camera at (0, CAMERA_HEIGHT_M, 0), ground plane at Y=0 in world,
            # and ray is expressed in the camera frame where this Y component points downward.
            X_m = 0.0
            Y_m = 0.0
            denom = float(ray[1, 0])
            if denom > 1e-6:
                t = CAMERA_HEIGHT_M / denom
                if t > 0.0:
                    X_m = float(ray[0, 0] * t)
                    Y_m = float(ray[2, 0] * t)
                    # Pack results: [hasTarget, u, v, X_m, Y_m, w, h, area]
                    llpython = [1, int(u), int(v), X_m, Y_m, int(w), int(h), area]
                else:
                    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
                    print("No intersection")
            else:
                llpython = [0, 0, 0, 0, 0, 0, 0, 0]
                print("No intersection")

                # Return the largest contour for LL crosshair, the modified image, and custom robot data
    return largestContour, image, llpython


# Test the pipeline locally with webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    largestContour, image, llpython = runPipeline(frame, None)
    print(llpython)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
