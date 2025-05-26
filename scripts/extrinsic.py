import cv2
import numpy as np
import glob
import re
import csv
import os

# Chessboard settings
chessboard_size = (6, 6)
square_size = 0.025  # meters

# Prepare object points (3D)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Load camera calibration
data = np.load("webcam_calibration_data.npz")
K = data["camera_matrix"]
dist = data["dist_coeffs"]

# Prepare folders
os.makedirs("projected_output", exist_ok=True)
with open("extrinsic_results.csv", "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["filename", "rvec", "tvec"])

    for i, path in enumerate(glob.glob("extrinsic_images/*.jpg")):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, chessboard_size)
        if not found:
            print(f"❌ No chessboard in {path}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)

        writer.writerow([os.path.basename(path), rvec.flatten().tolist(), tvec.flatten().tolist()])

        # ✅ Draw the origin (0,0,0)
        origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist)
        u, v = origin_2d.ravel()
        cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)
        cv2.putText(img, "World Origin", (int(u)+10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Draw X and Y axes (length = 5 cm)
        axis_length = 0.05  # 5 cm
        axes_3d = np.float32([
            [axis_length, 0, 0],  # X axis (red)
            [0, axis_length, 0],  # Y axis (green)
        ])
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)

        # Get 2D points
        x_end = tuple(axes_2d[0].ravel().astype(int))
        y_end = tuple(axes_2d[1].ravel().astype(int))
        origin_pt = (int(u), int(v))

        # Draw axes
        cv2.line(img, origin_pt, x_end, (0, 0, 255), 3)  # X in red
        cv2.line(img, origin_pt, y_end, (0, 255, 0), 3)  # Y in green

        out_path = os.path.join("projected_output", os.path.basename(path))
        cv2.imwrite(out_path, img)
        print(f"✅ Saved with origin + axes: {out_path}")
