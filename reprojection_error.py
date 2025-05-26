import cv2
import numpy as np
import glob

# === Load saved calibration data ===
data = np.load("webcam_calibration_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# === Chessboard settings ===
chessboard_size = (6, 6)
square_size = 0.025  # in meters

# Prepare 3D object points for one view
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                      0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image

# === Read calibration images ===
for path in glob.glob("*.jpg"):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size)
    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        objpoints.append(objp)
    else:
        print(f"❌ Chessboard not found in {path}")

# === Re-run calibration to get rvecs and tvecs
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, dist_coeffs)

# === Calculate total reprojection error
total_error = 0
total_points = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
    total_error += error**2
    total_points += len(objpoints[i])

mean_error = np.sqrt(total_error / total_points)
print(f"📐 Reprojection error (RMS): {mean_error:.4f} pixels")
