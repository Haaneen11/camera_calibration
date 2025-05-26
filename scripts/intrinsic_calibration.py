import cv2
import numpy as np
import os
# Chessboard settings (number of inner corners per row and column)
chessboard_size = (6, 6)
square_size = 0.025  # Size of a square (in meters or any consistent unit)

# Prepare 3D points for the chessboard pattern (e.g., (0,0,0), (1,0,0), ...)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                      0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Lists to store object points and image points from all the frames
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane
video_path = os.path.join('.','2.mp4')
cap = cv2.VideoCapture(video_path)
ret , frame = cap.read()
#output = cv2.VideoWriter(os.path.join('.','output.mp4'),cv2.VideoWriter_fourcc(*'MP4V'),int(cap.get(cv2.CAP_PROP_FPS)),(frame.shape[1],frame.shape[0]))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)
print(frame_interval)
frame_count = 0
while ret:
    if frame_count % frame_interval == 0:
        frame=cv2.resize(frame,(640,460))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, chessboard_size, corners, ret_corners)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1)

        if ret_corners:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            cv2.imwrite(f"calib_frame_{len(imgpoints)}.jpg", frame)
        
    frame_count+=1
    ret, frame = cap.read()

    

cap.release()
cv2.destroyAllWindows()

# Check if enough frames were collected
if len(objpoints) < 5:
    print("⚠️ Not enough calibration images. Please capture at least 5 valid frames.")
    exit()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("📷 Camera Matrix:\n", camera_matrix)
print("🔧 Distortion Coefficients:\n", dist_coeffs)

# Save calibration result to file
np.savez("webcam_calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("✅ Calibration data saved to webcam_calibration_data.npz")

