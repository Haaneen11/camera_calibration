from ultralytics import YOLO
import cv2
import os
import numpy as np
import csv

model = YOLO('yolov8n.pt')  

img = cv2.imread(os.path.join('./detect_image','camera.jpg'))
results = model(img)[0]
print(f"image shape is {img.shape}")

# === Load intrinsic calibration ===
data = np.load("webcam_calibration_data.npz")
K = data["camera_matrix"]
dist = data["dist_coeffs"]

# === Get ultrasonic input from user ===
angle_deg = float(input("🔢 Enter ultrasonic angle (in degrees): "))
dist_cm = float(input("📏 Enter ultrasonic distance (in cm): "))
dist_m = dist_cm / 100.0  # Convert to meters

# === Load extrinsic parameters from CSV ===
found = False

with open("extrinsic_results.csv", newline='') as f:
    reader = csv.DictReader(f)
    row = next(reader)  # take first row only
    rvec = np.array(eval(row["rvec"]), dtype=np.float32).reshape(3, 1)
    tvec = np.array(eval(row["tvec"]), dtype=np.float32).reshape(3, 1)
    found = True
          

if not found:
    print("❌ Couldn't find extrinsic parameters for that image.")
    exit()

# === Compute 3D point from ultrasonic reading ===
x = dist_m * np.cos(np.radians(angle_deg))
y = dist_m * np.sin(np.radians(angle_deg))
z = 0
point_3d = np.array([[x, y, z]], dtype=np.float32)

# === Project 3D point to 2D ===
projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist)
u, v = projected.ravel()
print(f"📍 Projected image location: (u={u:.1f}, v={v:.1f})")


# === Save output ===
output_path = f"ultrasonic_projection"

for box in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = box
    if score > 0.2:
        label = model.names[int(class_id)]
        print(f"Class: {label}, Score: {score:.2f}, Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        print(f"the center of the object is {(x_center  ,y_center)}")

cv2.imwrite("output.jpg", img)
u, v = int(u), int(v)


cv2.circle(img, (u, v), radius=6, color=(0, 0, 255), thickness=-1)

cv2.putText(img, "Ultrasonic point", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.imwrite("output_with_projection.jpg", img)

