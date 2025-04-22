import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Arm setup (6 markers for 6 joints)
NUM_MARKERS = 6

# Marker size in meters (update this with actual size used)
marker_length = 0.03  # 3 cm

# Camera calibration parameters (replace with your actual calibration)
camera_matrix = np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

# Initialize Matplotlib 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    joint_positions = {}

    if ids is not None:
        # Estimate pose for each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id < NUM_MARKERS:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, camera_matrix, dist_coeffs)

                # Store joint positions (marker centers)
                joint_positions[int(marker_id)] = tvec[0][0]

                # Draw the marker and axis
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.02)

        if len(joint_positions) >= 2:
            sorted_joints = [joint_positions[i] for i in sorted(joint_positions)]
            xs, ys, zs = zip(*sorted_joints)

            ax.clear()
            ax.set_xlim([-0.2, 0.2])
            ax.set_ylim([-0.2, 0.2])
            ax.set_zlim([0, 0.4])
            ax.set_title("3D Robotic Arm Joint Tracking")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.plot(xs, ys, zs, marker='o', linewidth=3, color='blue')

            plt.pause(0.01)

    cv2.imshow("Webcam View", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
