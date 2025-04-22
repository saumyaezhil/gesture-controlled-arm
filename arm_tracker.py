import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

L1, L2, L3 = 8, 6, 4
claw_len = 2

def map_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def forward_kinematics(base, shoulder, elbow, wrist, claw):
    base_rad = math.radians(base)
    shoulder_rad = math.radians(shoulder)
    elbow_rad = math.radians(elbow)
    wrist_rad = math.radians(wrist)

    x1 = L1 * math.cos(shoulder_rad) * math.cos(base_rad)
    y1 = L1 * math.cos(shoulder_rad) * math.sin(base_rad)
    z1 = L1 * math.sin(shoulder_rad)

    x2 = x1 + L2 * math.cos(shoulder_rad + elbow_rad * 0.01) * math.cos(base_rad)
    y2 = y1 + L2 * math.cos(shoulder_rad + elbow_rad * 0.01) * math.sin(base_rad)
    z2 = z1 + L2 * math.sin(shoulder_rad + elbow_rad * 0.01)

    x3 = x2 + L3 * math.cos(shoulder_rad + elbow_rad * 0.02 + wrist_rad * 0.01) * math.cos(base_rad)
    y3 = y2 + L3 * math.cos(shoulder_rad + elbow_rad * 0.02 + wrist_rad * 0.01) * math.sin(base_rad)
    z3 = z2 + L3 * math.sin(shoulder_rad + elbow_rad * 0.02 + wrist_rad * 0.01)

    claw_offset = claw_len * math.sin(math.radians(claw / 2))
    dx = claw_offset * math.cos(base_rad + math.pi / 2)
    dy = claw_offset * math.sin(base_rad + math.pi / 2)

    claw1 = (x3 + dx, y3 + dy, z3)
    claw2 = (x3 - dx, y3 - dy, z3)

    return [(0, 0, 0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), claw1, claw2]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                base_angle = map_range(lm[0].x, 0.0, 1.0, 0, 180)
                shoulder_angle = map_range(lm[5].y, 0.0, 1.0, 180, 0)
                elbow_angle = map_range(lm[9].y, 0.0, 1.0, 180, 0)
                wrist_angle = map_range(lm[13].y, 0.0, 1.0, 180, 0)

                thumb_tip = lm[4]
                index_tip = lm[8]
                claw_dist = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                claw_angle = map_range(claw_dist, 0.02, 0.1, 0, 180)

                print(f"Base: {base_angle}, Shoulder: {shoulder_angle}, Elbow: {elbow_angle}, Wrist: {wrist_angle}, Claw: {claw_angle}")

                joints = forward_kinematics(base_angle, shoulder_angle, elbow_angle, wrist_angle, claw_angle)

                ax.clear()
                ax.set_xlim([-20, 20])
                ax.set_ylim([-20, 20])
                ax.set_zlim([0, 20])
                ax.set_title("3D Robotic Arm Simulation")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                xs, ys, zs = zip(*joints[:4])
                ax.plot(xs, ys, zs, marker='o', linewidth=3, color='blue')

                cx, cy, cz = zip(*joints[3:])
                ax.plot(cx, cy, cz, marker='o', linewidth=2, color='red')

                plt.pause(0.01)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Webcam View", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
