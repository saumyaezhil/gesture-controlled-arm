This project lets you control a 6-DOF robotic arm using your hand gestures through a webcam. It uses MediaPipe for hand tracking and Arduino to move the servos.

 How It Works:
    The Python script captures hand landmarks using MediaPipe.
    It calculates gesture positions (like hand open, closed, or finger directions).
    These gestures are converted into servo angles.
    The angles are sent over Serial to an Arduino.
    The Arduino code reads this data and moves the servo motors of the robotic arm.
