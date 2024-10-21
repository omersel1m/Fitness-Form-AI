# Pose Estimation for Fitness Movement Accuracy Analysis
This project aims to evaluate the accuracy of fitness movements using computer vision technologies. The YOLOv8 Pose Estimation model detects human body keypoints in real-time images captured by a camera and determines whether the movement is performed correctly.

# Features:
Real-Time Analysis: Instant pose predictions are made on images captured from the user.
Elbow and Hip Positions: The user’s elbow and hip positions are monitored, and it is checked whether these positions are within tolerance compared to the starting points.
Accuracy Percentage Calculation: The accuracy percentage of the movement performed by the user is calculated and displayed on the screen.
Advanced Variance Calculation: Variance is calculated from the data obtained from the last 100 positions to assess the consistency of the movement.

# Usage:
This project is designed to assist users in performing their movements correctly during fitness training. Users receive instant feedback on the screen while performing exercises, allowing them to correct form errors and make their training more efficient.

# Requirements:
Python 3.x, 
OpenCV, 
NumPy, 
Ultralytics YOLOv8 model file (yolov8n-pose.pt), 
A suitable camera (e.g., laptop camera or external webcam)

# Installation:
Install the necessary libraries to run your project and place the yolov8n-pose.pt model file in the root directory of your project. Then, you can run the code to perform real-time pose estimation and movement accuracy analysis.


