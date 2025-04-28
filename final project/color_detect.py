import cv2
from picamera2 import Picamera2
import numpy as np
import time

color_dict = {'orange': [5, 18], 'yellow': [22, 37], 'green': [42, 85]}

kernel_5 = np.ones((5, 5), np.uint8)

color_dict = {'orange': [5, 18], 'yellow': [22, 37], 'green': [42, 85], 'blue': [100, 140], 'purple': [130, 160]}

kernel_5 = np.ones((5, 5), np.uint8)


def track_bottom_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color ranges for orange, blue, purple, yellow, and green
    color_ranges = {
        'orange': (np.array([5, 150, 150]), np.array([18, 255, 255])),
        'blue': (np.array([100, 150, 150]), np.array([140, 255, 255])),
        'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
    }

    # Focus on the bottom part of the image (bottom 30% of the image)
    height, width = hsv.shape[:2]
    bottom_area = hsv[int(height * 0.7):, :]

    # Create masks for each color and calculate their areas in the bottom region
    color_areas = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(bottom_area, lower, upper)
        color_area = np.sum(mask) / 255  # Count of non-zero pixels
        color_areas[color] = color_area

    # Sort colors by the area of detected pixels (largest to smallest)
    sorted_colors = sorted(color_areas.items(), key=lambda x: x[1], reverse=True)

    # Return the color with the largest area detected at the bottom
    return sorted_colors[0][0] if sorted_colors else None


def track_color_line_direction(img, color_name):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'orange': (np.array([5, 150, 150]), np.array([18, 255, 255])),
        'blue': (np.array([100, 150, 150]), np.array([140, 255, 255])),
        'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
    }

    if color_name not in color_ranges:
        print(f"[ERROR] Unsupported color: {color_name}")
        return 0

    lower, upper = color_ranges[color_name]
    mask = cv2.inRange(hsv, lower, upper)

    height, width = mask.shape
    bottom_half = mask[int(height * 0.4):, :]
    M = cv2.moments(bottom_half)

    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        center_x = width // 2
        diff = cx - center_x

        if diff < -20:
            return 1  # Turn left
        elif diff > 20:
            return 3  # Turn right
        else:
            return 2  # Move forward
    else:
        return 0  # No color detected


def detect_stop_sign(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    red_area = np.sum(mask) / 255
    height, width = mask.shape
    total_area = height * width

    red_ratio = red_area / total_area

    return red_ratio
