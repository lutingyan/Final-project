import time
import picar_4wd as fc
from color_detect import track_color_line_direction, detect_stop_sign, track_bottom_color
from picamera2 import Picamera2
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

movement_log = []


# New: Simple obstacle detection (check if the bottom of the image is too dark)
def detect_obstacle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    bottom_area = hsv[int(height * 0.7):, :]
    v = bottom_area[:, :, 2]
    dark_pixels = np.sum(v < 50)
    total_pixels = bottom_area.shape[0] * bottom_area.shape[1]
    dark_ratio = dark_pixels / total_pixels
    return dark_ratio > 0.4


def move(action, power_val):
    if action == "forward":
        fc.forward(power_val)
        time.sleep(0.2)
    elif action == "backward":
        fc.backward(power_val)
        time.sleep(0.2)
    elif action == "turn_left":
        fc.turn_left(power_val)
        time.sleep(0.2)
    elif action == "turn_right":
        fc.turn_right(power_val)
        time.sleep(0.2)
    else:
        fc.stop()


def stop():
    fc.stop()
    print("Stop")


def turn_left():
    global movement_log
    start_time = time.time()
    move("turn_left", 5)
    duration = time.time() - start_time
    movement_log.append(("turn_left", duration))
    stop()


def go_forward():
    global movement_log
    start_time = time.time()
    move("forward", 5)
    duration = time.time() - start_time
    movement_log.append(("forward", duration))
    stop()


def turn_right():
    global movement_log
    start_time = time.time()
    move("turn_right", 5)
    duration = time.time() - start_time
    movement_log.append(("turn_right", duration))
    stop()


def adjust_rotate(center_state):
    if center_state == 1:
        turn_left()
    elif center_state == 3:
        turn_right()
    else:
        go_forward()


def avoid_obstacle(camera):
    print("Obstacle detected! Starting avoidance maneuver...")
    stop()
    time.sleep(0.2)

    # Turn left slightly
    fc.turn_left(20)
    time.sleep(0.5)
    stop()
    time.sleep(0.2)

    # Move forward slightly
    fc.forward(20)
    time.sleep(0.6)
    stop()
    time.sleep(0.2)

    # Turn right slightly
    fc.turn_right(20)
    time.sleep(0.5)
    stop()
    time.sleep(0.2)

    # Check if the orange line is found again
    for _ in range(10):
        img = camera.capture_array()
        direction = track_color_line_direction(img, 'orange')
        if direction != 0:
            print("Found orange line after avoidance!")
            return True
        time.sleep(0.1)
    print("Failed to find orange line after avoidance.")
    return False


def follow_path():
    with Picamera2() as camera:
        print("Starting path tracking with color detection...")
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        try:
            while True:
                img = camera.capture_array()

                # Get the main color detected (either orange, blue, purple, etc.)
                main_color = track_bottom_color(img)
                print(f"Detected main color: {main_color}")

                if main_color:  # If we detect a color
                    # Determine movement direction based on the main color detected
                    direction = track_color_line_direction(img, main_color)
                    red_ratio = detect_stop_sign(img)

                    cv2.imshow(f"Main color {main_color} Line Tracking", img)

                    if direction != 0:
                        # Adjust behavior based on detected color
                        if main_color == 'orange':
                            if direction == 1:  # Left
                                turn_left()
                            elif direction == 3:  # Right
                                turn_right()
                            elif direction == 2:  # Forward
                                go_forward()
                        elif main_color == 'blue':
                            if direction == 1:  # Left
                                turn_left()
                            elif direction == 3:  # Right
                                turn_right()
                            elif direction == 2:  # Forward
                                go_forward()
                        elif main_color == 'purple':
                            if direction == 1:  # Left
                                turn_left()
                            elif direction == 3:  # Right
                                turn_right()
                            elif direction == 2:  # Forward
                                go_forward()

                    else:
                        print(f"No {main_color} line detected. Red ratio = {red_ratio:.4f}")
                        if red_ratio > 0.05:
                            print("STOP sign detected. Stopping car.")
                            stop()
                            break
                        else:
                            stop()
                else:
                    print("No color detected. Stopping.")
                    stop()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            print("Exiting...")
            fc.stop()
            cv2.destroyAllWindows()
            camera.stop()

    return movement_log


def plot_movement_log(movement_log):
    x, y = 0.0, 0.0  # Starting position
    angle = 90  # Assume the car starts at a 90-degree angle (facing upward)
    path = [(x, y)]  # List to store the path coordinates

    # Speed in cm per second (adjust if needed)
    speed_cm_per_sec = 5

    for action, duration in movement_log:
        if action == "forward":
            # Calculate the distance traveled in this duration
            distance = speed_cm_per_sec * duration
            # Convert angle to radians
            rad = math.radians(angle)
            # Update x and y based on the distance and angle
            x += distance * math.cos(rad)
            y += distance * math.sin(rad)
        elif action == "turn_left":
            # Turn left 90 degrees
            angle += 90
            angle %= 360  # Ensure the angle is always between 0 and 360 degrees
        elif action == "turn_right":
            # Turn right 90 degrees
            angle -= 90
            angle %= 360  # Ensure the angle is always between 0 and 360 degrees

        # Add the new position to the path
        path.append((x, y))

    # Extract the x and y positions for plotting
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    # Plot the path
    plt.plot(path_x, path_y, marker='o')
    plt.title("Car Movement Path")
    plt.xlabel("X position (cm)")
    plt.ylabel("Y position (cm)")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    follow_path()
    plot_movement_log(movement_log)

