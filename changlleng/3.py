import picar_4wd as fc
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from picamera2 import Picamera2

# Robot parameters
power_val = 50
FORWARD_SPEED = 0.35
TURN_SPEED = 0.5
current_position = (0, 0)
current_orientation = math.pi / 2
positions_x, positions_y = [], []

# Global variables
background_img = None
map_saved = False
map_ready_time = None
movement_started = False
PIXELS_PER_METER = None


def print_HSV(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print(f"HSV at ({x}, {y}) = {hsv[y, x]}")


def update_position(direction, time_elapsed):
    global current_position, current_orientation
    if direction == 'forward':
        distance = FORWARD_SPEED * time_elapsed
        dx = distance * math.cos(current_orientation)
        dy = distance * math.sin(current_orientation)
        current_position = (current_position[0] + dx, current_position[1] + dy)
    elif direction == 'turn_left':
        current_orientation = (current_orientation + TURN_SPEED * time_elapsed) % (2 * math.pi)
    elif direction == 'turn_right':
        current_orientation = (current_orientation - TURN_SPEED * time_elapsed) % (2 * math.pi)
    positions_x.append(current_position[0])
    positions_y.append(current_position[1])


def capture_background(camera):
    """Capture environment before objects placed."""
    global background_img
    print("Capturing background in 2 seconds...")
    time.sleep(2)
    background_img = camera.capture_array()
    cv2.imwrite("background.png", background_img)
    print("Background captured and saved as background.png.")


def detect_objects_by_difference(camera):
    global background_img
    if background_img is None:
        raise Exception("Background image not captured yet!")

    current_img = camera.capture_array()
    background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(background_gray, current_gray)

    _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    morph = cv2.dilate(morph, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    if len(contours) >= 2:

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= 1000:
                center_x = x + w // 2
                center_y = y + h // 2
                centers.append((center_x, center_y))

    return centers, current_img, morph


def generate_top_down_view(obj1_pos, obj2_pos):
    """Draw 2D top-down view and save it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    robot = np.array([0, 0])
    obj1 = np.array(obj1_pos)
    obj2 = np.array(obj2_pos)

    all_x = [0, obj1[0], obj2[0]]
    all_y = [0, obj1[1], obj2[1]]
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw robot
    robot_width = 0.3
    robot_height = 0.4
    ax.add_patch(patches.Rectangle((robot[0] - robot_width / 2, robot[1] - robot_height / 2),
                                   robot_width, robot_height, edgecolor='black', facecolor='red'))
    ax.add_patch(patches.Polygon([[robot[0], robot[1] + robot_height / 2],
                                  [robot[0] - 0.05, robot[1] + robot_height / 2 + 0.1],
                                  [robot[0] + 0.05, robot[1] + robot_height / 2 + 0.1]],
                                 closed=True, color='yellow'))

    # Draw objects
    ax.add_patch(patches.Circle(obj1, 0.1, color='green'))
    ax.add_patch(patches.Circle(obj2, 0.1, color='green'))

    ax.text(obj1[0], obj1[1] + 0.15, f"Object 1\n({obj1[0]:.2f},{obj1[1]:.2f})", ha='center', fontsize=9)
    ax.text(obj2[0], obj2[1] + 0.15, f"Object 2\n({obj2[0]:.2f},{obj2[1]:.2f})", ha='center', fontsize=9)

    # Distances
    def draw_dist(p1, p2, offset=(0, 0.1)):
        mid = (p1 + p2) / 2 + np.array(offset)
        dist = np.linalg.norm(p1 - p2)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='black')
        ax.text(mid[0], mid[1], f"{dist:.2f}m", ha='center', fontsize=9)

    draw_dist(robot, obj1, offset=(-0.1, 0.05))
    draw_dist(robot, obj2, offset=(0.1, 0.05))
    draw_dist(obj1, obj2, offset=(0, 0.2))

    # Scale bar
    scale_bar_start = [robot[0] - 0.25, robot[1] - 0.4]
    ax.plot([scale_bar_start[0], scale_bar_start[0] + 0.5],
            [scale_bar_start[1], scale_bar_start[1]], 'k-', lw=3)
    ax.text(scale_bar_start[0] + 0.25, scale_bar_start[1] - 0.05, "0.5m", ha='center')

    plt.savefig("challenge2_view.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(" challenge2_view.png saved.")


def control_movement(centers, last_seen_time):
    """Improved logic: move more smoothly and intelligently."""
    if len(centers) == 2:
        mid_x = (centers[0][0] + centers[1][0]) / 2
        object_distance_pixels = math.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2)
        if mid_x < 240:
            fc.turn_left(power_val * 0.3)
            update_position('turn_left', 0.1)
            print("Aligning: Turning left")
        elif mid_x > 400:
            fc.turn_right(power_val * 0.3)
            update_position('turn_right', 0.1)
            print("Aligning: Turning right")
        else:
            if object_distance_pixels > 400:
                fc.stop()
                print("Objects very close. Stopping.")
            else:
                fc.forward(power_val)
                update_position('forward', 0.1)
                print("Aligning: Moving forward")

        last_seen_time = time.time()

    else:
        # If objects are temporarily lost, keep moving forward for a short time
        if time.time() - last_seen_time < 2.0:  # tolerate 2 seconds
            fc.forward(power_val)
            update_position('forward', 0.1)
            print("Temporary lost view, keep moving forward")
        else:
            fc.stop()
            print(" No objects detected for a while. Stopping.")
    return last_seen_time


def follow_objects():
    """Main loop: capture, detect, map, navigate."""
    global map_saved, map_ready_time, movement_started, PIXELS_PER_METER

    with Picamera2() as camera:
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        last_seen_time = time.time()
        capture_background(camera)  # First capture background

        try:
            while True:
                centers, current_img, diff_mask = detect_objects_by_difference(camera)

                cv2.imshow("Current View", current_img)
                cv2.imshow("Difference Mask", diff_mask)
                cv2.setMouseCallback("Current View", print_HSV, current_img)

                if len(centers) == 2 and not map_saved:
                    (x1, y1), (x2, y2) = centers[0], centers[1]
                    dx = x1 - x2
                    dy = y1 - y2
                    pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

                    actual_distance_m = 1.5  # Assumed
                    PIXELS_PER_METER = pixel_distance / actual_distance_m

                    def pixel_to_meter(px, py):
                        dx_pixels = px - 320
                        dy_pixels = 240 - py
                        dx_m = dx_pixels / PIXELS_PER_METER
                        dy_m = dy_pixels / PIXELS_PER_METER
                        return (dx_m, dy_m)

                    obj1_pos = pixel_to_meter(*centers[0])
                    obj2_pos = pixel_to_meter(*centers[1])
                    generate_top_down_view(obj1_pos, obj2_pos)

                    map_saved = True
                    map_ready_time = time.time()
                    print("? Map saved! Waiting 5 seconds before moving...")

                if map_saved and not movement_started:
                    if time.time() - map_ready_time >= 5:
                        movement_started = True
                        print("? 5 seconds wait complete. Starting movement!")

                if movement_started:
                    last_seen_time = control_movement(centers, last_seen_time)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    print("?? Starting Challenge II: Build map ? Wait ? Drive!")
    try:
        follow_objects()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        fc.stop()
