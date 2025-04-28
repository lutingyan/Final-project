import picar_4wd as fc
import time
import math
import cv2
from picamera2 import Picamera2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# Define HSV ranges for red color (two ranges)
red_lower1 = np.array([0, 80, 80])
red_upper1 = np.array([4, 255, 255])
red_lower2 = np.array([165, 80, 80])
red_upper2 = np.array([180, 255, 255])

# Define HSV range for morning color (orange-yellow)
morning_lower = np.array([5, 80, 80])
morning_upper = np.array([37, 255, 255])

kernel_5 = np.ones((5, 5), np.uint8)

# Robot parameters
power_val = 50
FORWARD_SPEED = 0.35
TURN_SPEED = 0.5
current_position = (0, 0)
current_orientation = math.pi / 2
positions_x, positions_y = [], []
map_saved = False  # Save the map only once

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


def process_image(img):
    """Process the image to detect red and morning-colored objects."""
    resized = cv2.resize(img, (160, 120))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Detect red color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Detect morning color
    mask_morning = cv2.inRange(hsv, morning_lower, morning_upper)

    # Combine the two masks
    mask = cv2.bitwise_or(mask_red, mask_morning)

    # Morphological operation to remove noise
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    areas = []

    if len(contours) >= 2:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 6 and h >= 6:
                x, y, w, h = x * 4, y * 4, w * 4, h * 4
                center_x = x + w // 2
                center_y = y + h // 2
                centers.append((center_x, center_y))
                areas.append(w * h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    else:
        centers = []
        areas = []

    return img, mask, centers, areas


def generate_top_down_view(obj1_pos, obj2_pos):
    """Draw top-down 2D map with robot and two objects, distances, and scale bar."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    obj1 = np.array(obj1_pos)
    obj2 = np.array(obj2_pos)
    robot = np.array([0, 0])

    all_x = [0, obj1[0], obj2[0]]
    all_y = [0, obj1[1], obj2[1]]
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw robot
    robot_width = 0.3
    robot_height = 0.4
    ax.add_patch(patches.Rectangle((robot[0] - robot_width/2, robot[1] - robot_height/2),
                                   robot_width, robot_height, edgecolor='black', facecolor='red'))
    ax.add_patch(patches.Polygon([[robot[0], robot[1] + robot_height/2],
                                  [robot[0] - 0.05, robot[1] + robot_height/2 + 0.1],
                                  [robot[0] + 0.05, robot[1] + robot_height/2 + 0.1]],
                                 closed=True, color='yellow'))

    # Draw objects
    ax.add_patch(patches.Circle(obj1, 0.1, color='green'))
    ax.add_patch(patches.Circle(obj2, 0.1, color='green'))

    ax.text(obj1[0], obj1[1] + 0.15, f"Object 1\n({obj1[0]:.2f}, {obj1[1]:.2f})", ha='center', fontsize=9)
    ax.text(obj2[0], obj2[1] + 0.15, f"Object 2\n({obj2[0]:.2f}, {obj2[1]:.2f})", ha='center', fontsize=9)

    # Distances
    def draw_dist(p1, p2, offset=(0, 0.1), color='black'):
        mid = (p1 + p2) / 2 + np.array(offset)
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=color)
        ax.text(mid[0], mid[1], f"{dist:.2f} m", ha='center', fontsize=9, color=color)

    draw_dist(robot, obj1, offset=(-0.1, 0.05), color='blue')
    draw_dist(robot, obj2, offset=(0.1, 0.05), color='blue')
    draw_dist(obj1, obj2, offset=(0, 0.2), color='black')

    # Scale bar (0.5m)
    scale_bar_start = [robot[0] - 0.25, robot[1] - 0.4]
    ax.plot([scale_bar_start[0], scale_bar_start[0] + 0.5],
            [scale_bar_start[1], scale_bar_start[1]], 'k-', lw=3)
    ax.text(scale_bar_start[0] + 0.25, scale_bar_start[1] - 0.05, "0.5 m", ha='center')

    plt.savefig("challenge1_view.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ challenge1_view.png saved.")

def control_movement(centers, areas, last_seen_time):
    """Control the robot to pass between two objects correctly."""
    if len(centers) == 2:
        # Two objects detected, adjust heading if necessary
        mid_x = (centers[0][0] + centers[1][0]) / 2

        if mid_x < 280:
            fc.turn_left(power_val * 0.3)
            update_position('turn_left', 0.1)
            print("Aligning: Turning left")
        elif mid_x > 360:
            fc.turn_right(power_val * 0.3)
            update_position('turn_right', 0.1)
            print("Aligning: Turning right")
        else:
            fc.forward(power_val)
            update_position('forward', 0.1)
            print("Aligning: Moving forward")

        last_seen_time = time.time()

    else:
        # Less than 2 objects detected
        if time.time() - last_seen_time < 1.0:
            # Recently lost sight, assume still crossing, keep moving
            fc.forward(power_val)
            update_position('forward', 0.1)
            print("Lost view briefly, moving forward")
        else:
            # Lost view for longer time, assume passed, stop
            fc.stop()
            print("No objects detected for long time. Stopping.")

    return last_seen_time

def follow_objects():
    """Main detection and navigation loop."""
    global map_saved
    with Picamera2() as camera:
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        last_seen_time = time.time()
        try:
            while True:
                img = camera.capture_array()
                img_disp, mask, centers, areas = process_image(img)

                cv2.imshow("Camera View", img_disp)
                cv2.imshow("Color Mask", mask)
                cv2.setMouseCallback("Camera View", print_HSV, img_disp)

                if len(centers) == 2 and not map_saved:
                    (x1, y1), (x2, y2) = centers[0], centers[1]
                    dx = x1 - x2
                    dy = y1 - y2
                    pixel_distance = math.sqrt(dx**2 + dy**2)
                    print(f"Pixel distance between objects: {pixel_distance:.2f} px")

                    actual_distance_m = 1.5

                    PIXELS_PER_METER = pixel_distance / actual_distance_m
                    print(f"Auto-calculated PIXELS_PER_METER: {PIXELS_PER_METER:.2f}")

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

                last_seen_time = control_movement(centers, areas, last_seen_time)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Starting robot to find and pass between RED and MORNING color objects...")
    try:
        follow_objects()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        fc.stop()