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
PIXELS_PER_METER = None
CAMERA_VIEW_WIDTH = 0.42 #cm
CAMERA_VIEW_DEPTH = 0.40 #cm 0.297


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
    global background_img
    print("馃摲 Capturing background in 2 seconds...")
    time.sleep(2)
    background_img = camera.capture_array()
    cv2.imwrite("background.png", background_img)
    print("鉁?Background saved as background.png.")


def detect_objects_by_difference(camera):
    global background_img
    if background_img is None:
        raise Exception("鉂?Background image not captured!")

    current_img = camera.capture_array()
    cv2.imwrite("placed.png", current_img)
    background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (3,3), 0)  #blured to reduce noices
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (3,3), 0)
    diff = cv2.absdiff(background_gray, current_gray)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # lowered the threshhold for more sensetive detection
    kernel = np.ones((5, 5), np.uint8) # smaller kernel for less effective deteciton
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2) # multiple iteration for multi layered detections
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)
    # morph = cv2.dilate(morph, kernel, iterations=3)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    bounding_box = []
    if len(contours) >= 2:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= 1024: # lower the value to detect smaller objs
                center_x = x + w // 2
                center_y = y + h // 2
                centers.append((center_x, center_y))
                
                bounding_box.append(((x,y,w,h)))

    return bounding_box, centers, current_img, morph


def generate_top_down_view(obj1_pos, obj2_pos):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    robot = np.array([0.0, 0.0])
    obj1 = np.array(obj1_pos)
    obj2 = np.array(obj2_pos)

    all_x = [robot[0], obj1[0], obj2[0]]
    all_y = [robot[1], obj1[1], obj2[1]]
    margin = 0.8
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    triangle_size = 0.12
    triangle = np.array([
        [0, triangle_size],
        [-triangle_size / 2, -triangle_size / 2],
        [triangle_size / 2, -triangle_size / 2]
    ])
    triangle += robot
    ax.add_patch(patches.Polygon(triangle, closed=True, color='red', edgecolor='black', linewidth=1.0))

    circle_radius = 0.05
    for idx, obj in enumerate([obj1, obj2], start=1):
        ax.add_patch(patches.Circle(obj, circle_radius, color='green'))
        ax.text(obj[0], obj[1] + 0.1,
                f"Object {idx}\n({obj[0]:.2f}, {obj[1]:.2f})",
                ha='center', fontsize=8)

    def draw_dist(p1, p2, offset=(0, 0.15), color='black'):
        mid = (p1 + p2) / 2 + np.array(offset)
        dist = np.linalg.norm(p1 - p2)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=color, linewidth=1)
        ax.text(mid[0], mid[1], f"{dist:.2f} m", ha='center', fontsize=8, color=color)

    draw_dist(robot, obj1, offset=(-0.25, 0.15), color='blue')
    draw_dist(robot, obj2, offset=(0.25, 0.15), color='blue')
    draw_dist(obj1, obj2, offset=(0, 0.25), color='black')

    scale_x, scale_y = robot[0] - 0.25, robot[1] - 0.9
    ax.plot([scale_x, scale_x + 0.5], [scale_y, scale_y], 'k-', lw=2)
    ax.text(scale_x + 0.25, scale_y - 0.05, "0.5 m", ha='center', fontsize=8)

    ax.annotate("", xy=(0.8, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.85, 0.0, "+X", fontsize=9, ha='left', va='center')

    ax.annotate("", xy=(0, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.0, 0.85, "+Y", fontsize=9, ha='center', va='bottom')

    plt.savefig("challenge2_view.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("鉁?challenge2_view.png saved (car-triangle facing y+)")

def turn_left_angle(power, angle):
    time_per_degree = 0.8 / 90

    power_factor = power/50
    turn_time = time_per_degree * angle / power_factor

    fc.turn_left(power)
    time.sleep(turn_time)
    fc.stop()

def turn_right_angle(power, angle):
    time_per_degree = 0.8 / 90 # seconds per degree at power = 50

    power_factor = power/50
    turn_time = time_per_degree * angle / power_factor

    fc.turn_right(power)
    time.sleep(turn_time)
    fc.stop()

def move_forward(power, distance):
    time_per_meter = 1/0.5 # seconds per meter at power = 50
    power_factor = power / 50
    move_time = time_per_meter * distance / power_factor

    fc.forward(power)
    time.sleep(move_time)
    fc.stop()

def drive_towards(target_meter_pos, extra_distance=0.5):
    global current_position

    print("driving towards leftmost obj: (", target_meter_pos[0], ", ", target_meter_pos[1]),
    dist = math.sqrt(target_meter_pos[0] * target_meter_pos[0] + target_meter_pos[1] * target_meter_pos[1])
    if target_meter_pos[0] < 0:
        left_angle = math.atan(-1 * (target_meter_pos[0])/target_meter_pos[1]) * 180/math.pi
        turn_left_angle(power_val, left_angle)
        print("turnning left: ", left_angle, " degree")
    elif target_meter_pos[0] > 0:
        right_angle = math.atan((target_meter_pos[0])/target_meter_pos[1]) * 180/math.pi
        turn_right_angle(power_val, right_angle)
        print("turnning right: ", right_angle, " degree")

    print("moving forward: ", dist, "meters")
    if (dist < 10):
        dist -= CAMERA_VIEW_DEPTH/5
        dist -= extra_distance
        move_forward(power_val, dist)
    else:
        return
    
def drive_in_circle(radius, speed, duration):
    time.sleep(1)
    turn_right_angle(power_val, 60)

    segs = 18
    circumference = 2 * math.pi * radius
    seg_dist = circumference/segs

    cm_per_sec = 20
    forward_time = seg_dist/(cm_per_sec*speed/10)
    turn_time = 0.06

    start_time = time.time()
    try:
        while time.time()-start_time < duration:
            fc.forward(speed)
            time.sleep(forward_time)
            fc.stop()
            time.sleep(0.1)
            fc.turn_left(speed)
            time.sleep(turn_time)
    finally:
        fc.stop()  


def follow_objects():
    global PIXELS_PER_METER
    res_width = 1280 # 640
    res_height = 960 # 480
    with Picamera2() as camera:
        camera.preview_configuration.main.size = (res_width, res_height) #width * height
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        capture_background(camera)

        while True:
            start = input("ready to start: ")
            if start == "s":
                break

        try:
            while True:
                bbs, centers, current_img, diff_mask = detect_objects_by_difference(camera)

                cv2.imshow("Current View", current_img)
                cv2.imshow("Difference Mask", diff_mask)
                cv2.imwrite("diff_mask.png", diff_mask)
                cv2.setMouseCallback("Current View", print_HSV, current_img)

                if len(centers) == 2:
                    bb0 = bbs[0]
                    lower_bound0 = bb0[1] + (bb0[3] * 7/10) #get rid of shadows
                    left_bound0 = bb0[0]
                    right_bound0 = bb0[0] + bb0[2]

                    dist_scale0 = (res_height - lower_bound0)/(lower_bound0 - res_height/2) # the origin is at top left, y goes down, x goes right
                    dist0 = dist_scale0 * CAMERA_VIEW_DEPTH + CAMERA_VIEW_DEPTH
                    print("obj 0 is centered at (", centers[0][0], ", ", centers[0][1], ")")
                    print("obj 0 is bot at: ", lower_bound0)
                    print("dist_scale0: ", dist_scale0)
                    print("obj 0 is at " + str(dist0) + " away from pov")

                    gap_scale0 = (centers[0][0] - (res_width/2))/(res_width/2) #num pixels away from mid lane
                    img_gap0 = gap_scale0 * (CAMERA_VIEW_WIDTH/2)  # if negative obj to the left of mid lane, vice versa
                    # angle0 = math.atan(img_gap0/CAMERA_VIEW_DEPTH)# angle away from center lane
                    center_gap0 = img_gap0/CAMERA_VIEW_DEPTH * dist0
                    #get left bound distance from mid lane
                    lb_scale0 = (left_bound0 - (res_width/2))/(res_width/2)
                    img_lb0 = lb_scale0 * (CAMERA_VIEW_WIDTH/2)
                    lb0 = img_lb0/CAMERA_VIEW_DEPTH * dist0
                    # right bound dist
                    rb_scale0 = (right_bound0 - (res_width/2))/(res_width/2)
                    img_rb0 = rb_scale0 * (CAMERA_VIEW_WIDTH/2)
                    rb0 = img_rb0/CAMERA_VIEW_DEPTH * dist0
                    radius0 = abs(rb0-lb0)/2
                    
                    bb1 = bbs[1]
                    lower_bound1 = bb1[1] + (bb1[3] * 7/10)
                    left_bound1 = bb1[0]
                    right_bound1 = bb1[0] + bb1[2]

                    dist_scale1 = (res_height - lower_bound1)/(lower_bound1 - res_height/2) 
                    dist1 = dist_scale1 * CAMERA_VIEW_DEPTH + CAMERA_VIEW_DEPTH
                    print("obj 1 is centered at (", centers[1][0], ", ", centers[1][1], ")")
                    print("obj 1 is bot at: ", lower_bound1)
                    print("dist_scale1: ", dist_scale1)
                    print("obj 1 is at " + str(dist1) + " away from pov")

                    gap_scale1 = (centers[1][0] - (res_width/2))/(res_width/2) #num pixels away from mid lane
                    img_gap1 = gap_scale1 * (CAMERA_VIEW_WIDTH/2)  # get the actual meters on screen. if negative obj to the left of mid lane, vice versa
                    center_gap1 = img_gap1/CAMERA_VIEW_DEPTH * dist1
                    #get left bound distance from mid lane
                    lb_scale1 = (left_bound1 - (res_width/2))/(res_width/2)
                    img_lb1 = lb_scale1 * (CAMERA_VIEW_WIDTH/2)
                    lb1 = img_lb1/CAMERA_VIEW_DEPTH * dist1
                    # right bound dist
                    rb_scale1 = (right_bound1 - (res_width/2))/(res_width/2)
                    img_rb1 = rb_scale1 * (CAMERA_VIEW_WIDTH/2)
                    rb1 = img_rb1/CAMERA_VIEW_DEPTH * dist1
                    radius1 = abs(rb1-lb1)/2

                    obj0_pos = (center_gap0, dist0)
                    obj1_pos = (center_gap1, dist1)

                    # dx = centers[0][0] - centers[1][0]
                    # dy = centers[0][1] - centers[1][1]
                    # pixel_distance = math.hypot(dx, dy)

                    # actual_distance_m = 1.5
                    # PIXELS_PER_METER = pixel_distance / actual_distance_m

                    # def pixel_to_meter(px, py):
                    #     dx_pixels = px - 320
                    #     dy_pixels = 240 - py
                    #     return dx_pixels / PIXELS_PER_METER, dy_pixels / PIXELS_PER_METER

                    # obj1_pos = pixel_to_meter(*centers[0])
                    # obj2_pos = pixel_to_meter(*centers[1])

                    generate_top_down_view(obj0_pos, obj1_pos)

                    if bb0[0] < bb1[0]:
                        target_pos = obj0_pos
                        radius = radius0 + 0.06
                        print("target radius = ", radius)
                    else: 
                        target_pos = obj1_pos
                        radius = radius1 + 0.06
                        print("target radius = ", radius)

                    # time.sleep(2)
                    drive_towards(target_pos, radius)
                    drive_in_circle(radius*100, speed=20, duration=25)

                    break 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Drive Through Center")
    try:
        follow_objects()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        fc.stop()
