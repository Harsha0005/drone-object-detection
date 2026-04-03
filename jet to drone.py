import cv2
from ultralytics import YOLO
from dronekit import connect, VehicleMode
import time

# ================================
# 1. Connect to Pixhawk 4
# ================================
print("🔌 Connecting to Pixhawk 4...")
vehicle = connect("/dev/ttyTHS1", baud=57600, wait_ready=True)
print("✅ Connected to Pixhawk 4")

# ================================
# 2. Load YOLO model
# ================================
model = YOLO("runs/detect/train/weights/best.pt")  # path to your model

TARGET_CLASSES = [0, 1, 3, 5, 10, 11]
IGNORE_CLASSES = [2, 4, 6, 7, 8, 9, 12]

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 0, 0),
    3: (0, 255, 0),
    5: (255, 0, 255),
    10: (0, 128, 255),
    11: (128, 0, 128)
}

# ================================
# 3. Takeoff function
# ================================
def arm_and_takeoff(altitude):
    while not vehicle.is_armable:
        print("⏳ Waiting for vehicle to be armable...")
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print("⏳ Waiting for arming...")
        time.sleep(1)
    print(f"🚀 Taking off to {altitude} meters...")
    vehicle.simple_takeoff(altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"📡 Altitude: {alt:.2f} m")
        if alt >= altitude * 0.95:
            print("🟢 Reached target altitude")
            break
        time.sleep(1)

# ================================
# 4. Send velocity (NED) function
# ================================
def send_ned_velocity(vx, vy, vz):
    """
    vx: forward/backward
    vy: left/right
    vz: up/down
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ================================
# 5. Camera Setup (SIYI ZR10)
# ================================
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Cannot open SIYI ZR10 camera")
    exit()

frame_w = int(cap.get(3))
frame_h = int(cap.get(4))
center_x = frame_w // 2
center_y = frame_h // 2

TARGET_BOX_AREA = 40000
AREA_TOLERANCE = 10000

# ================================
# 6. Main Loop
# ================================
airborne = False
print("Press 't' to takeoff, 'l' to land, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed")
        break

    results = model(frame, stream=True)
    detected = False

    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            cls_id = int(cls_id)
            if cls_id in IGNORE_CLASSES:
                continue
            if cls_id not in TARGET_CLASSES:
                continue

            detected = True
            x1, y1, x2, y2 = box
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            area = (x2 - x1) * (y2 - y1)
            x_offset = cx - center_x

            if airborne:
                # Left-right control
                if x_offset < -50:
                    send_ned_velocity(0, -0.5, 0)
                elif x_offset > 50:
                    send_ned_velocity(0, 0.5, 0)
                else:
                    send_ned_velocity(0, 0, 0)
                # Forward/backward
                if area < (TARGET_BOX_AREA - AREA_TOLERANCE):
                    send_ned_velocity(0.5, 0, 0)
                elif area > (TARGET_BOX_AREA + AREA_TOLERANCE):
                    send_ned_velocity(-0.5, 0, 0)
                else:
                    send_ned_velocity(0, 0, 0)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            class_name = model.names.get(cls_id, "Unknown")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not detected and airborne:
        send_ned_velocity(0, 0, 0)  # hover

    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.imshow("ZR10 Live Feed - YOLO Follow", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t') and not airborne:
        airborne = True
        arm_and_takeoff(3)  # takeoff to 3m
    elif key == ord('l') and airborne:
        airborne = False
        vehicle.mode = VehicleMode("LAND")

cap.release()
cv2.destroyAllWindows()
vehicle.close()
