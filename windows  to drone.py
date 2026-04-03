import cv2
from ultralytics import YOLO

# ================================
# 1. Load YOLO model
# ================================
model = YOLO(r"runs\detect\train\weights\best.pt")  # Your trained model

# Define target and ignore class IDs
TARGET_CLASSES = [0, 1, 3, 5, 10, 11]
IGNORE_CLASSES = [2, 4, 6, 7, 8, 9, 12]

# Assign a color for each class
CLASS_COLORS = {
    0: (0, 255, 255),   # ATAGS
    1: (255, 0, 0),     # Arjun
    3: (0, 255, 0),     # D30
    5: (255, 0, 255),   # K9-Vajra
    10: (0, 128, 255),  # T72
    11: (128, 0, 128)   # T90
}

# ================================
# 2. Camera/video input
# ================================
WEBCAM_INDEX = 0 # External webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

frame_w = int(cap.get(3))
frame_h = int(cap.get(4))
center_x = frame_w // 2
center_y = frame_h // 2

# Target distance tuning (bounding box area)
TARGET_BOX_AREA = 40000
AREA_TOLERANCE = 10000

# Drone simulated state
drone_airborne = False

# ================================
# 3. Drone control simulation
# ================================
def drone_control(x_offset, area, detected):
    global drone_airborne

    if not drone_airborne:
        print("🛑 Drone on ground. Press 't' to takeoff.")
        return

    if not detected:
        print("🌀 Hovering: No target detected")
        return

    # Left/right control
    if x_offset < -50:
        print("⬅ Move Left")
    elif x_offset > 50:
        print("➡ Move Right")
    else:
        print("🎯 Centered")

    # Forward/backward based on bounding box area
    if area < (TARGET_BOX_AREA - AREA_TOLERANCE):
        print("⬆ Move Forward (object far)")
    elif area > (TARGET_BOX_AREA + AREA_TOLERANCE):
        print("⬇ Move Backward (object close)")
    else:
        print("🟢 Correct distance")

# ================================
# 4. Main loop
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
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
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)
            x_offset = cx - center_x

            # Get class name and color
            class_name = model.names.get(cls_id, "Unknown")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # Simulate drone control
            drone_control(x_offset, area, True)

            # Draw bounding box, center, and class name + confidence
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            label = f"{class_name} ({cls_id})"
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not detected:
        drone_control(0, 0, False)

    # Draw frame center
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.imshow("Drone Simulation YOLO", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🛑 Exiting simulation...")
        break
    elif key == ord('t'):
        drone_airborne = True
        print("🚀 Drone Takeoff (simulated)")
    elif key == ord('l'):
        drone_airborne = False
        print("🛬 Drone Landing (simulated)")

# Cleanup
cap.release()
cv2.destroyAllWindows()
