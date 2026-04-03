import cv2
from ultralytics import YOLO
import time

# ================================
# 1. Load YOLO model
# ================================
model = YOLO(r"runs\detect\train16\weights\best.pt")  # your trained model

# ================================
# 2. Camera/video input
# ================================
cap = cv2.VideoCapture(0)  # external webcam
frame_w = int(cap.get(3))
frame_h = int(cap.get(4))
center_x = frame_w // 2
center_y = frame_h // 2

TARGET_BOX_AREA = 40000
AREA_TOLERANCE = 10000
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

    if x_offset < -50:
        print("⬅ Move Left")
    elif x_offset > 50:
        print("➡ Move Right")
    else:
        print("🎯 Centered")

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
        # Iterate through each detection
        for box in r.boxes:
            detected = True

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)
            x_offset = cx - center_x

            # Class name and confidence
            cls_id = int(box.cls[0])
            class_name = model.names.get(cls_id, "Unknown")
            conf = float(box.conf[0])  # confidence score

            drone_control(x_offset, area, True)

            # Draw rectangle and center
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw class name + confidence BELOW box (to avoid going out of frame)
            label = f"{class_name} {conf:.2f}"
            text_x = int(x1)
            text_y = int(y2) + 25  # draw below box
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

cap.release()
cv2.destroyAllWindows()