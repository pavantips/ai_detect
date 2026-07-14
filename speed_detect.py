import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Store centroid history per track ID
positions = defaultdict(list)
speeds    = defaultdict(float)

# --- Speed calibration ---
# This converts pixel movement to a speed estimate.
# Without a physical reference, treat this as "relative speed" not real mph.
# If you know roughly how wide the frame is in real world (e.g. 3 meters),
# set FRAME_WIDTH_METERS to that value for a rough real-world estimate.
FRAME_WIDTH_METERS = 3.0   # adjust for your scene
HISTORY_FRAMES     = 8     # smooth over N frames to reduce jitter

cap = cv2.VideoCapture(0)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prev_time    = time.time()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps       = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # Run tracking (ByteTrack built in)
    results = model.track(frame, persist=True, verbose=False, conf=0.4)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes  = results[0].boxes.xyxy.cpu().numpy()
        ids    = results[0].boxes.id.cpu().numpy().astype(int)
        labels = [model.names[int(c)] for c in results[0].boxes.cls]
        confs  = results[0].boxes.conf.cpu().numpy()

        for box, track_id, label, conf in zip(boxes, ids, labels, confs):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Store centroid history
            positions[track_id].append((cx, cy, curr_time))
            if len(positions[track_id]) > HISTORY_FRAMES:
                positions[track_id].pop(0)

            # Calculate speed from oldest to newest position
            if len(positions[track_id]) >= 2:
                oldest = positions[track_id][0]
                newest = positions[track_id][-1]
                dx = newest[0] - oldest[0]
                dy = newest[1] - oldest[1]
                dt = newest[2] - oldest[2]
                pixel_dist = (dx**2 + dy**2) ** 0.5
                pixel_per_sec = pixel_dist / max(dt, 1e-6)

                # Convert to real-world estimate
                meters_per_sec = pixel_per_sec * (FRAME_WIDTH_METERS / frame_width)
                mph = meters_per_sec * 2.237

                speeds[track_id] = mph

            spd = speeds.get(track_id, 0)

            # Color by speed: green=slow, yellow=medium, red=fast
            if spd < 2:
                color = (0, 200, 0)
            elif spd < 6:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw centroid trail
            pts = positions[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame,
                         (pts[i-1][0], pts[i-1][1]),
                         (pts[i][0],   pts[i][1]),
                         color, 1)

            # Label
            text = f"{label} #{track_id} | {spd:.1f} mph"
            cv2.rectangle(frame, (int(x1), int(y1)-24), (int(x1)+len(text)*9, int(y1)), color, -1)
            cv2.putText(frame, text, (int(x1)+4, int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # FPS overlay
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()