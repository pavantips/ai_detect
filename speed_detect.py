import csv
import os
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# --- Camera source ---
# 0 = built-in/default camera. If using an external USB webcam, try 1, 2, ...
CAMERA_INDEX = 0

# --- Vehicle classes only (COCO ids) ---
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- Speed calibration ---
# A road receding into the distance is NOT flat in pixel space: a pixel near
# the top of the frame (far from camera) spans much more real-world distance
# than a pixel near the bottom (close to camera). We linearly interpolate
# meters-per-pixel between a "near" and "far" estimate based on vertical
# position in the frame.
#
# These are rough placeholders -- to calibrate properly:
#   1. Pause on a frame (e.g. save it with cv2.imwrite).
#   2. Pick two points near the BOTTOM of the frame whose real-world distance
#      apart you know (e.g. lane width, distance between two driveway lines).
#      METERS_PER_PIXEL_NEAR = real_distance_m / pixel_distance
#   3. Do the same for two points near the TOP of the frame (further away).
#      METERS_PER_PIXEL_FAR = real_distance_m / pixel_distance
METERS_PER_PIXEL_NEAR = 0.02   # at the bottom of the frame (closest to camera)
METERS_PER_PIXEL_FAR  = 0.08   # at the top of the frame (farthest from camera)

HISTORY_FRAMES = 8    # smooth speed over N frames to reduce jitter
TRACK_TIMEOUT  = 1.5  # seconds without a detection before a vehicle is considered "gone"

LOG_PATH = "vehicle_log.csv"


def meters_per_pixel(y, frame_height):
    t = max(0.0, min(1.0, y / frame_height))  # 0 = top (far), 1 = bottom (near)
    return METERS_PER_PIXEL_FAR + t * (METERS_PER_PIXEL_NEAR - METERS_PER_PIXEL_FAR)


def init_log(path):
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["timestamp", "track_id", "vehicle_type", "max_speed_mph", "avg_speed_mph", "duration_sec"])
    return f, writer


# Per-track state
positions   = defaultdict(list)   # track_id -> [(cx, cy, t), ...]
speeds      = defaultdict(float)  # track_id -> current smoothed mph
max_speeds  = defaultdict(float)  # track_id -> highest mph seen
speed_sum   = defaultdict(float)  # track_id -> sum of mph samples (for avg)
speed_count = defaultdict(int)
vehicle_type = {}                 # track_id -> label
first_seen  = {}                  # track_id -> t
last_seen   = {}                  # track_id -> t

log_file, log_writer = init_log(LOG_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev_time    = time.time()

# Rolling FPS stats, printed periodically so real throughput can be measured
fps_window_start = time.time()
fps_frame_count   = 0
FPS_REPORT_SEC    = 5

print(f"Camera resolution: {frame_width}x{frame_height}")
print(f"Logging vehicle speeds to {LOG_PATH}")
print("Press 'q' to quit")


def finalize_track(track_id):
    duration = last_seen[track_id] - first_seen[track_id]
    avg_speed = speed_sum[track_id] / speed_count[track_id] if speed_count[track_id] else 0.0
    log_writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_seen[track_id])),
        track_id,
        vehicle_type.get(track_id, "unknown"),
        f"{max_speeds[track_id]:.1f}",
        f"{avg_speed:.1f}",
        f"{duration:.1f}",
    ])
    log_file.flush()
    for d in (positions, speeds, max_speeds, speed_sum, speed_count, vehicle_type, first_seen, last_seen):
        d.pop(track_id, None)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        fps_frame_count += 1
        if curr_time - fps_window_start >= FPS_REPORT_SEC:
            avg_fps = fps_frame_count / (curr_time - fps_window_start)
            print(f"[perf] avg FPS over last {FPS_REPORT_SEC}s: {avg_fps:.1f}")
            fps_window_start = curr_time
            fps_frame_count = 0

        results = model.track(frame, persist=True, verbose=False, conf=0.4, classes=list(VEHICLE_CLASSES))

        active_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                active_ids.add(track_id)

                if track_id not in first_seen:
                    first_seen[track_id] = curr_time
                last_seen[track_id] = curr_time
                vehicle_type[track_id] = VEHICLE_CLASSES.get(cls_id, "vehicle")

                positions[track_id].append((cx, cy, curr_time))
                if len(positions[track_id]) > HISTORY_FRAMES:
                    positions[track_id].pop(0)

                if len(positions[track_id]) >= 2:
                    oldest = positions[track_id][0]
                    newest = positions[track_id][-1]
                    dx = newest[0] - oldest[0]
                    dy = newest[1] - oldest[1]
                    dt = newest[2] - oldest[2]
                    pixel_dist = (dx**2 + dy**2) ** 0.5
                    pixel_per_sec = pixel_dist / max(dt, 1e-6)

                    avg_cy = (oldest[1] + newest[1]) / 2
                    m_per_px = meters_per_pixel(avg_cy, frame_height)
                    mph = pixel_per_sec * m_per_px * 2.237

                    speeds[track_id] = mph
                    max_speeds[track_id] = max(max_speeds[track_id], mph)
                    speed_sum[track_id] += mph
                    speed_count[track_id] += 1

                spd = speeds.get(track_id, 0)

                if spd < 2:
                    color = (0, 200, 0)
                elif spd < 6:
                    color = (0, 200, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                pts = positions[track_id]
                for i in range(1, len(pts)):
                    cv2.line(frame,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             color, 1)

                label = vehicle_type[track_id]
                text = f"{label} #{track_id} | {spd:.1f} mph"
                cv2.rectangle(frame, (int(x1), int(y1)-24), (int(x1)+len(text)*9, int(y1)), color, -1)
                cv2.putText(frame, text, (int(x1)+4, int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Finalize (log + drop) tracks that haven't been seen recently
        stale_ids = [tid for tid in last_seen if curr_time - last_seen[tid] > TRACK_TIMEOUT]
        for tid in stale_ids:
            finalize_track(tid)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Flush any vehicles still on screen when we quit
    for tid in list(last_seen.keys()):
        finalize_track(tid)
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
