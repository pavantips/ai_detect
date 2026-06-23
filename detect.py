import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano = fastest, good for real-time)
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (larger = more accurate, slower)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera; try 1 if you have multiple)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the frame
    results = model(frame, verbose=False)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Object Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()