from ultralytics import YOLO
import cv2
import time

# Load your trained model
model = YOLO(r"C:\Users\anujt\OneDrive\Desktop\ML\Face_detection_Yolov8s\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit() 

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Run YOLO prediction (disable internal show for smoother processing)
    results = model(frame, verbose=False)[0]  # get first result object

    # Draw bounding boxes and labels
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get box coordinates
        conf = box.conf[0].item()  # confidence
        cls = int(box.cls[0].item())  # class index
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw rectangle and label
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # green for class 0, red for others
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
