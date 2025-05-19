import cv2
import numpy as np

def detect_fire_smoke(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([18, 50, 50])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_output = cv2.bitwise_and(frame, frame, mask=fire_mask)
    fire_detected = cv2.countNonZero(fire_mask) > 5000

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    smoke_mask = cv2.absdiff(blurred, gray)
    _, smoke_thresh = cv2.threshold(smoke_mask, 25, 255, cv2.THRESH_BINARY)
    smoke_detected = cv2.countNonZero(smoke_thresh) > 5000

    if fire_detected:
        cv2.putText(frame, "Fire Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if smoke_detected:
        cv2.putText(frame, "Smoke Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = detect_fire_smoke(frame)
    cv2.imshow("Fire and Smoke Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()