import cv2
import numpy as np

# ─── Load DNN face detector ───────────────────────────────────────────────────
# Download these two files from OpenCV's GitHub:
# https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
#   - deploy.prototxt
#   - res10_300x300_ssd_iter_140000.caffemodel

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

CONFIDENCE_THRESHOLD = 0.5   # Ignore detections below 50% confidence
BLUR_PADDING = 10         # Extra pixels around face for better coverage

# ─── Start webcam ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ─── Prepare blob for DNN (resizes to 300x300 internally) ────────────────
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        scalefactor=1.0, 
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)  # Mean BGR values the model was trained with
    )

    net.setInput(blob)
    detections = net.forward()

    # ─── Process each detection ───────────────────────────────────────────────
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Scale bounding box back to original frame size
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Add padding and clamp to frame bounds
        x1 = max(0, x1 - BLUR_PADDING)
        y1 = max(0, y1 - BLUR_PADDING)
        x2 = min(w, x2 + BLUR_PADDING)
        y2 = min(h, y2 + BLUR_PADDING)

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            continue

        # Blur strength scales with face size for consistent effect
        PIXEL_SIZE = 15
        face_h, face_w = face_region.shape[:2]

        small_w = max(1, face_w // PIXEL_SIZE)
        small_h = max(1, face_h // PIXEL_SIZE)

        temp = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (face_w, face_h), interpolation=cv2.INTER_NEAREST)

        frame[y1:y2, x1:x2] = pixelated

        #---

        # Optional: show confidence score above the blurred region
        label = f"{confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Blur (DNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()