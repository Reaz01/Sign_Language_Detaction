

import os
import sys

import cv2
import joblib
import numpy as np

from function import (
    MODEL_PATH,
    create_detector, mediapipe_detection, extract_keypoints, draw_landmarks,
)

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(
        f"[ERROR] {MODEL_PATH} not found.\n"
        "Run the pipeline first:\n"
        "  1. python collectdata.py\n"
        "  2. python data.py\n"
        "  3. python train_model.py\n"
    )
    sys.exit(1)

payload = joblib.load(MODEL_PATH)
model = payload['model']
label_names = payload['label_names']

print(f"Model loaded. Classes: {', '.join(label_names)}")

# ── Open camera ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(
        "\n[ERROR] Could not open the camera.\n"
        "On macOS grant camera access:\n"
        "  System Settings → Privacy & Security → Camera → enable Terminal / VS Code\n"
    )
    sys.exit(1)

print("Camera opened. Press q to quit.")

# ── Detector ──────────────────────────────────────────────────────────────────
detector = create_detector()
threshold = 0.65    # minimum confidence to display a prediction
sentence = []
confidence = []

# ── Main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # ROI for detection
    roi = frame[40:400, 0:300]
    results = mediapipe_detection(roi, detector)

    # Draw landmarks on the ROI region of the display frame
    roi_display = display[40:400, 0:300]
    draw_landmarks(roi_display, results)

    # ROI rectangle
    cv2.rectangle(display, (0, 40), (300, 400), (255, 255, 255), 2)

    # ── Predict ───────────────────────────────────────────────────────────────
    keypoints = extract_keypoints(results)
    label_text = ""
    confidence_text = ""

    if not np.all(keypoints == 0):
        probabilities = model.predict_proba([keypoints])[0]
        best_idx = int(probabilities.argmax())
        best_prob = float(probabilities[best_idx])

        if best_prob >= threshold:
            predicted = label_names[best_idx]
            label_text = predicted
            confidence_text = f"{best_prob:.0%}"

            # Update sentence (keep only the most recent letter)
            if not sentence or sentence[-1] != predicted:
                sentence = [predicted]
                confidence = [confidence_text]
        else:
            label_text = "?"
            confidence_text = f"{best_prob:.0%}"

    # ── Output banner ─────────────────────────────────────────────────────────
    cv2.rectangle(display, (0, 0), (640, 40), (245, 117, 16), -1)
    output = f"Output: {' '.join(sentence)}  {' '.join(confidence)}"
    cv2.putText(display, output, (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection  A-Z  (Press 1 to quit)", display)

    if cv2.waitKey(10) & 0xFF == ord('1'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
