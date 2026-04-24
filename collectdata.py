import os
import sys
import cv2

IMAGE_DIR = 'Image'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Create base folder
os.makedirs(IMAGE_DIR, exist_ok=True)

# Create A-Z folders
for letter in LETTERS:
    os.makedirs(os.path.join(IMAGE_DIR, letter), exist_ok=True)

print("Images will be saved in:", os.path.abspath(IMAGE_DIR))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    sys.exit(1)

print("Camera opened.")
print("Press a-z to save image for that letter.")
print("Press 1 to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)

    # ROI coordinates
    x1, y1 = 0, 40
    x2, y2 = 300, 400

    # Make sure frame is large enough
    h, w = frame.shape[:2]
    if h < y2 or w < x2:
        print("[ERROR] Frame too small for ROI.")
        break

    roi = frame[y1:y2, x1:x2]

    # Count images
    count = {}
    for letter in LETTERS:
        folder_path = os.path.join(IMAGE_DIR, letter)
        count[letter] = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Show counts
    for i, letter in enumerate(LETTERS):
        x_pos = (i % 13) * 48 + 5
        y_pos = frame.shape[0] - 40 + (i // 13) * 18
        cv2.putText(
            frame,
            f"{letter}:{count[letter]}",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_PLAIN,
            0.9,
            (0, 255, 255),
            1
        )

    cv2.putText(
        frame,
        "Press a-z to save | Press 1 to quit",
        (5, 30),
        cv2.FONT_HERSHEY_PLAIN,
        1.1,
        (0, 255, 0),
        1
    )

    cv2.imshow("Collect Data", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('1'):
        break

    for letter in LETTERS:
        if key == ord(letter.lower()):
            path = os.path.join(IMAGE_DIR, letter, f"{count[letter]}.png")

            if roi.size == 0:
                print("[ERROR] ROI is empty. Image not saved.")
                break

            success = cv2.imwrite(path, roi)

            if success:
                print(f"[SAVED] {path}")
            else:
                print(f"[ERROR] Failed to save {path}")
            break

cap.release()
cv2.destroyAllWindows()
