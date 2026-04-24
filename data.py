"""
Step 2 – Extract MediaPipe hand landmarks from collected images and save as a dataset.

Reads  : Image/<LETTER>/*.png
Writes : dataset.npz  (X = landmark vectors, y = label indices, label_names = A-Z)

Images where no hand is detected are skipped automatically.
"""

import os
import sys
import cv2
import numpy as np

from function import (
    actions, IMAGE_DIR, DATASET_PATH,
    create_detector, mediapipe_detection, extract_keypoints,
)

print("Loading MediaPipe hand detector …")
detector = create_detector()

features     = []
labels       = []
label_names  = []
skipped      = []
samples_seen = 0

for idx, action in enumerate(actions):
    img_dir = os.path.join(IMAGE_DIR, action)
    if not os.path.isdir(img_dir):
        print(f"  [SKIP] No folder for '{action}'")
        continue

    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0,
    )

    if not image_files:
        print(f"  [SKIP] No images in {img_dir}")
        continue

    kept = 0
    for fname in image_files:
        samples_seen += 1
        path  = os.path.join(img_dir, fname)
        image = cv2.imread(path)
        if image is None:
            skipped.append(path)
            continue

        results   = mediapipe_detection(image, detector)
        keypoints = extract_keypoints(results)

        if np.all(keypoints == 0):
            skipped.append(path)
            continue

        features.append(keypoints)
        labels.append(idx)
        kept += 1

    if kept > 0:
        label_names.append(action)
        print(f"  {action}: {kept} / {len(image_files)} images kept")
    else:
        print(f"  [WARN] '{action}': 0 hands detected in {len(image_files)} images — skipped")

detector.close()

if not features:
    print(
        "\n[ERROR] No hand landmarks detected in any image.\n"
        "Make sure hands are clearly visible inside the ROI when collecting images.\n"
    )
    sys.exit(1)

X = np.vstack(features).astype(np.float32)
y = np.array(labels, dtype=np.int64)

np.savez_compressed(
    DATASET_PATH,
    X=X,
    y=y,
    label_names=np.array(label_names, dtype=object),
)

print(f"\nDataset saved → {DATASET_PATH}")
print(f"Samples seen : {samples_seen}")
print(f"Samples kept : {len(features)}")
print(f"Labels       : {', '.join(label_names)}")
if skipped:
    print(f"Skipped      : {len(skipped)} images (no hand detected)")
print("\nNext → run train_model.py")
