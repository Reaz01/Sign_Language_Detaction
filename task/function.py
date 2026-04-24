"""
Shared utilities for the Sign Language Detection pipeline.
Uses the mediapipe Tasks API (mediapipe >= 0.10) and scikit-learn.
No TensorFlow / Keras required.
"""

import os
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
])

DATASET_PATH = 'dataset.npz'
MODEL_PATH   = 'model.joblib'
IMAGE_DIR    = 'Image'

# Hand landmarker model (reuse the one already downloaded by the sub-package)
_TASK_PATHS = [
    Path(__file__).parent / 'SignLanguageDetection' / 'artifacts' / 'hand_landmarker.task',
    Path(__file__).parent / 'hand_landmarker.task',
]
_TASK_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Model path helper ─────────────────────────────────────────────────────────

def _get_task_path() -> Path:
    for p in _TASK_PATHS:
        if p.exists():
            return p
    # Download to the first candidate location
    dest = _TASK_PATHS[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model → {dest} …")
    with urlopen(_TASK_URL) as r, open(dest, 'wb') as f:
        f.write(r.read())
    return dest

# ── Detector factory ──────────────────────────────────────────────────────────

def create_detector():
    """Return a MediaPipe HandLandmarker (IMAGE running mode)."""
    task_path = _get_task_path()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

# ── Detection helpers ─────────────────────────────────────────────────────────

def mediapipe_detection(image: np.ndarray, detector):
    """Run the detector on a BGR image; returns the raw result object."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect(mp_image)


def extract_keypoints(results) -> np.ndarray:
    """
    Return a normalized 63-element float32 vector (21 landmarks × xyz).
    Normalised: wrist at origin, scale = max 2-D landmark spread.
    Returns zeros when no hand is detected.
    """
    hand_landmarks_list = getattr(results, 'hand_landmarks', None)
    if not hand_landmarks_list:
        return np.zeros(63, dtype=np.float32)

    pts = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks_list[0]],
        dtype=np.float32,
    )
    pts -= pts[0]                                        # wrist to origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))  # max 2-D spread
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_landmarks(image: np.ndarray, results) -> None:
    """Draw hand landmarks and connections onto *image* in-place."""
    hand_landmarks_list = getattr(results, 'hand_landmarks', None)
    if not hand_landmarks_list:
        return
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks_list[0]]
    for pt in pts:
        cv2.circle(image, pt, 4, (0, 255, 255), -1)
    for s, e in HAND_CONNECTIONS:
        cv2.line(image, pts[s], pts[e], (0, 255, 0), 2)
