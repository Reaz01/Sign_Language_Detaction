"""
Step 3 – Train a RandomForest classifier on the landmark dataset.

Reads  : dataset.npz
Writes : model.joblib
"""

import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from function import DATASET_PATH, MODEL_PATH

# ── Load dataset ──────────────────────────────────────────────────────────────
if not __import__('os').path.exists(DATASET_PATH):
    print(f"[ERROR] {DATASET_PATH} not found. Run data.py first.")
    sys.exit(1)

data        = np.load(DATASET_PATH, allow_pickle=True)
X           = data['X']
y           = data['y']
label_names = data['label_names']

print(f"Dataset: {X.shape[0]} samples, {len(label_names)} classes → {', '.join(label_names)}")

unique, counts = np.unique(y, return_counts=True)
if len(unique) < 2:
    print("[ERROR] Need at least 2 classes with samples.")
    sys.exit(1)
if counts.min() < 2:
    print("[ERROR] Every class needs at least 2 samples. Collect more images.")
    sys.exit(1)

# ── Split ─────────────────────────────────────────────────────────────────────
test_size = max(len(unique), int(round(len(y) * 0.20)))
test_size = min(test_size, len(y) - len(unique))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training RandomForest …")
model = Pipeline([
    ('scaler',     StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight='balanced',
    )),
])
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds    = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
report   = classification_report(y_test, preds, target_names=label_names, zero_division=0)

print(f"\nValidation accuracy : {accuracy:.3f}  ({len(y_test)} samples)")
print(report)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump({'model': model, 'label_names': label_names}, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
print("Next → run app.py")
