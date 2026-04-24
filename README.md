# Sign_Language_Detaction
Real-Time English Alphabet Recognition Using MediaPipe and scikit-learn
# Sign Language Detection (A–Z)

Real-time hand sign recognition for all 26 English alphabet letters using MediaPipe + LSTM.

## Project layout

```
task/
├── app.py            ← live detection  (run this to use the app)
├── collectdata.py    ← Step 1: capture training images from webcam
├── data.py           ← Step 2: extract MediaPipe keypoints from images
├── train_model.py    ← Step 3: train the LSTM model
├── function.py       ← shared utilities and config
├── Image/            ← collected images  (Image/A/, Image/B/, …)
├── MP_Data/          ← .npy keypoint sequences (created by data.py)
├── model.json        ← saved model architecture (created by train_model.py)
├── model.h5          ← saved model weights    (created by train_model.py)
└── SignLanguageDetection/   ← original sklearn-based package (still works)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run (full pipeline)

### Step 1 – Collect images
```bash
python collectdata.py
```
- A webcam window opens with a white ROI rectangle.
- Press **a–z** to save a snapshot of that letter into `Image/<LETTER>/`.
- Collect at least **30 images per letter** for good results.
- Press **1** to quit.

### Step 2 – Extract keypoints
```bash
python data.py
```
Reads every image in `Image/<LETTER>/`, runs MediaPipe Hands, and saves
the 63-feature landmark vectors as `.npy` files under `MP_Data/`.

### Step 3 – Train the model
```bash
python train_model.py
```
Trains an LSTM network (64→128→64 → Dense 64→32→26) for 200 epochs and
saves `model.json` + `model.h5`.

### Step 4 – Run live detection
```bash
python app.py
```
Opens the webcam. Hold a hand sign inside the white rectangle and the
predicted letter appears in the orange banner at the top.  Press **q** to quit.

## Notes

- Each letter needs at least 30 images in its `Image/` folder before running `data.py`.
- The model predicts A–Z; confidence threshold is 0.8 (adjustable in `app.py`).
- The `SignLanguageDetection/` sub-package (sklearn/RandomForest approach) still works independently via `python -m SignLanguageDetection predict`.

