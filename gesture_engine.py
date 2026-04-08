"""
gesture_engine.py
-----------------
Core prediction engine for Hand Gesture Recognition.
Handles model loading, preprocessing, and inference.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# ── Config ────────────────────────────────────────────────────────────────────
OFFSET       = 20
IMG_SIZE     = 300
CONFIDENCE_THRESHOLD = 0.60          # ignore predictions below 60 % confidence
MODEL_PATH   = "model/keras_model.h5"
LABELS_PATH  = "model/labels.txt"

LABELS = ["Hello", "I Love You", "No", "Okay", "Please", "Thank You", "Yes"]

# ── Initialise (once) ─────────────────────────────────────────────────────────
detector   = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

# ── Dynamic label loading (falls back to hard-coded list) ────────────────────
def _load_labels(path: str) -> list[str]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return LABELS

labels = _load_labels(LABELS_PATH)


# ── Helper: safe crop with bounds guard ──────────────────────────────────────
def _safe_crop(img, y, h, x, w, offset):
    H, W = img.shape[:2]
    y1 = max(0, y - offset)
    y2 = min(H, y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(W, x + w + offset)
    crop = img[y1:y2, x1:x2]
    return crop


# ── Helper: place hand on white canvas (preserving aspect ratio) ─────────────
def _hand_to_canvas(imgCrop) -> np.ndarray:
    h, w = imgCrop.shape[:2]
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    aspectRatio = h / w

    if aspectRatio > 1:                          # taller than wide
        k    = IMG_SIZE / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
        wGap = math.ceil((IMG_SIZE - wCal) / 2)
        canvas[:, wGap: wCal + wGap] = imgResize
    else:                                        # wider than tall
        k    = IMG_SIZE / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
        hGap = math.ceil((IMG_SIZE - hCal) / 2)
        canvas[hGap: hCal + hGap, :] = imgResize

    return canvas


# ── Public API ────────────────────────────────────────────────────────────────
def predict_gesture(frame_rgb: np.ndarray) -> dict:
    """
    Parameters
    ----------
    frame_rgb : np.ndarray
        RGB image frame from camera.

    Returns
    -------
    dict with keys:
        label      : str   – predicted gesture or 'No Hand Detected'
        confidence : float – prediction confidence (0–1)
        hand_bbox  : tuple – (x, y, w, h) or None
        canvas     : np.ndarray | None – preprocessed 300×300 canvas
    """
    # CVZone works in BGR → convert
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hands, _  = detector.findHands(frame_bgr, draw=False)

    if not hands:
        return {"label": "No Hand Detected", "confidence": 0.0,
                "hand_bbox": None, "canvas": None}

    hand = hands[0]
    x, y, w, h = hand["bbox"]

    imgCrop = _safe_crop(frame_bgr, y, h, x, w, OFFSET)
    if imgCrop.size == 0:
        return {"label": "No Hand Detected", "confidence": 0.0,
                "hand_bbox": (x, y, w, h), "canvas": None}

    canvas = _hand_to_canvas(imgCrop)
    prediction, index = classifier.getPrediction(canvas, draw=False)

    confidence = float(max(prediction))
    if confidence < CONFIDENCE_THRESHOLD:
        return {"label": "Uncertain…", "confidence": confidence,
                "hand_bbox": (x, y, w, h), "canvas": canvas}

    return {
        "label":     labels[index],
        "confidence": confidence,
        "hand_bbox": (x, y, w, h),
        "canvas":    canvas,
    }
