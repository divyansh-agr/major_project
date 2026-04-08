"""
gesture_engine.py
-----------------
Core prediction engine for Hand Gesture Recognition.
Compatible with Streamlit Cloud (Python 3.12, no tensorflow direct import).
"""

import cv2
import numpy as np
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF noise

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# ── Config ────────────────────────────────────────────────────────────────────
OFFSET      = 20
IMG_SIZE    = 300
CONFIDENCE_THRESHOLD = 0.60
MODEL_PATH  = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

FALLBACK_LABELS = ["Hello", "I Love You", "No", "Okay", "Please", "Thank You", "Yes"]

# ── Label loader ──────────────────────────────────────────────────────────────
def _load_labels(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            # Teachable Machine prefixes lines with "0 Hello", "1 No" etc.
            cleaned = []
            for line in lines:
                parts = line.split(" ", 1)
                cleaned.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
            return cleaned
    return FALLBACK_LABELS

labels = _load_labels(LABELS_PATH)

# ── Lazy init (avoids crash if model not present on cold start) ───────────────
_detector   = None
_classifier = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = HandDetector(maxHands=1)
    return _detector

def _get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please add model/keras_model.h5 and model/labels.txt to your repo."
            )
        _classifier = Classifier(MODEL_PATH, LABELS_PATH)
    return _classifier

# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_crop(img, y, h, x, w, offset):
    H, W = img.shape[:2]
    return img[max(0, y-offset): min(H, y+h+offset),
               max(0, x-offset): min(W, x+w+offset)]

def _hand_to_canvas(crop) -> np.ndarray:
    h, w   = crop.shape[:2]
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    ratio  = h / w
    if ratio > 1:
        wc = math.ceil(IMG_SIZE / h * w)
        wg = math.ceil((IMG_SIZE - wc) / 2)
        canvas[:, wg: wc + wg] = cv2.resize(crop, (wc, IMG_SIZE))
    else:
        hc = math.ceil(IMG_SIZE / w * h)
        hg = math.ceil((IMG_SIZE - hc) / 2)
        canvas[hg: hc + hg, :] = cv2.resize(crop, (IMG_SIZE, hc))
    return canvas

# ── Public API ────────────────────────────────────────────────────────────────
def predict_gesture(frame_rgb: np.ndarray) -> dict:
    """
    Parameters
    ----------
    frame_rgb : np.ndarray  — RGB image from camera

    Returns
    -------
    dict: label, confidence (0-1), hand_bbox (x,y,w,h)|None, canvas|None
    """
    NO_HAND = {"label": "No Hand Detected", "confidence": 0.0,
               "hand_bbox": None, "canvas": None}

    try:
        detector   = _get_detector()
        classifier = _get_classifier()
    except FileNotFoundError as e:
        return {**NO_HAND, "label": str(e)}

    frame_bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hands, _   = detector.findHands(frame_bgr, draw=False)

    if not hands:
        return NO_HAND

    x, y, w, h = hands[0]["bbox"]
    crop = _safe_crop(frame_bgr, y, h, x, w, OFFSET)
    if crop.size == 0:
        return {**NO_HAND, "hand_bbox": (x, y, w, h)}

    canvas     = _hand_to_canvas(crop)
    prediction, index = classifier.getPrediction(canvas, draw=False)
    confidence = float(max(prediction))

    if confidence < CONFIDENCE_THRESHOLD:
        return {"label": "Uncertain…", "confidence": confidence,
                "hand_bbox": (x, y, w, h), "canvas": canvas}

    return {"label": labels[index], "confidence": confidence,
            "hand_bbox": (x, y, w, h), "canvas": canvas}
