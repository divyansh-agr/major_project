"""
datacollection.py
-----------------
Interactive data collection tool for training the gesture model.

Usage:
    python datacollection.py --gesture Hello --target 300

Controls:
    S  →  Save current frame
    Q  →  Quit
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import argparse

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gesture Data Collection")
parser.add_argument("--gesture", type=str, default="Hello",
                    help="Gesture label / folder name")
parser.add_argument("--target",  type=int, default=300,
                    help="Target number of images to collect")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
OFFSET   = 20
IMG_SIZE = 300
FOLDER   = f"Data/{args.gesture}"
TARGET   = args.target
os.makedirs(FOLDER, exist_ok=True)

# ── Init ──────────────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
counter  = 0

print(f"\n{'='*50}")
print(f"  Collecting data for gesture: [{args.gesture}]")
print(f"  Target: {TARGET} images  →  Folder: {FOLDER}")
print(f"  Press  S  to save   |   Q  to quit")
print(f"{'='*50}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_crop(img, y, h, x, w, offset):
    H, W = img.shape[:2]
    return img[max(0, y-offset): min(H, y+h+offset),
               max(0, x-offset): min(W, x+w+offset)]


def hand_to_canvas(crop):
    h, w  = crop.shape[:2]
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    ratio  = h / w
    if ratio > 1:
        k, wCal = IMG_SIZE / h, math.ceil((IMG_SIZE / h) * w)
        wGap = math.ceil((IMG_SIZE - wCal) / 2)
        canvas[:, wGap: wCal + wGap] = cv2.resize(crop, (wCal, IMG_SIZE))
    else:
        k, hCal = IMG_SIZE / w, math.ceil((IMG_SIZE / w) * h)
        hGap = math.ceil((IMG_SIZE - hCal) / 2)
        canvas[hGap: hCal + hGap, :] = cv2.resize(crop, (IMG_SIZE, hCal))
    return canvas


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        print("Camera read failed.")
        break

    display = img.copy()
    hands, img_drawn = detector.findHands(img)

    if hands:
        hand   = hands[0]
        x, y, w, h = hand["bbox"]
        crop   = safe_crop(img, y, h, x, w, OFFSET)

        if crop.size > 0:
            canvas = hand_to_canvas(crop)

            # Overlay stats on canvas
            cv2.putText(canvas, f"Gesture: {args.gesture}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 120, 255), 2)
            cv2.putText(canvas, f"Count: {counter}/{TARGET}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 180, 0), 2)

            cv2.imshow("Canvas (press S to save)", canvas)
            cv2.imshow("Crop",   crop)

        # Draw bbox on main view
        cv2.rectangle(display, (x-OFFSET, y-OFFSET),
                      (x+w+OFFSET, y+h+OFFSET), (0, 229, 255), 2)
        cv2.putText(display, f"{counter}/{TARGET}",
                    (x-OFFSET, y-OFFSET-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 229, 255), 2)

    # Progress bar on main view
    bar_w = int((counter / TARGET) * display.shape[1])
    cv2.rectangle(display, (0, 0), (bar_w, 8), (0, 229, 255), -1)
    cv2.putText(display, f"Collecting: {args.gesture}  [{counter}/{TARGET}]",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    cv2.imshow("Data Collection  (S=Save  Q=Quit)", display)

    key = cv2.waitKey(1)
    if key == ord("s") and hands and crop.size > 0:
        fname = f"{FOLDER}/img_{int(time.time()*1000)}.jpg"
        cv2.imwrite(fname, canvas)
        counter += 1
        print(f"  Saved [{counter}/{TARGET}]  →  {fname}")

        if counter >= TARGET:
            print(f"\n✅  Collected {TARGET} images for [{args.gesture}]. Done!")
            break

    if key == ord("q"):
        print("\nQuit by user.")
        break

cap.release()
cv2.destroyAllWindows()
