# ✋ SignSense — Hand Gesture Recognition System
### Major Project | Computer Vision & Deep Learning

---

## 📁 File Structure

```
project/
├── gesture_engine.py          ← Core prediction engine (fixed & improved)
├── app.py                     ← Streamlit web app (premium dark UI)
├── datacollection.py          ← Training data collector
├── GestureRecognition_MajorProject.ipynb  ← Full Jupyter notebook
├── model/
│   ├── keras_model.h5         ← Trained Keras model (you provide)
│   └── labels.txt             ← Class labels (you provide)
└── Data/
    ├── Hello/
    ├── I Love You/
    ├── No/
    ├── Okay/
    ├── Please/
    ├── Thank You/
    └── Yes/
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install opencv-python cvzone tensorflow streamlit numpy Pillow
```

### Step 2 — Collect training data
```bash
python datacollection.py --gesture Hello --target 300
# Repeat for each gesture class
```

### Step 3 — Train model
1. Go to https://teachablemachine.withgoogle.com/
2. Image Project → Standard image model
3. Upload images from `Data/` folders
4. Train → Export as Keras (.h5)
5. Place `keras_model.h5` and `labels.txt` in the `model/` folder

### Step 4 — Run the app
```bash
streamlit run app.py
```

### Step 5 — Or use the Jupyter notebook
```bash
jupyter notebook GestureRecognition_MajorProject.ipynb
```

---

## 🖐 Supported Gestures

| Gesture | Label |
|:---|:---|
| ✋ Open palm | Hello |
| 🤟 ILY sign | I Love You |
| 👎 Thumbs down | No |
| 👌 OK sign | Okay |
| 🙏 Hands together | Please |
| 🤙 Hang loose | Thank You |
| 👍 Thumbs up | Yes |

---

## ⚙️ Key Fixes Made

| File | Fix |
|:---|:---|
| `gesture_engine.py` | BGR/RGB mismatch fixed · safe crop with bounds guard · confidence threshold · dynamic label loading |
| `app.py` | `st.session_state` camera management · no blocking while loop · live FPS counter · history tracking |
| `datacollection.py` | CLI args for gesture/target · bounds-safe crop · progress bar · auto-naming |

---

*Tech Stack: OpenCV · CVZone · Keras / TensorFlow · MediaPipe · Streamlit*
