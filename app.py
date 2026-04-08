"""
app.py  –  Hand Gesture Recognition  |  Major Project
======================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from gesture_engine import predict_gesture, labels

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSense · Gesture Recognition",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #080b12;
    --surface:   #0f1420;
    --border:    #1e2640;
    --accent:    #00e5ff;
    --accent2:   #7b61ff;
    --success:   #00ffb3;
    --warn:      #ffb700;
    --danger:    #ff4566;
    --text:      #e8eaf6;
    --muted:     #5c6080;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: .7rem;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: .25rem .75rem;
    border-radius: 2px;
    margin-bottom: 1rem;
    opacity: .85;
}
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: .85rem;
    margin-top: .6rem;
    letter-spacing: .05em;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* ── Prediction display ── */
.pred-wrap {
    text-align: center;
    padding: 2rem 1rem;
}
.pred-label {
    font-family: var(--font-head);
    font-size: clamp(2.5rem, 6vw, 5rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--success), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.02em;
    line-height: 1;
    animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50%       { opacity: .75; }
}
.pred-none {
    font-family: var(--font-head);
    font-size: 2rem;
    color: var(--muted);
    font-weight: 600;
}
.pred-conf {
    font-family: var(--font-mono);
    font-size: .8rem;
    color: var(--muted);
    margin-top: .4rem;
    letter-spacing: .1em;
}

/* ── Confidence bar ── */
.conf-bar-wrap { margin: .8rem 0 .3rem; }
.conf-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent2), var(--accent), var(--success));
    transition: width .3s ease;
}

/* ── Stats pill ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: var(--border);
    border-radius: 99px;
    padding: .3rem .9rem;
    font-size: .75rem;
    font-family: var(--font-mono);
    color: var(--text);
    margin: .2rem;
}
.stat-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--success);
    animation: blink 1.2s infinite;
}
.stat-dot.off { background: var(--danger); animation: none; }
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: .2; }
}

/* ── History chips ── */
.hist-wrap { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
.hist-chip {
    font-family: var(--font-mono);
    font-size: .7rem;
    background: var(--border);
    border: 1px solid var(--accent2);
    color: var(--accent);
    border-radius: 4px;
    padding: .15rem .55rem;
}

/* ── Sidebar labels ── */
.label-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .35rem;
    margin-top: .5rem;
}
.label-item {
    background: var(--border);
    border-radius: 6px;
    padding: .3rem .5rem;
    font-size: .72rem;
    font-family: var(--font-mono);
    color: var(--text);
    text-align: center;
    border-left: 2px solid var(--accent2);
}

/* ── Streamlit overrides ── */
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
    width: 100% !important;
}
.stCheckbox label { color: var(--text) !important; font-family: var(--font-mono) !important; }
.stSlider label   { color: var(--text) !important; }
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .8rem 1rem !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "camera"       not in st.session_state: st.session_state.camera       = None
if "running"      not in st.session_state: st.session_state.running      = False
if "history"      not in st.session_state: st.session_state.history      = deque(maxlen=12)
if "fps_times"    not in st.session_state: st.session_state.fps_times    = deque(maxlen=30)
if "total_frames" not in st.session_state: st.session_state.total_frames = 0
if "detections"   not in st.session_state: st.session_state.detections   = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.8rem 0 1.2rem'>
        <div style='font-family:var(--font-head); font-size:1.4rem; font-weight:800;
                    background:linear-gradient(135deg,#00e5ff,#7b61ff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ✋ SignSense
        </div>
        <div style='font-size:.68rem; color:#5c6080; letter-spacing:.12em; margin-top:.2rem'>
            GESTURE RECOGNITION SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    run = st.checkbox("▶  Start Camera", value=st.session_state.running)

    st.markdown("**Camera Index**")
    cam_index = st.number_input("", min_value=0, max_value=5, value=0,
                                 label_visibility="collapsed")

    st.markdown("**Confidence Threshold**")
    conf_threshold = st.slider("", 0.3, 0.99, 0.60, 0.01,
                                label_visibility="collapsed")

    show_canvas = st.checkbox("Show preprocessed canvas", value=False)

    st.divider()
    st.markdown("**Recognisable Gestures**")
    label_html = "".join(f'<div class="label-item">{l}</div>' for l in labels)
    st.markdown(f'<div class="label-grid">{label_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:.65rem; color:#5c6080; line-height:1.6; text-align:center'>
        Major Project · Computer Vision<br>Hand Gesture → Text Translation<br>
        <span style='color:#00e5ff'>cvzone · keras · streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🎓 Major Project Presentation</div>
    <h1 class="hero-title">Hand Gesture Recognition</h1>
    <p class="hero-sub">Real-time Sign Language Detection using Deep Learning & Computer Vision</p>
</div>
""", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--font-mono); font-size:.7rem; color:var(--muted);
                letter-spacing:.15em; text-transform:uppercase; margin-bottom:.8rem'>
        📷 Live Camera Feed
    </div>
    """, unsafe_allow_html=True)
    frame_placeholder = st.empty()
    canvas_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    # ── Prediction card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--font-mono); font-size:.7rem; color:var(--muted);
                letter-spacing:.15em; text-transform:uppercase; margin-bottom:.5rem'>
        🧠 Current Prediction
    </div>
    """, unsafe_allow_html=True)
    pred_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Stats ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--font-mono); font-size:.7rem; color:var(--muted);
                letter-spacing:.15em; text-transform:uppercase; margin-bottom:.8rem'>
        📊 Session Stats
    </div>
    """, unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    fps_ph    = mc1.empty()
    frames_ph = mc2.empty()
    detect_ph = mc3.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── History card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:var(--font-mono); font-size:.7rem; color:var(--muted);
                letter-spacing:.15em; text-transform:uppercase; margin-bottom:.6rem'>
        🕑 Detection History
    </div>
    """, unsafe_allow_html=True)
    hist_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_prediction(result: dict):
    label = result["label"]
    conf  = result["confidence"]
    pct   = int(conf * 100)

    if label in ("No Hand Detected", "Uncertain…"):
        html = f"""
        <div class="pred-wrap">
            <div class="pred-none">{label}</div>
        </div>"""
    else:
        html = f"""
        <div class="pred-wrap">
            <div class="pred-label">{label}</div>
            <div class="pred-conf">confidence · {pct}%</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%"></div>
                </div>
            </div>
        </div>"""
    pred_placeholder.markdown(html, unsafe_allow_html=True)


def render_history():
    if not st.session_state.history:
        hist_placeholder.markdown(
            '<div style="color:var(--muted); font-size:.75rem">No gestures detected yet…</div>',
            unsafe_allow_html=True)
        return
    chips = "".join(f'<span class="hist-chip">{g}</span>'
                    for g in reversed(st.session_state.history))
    hist_placeholder.markdown(
        f'<div class="hist-wrap">{chips}</div>', unsafe_allow_html=True)


def annotate_frame(frame_rgb: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding box and label on the frame."""
    img = frame_rgb.copy()
    bbox = result.get("hand_bbox")
    if bbox:
        x, y, w, h = bbox
        label = result["label"]
        conf  = result["confidence"]

        # Box
        cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20),
                       (0, 229, 255), 2)
        # Background chip
        tag = f"{label}  {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, .7, 2)
        cv2.rectangle(img, (x - 20, y - 50), (x - 20 + tw + 10, y - 20),
                       (0, 229, 255), -1)
        cv2.putText(img, tag, (x - 15, y - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (8, 11, 18), 2)
    return img


# ── Camera management ─────────────────────────────────────────────────────────
def open_camera(index):
    if st.session_state.camera is not None:
        st.session_state.camera.release()
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    st.session_state.camera = cap

def close_camera():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None


# ── Idle state ────────────────────────────────────────────────────────────────
if not run:
    close_camera()
    st.session_state.running = False

    frame_placeholder.markdown("""
    <div style='
        background: var(--surface);
        border: 2px dashed var(--border);
        border-radius: 10px;
        height: 340px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        font-family: var(--font-mono);
        font-size: .85rem;
        gap: .6rem;
    '>
        <span style="font-size:2.5rem">📷</span>
        <span>Enable camera from the sidebar</span>
        <span style="font-size:.7rem; opacity:.5">▶  Start Camera → toggle</span>
    </div>
    """, unsafe_allow_html=True)

    render_prediction({"label": "No Hand Detected", "confidence": 0.0,
                        "hand_bbox": None, "canvas": None})
    fps_ph.metric("FPS", "–")
    frames_ph.metric("Frames", "–")
    detect_ph.metric("Detections", "–")
    render_history()
    st.stop()


# ── Running ───────────────────────────────────────────────────────────────────
if not st.session_state.running:
    open_camera(cam_index)
    st.session_state.running      = True
    st.session_state.total_frames = 0
    st.session_state.detections   = 0
    st.session_state.fps_times.clear()

cap = st.session_state.camera

# Override engine threshold from slider
import gesture_engine as _ge
_ge.CONFIDENCE_THRESHOLD = conf_threshold

last_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Camera not accessible. Check index or permissions.")
        close_camera()
        st.session_state.running = False
        break

    # FPS tracking
    now = time.time()
    st.session_state.fps_times.append(now)
    times = st.session_state.fps_times
    fps   = len(times) / (times[-1] - times[0] + 1e-9) if len(times) > 1 else 0

    # Predict
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = predict_gesture(frame_rgb)

    st.session_state.total_frames += 1
    real_detect = result["label"] not in ("No Hand Detected", "Uncertain…")
    if real_detect:
        st.session_state.detections += 1
        if result["label"] != last_label:
            st.session_state.history.append(result["label"])
            last_label = result["label"]

    # Annotated frame
    annotated = annotate_frame(frame_rgb, result)
    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

    # Canvas
    if show_canvas and result["canvas"] is not None:
        canvas_placeholder.image(
            cv2.cvtColor(result["canvas"], cv2.COLOR_BGR2RGB),
            caption="Preprocessed canvas (model input)",
            width=150)
    else:
        canvas_placeholder.empty()

    # Prediction
    render_prediction(result)

    # Stats
    fps_ph.metric("FPS",        f"{fps:.1f}")
    frames_ph.metric("Frames",  st.session_state.total_frames)
    detect_ph.metric("Detections", st.session_state.detections)

    # History
    render_history()

    time.sleep(0.03)       # ~30 fps cap
