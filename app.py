"""
Pothole Detection - Streamlit app (ResNet-50)

Run: streamlit run app.py
Needs: models/pothole_resnet50.keras and models/class_names.json
"""
import hashlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pothole_resnet50.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)

# Your train.py applies ResNet preprocessing before the model.
PREPROCESS_INSIDE_MODEL = False

# Fixed decision threshold (Settings section removed)
DEFAULT_THRESHOLD = 0.50

st.set_page_config(page_title="Pothole Detection", page_icon="🛣️", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp { 
        font-family: "Inter", sans-serif; 
        color: #111827 !important; 
    }
    
    .stApp { background: #f7f7fc; }
    .block-container { max-width: 1100px; padding: 2rem 2rem 3rem; }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* Top bar */
    .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .brand-icon { width:42px; height:42px; border-radius:12px; background:#7b7dfc;
                  display:flex; align-items:center; justify-content:center; font-size:20px;
                  box-shadow:0 6px 16px rgba(123,125,252,.25); }
    .brand-name { font-size:19px; font-weight:800; color:#111827; line-height:1.1; }
    .brand-sub { font-size:11px; color:#4b5563; margin-top:3px; }
    .pill { background:#eeedff; color:#4f46e5; border:1px solid #dfdefd; border-radius:30px;
            padding:7px 14px; font-size:11px; font-weight:700; }

    /* Hero */
    .hero { position:relative; overflow:hidden; background:linear-gradient(135deg,#7b7dfc,#686be2);
            border-radius:24px; padding:34px 40px; margin-bottom:26px;
            box-shadow:0 16px 36px rgba(93,95,205,.18); }
    .hero::after { content:""; position:absolute; width:220px; height:220px; border-radius:50%;
                   right:-90px; top:-115px; background:rgba(255,255,255,.08); }
    .hero-content { position:relative; z-index:1; }
    .hero-badge { display:inline-block; background:rgba(255,255,255,.15);
                  border:1px solid rgba(255,255,255,.22); color:#fff; border-radius:30px;
                  padding:7px 13px; font-size:10px; font-weight:700; letter-spacing:.4px;
                  margin-bottom:14px; }
    .hero-title { color:#fff; font-size:34px; font-weight:800; line-height:1.15; margin-bottom:10px; }
    .hero-text { color:rgba(255,255,255,.88); font-size:15px; line-height:1.65; max-width:620px; }

    /* Headings & Subtitles */
    .section-title { font-size:20px; font-weight:800; color:#111827; margin:6px 0 7px; }
    .section-subtitle { color:#4b5563; font-size:12px; margin-bottom:15px; }

    /* Radio button options */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] span {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e7e7ef !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(31,34,65,.03) !important;
        overflow: hidden;
        margin-bottom: 12px;
    }

    details summary,
    details summary span,
    details summary p,
    div[data-testid="stExpander"] summary * {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    details summary svg {
        fill: #111827 !important;
        stroke: #111827 !important;
    }

    /* Source card */
    .source-card { background:#fff; border:1px solid #e7e7ef; border-radius:17px;
                   padding:16px 18px 10px; box-shadow:0 7px 23px rgba(31,34,65,.04); margin-bottom:12px; }
    .source-title { color:#111827; font-size:14px; font-weight:700; margin-bottom:4px; }
    .source-text { color:#4b5563; font-size:11px; line-height:1.5; }

    /* File uploader & camera container */
    div[data-testid="stFileUploader"] section,
    div[data-testid*="stFileUploaderDropzone"],
    div[data-testid*="FileUploadDropzone"],
    div[data-testid="stCameraInput"] section {
        background-color: #797BF9 !important;
        background: #797BF9 !important;
        border: 2px dashed rgba(255, 255, 255, 0.5) !important;
        border-radius: 16px !important;
    }

    div[data-testid="stFileUploader"] section *,
    div[data-testid*="stFileUploaderDropzone"] *,
    div[data-testid*="FileUploadDropzone"] *,
    div[data-testid="stCameraInput"] section * {
        color: #ffffff !important;
    }

    /* Uploader inner button */
    div[data-testid="stFileUploader"] section button,
    div[data-testid="stCameraInput"] section button {
        background-color: rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        border: 1.5px solid #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] section button *,
    div[data-testid="stCameraInput"] section button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Regular Buttons (Download CSV, Clear History) */
    div.stButton > button,
    div.stDownloadButton > button {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        padding: 0.5rem 1rem !important;
    }

    div.stButton > button *,
    div.stDownloadButton > button * {
        color: #111827 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #7b7dfc !important;
        color: #4f46e5 !important;
    }

    /* Result Card & Stats */
    .image-card { background:#fff; border:1px solid #e7e7ef; border-radius:20px; padding:15px;
                  box-shadow:0 8px 26px rgba(31,34,65,.05); }
    .image-label { color:#4b5563; font-size:10px; font-weight:700; letter-spacing:.8px;
                   text-transform:uppercase; margin-bottom:10px; }
    div[data-testid="stImage"] { display:flex; justify-content:center; }
    div[data-testid="stImage"] img { width:100%; max-height:460px; object-fit:contain; border-radius:14px; }

    .result { background:#fff; border:1px solid #e7e7ef; border-radius:20px; padding:22px;
              box-shadow:0 8px 26px rgba(31,34,65,.05); }
    .result.pothole { border-top:5px solid #d9534f; }
    .result.normal { border-top:5px solid #39845f; }
    .result-label { font-size:10px; font-weight:700; letter-spacing:.8px; color:#4b5563; }
    .result-title { font-size:25px; font-weight:800; margin:5px 0 16px; }
    .result-title.pothole { color:#d9534f; }
    .result-title.normal { color:#39845f; }

    .stats { display:flex; gap:10px; margin-bottom:16px; }
    .stat { flex:1; background:#f7f7ff; border:1px solid #ededfa; border-radius:14px; padding:12px 14px; }
    .stat-value { font-size:20px; font-weight:800; color:#111827; }
    .stat-name { font-size:10px; font-weight:700; letter-spacing:.4px; color:#4b5563; margin-top:2px; }
    
    .bar-label { font-size:11px; font-weight:700; color:#111827; margin-bottom:6px; }
    .bar { position:relative; height:10px; background:#e4e4f1; border-radius:20px; }
    .bar-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#7b7dfc,#6769e0); }
    .bar-mark { position:absolute; top:-4px; width:2px; height:18px; background:#111827; opacity:.65; }
    .bar-scale { display:flex; justify-content:space-between; font-size:10px; color:#4b5563; margin-top:6px; }
    .note { color:#4b5563; font-size:11px; line-height:1.5; margin-top:14px; }

    .foot { text-align:center; color:#4b5563; font-size:11px; margin-top:34px; padding-top:18px;
            border-top:1px solid #e5e5ed; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names


def open_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(image).convert("RGB")


@st.cache_data(show_spinner=False)
def predict_pothole_probability(data: bytes):
    model, class_names = load_model_and_classes()
    names = [c.lower() for c in class_names]
    pothole_index = names.index("pothole") if "pothole" in names else 1

    array = np.asarray(open_image(data).resize(IMAGE_SIZE), dtype=np.float32)
    if not PREPROCESS_INSIDE_MODEL:
        array = preprocess_input(array)

    start = time.perf_counter()
    score = float(model.predict(array[None, ...], verbose=0)[0][0])
    elapsed_ms = (time.perf_counter() - start) * 1000

    probability = score if pothole_index == 1 else 1.0 - score
    return probability, elapsed_ms


if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("The trained model was not found.")
    st.markdown(
        f"""
        The app expects these two files:

        - `{MODEL_PATH.relative_to(ROOT)}`
        - `{CLASS_NAMES_PATH.relative_to(ROOT)}`

        Train the model first, then restart the app.
        """
    )
    st.stop()


st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">🛣️</div>
            <div>
                <div class="brand-name">Pothole Detection</div>
                <div class="brand-sub">ML-based road analysis</div>
            </div>
        </div>
        <div class="pill">RESNET-50</div>
    </div>

    <div class="hero">
        <div class="hero-content">
            <div class="hero-badge">✦ ML-POWERED DETECTION</div>
            <div class="hero-title">Is this road safe?<br>Check it in seconds.</div>
            <div class="hero-text">
                Upload a road photo or take one with your camera. The model tells you
                whether the image contains a pothole, and how sure it is.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">1. Choose an image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Upload an existing road photo or use your device camera.</div>',
    unsafe_allow_html=True,
)

source = st.radio(
    "Image source",
    ["📁 Upload a photo", "📷 Use camera"],
    horizontal=True,
    label_visibility="collapsed",
)

if source.startswith("📁"):
    st.markdown(
        """
        <div class="source-card">
            <div class="source-title">📁 Upload a road photo</div>
            <div class="source-text">JPG, JPEG, PNG, BMP or WebP images are supported.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chosen = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
else:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-title">📷 Capture a road photo</div>
            <div class="source-text">Use your device camera to capture the road directly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chosen = st.camera_input("Take a road picture", label_visibility="collapsed")


if chosen is not None:
    data = chosen.getvalue()

    try:
        image = open_image(data)
    except Exception:
        st.error("Could not read this file as an image. Please try another image.")
        st.stop()

    st.markdown('<div class="section-title" style="margin-top:26px;">2. Detection result</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing road image..."):
        probability, elapsed_ms = predict_pothole_probability(data)

    threshold = DEFAULT_THRESHOLD
    is_pothole = probability >= threshold
    confidence = probability if is_pothole else 1.0 - probability
    css = "pothole" if is_pothole else "normal"
    title = "⚠️ Pothole detected" if is_pothole else "✅ Normal road"

    image_col, result_col = st.columns([1.25, 1], gap="large")

    with image_col:
        st.markdown(
            '<div class="image-card"><div class="image-label">Selected road image</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        st.markdown(
            f"""
            <div class="result {css}">
                <div class="result-label">DETECTION RESULT</div>
                <div class="result-title {css}">{title}</div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{confidence * 100:.1f}%</div>
                        <div class="stat-name">CONFIDENCE</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{elapsed_ms:.0f} ms</div>
                        <div class="stat-name">ANALYSIS TIME</div>
                    </div>
                </div>
                <div class="bar-label">POTHOLE SCORE &nbsp;{probability * 100:.1f}%</div>
                <div class="bar">
                    <div class="bar-fill" style="width:{probability * 100:.1f}%"></div>
                    <div class="bar-mark" style="left:{threshold * 100:.0f}%"></div>
                </div>
                <div class="bar-scale"><span>Normal</span><span>Pothole</span></div>
                <div class="note">
                    The dark marker is the decision threshold ({threshold:.2f}).
                    This model checks the whole image; it does not show where the pothole is.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if abs(probability - threshold) < 0.15:
            st.warning(
                "The model is not very sure about this image. Try a clearer, well-lit road photo."
            )

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.last_key = None

    key = hashlib.md5(data).hexdigest()
    if key != st.session_state.last_key:
        st.session_state.last_key = key
        st.session_state.history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "file": getattr(chosen, "name", "camera"),
                "result": "Pothole" if is_pothole else "Normal",
                "confidence_%": round(confidence * 100, 1),
                "pothole_score_%": round(probability * 100, 1),
            }
        )


history = st.session_state.get("history", [])
if history:
    with st.expander(f"🕘 This session ({len(history)} image{'s' if len(history) != 1 else ''} checked)"):
        df = pd.DataFrame(history)
        st.dataframe(df, hide_index=True, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download results (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="pothole_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with right:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_key = None
                st.rerun()


with st.expander("ℹ️ About this system"):
    st.markdown(
        """
        A ResNet-50 network, pretrained on ImageNet and fine-tuned on labelled road images,
        classifies each photo as **pothole** or **normal road**. Accuracy depends on how varied
        the training data is, so results on very dark, blurry or unusual roads can be less reliable.
        """
    )


st.markdown(
    '<div class="foot">Pothole Detection System &nbsp;•&nbsp; ResNet-50 &nbsp;•&nbsp; TensorFlow &nbsp;•&nbsp; Streamlit</div>',
    unsafe_allow_html=True,
)