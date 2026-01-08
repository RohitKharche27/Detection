import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🚀",
    layout="wide"
)

# -------------------- ULTRA ATTRACTIVE CSS --------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #1e293b, #020617);
    color: #f8fafc;
    font-family: 'Poppins', sans-serif;
}

/* Header card */
.header-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(14,165,233,0.15));
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    box-shadow: 0 25px 60px rgba(0,0,0,0.6);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 40px;
}

/* Title */
.header-card h1 {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.header-card p {
    font-size: 1.2rem;
    color: #cbd5f5;
    margin-top: 10px;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Slider */
.stSlider > div {
    padding: 10px 0;
}

/* Success badge */
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: #022c22;
    padding: 8px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
}

/* Class pills */
.pill {
    display: inline-block;
    background: rgba(59,130,246,0.25);
    color: #e0f2fe;
    padding: 6px 14px;
    border-radius: 999px;
    margin: 5px 5px 0 0;
    font-size: 0.85rem;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    margin: 40px 0;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="header-card">
    <h1>🚀 AI Object Detection</h1>
    <p>Next-generation multi-image detection using YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## ⚙️ Detection Controls")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.sidebar.markdown("""
---
**🧠 Model:** YOLOv8 Nano  
**⚡ Speed:** Fast  
**🎯 Use Case:** Object Detection  
""")

# -------------------- UPLOAD --------------------
uploaded_files = st.file_uploader(
    "📤 Upload images (multiple allowed)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# -------------------- PROCESS --------------------
if uploaded_files:
    st.markdown(f"<div class='badge'>📸 {len(uploaded_files)} image(s) uploaded</div>", unsafe_allow_html=True)

    for i, file in enumerate(uploaded_files, 1):
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"<div class='glass'><h3>🖼 Image {i}: {file.name}</h3></div>", unsafe_allow_html=True)

        image = Image.open(file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='glass'><h4>Original</h4></div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("<div class='glass'><h4>Detection Result</h4></div>", unsafe_allow_html=True)
            results = model.predict(img_np, conf=conf, save=False, verbose=False)
            annotated = results[0].plot()
            st.image(annotated, use_container_width=True)

        boxes = results[0].boxes
        count = 0 if boxes is None else len(boxes)

        st.markdown(f"<div class='badge'>✅ Objects detected: {count}</div>", unsafe_allow_html=True)

        if count > 0:
            class_ids = boxes.cls.cpu().numpy()
            class_names = [results[0].names[int(i)] for i in class_ids]

            st.markdown("<br>", unsafe_allow_html=True)
            for name in class_names:
                st.markdown(f"<span class='pill'>{name}</span>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass" style="text-align:center">
        <h3>👆 Upload images to start detection</h3>
        <p>Supports JPG, PNG, WEBP • Multiple images allowed</p>
    </div>
    """, unsafe_allow_html=True)
