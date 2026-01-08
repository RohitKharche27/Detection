import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 700;
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.caption {
    text-align: center;
    font-size: 1.1rem;
    color: #d1d5db;
    margin-bottom: 30px;
}
.stImage, .stFileUploader, .stSlider {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}
.stSuccess {
    background: rgba(16, 185, 129, 0.15);
    border-left: 6px solid #10b981;
    border-radius: 10px;
}
.stInfo {
    background: rgba(59, 130, 246, 0.15);
    border-left: 6px solid #3b82f6;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🎯 YOLO Object Detection")
st.markdown(
    "<p class='caption'>Multiple Image Object Detection using YOLOv8 & Streamlit</p>",
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Detection Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# -------------------- MULTIPLE IMAGE UPLOAD --------------------
uploaded_files = st.file_uploader(
    "📤 Upload Images (Multiple allowed)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# -------------------- PROCESS IMAGES --------------------
if uploaded_files:
    st.success(f"📸 {len(uploaded_files)} image(s) uploaded")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"---\n## 🖼 Image {idx}: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("🔍 Running YOLO detection..."):
            results = model.predict(
                source=img_np,
                conf=conf,
                save=False,
                verbose=False
            )
            annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detection Result", use_container_width=True)

        # -------- Detection Summary --------
        boxes = results[0].boxes
        count = 0 if boxes is None else len(boxes)
        st.success(f"✅ Objects detected: {count}")

        if count > 0:
            class_ids = boxes.cls.cpu().numpy()
            class_names = [results[0].names[int(i)] for i in class_ids]
            st.write("**🏷 Detected Classes:**")
            st.write(class_names)

else:
    st.info("👆 Upload one or more images to start detection")
