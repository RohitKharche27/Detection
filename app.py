import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 YOLO Object Detection")
st.caption("Reliable & stable Streamlit deployment")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # ✅ STABLE MODEL

model = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")
conf = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("🖼 Original Image")
    st.image(image, use_container_width=True)

    st.subheader("📌 Detection Result")

    with st.spinner("Running YOLO detection..."):
        results = model.predict(
            source=img_np,
            conf=conf,
            save=False,
            verbose=False
        )

        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, use_container_width=True)

    # Detection summary
    boxes = results[0].boxes
    count = 0 if boxes is None else len(boxes)

    st.success(f"✅ Objects detected: {count}")

    if count > 0:
        class_ids = boxes.cls.cpu().numpy()
        class_names = [results[0].names[int(i)] for i in class_ids]
        st.write("### 🏷 Detected Classes")
        st.write(class_names)

else:
    st.info("👆 Upload an image to start detection")
