import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

model = YOLO("best (1).pt")

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")
st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image, and the model will detect bone fractures.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.image(img_array, caption='Uploaded Image', use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        img.save(tmp_file.name)
        results = model(tmp_file.name)
    for r in results:
        result_img = r.plot()
        st.image(result_img, caption="Detection Result", use_column_width=True)

    st.success("✅ Detection completed!")
