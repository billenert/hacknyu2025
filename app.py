from ultralytics import YOLO
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import kagglehub


# path=r"C:\Users\ryanj\.cache\kagglehub\datasets\nadavishai\yolov8-glasses-dataset-v1\versions\4"

# Load YOLO model
model = YOLO("runs/detect/train16/weights/best.pt")  # 👈 Load your trained weights

st.title("YOLO Object Detection with Streamlit")

# Image Upload and Processing
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    results = model(image)
    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Processed Image with Detections", use_column_width=True)

# Video Upload and Processing
uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(tfile, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        stframe.image(annotated_frame, channels="BGR", caption="Processed Video Frame")

    cap.release()

# Webcam Processing with Selection
use_webcam = st.checkbox("Use Webcam for Live Detection")

if use_webcam:
    # Select webcam index
    webcam_index = st.selectbox("Select Webcam", options=[0, 1, 2, 3], index=0, help="Choose the webcam device (0 is the default).")

    cap = cv2.VideoCapture(webcam_index)
    stframe = st.empty()

    if not cap.isOpened():
        st.error(f"Unable to open webcam {webcam_index}. Try selecting another one.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", caption=f"Live Webcam Detection (Camera {webcam_index})")

        cap.release()