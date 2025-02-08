from ultralytics import YOLO
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import sounddevice as sd
import whisper
import tempfile
import os

# Load YOLO model
model = YOLO("yolov8n.pt")  # 👈 Load a pretrained model (e.g., YOLOv8 Nano)

# Load Whisper model
whisper_model = whisper.load_model("base")  # Load Whisper base model (smallest and fastest)

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

# Microphone and Whisper Integration
st.markdown("---")
st.header("Microphone Input with Whisper")

# Add a big button to activate the microphone
if st.button("🎤 Press to Speak", use_container_width=True):
    st.write("Listening... Speak now!")

    # Record audio from the microphone
    sample_rate = 16000  # Whisper expects 16kHz audio
    duration = 5  # Record for 5 seconds
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until the recording is finished

    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        import scipy.io.wavfile as wav
        wav.write(tmpfile.name, sample_rate, audio)  # Save as WAV file
        tmpfile_path = tmpfile.name

    # Transcribe the audio using Whisper
    result = whisper_model.transcribe(tmpfile_path)
    transcription = result["text"]

    # Print the transcription to the console
    print("Transcription:", transcription)

    # Display the transcription on the Streamlit app
    st.write("You said:")
    st.write(transcription)

    # Clean up the temporary file
    os.remove(tmpfile_path)