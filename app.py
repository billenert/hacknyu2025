from ultralytics import YOLO
import numpy as np
import streamlit as st
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import pygame.mixer 

pygame.mixer.init()

def load_audio_custom(file_path, sr=16000):
    """Load a WAV file using SciPy instead of FFmpeg."""
    rate, audio = wav.read(file_path)
    if rate != sr:
        raise ValueError(f"Expected sample rate {sr}, but got {rate}")
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max  


whisper.audio.load_audio = load_audio_custom
import tempfile
import os
from openai import OpenAI  # Import the new OpenAI client
from dotenv import load_dotenv

os.environ["STREAMLIT_WATCHER_IGNORE"] = "1"
load_dotenv()

# Load YOLO model
model = YOLO("yolov8n.pt")
default_model = YOLO("yolov8n.pt") 
glasses_model =YOLO("best.pt")
# Load Whisper model
whisper_model = whisper.load_model("base") 

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
# List of COCO dataset nouns
COCO_NOUNS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "glasses"
]

# Function to extract the most relevant COCO noun using OpenAI API
def extract_most_relevant_noun(text):
    step1_prompt = f"""
    Analyze the following sentence and extract the most important noun:
    "{text}"

    Return only the noun. Do not include any additional text or explanation.
    """
    step1_response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts the most important noun from a sentence."},
            {"role": "user", "content": step1_prompt}
        ],
        max_tokens=10,
        temperature=0.2
    )
    extracted_noun = step1_response.choices[0].message.content.strip()

    step2_prompt = f"""
    The following is a list of nouns from the COCO dataset:
    {', '.join(COCO_NOUNS)}

    If you find the matched word within the list, return the same word. Else, find the most relevant noun exclusively from the list that matches:
    "{extracted_noun}"

    Return only the noun. Do not include any additional text or explanation.
    """
    step2_response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that maps a noun to the most relevant COCO dataset noun."},
            {"role": "user", "content": step2_prompt}
        ],
        max_tokens=10,
        temperature=0.2
    )
    relevant_noun = step2_response.choices[0].message.content.strip()

    return relevant_noun

if "relevant_noun" not in st.session_state:
    st.session_state.relevant_noun = ""

# Load sound for detection
sound = pygame.mixer.Sound("ping.wav")  # Replace with the path to your sound file

st.title("YOLO Object Detection with Streamlit")

use_webcam = st.checkbox("Use Webcam for Live Detection")

if use_webcam:
    webcam_index = st.selectbox("Select Webcam", options=[0, 1, 2, 3], index=1, help="Choose the webcam device (0 is the default).")

    cap = cv2.VideoCapture(webcam_index)
    stframe = st.empty()

    if not cap.isOpened():
        st.error(f"Unable to open webcam {webcam_index}. Try selecting another one.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            if st.session_state.relevant_noun == "glasses" or st.session_state.relevant_noun == "sunglasses": 
                st.session_state.relevant_noun = "glasses"
                model = glasses_model
            else:
                model = default_model

            if st.session_state.relevant_noun:
                for box in results[0].boxes:
                    detected_object = model.names[int(box.cls)]
                    confidence = box.conf  # Confidence score

                    if detected_object == st.session_state.relevant_noun and confidence > 0.9:
                        sound.play() 

                filtered_results = [r for r in results[0].boxes if model.names[int(r.cls)] == st.session_state.relevant_noun]
                results[0].boxes = filtered_results

            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", caption=f"Live Webcam Detection (Camera {webcam_index})")

        cap.release()

# Microphone and Whisper Integration
st.markdown("---")
st.header("Microphone Input with Whisper")
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        font-size: 30px;
        height: 100px;
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def save_audio_without_ffmpeg(audio_data, sample_rate):
    """Save the recorded audio as a WAV file using SciPy."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        import scipy.io.wavfile as wav
        wav.write(tmpfile.name, sample_rate, audio_data)
        return tmpfile.name

if st.button("🎤 Press to Speak", key="big_button"):
    st.write("Listening... Speak now!")

    # Record audio from the microphone
    sample_rate = 16000 
    duration = 3 
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    tmpfile_path = save_audio_without_ffmpeg(audio, sample_rate)
    result = whisper_model.transcribe(tmpfile_path)
    transcription = result["text"]

    print("Transcription:", transcription)

    st.write("You said:")
    st.write(transcription)

    relevant_noun = extract_most_relevant_noun(transcription)
    st.session_state.relevant_noun = relevant_noun
    
    st.write(f"The most relevant COCO noun is: **{relevant_noun}**")
    os.remove(tmpfile_path)