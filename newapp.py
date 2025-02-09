import streamlit as st
import hashlib
import os
import tempfile
import numpy as np
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import pygame.mixer
import whisper
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI


def load_audio_custom(file_path, sr=16000):
    """Load a WAV file using SciPy instead of FFmpeg."""
    rate, audio = wav.read(file_path)
    if rate != sr:
        raise ValueError(f"Expected sample rate {sr}, but got {rate}")
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max

whisper.audio.load_audio = load_audio_custom

os.environ["STREAMLIT_WATCHER_IGNORE"] = "1"

load_dotenv()

pygame.mixer.init()
sound = pygame.mixer.Sound("ping.wav")

# MongoDB Intitialization
MONGO_URI = "mongodb+srv://awesomeryank:Necro563808ryk!!@cluster0.ro9a0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["my_database"]
users_collection = db["users"]
logs_collection = db["logs"]

# User Authentication
def hash_password(password):
    """Hashes the password for security."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    """Authenticates the user."""
    user = users_collection.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return user["_id"]
    return None

# Streamlit Styling

st.set_page_config(page_title="Object Detection & Speech Recognition", page_icon="eye.png",layout="wide")
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .stButton > button {
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
""", unsafe_allow_html=True)
st.sidebar.image("eye.png", width=100)
st.sidebar.header("Login / Sign Up")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    user_id = authenticate(username, password)
    if user_id:
        st.session_state["user_id"] = str(user_id)
        st.session_state["username"] = username
        st.sidebar.success(f"Welcome, {username}!")
    else:
        st.sidebar.error("Invalid credentials. Try again.")

if st.sidebar.checkbox("New user? Sign up here"):
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Sign Up"):
        if users_collection.find_one({"username": new_username}):
            st.sidebar.error("Username already exists. Try another.")
        else:
            user_data = {"username": new_username, "password": hash_password(new_password)}
            users_collection.insert_one(user_data)
            st.sidebar.success("Account created! Please log in.")



# Login
if "user_id" not in st.session_state:
    st.warning("Please log in to access object detection.")
    st.stop()

st.title("HorusAI: Finding Objects For The Visually Impaired")

# Load YOLO Model
model = YOLO("yolov8n.pt")
default_model = YOLO("yolov8n.pt")
glasses_model = YOLO(r"runs/detect/train12/weights/best.pt")

# Load Whisper Model
whisper_model = whisper.load_model("base")

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# COCO Objects
COCO_NOUNS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "glasses"
]

# Search Bar for Object Detectoin
st.sidebar.subheader("Search for Object to Detect")
search_object = st.sidebar.text_input("Enter an object name")

if st.sidebar.button("Set Detection Object"):
    if search_object.lower() in COCO_NOUNS:
        st.session_state.relevant_noun = search_object.lower()
        st.sidebar.success(f"Detection object set to: {search_object}")
    else:
        st.sidebar.error(f"'{search_object}' is not a valid object for detection.")

# OpenAI Object Extraction
def extract_most_relevant_noun(text):
    prompt = f"""
    Identify the most important noun from the sentence:
    "{text}"
    Return only the noun.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.2
    )
    noun = response.choices[0].message.content.strip()

    if noun in COCO_NOUNS:
        return noun
    return "person"

sound = pygame.mixer.Sound("ping.wav") 


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
            current_model = default_model 
            if "relevant_noun" not in st.session_state:
                st.session_state.relevant_noun = "person" 

            if st.session_state.relevant_noun.lower() in ["glasses", "sunglasses"]:
                st.session_state.relevant_noun = "glasses"
                current_model = glasses_model

            results = current_model(frame)

            if st.session_state.relevant_noun:
                class_names = current_model.names
                
                for box in results[0].boxes:
                    try:
                        detected_object = class_names[int(box.cls)]
                        confidence = box.conf.item() 
                        
                        if (detected_object == st.session_state.relevant_noun and confidence > 0.5 and st.session_state.relevant_noun=="glasses") or (detected_object == st.session_state.relevant_noun and confidence > 0.9):
                            sound.play()
                    except KeyError:
                        continue 

                filtered_results = [
                    r for r in results[0].boxes 
                    if class_names.get(int(r.cls), "") == st.session_state.relevant_noun
                ]
                results[0].boxes = filtered_results
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", caption=f"Live Webcam Detection (Camera {webcam_index})")

        cap.release()


# Last Used Object from MongoDB
def get_past_relevant_noun(user_id):
    """Retrieve the most recent relevant noun for the user from MongoDB."""
    log_entry = list(logs_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(1))
    if log_entry:
        return log_entry[0]["relevant_noun"]
    return None

def get_most_frequent_relevant_noun(user_id):
    """Retrieve the most frequently asked relevant noun for the user from MongoDB."""
    pipeline = [
        {"$match": {"user_id": user_id}}, 
        {"$group": {"_id": "$relevant_noun", "count": {"$sum": 1}}}, 
        {"$sort": {"count": DESCENDING}}, 
        {"$limit": 1} 
    ]
    result = list(logs_collection.aggregate(pipeline))
    if result:
        return result[0]["_id"]
    return None

if "user_id" not in st.session_state:
    st.warning("Please log in to access object detection.")
    st.stop()

past_relevant_noun = get_past_relevant_noun(st.session_state["user_id"])

if past_relevant_noun:
    st.sidebar.subheader("Past Relevant Word Suggestion")
    if st.sidebar.button(f"Set as Relevant Word: {past_relevant_noun}", key=f"past_{past_relevant_noun}"):
        st.session_state.relevant_noun = past_relevant_noun
        st.sidebar.success(f"Relevant word set to: {past_relevant_noun}")
most_frequent_noun = get_most_frequent_relevant_noun(st.session_state["user_id"])

if most_frequent_noun:
    st.sidebar.subheader("Most Frequently Asked Word Suggestion")
    if st.sidebar.button(f"Set as Relevant Word: {most_frequent_noun}", key=f"freq_{most_frequent_noun}"):
        st.session_state.relevant_noun = most_frequent_noun
        st.sidebar.success(f"Relevant word set to: {most_frequent_noun}")

st.markdown("---")
st.header("Microphone Input")

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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, sample_rate, audio_data)
        return tmpfile.name

if st.button("🎤 Press to Speak", key="big_button"):
    st.write("Listening... Speak now!")


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
    logs_collection.insert_one({
        "user_id": st.session_state["user_id"],
        "text": transcription,
        "relevant_noun": relevant_noun,
        "timestamp": datetime.utcnow()
    })
    st.write(f"The most relevant COCO noun is: **{relevant_noun}**")

    os.remove(tmpfile_path)
