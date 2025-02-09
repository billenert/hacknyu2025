# 🦉 HorusAI

# HackNYU 2025 Project

**Empowering the visually impaired to locate and identify everyday objects using voice-guided YOLOv8 object detection.**  

HorusAI leverages state-of-the-art AI and computer vision to provide real-time auditory feedback for object recognition, improving accessibility and independence.

---

## 🚀 Features
- 🎤 **Voice-Guided Interaction** – Uses OpenAI Whisper for speech-to-text processing.
- 🔍 **Real-Time Object Detection** – Powered by YOLOv8 for accurate identification.
- 🗣 **AI-Powered Descriptions** – OpenAI API enhances object recognition with contextual information.
- 🏗 **User-Friendly Interface** – Built with Streamlit for easy deployment and accessibility.
- 📦 **Lightweight & Scalable** – Cloud integration with MongoDB Atlas for data management.
- 🎮 **Audio Feedback** – Pygame enables dynamic sound responses.

---

## 🛠️ Technologies Used
| Technology      | Purpose |
|----------------|---------|
| 🐍 Python | Core programming language |
| 🎨 Streamlit | Web application framework |
| 🤖 OpenAI API | AI-powered descriptions |
| 🗣 OpenAI Whisper | Speech-to-text processing |
| 📦 MongoDB Atlas | Cloud-based database |
| 🎯 YOLOv8 | Real-time object detection |
| 🎵 Pygame | Audio feedback system |
| 📸 OpenCV | Image processing |

---

## 📂 Project Structure
📁 HorusAI 
│── 📝 README.md # Project documentation 
│── 📜 newapp.py # Main application script 
│── 🎯 yolov8n.pt # General pre-trained YOLOv8 model 
│── 🔬 best.pt # Custom YOLOv8 model trained on glasses dataset

📌 **Pretrained Model Dataset:**  
[YOLOv8 Glasses Dataset on Kaggle](https://www.kaggle.com/datasets/nadavishai/yolov8-glasses-dataset-v1)

---

## 🎯 How to Use
1. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
2. **Run app in Streamlit!**
   ```bash
   streamlit run newapp.py
