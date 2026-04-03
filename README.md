# 🚁 AI-Based Drone Object Detection & GPS Tracking System

## 📌 Overview

This project presents an intelligent **drone-based object detection system** that uses **YOLO (You Only Look Once)** for real-time detection and integrates **GPS tracking** to identify the exact location of detected objects.

The system is designed for applications such as **surveillance, search & rescue, smart monitoring, and defense systems**.

---

## 🎯 Key Features

* 🎥 Real-time object detection using YOLO
* 📍 GPS-based object location tracking
* 🚁 Drone integration using DroneKit
* 🧠 Custom model training using Roboflow
* ⚡ High-speed detection with optimized performance

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** OpenCV, NumPy
* **Model:** YOLOv5 / YOLOv8
* **Tools:** Roboflow (dataset + training)
* **Drone Integration:** DroneKit
* **Others:** GPS modules, MAVLink

---

## 📂 Project Structure

```id="projstr01"
drone-object-detection/
│── dataset/                # Training & testing images 
│── models/                 # Trained YOLO weights
│── src/
│   │── detect.py           # Main detection script
│   │── gps_module.py       # GPS data handling
│   │── drone_control.py    # DroneKit integration
│── results/                # Output images/videos
│── requirements.txt
│── README.md
```

---
## 📊 Dataset

The dataset used in this project is available here:

🔗 [Download Dataset](https://drive.google.com/drive/folders/1GERXKlJEynWNzlz7XQfswKZkFgSaBAUY?usp=sharing)

Description:
- Contains features like [mention features]
- Used for [prediction/classification task]
## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash id="clone01"
git clone https://github.com/your-username/drone-object-detection.git
cd drone-object-detection
```

### 2️⃣ Install Dependencies

```bash id="install01"
pip install -r requirements.txt
```

### 3️⃣ Setup YOLO Model

* Download trained weights (`.pt` file)
* Place it inside the `models/` folder

---

## ▶️ How It Works

1. Drone captures live video feed
2. Frames are processed using YOLO model
3. Objects are detected with bounding boxes
4. GPS coordinates are fetched using DroneKit
5. Object location is mapped and displayed

---

## ▶️ Usage

### Run Detection

```bash id="run01"
python src/detect.py
```

### Output

* Detected objects with bounding boxes
* Real-time GPS coordinates of detected objects
* Saved results in `/results` folder

---

## 📸 Sample Output

(Add your screenshots here)

---

## 🚀 Applications

* 🛰️ Surveillance & security
* 🚑 Search and rescue missions
* 🌾 Smart agriculture monitoring
* 🏙️ Traffic and crowd analysis
* 🛡️ Defense & border monitoring

---

## 📌 Future Enhancements

* 🔥 Live drone deployment
* 🌐 Web dashboard (Streamlit/Flask)
* 📡 Cloud-based monitoring system
* 📊 Advanced analytics & tracking history

---

## ⚠️ Notes

* Large datasets are not included (use Roboflow link instead)
* Ensure proper drone permissions before real deployment
* Model performance depends on training dataset quality
Dataset Link:https://drive.google.com/drive/folders/1GERXKlJEynWNzlz7XQfswKZkFgSaBAUY?usp=sharing
---

## 🤝 Contributing

Contributions are welcome! Fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* Ultralytics YOLO
* OpenCV
* Roboflow
* DroneKit

---

## 👨‍💻 Author

**HARSHAVARTHAN KS**
www.linkedin.com/in/harshavarthan-ks-47123a295
https://github.com/Harsha0005

