# TinyML-Based Real-Time Image Classification

## Project Description
This project implements a TinyML-based real-time image classification system using TensorFlow Lite. 
The model is trained using MobileNetV2 and optimized for low memory usage. 
The final model is converted to TFLite format and tested using real-time webcam inference.

---

## Features
- Lightweight MobileNetV2 model
- TensorFlow Lite conversion
- Model size < 3MB
- Real-time webcam classification
- Deployment-ready for microcontrollers (ESP32, Raspberry Pi)

---

## Technologies Used
- Python
- TensorFlow
- TensorFlow Lite
- OpenCV
- NumPy

---

##  Project Structure

tinyml-image-classifier/
│
├── dataset/
├── train.py
├── convert_to_tflite.py
├── infer_webcam.py
├── model.tflite
└── README.md

---

## ▶️ How to Run

### Train Model
python train.py

### Convert to TFLite
python convert_to_tflite.py

### Run Real-Time Inference
python infer_webcam.py

---

## Model Performance
Validation Accuracy: ~94-95%  
Model Size: < 3MB  

---

## TinyML Aspect
The trained model is optimized and converted into TensorFlow Lite format, 
making it suitable for deployment on low-power devices and microcontrollers.
