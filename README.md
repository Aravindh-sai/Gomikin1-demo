# Gomikin1-demo
Gomikin - Smart Waste Classifier Demo

This repository contains the core software for the Gomikin Smart Waste Bin, a college project to design an apparatus for automatic waste segregation and in-situ composting.

This code represents the "brain" of the project. It's a Python script that uses a TensorFlow Lite (TFLite) Machine Learning model to classify waste from a laptop's webcam in real-time.

Project Status

Status: Design & Feasibility Phase Complete

The core ML classification model (the "brain") is complete, trained, and validated. This demo script proves the project's core concept is feasible and works.

The next phase of the project would be to integrate this software with the physical hardware (Arduino, servos, sensors) as detailed in the project report.

Features

Real-Time Classification: Runs a TFLite model to classify waste as "Organic" or "Inorganic".

Webcam Input: Uses OpenCV to capture a live video feed from your laptop's webcam.

Interactive Demo: Press 'c' to capture a frame and run the classification.

Demo-Only Mode: Safely runs and prints its intended actions (e.g., (Demo Mode) Would send 'O' signal.) without crashing if no Arduino hardware is connected.

How to Run the Demo

1. Prerequisites

You must have Python 3.8+ installed.

2. Setup

Get the Files:
Download or clone this repository to a folder on your computer.

Add Your Model:
Place your trained TFLite model in the same folder. This script assumes the model is named model_unquant.tflite.

Install Dependencies:
Open a terminal or Command Prompt and run the following command to install the required libraries:

pip install tensorflow opencv-python numpy pyserial


3. Run the Demo

Navigate to the project folder in your terminal:

cd path/to/your/Gomikin_Demo


Run the script:

python gomikin_demo.py


4. How to Use

A window named "Gomikin Demo" will appear, showing your webcam feed.

Point a piece of waste (e.g., a banana peel, a plastic bottle) at the camera.

Press the 'c' key while the webcam window is active.

Look at your terminal window. It will print the classification result and confidence.

Example Output:

Classifying...
  ==> Prediction: Organic (Confidence: 98.72%)
  ==> (Demo Mode) Would send 'O' signal.


How to Get Model Performance Metrics

This project includes a separate script, get_metrics.py, to test your TFLite model against a test dataset and generate the classification report needed for your project documentation.

1. Additional Setup

Install Dependencies:
You will need scikit-learn and tqdm.

pip install scikit-learn tqdm


Unzip Your Test Data:
Make sure your test dataset is unzipped into a regular folder (e.g., C:\Users\YourName\Downloads\DATASET\TEST).

Edit get_metrics.py:
You must edit these two lines at the top of the get_metrics.py script:

TEST_DATA_PATH: Change this string to the full path of your TEST folder.

CLASSES: Make sure the folder names match what's in your TEST folder (e.g., "O" and "R").

# Example edit in get_metrics.py
TEST_DATA_PATH = r"C:\Users\ARAVINDH\Downloads\DATASET\TEST"
CLASSES = ["O", "R"] 


2. Run the Test

In your terminal, run the metrics script:

python get_metrics.py


The script will show a progress bar as it tests every image.

When finished, it will print the Classification Report and Confusion Matrix to your console, ready to be copied into your project report.