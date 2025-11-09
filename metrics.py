import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

MODEL_PATH = "model_unquant.tflite"
TEST_DATA_PATH = r"C:\Users\ARAVINDH\Downloads\test" 
CLASSES = ["Organic", "Inorganic"]

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
except Exception as e:
    print(f"FATAL ERROR: Could not load TFLite model from {MODEL_PATH}. {e}")
    exit()

def preprocess_image(image_path):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (input_width, input_height))
        normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
        return np.expand_dims(normalized_frame, axis=0)
    except Exception as e:
        print(f"Warning: Skipping broken image {image_path}. Error: {e}")
        return None

def get_prediction(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data[0])

def evaluate_model():
    print(f"Loading test images from: {TEST_DATA_PATH}")
    
    y_true = []
    y_pred = []

    if not os.path.isdir(TEST_DATA_PATH):
        print(f"FATAL ERROR: Path not found. Please edit TEST_DATA_PATH to point to your 'test' folder.")
        print(f"Current path: {TEST_DATA_PATH}")
        return

    try:
        organic_path = os.path.join(TEST_DATA_PATH, "O")
        inorganic_path = os.path.join(TEST_DATA_PATH, "R") 
        
        organic_images = [os.path.join(organic_path, f) for f in os.listdir(organic_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        inorganic_images = [os.path.join(inorganic_path, f) for f in os.listdir(inorganic_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    except Exception as e:
        print(f"FATAL ERROR: Could not find 'O' or 'R' subfolders.")
        print(f"Please make sure {TEST_DATA_PATH} contains them. Error: {e}")
        return

    print(f"Found {len(organic_images)} Organic test images.")
    print(f"Found {len(inorganic_images)} Inorganic test images.")
    print("\nRunning predictions... (This may take a minute)")

    for img_path in tqdm(organic_images, desc="Organic"):
        input_data = preprocess_image(img_path)
        if input_data is not None:
            y_true.append(0)
            y_pred.append(get_prediction(input_data))

    for img_path in tqdm(inorganic_images, desc="Inorganic"):
        input_data = preprocess_image(img_path)
        if input_data is not None:
            y_true.append(1)
            y_pred.append(get_prediction(input_data))

    if not y_true:
        print("\nFATAL ERROR: No test images were successfully processed.")
        return
        
    print("\n" + "="*50)
    print("           CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    print("\n" + "="*50)
    print("             CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred)
    print(f"         {CLASSES[0]}   {CLASSES[1]} (Predicted)")
    print(f"{CLASSES[0]:<9} {cm[0][0]:<10} {cm[0][1]}")
    print(f"{CLASSES[1]:<9} {cm[1][0]:<10} {cm[1][1]}")
    print("(Actual)")
    print("\n" + "="*50)

if __name__ == "__main__":
    evaluate_model()