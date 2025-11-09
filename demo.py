import tensorflow as tf
import cv2
import numpy as np
import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
MODEL_PATH = "model_unquant.tflite" 
CAMERA_INDEX = 0           



print("Attempting to connect to Arduino...")
try:
    
    raise Exception("Demo Mode")
    
except Exception as e:
    print(f"Warning: Could not connect to Arduino.")
    print("Running in DEMO-ONLY mode (no hardware signals).")
    arduino = None

print("\nLoading ML Model...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    print("ML Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load TFLite model from {MODEL_PATH}. {e}")
    print("Did you place 'model.tflite' in the same folder as this script?")
    exit()
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (input_width, input_height))
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_frame, axis=0) 

def get_prediction(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] 

def main_loop():
    print("\nAttempting to open webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX) 
    
    if not cap.isOpened():
        print(f"Warning: Could not open camera {CAMERA_INDEX}. Trying index 1...")
        cap.release()
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print(f"FATAL ERROR: Cannot open any camera (tried 0 and 1).")
        print("Please check camera permissions or if another app is using it.")
        return

    print("\n" + "="*30)
    print("  GOMIKIN WASTE CLASSIFIER (DEMO)")
    print("="*30 + "\n")
    print("System Ready. Point waste at the camera.")
    print("A new window named 'Gomikin Demo' should appear.")
    print("\nPress 'c' in that window to classify.")
    print("Press 'q' in that window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.imshow('Gomikin Demo - Press "c" to classify', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("\nClassifying...")
            
            input_data = preprocess_frame(frame)
            
            prediction = get_prediction(input_data)
            pred_index = np.argmax(prediction)
            confidence = prediction[pred_index]

            if pred_index == 0: 
                label = "Organic"
                command = b'O'
            else:
                label = "Inorganic"
                command = b'I'

            print(f"  ==> Prediction: {label} (Confidence: {confidence*100:.2f}%)")
            
            if arduino: 
                arduino.write(command)
                print(f"  ==> Sent '{command.decode()}' signal to Arduino.")
            else:
                print(f"  ==> (Demo Mode) Would send '{command.decode()}' signal.")
        
        elif key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print("Demo finished.")

if __name__ == "__main__":
    main_loop()