import tensorflow as tf
import cv2
import numpy as np
import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyUSB0' # Linux/Mac. For Windows, use 'COM3', 'COM4', etc.
BAUD_RATE = 9600
MODEL_PATH = "model_unquant.tflite"  # <-- It will look for this file!
CAMERA_INDEX = 0             # <--- We will try 0, then 1
# ---------------------

# --- Initialize Serial (Arduino) ---
print("Attempting to connect to Arduino...")
try:
    # We are forcing demo mode since we have no hardware
    raise Exception("Demo Mode")
    
    # arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # time.sleep(2) # Wait for Arduino to reset
    # print("Arduino connected successfully.")
except Exception as e:
    print(f"Warning: Could not connect to Arduino.")
    print("Running in DEMO-ONLY mode (no hardware signals).")
    arduino = None

# --- Load TFLite Model ---
print("\nLoading ML Model...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Get input size
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    print("ML Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load TFLite model from {MODEL_PATH}. {e}")
    print("Did you place 'model.tflite' in the same folder as this script?")
    exit()

# --- Function for Preprocessing ---
def preprocess_frame(frame):
    # Preprocessing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (input_width, input_height))
    # Normalize the image to be in range [-1, 1] for floating-point models
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_frame, axis=0) # Add batch dimension

# --- Function for Prediction ---
def get_prediction(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # Return prediction array

# --- Main Loop ---
def main_loop():
    print("\nAttempting to open webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX) # Init webcam
    
    # --- THIS IS THE FIX ---
    # If camera 0 didn't work, try camera 1
    if not cap.isOpened():
        print(f"Warning: Could not open camera {CAMERA_INDEX}. Trying index 1...")
        cap.release()
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print(f"FATAL ERROR: Cannot open any camera (tried 0 and 1).")
        print("Please check camera permissions or if another app is using it.")
        return
    # ---------------------

    print("\n" + "="*30)
    print("  GOMIKIN WASTE CLASSIFIER (DEMO)")
    print("="*30 + "\n")
    print("System Ready. Point waste at the camera.")
    print("A new window named 'Gomikin Demo' should appear.")
    print("\nPress 'c' in that window to classify.")
    print("Press 'q' in that window to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the live feed
        cv2.imshow('Gomikin Demo - Press "c" to classify', frame)
        key = cv2.waitKey(1) & 0xFF

        # --- ON 'c' KEY PRESS (CLASSIFY) ---
        if key == ord('c'):
            print("\nClassifying...")
            
            # 1. Preprocess the current frame
            input_data = preprocess_frame(frame)
            
            # 2. Get prediction
            prediction = get_prediction(input_data)
            pred_index = np.argmax(prediction)
            confidence = prediction[pred_index]

            # 3. Send command
            if pred_index == 0: 
                label = "Organic"
                command = b'O'
            else:
                label = "Inorganic"
                command = b'I'

            print(f"  ==> Prediction: {label} (Confidence: {confidence*100:.2f}%)")
            
            if arduino: # This will be 'None' (False)
                arduino.write(command)
                print(f"  ==> Sent '{command.decode()}' signal to Arduino.")
            else:
                print(f"  ==> (Demo Mode) Would send '{command.decode()}' signal.")
        
        # --- ON 'q' KEY PRESS (QUIT) ---
        elif key == ord('q'):
            print("Quitting...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print("Demo finished.")

if __name__ == "__main__":
    main_loop()