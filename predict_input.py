import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

# --- Configuration (Based on your successful training) ---
# NOTE: Removed the fixed IMAGE_TO_PREDICT_PATH variable
MODEL_FILE = 'best_transfer_model.keras' 
IMG_SIZE = (224, 224) 
CLASS_NAMES = ['Eczema', 'Non-Eczema'] 

def predict_single_image(model, image_path):
    """
    Loads an image, preprocesses it, and makes a prediction.
    """
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found at path: {image_path}")
        print("Please ensure the path is correct and enclosed in quotes if it contains spaces.")
        return

    print(f"\n--- Analyzing Image: {os.path.basename(image_path)} ---")
    
    try:
        # 1. Load and Preprocess Image
        # Use PIL's Image.open for reliable image reading
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Resize and Rescale using cv2
        img_resized = cv2.resize(img_np, IMG_SIZE)
        img_scaled = img_resized / 255.0
        
        # Add batch dimension
        img_final = np.expand_dims(img_scaled, axis=0)
        
        # 2. Predict
        prediction = model.predict(img_final, verbose=0)
        
        # 3. Format Output
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(prediction) * 100
        
        # 4. Print Results
        print(f"✅ Prediction Successful:")
        print(f"   -> Predicted Class: **{predicted_class}**")
        print(f"   -> Confidence: **{confidence:.2f}%**")
        
        # 5. Display the image
        plt.figure()
        plt.imshow(img_resized)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")


def main():
    # Load the Model
    try:
        # We load the best model saved from your training script
        model = load_model(MODEL_FILE)
        print(f"✅ Model loaded successfully from {MODEL_FILE}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model. Ensure {MODEL_FILE} is in the current directory.")
        return

    # --- NEW: PROMPT USER FOR IMAGE PATH ---
    image_path = input("\n🖼️ Enter the full path to the image you want to predict (e.g., D:/images/test.jpg): ")

    # Run Prediction
    predict_single_image(model, image_path.strip()) # .strip() removes accidental spaces


if __name__ == "__main__":
    main()