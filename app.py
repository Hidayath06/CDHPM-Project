import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
import warnings
import requests
import cv2 
import tempfile
from skin_disease_detection import SkinDiseaseDetector # Correct dependency
import shutil

warnings.filterwarnings('ignore')

# --- CONFIGURATION (UPDATE THIS) ---
# ⚠️ 1. REPLACE THIS LINK with your actual public download URL for the .h5 file!
# Example: https://huggingface.co/Hidayathulla06/eczema-detector-resnet50/resolve/main/best_transfer_model.h5
MODEL_URL = "https://huggingface.co/Hidayathulla06/eczema-detector-resnet50/resolve/main/best_transfer_model.h5" 
LOCAL_MODEL_PATH = "downloaded_model.h5" 
TARGET_CLASSES = ['Eczema', 'Non-Eczema'] 
# -----------------------------------

# Set page config and custom CSS (Your original code here)
st.set_page_config(page_title="Eczema Detection System", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keep your CSS here

# 🛠️ FINAL FIXES: Download, Save, and Load Locally
@st.cache_resource
def load_model_from_url():
    """
    Downloads the model from the URL to a local temp file and loads it into Keras.
    """
    # Use a temporary directory path for safe local storage
    local_path = os.path.join(tempfile.gettempdir(), LOCAL_MODEL_PATH)
    
    # 1. Download the Model
    if not os.path.exists(local_path):
        try:
            st.info(f"Downloading model from Hugging Face...")
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status() 

            # Write content directly to the temporary file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")

        except Exception as e:
            st.error("Model download failed. Check the URL and try again.")
            st.error(f"Details: Download/Save Error: {e}")
            return None
    
    # 2. Load the Model from Local Disk
    try:
        detector = SkinDiseaseDetector() 
        detector.class_names = TARGET_CLASSES 
        
        # 🟢 FINAL FIX: The load_saved_model function handles the internal Keras load
        if detector.load_saved_model(local_path):
            st.success("Model initialized and ready!")
            return detector
        else:
            raise RuntimeError("Keras load failed after file download.")
        
    except Exception as e:
        st.error("Model loading failed. The downloaded file may be corrupted (try clearing cache).")
        st.error(f"Details: Load Error: {e}")
        return None


def preprocess_image(image):
    # ... (Keep your existing preprocess_image logic) ...
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ... (Keep your get_confidence_color and other helper functions) ...

def main():
    st.markdown('<h1 class="main-header">🏥 Eczema Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar logic...
    page = st.sidebar.selectbox("Choose a page", ["Home", "Upload & Predict", "Model Information"])

    # ⚠️ Use load_model_from_url instead of load_model
    detector_instance = load_model_from_url() 
    
    if page == "Home":
        # ... (Keep your Home page logic) ...
        st.markdown("""...""") 
        
    elif page == "Upload & Predict":
        st.header("📤 Upload Image for Eczema Diagnosis")
        
        if detector_instance is None:
            st.error("Application cannot run because the model failed to load.")
            st.stop()
        
        uploaded_file = st.file_uploader(
            "Choose an image file", type=['png', 'jpg', 'jpeg'], help="Upload a clear image..."
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("🔍 Analysis")
                
                img_array = preprocess_image(image)
                
                with st.spinner("Analyzing image..."):
                    # 🟢 FIX: Use the correct attribute path (detector_instance.model)
                    prediction = detector_instance.model.predict(img_array)
                    
                    probabilities = prediction[0][:2]
                    predicted_class_index = np.argmax(probabilities)
                    confidence = probabilities[predicted_class_index]
                    disease_name = detector_instance.class_names[predicted_class_index]
                
                # ... (Rest of the results display logic) ...
                
if __name__ == "__main__":
    main()