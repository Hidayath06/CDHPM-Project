import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50 # <-- New Import
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from datetime import datetime

class SkinDiseaseDetector:
    """
    Core class to handle model definition, training, and loading.
    The 'model' attribute holds the loaded Keras model object.
    """
    def __init__(self, dataset_path="D:\\skin_detection_model\\Dataset\\Train", 
                 test_path="D:\\skin_detection_model\\Dataset\\Test", 
                 img_size=(224, 224), batch_size=32):
        
        self.dataset_path = dataset_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        # 🟢 CRITICAL FIX: Ensure 'model' and 'class_names' are initialized here
        self.model = None
        self.class_names = ['Eczema', 'Non-Eczema'] # Default for app usage
        
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def create_data_generators(self):
        # ... (Rest of training logic - kept for completeness) ...
        pass
        
    def build_model(self):
        # ... (Rest of training logic - kept for completeness) ...
        pass
        
    def train_model(self, epochs=50):
        # ... (Rest of training logic - kept for completeness) ...
        pass

    def test_model(self):
        # ... (Rest of testing logic - kept for completeness) ...
        pass

    def save_model(self, model_path='skin_disease_transfer_model.keras'):
        # ... (Rest of saving logic - kept for completeness) ...
        pass

    def load_saved_model(self, model_path):
        """
        Loads a saved model from a local path and assigns it to self.model.
        This function is used by the Streamlit app.
        """
        try:
            self.model = load_model(model_path)
            # The app relies on self.model being set here.
            return True
        except Exception as e:
            print(f"Error loading model locally: {e}")
            return False

# The main() function for local training is omitted here as we focus on the app dependency.