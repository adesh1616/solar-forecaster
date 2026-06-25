import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Path to the trained model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "solar_model.pkl")

def load_trained_model():
    """
    Tries to load models/solar_model.pkl using pickle.
    If it doesn't exist, trains a dummy LinearRegression model and returns it.
    """
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}. Falling back to dummy model.")
    
    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Fallback dummy model
    print("solar_model.pkl not found or failed to load. Creating a fallback dummy LinearRegression model...")
    # Features: hour, temperature_c, cloud_cover_pct, irradiance_wm2
    # Hardcoded dummy data
    X_dummy = pd.DataFrame([
        # hour, temperature_c, cloud_cover_pct, irradiance_wm2
        [6, 15.0, 20.0, 50.0],
        [12, 25.0, 10.0, 800.0],
        [18, 20.0, 30.0, 100.0],
        [24, 12.0, 5.0, 0.0]
    ], columns=["hour", "temperature_c", "cloud_cover_pct", "irradiance_wm2"])
    
    # Target solar output in MW
    y_dummy = np.array([5.0, 85.0, 10.0, 0.0])
    
    dummy_model = LinearRegression()
    dummy_model.fit(X_dummy, y_dummy)
    
    # Save the dummy model to make subsequent loads faster/persistent
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(dummy_model, f)
    except Exception as e:
        print(f"Failed to save dummy model: {e}")
        
    return dummy_model

# Load model globally on import
model = load_trained_model()

def predict_solar_output(features_df: pd.DataFrame) -> np.ndarray:
    """
    Takes a Pandas DataFrame, runs prediction, clips negative predictions to 0.0,
    and returns the numpy array of predictions.
    """
    predictions = model.predict(features_df)
    # Clip any negative predictions to 0.0
    predictions = np.clip(predictions, 0.0, None)
    return predictions
