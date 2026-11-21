
# ============================================
# 1. IMPORTS
# ============================================

from pathlib import Path
import pickle

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================
# 2. APPLICATION SETUP
# ============================================

app = FastAPI(
    title="Crop Recommendation API",
    description="API for predicting suitable crops using ML models",
    version="1.0.0"
)


# ============================================
# 3. CORS CONFIGURATION
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================
# 4. MODEL FILE PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

NB_MODEL_PATH = MODEL_DIR / "NBClassifier.pkl"

RF_MODEL_PATH = MODEL_DIR / "RandomForest.pkl"


# ============================================
# 5. MODEL LOADING FUNCTION
# ============================================

def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    try:
        # First, try loading using joblib
        return joblib.load(model_path)

    except Exception:
        # If that fails, try standard Python pickle
        with open(model_path, "rb") as file:
            return pickle.load(file)


# ============================================
# 6. LOAD BOTH ML MODELS
# ============================================

try:

    nb_model = load_model(NB_MODEL_PATH)

    rf_model = load_model(RF_MODEL_PATH)

    print("Naive Bayes model loaded successfully!")

    print("Random Forest model loaded successfully!")

except Exception as error:

    raise RuntimeError(
        f"Could not load ML models: {error}"
    ) from error


# Store both models in a dictionary
models = {
    "naive_bayes": nb_model,
    "random_forest": rf_model
}


# ============================================
# 7. INPUT DATA VALIDATION
# ============================================

class CropInput(BaseModel):

    N: float = Field(ge=0, description="Nitrogen value")

    P: float = Field(ge=0, description="Phosphorus value")

    K: float = Field(ge=0, description="Potassium value")

    temperature: float = Field(
        description="Temperature in Celsius"
    )

    humidity: float = Field(
        ge=0,
        le=100,
        description="Humidity percentage"
    )

    ph: float = Field(
        ge=0,
        le=14,
        description="Soil pH value"
    )

    rainfall: float = Field(
        ge=0,
        description="Rainfall in mm"
    )

    model_name: str = Field(
        default="naive_bayes",
        description="naive_bayes or random_forest"
    )


# ============================================
# 8. FEATURE PREPARATION
# ============================================

def prepare_features(data: CropInput):

    features = np.array([[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    return features


# ============================================
# 9. HOME ENDPOINT
# ============================================

@app.get("/")
def home():

    return {
        "message": "Crop Recommendation API is running!"
    }


# ============================================
# 10. HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "loaded_models": list(models.keys())
    }


# ============================================
# 11. PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict_crop(data: CropInput):

    # Check whether the requested model exists
    if data.model_name not in models:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model_name. Choose "
                "'naive_bayes' or 'random_forest'."
            )
        )

    # Select the requested model
    selected_model = models[data.model_name]

    # Prepare the input features
    features = prepare_features(data)

    try:

        # Predict the crop
        prediction = selected_model.predict(features)[0]

        # Get probabilities for every crop class
        probabilities = selected_model.predict_proba(features)[0]

        # Get the class labels
        crop_classes = selected_model.classes_

        # Find the indexes of the three highest probabilities
        top_indices = np.argsort(probabilities)[::-1][:3]

        # Prepare the top three predictions
        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "crop": str(crop_classes[index]),
                "probability": round(
                    float(probabilities[index]) * 100, 2
                )
            })

        # Probability of the predicted crop
        prediction_probability = float(
            np.max(probabilities) * 100
        )

        # Return the result
        return {
            "recommended_crop": str(prediction),

            "prediction_probability": round(
                prediction_probability, 2
            ),

            "selected_model": data.model_name,

            "top_predictions": top_predictions
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
