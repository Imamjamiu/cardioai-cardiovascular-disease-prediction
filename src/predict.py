
import joblib
import pandas as pd
from pathlib import Path

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the saved model
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"


def load_model():
    """Load the saved machine learning pipeline."""
    model = joblib.load(MODEL_PATH)
    return model


def predict_cvd(input_data):
    """Predict the target class and estimated probability."""

    model = load_model()

    # Convert input into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Generate prediction
    prediction = model.predict(input_df)[0]

    # Generate probability for class 1
    probability = model.predict_proba(input_df)[0][
        list(model.classes_).index(1)
    ]

    return prediction, probability


if __name__ == "__main__":

    sample_input = {
        "age": 50,
        "gender": 1,
        "height": 165,
        "weight": 70,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "cholesterol": 1,
        "glucose": 1,
        "smoke": 0,
        "alcohol": 0,
        "physical_activity": 1
    }

    prediction, probability = predict_cvd(sample_input)

    print("\nCARDIOVASCULAR PREDICTION")
    print("Predicted class:", prediction)
    print(f"Estimated class 1 probability: {probability:.4f}")
