
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocessing import load_data, clean_data, prepare_features


# Locate project directories
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"


def evaluate_model():
    """Evaluate the saved Logistic Regression pipeline."""

    # Load dataset
    df = load_data()

    # Clean and prepare dataset
    df = clean_data(df)
    X, y = prepare_features(df)

    # Recreate the original train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load saved model
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    y_pred = model.predict(X_test)

    class_1_index = list(model.classes_).index(1)
    y_probability = model.predict_proba(X_test)[:, class_1_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probability)

    print("\nMODEL EVALUATION REPORT")

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(f"{auc:.4f}")


if __name__ == "__main__":
    evaluate_model()
