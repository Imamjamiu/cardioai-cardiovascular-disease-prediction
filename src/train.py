import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from preprocessing import load_data, clean_data, prepare_features
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


# Load the dataset
df = load_data()

# Clean the dataset
cleaned_df = clean_data(df)

# Prepare features and target
X, y = prepare_features(cleaned_df)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Define numerical features
numeric_features = [
    "age",
    "height",
    "weight",
    "systolic_bp",
    "diastolic_bp"
]

# Define categorical features
categorical_features = [
    "gender",
    "cholesterol",
    "glucose",
    "smoke",
    "alcohol",
    "physical_activity"
]

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Define candidate models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

print("\nMODEL COMPARISON USING 5-FOLD CROSS-VALIDATION")

for name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=1
    )

    print(f"\n{name}")
    print("Fold scores:", scores)
    print("Mean ROC-AUC:", scores.mean())
    print("Standard deviation:", scores.std())


# Build the complete machine-learning pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

# Train the model
print("\nTRAINING LOGISTIC REGRESSION MODEL")

# Perform 5-fold cross-validation on training data
print("\n5-FOLD CROSS-VALIDATION")

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)

print("Individual fold ROC-AUC scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())



model_pipeline.fit(X_train, y_train)

# SAVE TRAINED MODEL
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"

joblib.dump(model_pipeline, MODEL_PATH)

print("\nMODEL SAVED SUCCESSFULLY")
print("Saved to:", MODEL_PATH)


# Generate predictions
y_pred = model_pipeline.predict(X_test)

# Generate prediction probabilities for class 1
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

y_probability = model_pipeline.predict_proba(X_test)[:, 1]

print("\nFIRST 10 PREDICTED PROBABILITIES:")
print(y_probability[:10])

# Evaluate the model
print("\nMODEL EVALUATION")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))


# Display results
print("\nPREPROCESSING PIPELINE")

# Display split information
print("\nTRAINING AND TESTING SPLIT")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
