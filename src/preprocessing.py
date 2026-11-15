
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "cardio_train.csv"


def load_data():
    """Load the cardiovascular disease dataset."""
    df = pd.read_csv(DATA_PATH, sep=";")
    return df




def clean_data(df):

    """Remove records with invalid measurements."""

    df = df.copy()

    invalid_systolic = (
        (df["ap_hi"] < 60) |
        (df["ap_hi"] > 300)
    )

    invalid_diastolic = (
        (df["ap_lo"] < 30) |
        (df["ap_lo"] > 200)
    )

    invalid_relationship = df["ap_hi"] <= df["ap_lo"]

    invalid_height = (
        (df["height"] < 100) |
        (df["height"] > 230)
    )

    invalid_weight = df["weight"] < 20

    invalid_records = (
        invalid_systolic |
        invalid_diastolic |
        invalid_relationship |
        invalid_height |
        invalid_weight
    )

    # Remove invalid records
    cleaned_df = df.loc[~invalid_records].copy()

    # Cleaning report
    print("\nCLEANING REPORT")
    print("Invalid height records:", invalid_height.sum())
    print("Invalid weight records:", invalid_weight.sum())
    print("Total records removed:", invalid_records.sum())
    print("Remaining records:", len(cleaned_df))

    return cleaned_df


def prepare_features(df):
    """Prepare features and target for machine learning."""

    df = df.copy()

    # Remove the unique identifier
    df = df.drop(columns=["id"])

    # Convert age from days to years
    df["age"] = (df["age"] / 365.25).round(2)

    # Rename columns
    df = df.rename(columns={
        "ap_hi": "systolic_bp",
        "ap_lo": "diastolic_bp",
        "gluc": "glucose",
        "alco": "alcohol",
        "active": "physical_activity",
        "cardio": "target"
    })

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y



if __name__ == "__main__":
    df = load_data()

    cleaned_df = clean_data(df)

    X, y = prepare_features(cleaned_df)

    print("\nFEATURE PREPARATION")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Feature columns:", X.columns.tolist())
    print("\nTarget distribution:")
    print(y.value_counts())

    unusual_body_measurements = cleaned_df[
        (cleaned_df["height"] < 100) |
        (cleaned_df["height"] > 230) |
        (cleaned_df["weight"] < 20)
    ]

    print("Unusual records:", len(unusual_body_measurements))

    print(
        unusual_body_measurements[
            ["age", "gender", "height", "weight", "cardio"]
        ]
    )


    print("Dataset Shape:")

    print(df.shape)

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())



    print("\nUnique Values in Categorical Columns:")
    categorical_columns = [
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "cardio"
    ]

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts().sort_index())

    print("\nNumerical Feature Statistics:")
    numerical_columns = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo"
    ]

    print(df[numerical_columns].describe().round(2))


    print("\nDATA QUALITY INVESTIGATION")

    # Convert age from days to years for inspection
    age_years = df["age"] / 365.25

    print("\nAge range in years:")
    print(age_years.min(), "to", age_years.max())

    # Count suspicious measurements separately
    checks = {
        "Non-positive systolic BP": df["ap_hi"] <= 0,
        "Non-positive diastolic BP": df["ap_lo"] <= 0,
        "Systolic BP below diastolic BP": df["ap_hi"] < df["ap_lo"],
        "Systolic BP above 300": df["ap_hi"] > 300,
        "Diastolic BP above 200": df["ap_lo"] > 200,
        "Height below 100 cm": df["height"] < 100,
        "Height above 230 cm": df["height"] > 230,
        "Weight below 20 kg": df["weight"] < 20,
    }

    for description, condition in checks.items():
        print(f"{description}: {condition.sum()}")

    # Display examples of suspicious blood pressure records
    suspicious_bp = df[
        (df["ap_hi"] <= 0) |
        (df["ap_lo"] <= 0) |
        (df["ap_hi"] < df["ap_lo"]) |
        (df["ap_hi"] > 300) |
        (df["ap_lo"] > 200)
    ]

    print("\nSuspicious blood pressure examples:")
    print(
        suspicious_bp[
            ["age", "ap_hi", "ap_lo", "cardio"]
        ].head(15)
    )

    print("\nTotal suspicious BP records:", len(suspicious_bp))


