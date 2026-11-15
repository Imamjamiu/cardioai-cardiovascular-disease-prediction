# CardioAI — Cardiovascular Disease Prediction

**CardioAI** is a machine-learning-powered cardiovascular disease prediction application built with Python, Scikit-learn, and Streamlit.

The project demonstrates a structured machine-learning workflow, including data cleaning, preprocessing, model comparison, evaluation, and interactive prediction.

It is an independent engineering implementation and is separate from my earlier research project using Random Forest and KNN.

## Project Overview

The objective of CardioAI is to explore how health and lifestyle attributes can be processed by a machine-learning model to identify patterns associated with cardiovascular disease.

### Key Features

- Data cleaning and preprocessing
- Feature scaling and categorical encoding
- Comparison of multiple machine-learning algorithms
- Logistic Regression classification pipeline
- Model evaluation using classification metrics and ROC-AUC
- Interactive Streamlit prediction interface
- Estimated probability output
- Modular and reusable Python implementation

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Visualization:** Plotly
- **Web Application:** Streamlit

## Dataset

The project uses a cardiovascular disease dataset containing health and lifestyle attributes.

The dataset file is not included in this repository. To use the project with the dataset:

1. Obtain the dataset from its original source.
2. Create a `data` directory in the project root if it does not exist.
3. Place the dataset file inside that directory.
4. Ensure the filename is `cardio_train.csv`.

The expected file path is:

`data/cardio_train.csv`

The dataset is excluded from this repository pending verification of its redistribution licence. Please follow the dataset provider's terms when obtaining and using it.

## Data Preprocessing

The preprocessing workflow includes:

- Converting age from days into years
- Removing implausible or invalid measurements using operational data-quality filters
- Removing the identifier column
- Renaming columns for readability
- Separating input features from the target variable
- Scaling numerical features
- One-hot encoding categorical features

The cleaning filters are data-quality rules for this dataset, not clinical diagnostic thresholds.

## Model Development

Three algorithms were compared using five-fold cross-validation on the training data:

- Logistic Regression
- Random Forest
- Extra Trees

Logistic Regression was selected based on its cross-validation ROC-AUC performance.

The selected model is implemented as a Scikit-learn pipeline that combines preprocessing and classification.

## Model Evaluation

The selected Logistic Regression model achieved the following results on the project's held-out test split:

| Metric | Score |
|---|---:|
| Accuracy | 72.66% |
| ROC-AUC | 0.7940 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 | 0.71 | 0.79 | 0.74 |
| 1 | 0.75 | 0.67 | 0.71 |

These results describe performance on the project's test split. They do not establish clinical validity or performance on external populations.

## Project Structure

```text
Cardiovascular prediction/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── logistic_regression_pipeline.joblib
│
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
└── data/
    └── cardio_train.csv
```

The dataset directory and CSV file are required locally but are not committed to the public repository.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/imamjamiu/cardioai-cardiovascular-disease-prediction.git
```

### 2. Navigate into the project

```bash
cd cardioai-cardiovascular-disease-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

## Related Research Project

This engineering implementation is related to my earlier cardiovascular disease prediction research, which investigated Random Forest and KNN.

[Explore the Original Research — Prediction of Cardiovascular Disease Using Random Forest and KNN](https://github.com/Imamjamiu/cardiovascular-Disease-Prediction)

The two repositories are separate projects with different implementations, model choices, and experimental workflows.

## Disclaimer

CardioAI is intended strictly for educational, research, and machine-learning demonstration purposes. It is not a medical diagnostic tool, does not provide a clinically validated personal risk assessment, and must not replace professional medical advice or evaluation.
