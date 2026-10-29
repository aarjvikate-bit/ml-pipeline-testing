import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def process_data(data):
    if data.empty:
        raise ValueError("Dataset is empty")

    if "target" not in data.columns:
        raise ValueError("Missing target column")

    result = data.copy()
    feature_columns = [
        c for c in result.columns if c != "target"
    ]

    if not feature_columns:
        raise ValueError("No feature columns found")

    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(result[col]):
            raise TypeError(f"Feature '{col}' must be numeric")

    result[feature_columns] = result[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    for col in feature_columns:
        if result[col].isna().all():
            raise ValueError(
                f"Feature '{col}' contains only missing values"
            )
        result[col] = result[col].fillna(result[col].median())

    if result["target"].isna().any():
        raise ValueError("Target contains missing values")

    return result


def train_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ])


def validate_predictions(y_true, predictions):
    if len(y_true) == 0:
        raise ValueError("Test labels are empty")

    if len(y_true) != len(predictions):
        raise ValueError(
            "Prediction count does not match test labels"
        )

    if pd.isna(predictions).any():
        raise ValueError("Predictions contain missing values")

    allowed_classes = set(np.unique(y_true))
    predicted_classes = set(np.unique(predictions))

    if not predicted_classes.issubset(allowed_classes):
        raise ValueError("Predictions contain unknown class labels")

    return True


def evaluate_model(y_true, predictions):
    validate_predictions(y_true, predictions)
    score = accuracy_score(y_true, predictions)

    if not 0.0 <= score <= 1.0:
        raise ValueError("Accuracy is outside the valid range")

    return score
