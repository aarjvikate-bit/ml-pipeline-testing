import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_pipeline import (
    process_data,
    train_model,
    validate_predictions,
    evaluate_model
)


@pytest.fixture
def sample_data():
    return load_iris(as_frame=True).frame.copy()


def test_process_data_valid_input(sample_data):
    result = process_data(sample_data)
    assert not result.empty
    assert "target" in result.columns
    assert result.isnull().sum().sum() == 0


def test_process_data_handles_missing_feature(sample_data):
    sample_data.loc[0, "sepal length (cm)"] = np.nan
    result = process_data(sample_data)
    assert result["sepal length (cm)"].isnull().sum() == 0


def test_process_data_missing_target_column(sample_data):
    with pytest.raises(ValueError, match="Missing target column"):
        process_data(sample_data.drop(columns=["target"]))


def test_process_data_rejects_non_numeric_feature(sample_data):
    sample_data["sepal length (cm)"] = "invalid"
    with pytest.raises(TypeError):
        process_data(sample_data)


def test_process_data_rejects_empty_dataset(sample_data):
    with pytest.raises(ValueError, match="Dataset is empty"):
        process_data(sample_data.iloc[0:0])


def test_training_pipeline_produces_predictions(sample_data):
    data = process_data(sample_data)
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_validation_rejects_wrong_length():
    with pytest.raises(ValueError, match="Prediction count"):
        validate_predictions([0, 1, 2], [0, 1])


def test_prediction_validation_rejects_unknown_class():
    with pytest.raises(ValueError, match="unknown class"):
        validate_predictions([0, 1, 2], [0, 1, 9])


def test_evaluation_returns_valid_accuracy(sample_data):
    data = process_data(sample_data)
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = evaluate_model(y_test, predictions)
    assert 0.0 <= accuracy <= 1.0
