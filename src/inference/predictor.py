from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_v1.cbm"

FEATURES = [
    "start_year",
    "area",
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]

CATEGORICAL_FEATURES = [
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]


class AgroYieldPredictor:
    """Loads the frozen CatBoost v1 model and performs inference."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_path))

        model_features = self.model.feature_names_

        if model_features != FEATURES:
            raise ValueError(
                "Model feature contract mismatch. "
                f"Expected {FEATURES}, received {model_features}."
            )

    def predict(self, payload: dict[str, Any]) -> float:
        """Return the raw model prediction in tonnes per hectare."""

        input_data = pd.DataFrame([payload], columns=FEATURES)

        for column in CATEGORICAL_FEATURES:
            input_data[column] = input_data[column].astype(str)

        prediction = self.model.predict(input_data)
        return float(prediction[0])

    @staticmethod
    def validate_prediction(prediction: float) -> bool:
        """Check whether the model output is a valid yield estimate."""
        return bool(np.isfinite(prediction) and prediction >= 0)
