import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from stroke_detection_model import __version__ as _version
from stroke_detection_model.config.core import config
from stroke_detection_model.processing.data_manager import load_pipeline

# from stroke_detection_model.processing.data_manager import pre_pipeline_preparation
from stroke_detection_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model"""

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    # validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday',
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns=config.model_config.features)

    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = bikeshare_pipe.predict(validated_data)
        results = {
            "predictions": np.floor(predictions),
            "version": _version,
            "errors": errors,
        }
        print(results)

    return results


if __name__ == "__main__":

    data_in = {
        "gender": ["Male"],
        "age": [85.0],
        "hypertension": [1],
        "heart_disease": [1],
        "ever_married": ["Yes"],
        "work_type": ["Private"],
        "Residence_type": ["Urban"],
        "avg_glucose_level": [125.0],
        "bmi": [50.0],
        "smoking_status": ["smokes"],
    }

    make_prediction(input_data=data_in)
