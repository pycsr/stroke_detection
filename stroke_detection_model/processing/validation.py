import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from stroke_detection_model.config.core import config
from stroke_detection_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    gender: Optional[str]
    age: Optional[float]
    holiday: Optional[str]
    hypertension: Optional[int]
    ever_married: Optional[str]
    work_type: Optional[str]
    Residence_type: Optional[str]
    avg_glucose_level: Optional[float]
    bmi: Optional[float]
    smoking_status: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
