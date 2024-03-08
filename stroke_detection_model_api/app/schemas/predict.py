from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from stroke_detection_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    # predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "gender": "Male",
                        "age": 38.5,
                        "hypertension": 0,
                        "heart_disease": 1,
                        "ever_married": "Yes",
                        "work_type": "Private",
                        "Residence_type": "Urban",
                        "avg_glucose_level": 228.69,
                        "bmi": 36.6,
                    }
                ]
            }
        }
