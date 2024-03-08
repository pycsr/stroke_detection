"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from stroke_detection_model.config.core import config
from stroke_detection_model.processing.features import (
    BmiImputer,
    Mapper,
    WorkTypeOneHotEncoder,
    ResidenceTypeOneHotEncoder,
    SmokingStatusTypeOneHotEncoder,
)


def test_bmi_variable_imputer(sample_input_data):
    # Given
    imputer = BmiImputer(variables=config.model_config.bmi_var)
    print(sample_input_data[0].loc[19, "bmi"])
    assert np.isnan(sample_input_data[0].loc[19, "bmi"])

    # When
    subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[19, "bmi"].round(2) == 28.66


def test_gender_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(
        variables=config.model_config.gender_var,
        mappings=config.model_config.gender_mappings,
    )
    assert sample_input_data[0].loc[19, "gender"] == "Male"

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[19, "gender"] == 0


def test_worktype_variable_encoder(sample_input_data):
    # Given
    encoder = WorkTypeOneHotEncoder(variables=config.model_config.work_type_var)
    assert sample_input_data[0].loc[19, "work_type"] == "Govt_job"

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[19, "work_type_Govt_job"] == 1.0


def test_residence_variable_encoder(sample_input_data):
    # Given
    encoder = ResidenceTypeOneHotEncoder(
        variables=config.model_config.Residence_type_var
    )
    assert sample_input_data[0].loc[19, "Residence_type"] == "Urban"

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[19, "Residence_type_Urban"] == 1.0


def test_smoking_variable_encoder(sample_input_data):
    # Given
    encoder = SmokingStatusTypeOneHotEncoder(
        variables=config.model_config.smoking_status_var
    )
    assert sample_input_data[0].loc[19, "smoking_status"] == "Unknown"

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[19, "smoking_status_Unknown"] == 1.0
