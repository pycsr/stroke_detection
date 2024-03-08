import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from stroke_detection_model.config.core import config
from stroke_detection_model.processing.features import (
    BmiImputer,
    WorkTypeOneHotEncoder,
    ResidenceTypeOneHotEncoder,
    SmokingStatusTypeOneHotEncoder,
)
from stroke_detection_model.processing.features import Mapper

stroke_detection_pipe = Pipeline(
    [
        ######### Imputation ###########
        ("bmi_imputation", BmiImputer(variables=config.model_config.bmi_var)),
        ######### Mapper ###########
        (
            "map_gender",
            Mapper(
                variables=config.model_config.gender_var,
                mappings=config.model_config.gender_mappings,
            ),
        ),
        (
            "map_ever_married",
            Mapper(
                variables=config.model_config.ever_married_var,
                mappings=config.model_config.ever_married_mappings,
            ),
        ),
        ######## Handle outliers ########
        # NA
        ######## One-hot encoding ########
        (
            "encode_work_type",
            WorkTypeOneHotEncoder(variables=config.model_config.work_type_var),
        ),
        (
            "encode_Residence_type",
            ResidenceTypeOneHotEncoder(
                variables=config.model_config.Residence_type_var
            ),
        ),
        (
            "encode_smoking_status",
            SmokingStatusTypeOneHotEncoder(
                variables=config.model_config.smoking_status_var
            ),
        ),
        # Scale features
        ("scaler", StandardScaler()),
        # Regressor
        (
            "model_rf",
            RandomForestClassifier(
                n_estimators=config.model_config.n_estimators,
                max_depth=config.model_config.max_depth,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
# for xgboost
# XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=config.model_config.n_estimators,
#     max_depth=config.model_config.max_depth,
#     random_state=config.model_config.random_state,
# ),
