from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class BmiImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'BMI' column by replacing them with the mean value of BMI column."""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variable name should be a string")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.fill_value = X[self.variables].mean()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        # print("Mapper.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # for feature in self.variables:
        # Code used to debug the na values in mapper , turned out it was first mapper yr causing the problem
        # hardcoded to yr values , need to check later what is the problem
        # print(X.isna().sum())
        # numeric_columns = X.select_dtypes(include=[np.number]).columns
        # print((X[numeric_columns].isna() | np.isinf(X[numeric_columns])).sum())
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X


class WorkTypeOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode any feature column"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variable name should be a string")

        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variables]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out(
            [self.variables]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        encoded_work_type = self.encoder.transform(X[[self.variables]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_work_type

        # drop 'WorkType' column after encoding
        X.drop(self.variables, axis=1, inplace=True)

        return X


class ResidenceTypeOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode any feature column"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variable name should be a string")

        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variables]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out(
            [self.variables]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        encoded_Residence_type = self.encoder.transform(X[[self.variables]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_Residence_type

        # drop 'ResidenceType' column after encoding
        X.drop(self.variables, axis=1, inplace=True)

        return X


class SmokingStatusTypeOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode any feature column"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variable name should be a string")

        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variables]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out(
            [self.variables]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        encoded_smoking_status = self.encoder.transform(X[[self.variables]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_smoking_status

        # drop 'smokingstatus' column after encoding
        X.drop(self.variables, axis=1, inplace=True)

        return X
