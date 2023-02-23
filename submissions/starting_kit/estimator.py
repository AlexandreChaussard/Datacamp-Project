import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.preprocessing as preprocessing
import numpy as np
import problem as pb

from sklearn.tree import DecisionTreeClassifier


class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def add_cgm_feature(self, clinical_data, feature_name, compute_feature_function):
        n_individuals = len(clinical_data)
        feature_column = np.zeros((n_individuals,))

        for i, user_id in enumerate(clinical_data.index.values):
            cgm_data = pb.get_cgm_data(int(user_id) + 1)
            feature = compute_feature_function(cgm_data)
            feature_column[i] = feature

        clinical_data[feature_name] = feature_column
        return clinical_data

    def compute_variance(self, cgm_data):
        return cgm_data["glycemia"].var()

    def compute_mean(self, cgm_data):
        return cgm_data["glycemia"].mean()

    def compute_average_time_in_range(self, cgm_data, normal_range=None):
        if normal_range is None:
            normal_range = [70, 127]

        index_in_range = cgm_data[
            (cgm_data["glycemia"] >= normal_range[0]) & (cgm_data["glycemia"] <= normal_range[1])
            ].index
        return len(index_in_range) / len(cgm_data.index)

    def compute_maximum(self, cgm_data):
        return cgm_data["glycemia"].max()

    def transform(self, X: pd.DataFrame):
        X = self.add_cgm_feature(X, "cgm_variance", self.compute_variance)
        X = self.add_cgm_feature(X, "cgm_mean", self.compute_mean)
        X = self.add_cgm_feature(X, "cgm_time_in_range", self.compute_average_time_in_range)
        X = self.add_cgm_feature(X, "cgm_max", self.compute_maximum)
        X = X.drop(["HbA1c", "BMI"], axis=1)

        return X


def get_preprocessing():
    return preprocessing.StandardScaler(), preprocessing.MinMaxScaler()


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = DecisionTreeClassifier(
        random_state=0
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe
