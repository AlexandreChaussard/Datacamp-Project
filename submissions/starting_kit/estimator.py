import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.preprocessing as preprocessing
import numpy as np

from sklearn.linear_model import LogisticRegression


def get_cgm_data():
    data = pd.read_csv('external_data.csv')
    data.set_index('patient_id', inplace=True)
    return data


def get_patient_cgm_data(patient_id):
    cgm_data = get_cgm_data()
    patient_cgm = cgm_data.loc[patient_id]
    patient_cgm.dropna(inplace=True)
    return patient_cgm


class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def add_cgm_feature(self, clinical_data, feature_name, compute_feature_function):
        n_individuals = len(clinical_data)
        feature_column = np.zeros((n_individuals,))

        for i, user_id in enumerate(clinical_data.index.values):
            cgm_data = get_patient_cgm_data(int(user_id))
            feature = compute_feature_function(cgm_data)
            feature_column[i] = feature

        clinical_data[feature_name] = feature_column
        return clinical_data

    def compute_variance(self, cgm_data):
        return cgm_data.var()

    def compute_mean(self, cgm_data):
        return cgm_data.mean()

    def compute_average_time_in_range(self, cgm_data, normal_range=None):
        if normal_range is None:
            normal_range = [70, 127]

        index_in_range = cgm_data[
            (cgm_data >= normal_range[0]) & (cgm_data <= normal_range[1])
            ].index
        return len(index_in_range) / len(cgm_data.index)

    def compute_maximum(self, cgm_data):
        return cgm_data.max()

    def transform(self, X: pd.DataFrame):
        X = self.add_cgm_feature(X, "cgm_variance", self.compute_variance)
        X = self.add_cgm_feature(X, "cgm_mean", self.compute_mean)
        X = self.add_cgm_feature(X, "cgm_time_in_range", self.compute_average_time_in_range)
        X = self.add_cgm_feature(X, "cgm_max", self.compute_maximum)

        return X


def get_preprocessing():
    return preprocessing.StandardScaler(), preprocessing.MinMaxScaler()


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = LogisticRegression(
        random_state=0
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe
