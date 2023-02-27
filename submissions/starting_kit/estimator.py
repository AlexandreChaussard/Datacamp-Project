import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.preprocessing as preprocessing
import numpy as np
from numpy import trapz
from sklearn.ensemble import HistGradientBoostingClassifier


def get_cgm_data(path=""):
    data = pd.read_csv(path+'external_data.csv')
    data.set_index('patient_id', inplace=True)
    return data


def compute_estimate_hba1c(cmg_data):
    return 0.0296 * cmg_data.mean() + 2.419


def compute_variance(cgm_data):
    return cgm_data.var()


def compute_mean(cgm_data):
    return cgm_data.mean()


def compute_average_time_in_range(cgm_data, normal_range=None):
    if normal_range is None:
        normal_range = [70, 127]

    index_in_range = cgm_data[
        (cgm_data >= normal_range[0]) & (cgm_data <= normal_range[1])
        ].index
    return len(index_in_range) / len(cgm_data.index)


def compute_maximum(cgm_data):
    return cgm_data.max()


def compute_skewness(cgm_data):
    return cgm_data.skew()


def area_norm_points(cgm_data):
    if cgm_data.isna().sum() !=0:
        area = trapz(cgm_data[:288])/288
    else:
        area = trapz(cgm_data)/576
    return area


def compute_Q3(cgm_data):
    return cgm_data.quantile(0.75)


def sampen(L, m=2, r=3):
    """Sample entropy."""
    N = len(L)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    return -np.log(A / B)


def compute_sd1(cgm_val):
    # consider only 24 hours of data for each patient
    cgm_val = cgm_val[:288]
    cgm_data_n = np.array(cgm_val[:-1])
    cgm_data_n1 = np.array(cgm_val[1:])

    sd1 = np.sqrt(0.5) * np.std(cgm_data_n1 - cgm_data_n)
    return sd1


def compute_sd2(cgm_val):
    # consider only 24 hours of data for each patient
    cgm_val = cgm_val[:288]
    cgm_data_n = np.array(cgm_val[:-1])
    cgm_data_n1 = np.array(cgm_val[1:])

    sd2 = np.sqrt(0.5) * np.std(cgm_data_n1 + cgm_data_n)
    return sd2


class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def add_cgm_feature(self, clinical_data, feature_name, compute_feature_function):

        cgm_df = get_cgm_data()
        
        feature = cgm_df.apply(compute_feature_function, axis=1)

        clinical_data[feature_name] = feature.loc[clinical_data.index]

        return clinical_data

    def transform(self, X: pd.DataFrame):
        X = self.add_cgm_feature(X, "hba1c_estimate", compute_estimate_hba1c)
        X = self.add_cgm_feature(X, "cgm_variance", compute_variance)
        X = self.add_cgm_feature(X, "cgm_mean", compute_mean)
        X = self.add_cgm_feature(X, "cgm_time_in_range", compute_average_time_in_range)
        X = self.add_cgm_feature(X, "cgm_max", compute_maximum)
        X = self.add_cgm_feature(X, "cgm_area_under", area_norm_points)
        X = self.add_cgm_feature(X, "skewness", compute_skewness)
        X = self.add_cgm_feature(X, "Q3", compute_Q3)
        X = self.add_cgm_feature(X, 'sd1', compute_sd1)
        X = self.add_cgm_feature(X, 'sd2', compute_sd2)
    
        return X


def get_preprocessing():
    return preprocessing.StandardScaler(), preprocessing.MinMaxScaler()


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()
    classifier = HistGradientBoostingClassifier(
        l2_regularization=0.5,
        learning_rate=0.01,
        max_iter=500,
        class_weight={1: 10, 0: 1},
        random_state=42
    )
    """
        max_depth=26,
        learning_rate=0.0045,
        l2_regularization=0.349,
        min_samples_leaf=32,
        max_iter=99,
        class_weight={1: 10, 0: 1},
        random_state=42
    )"""

    pipe = make_pipeline(
        feature_extractor,
        preprocessing.MinMaxScaler(),
        classifier
    )

    return pipe
