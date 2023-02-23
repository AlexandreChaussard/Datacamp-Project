import os
import pandas as pd
import rampwf as rw
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

problem_title = 'Types 2 Diabetes prediction'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1]
)
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()

score_types = [
    rw.score_types.BalancedAccuracy(name='balanced_accuracy'),
    rw.score_types.ROCAUC(name="auc"),
    rw.score_types.ClassificationError(name='error')
]


# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------

def get_cv(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    return cv.split(X, y)


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------

def _read_raw_clinical_data(path='.'):
    """
    Helper function to simply fetch the raw clinical data
    """
    # Fetching the clinical data
    clinical_data = pd.read_csv(os.path.join(path, 'data', 'clinical_data.txt'), sep=" ")
    # We have an indexing issue on the clinical data as "79" has been skipped
    clinical_data = clinical_data.reset_index()
    clinical_data.index += 1
    # Dropping the old index and the "follow up" that isn't relevant in our study
    clinical_data = clinical_data.drop(columns=["index", "follow.up"])
    return clinical_data


def get_HbA1c_and_labels_data(path='.'):
    """
    Fetch the HbA1c feature for all individual in the dataset, with its matching DT2 label.
    This feature was originally removed from the dataset as it is a blood measurement, which we want to avoid.
    However we would still like to pursue statistical analysis on the HbA1c, so we leave that function available.
    """
    # First we fetch the clinical data
    clinical_data = _read_raw_clinical_data(path)
    # Then we output the HbA1c and the labels dataframe, for each user, may it be missing or not
    return clinical_data[["HbA1c", "T2DM"]]


def _read_clinical_data_and_labels(path='.'):
    """
    Fetch the clinical data for every individual (this is meant to be used as a private shortcut function)
    """
    # We fetch the raw clinical data first
    clinical_data = _read_raw_clinical_data(path)

    # We also drop the "HbA1c" feature as it is part of our study to try to infer with blood measurement
    clinical_data = clinical_data.drop(columns=["HbA1c"])

    # Now we fetch the labels and seperate them from the original dataframe
    y = clinical_data["T2DM"].values.astype(np.int32)
    X = clinical_data.drop(columns=["T2DM"])

    # Filling the one missing value of the BIM with the mean
    X["BMI"].fillna(value=X["BMI"].mean(), inplace=True)

    return X, y


# Define the test size of the dataset
test_size = 0.33


def get_train_data(path='.'):
    """
    Fetch the training dataset (X_train, y_train)
    X_train is a pandas Dataframe
    y_train is a numpy vector of labels (1 if turned out to be diabetic, 0 otherwise)
    """
    X, y = _read_clinical_data_and_labels(path)
    # For now we put back the indexes in X so we can put them in the final result (this is temporary)
    X = X.reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=0
    )

    X_train = pd.DataFrame(X_train, index=X_train[:, 0].astype(np.int32), columns=X.columns)
    # And we remove the column index
    X_train = X_train.drop(columns=["index"])
    return X_train, y_train


def get_test_data(path='.'):
    """
    Fetch the test dataset (X_test, y_test)
    X_test is a pandas Dataframe
    y_test is a numpy vector of labels (1 if turned out to be diabetic, 0 otherwise)
    """
    X, y = _read_clinical_data_and_labels(path)
    # For now we put back the indexes in X so we can put them in the final result (this is temporary)
    X = X.reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=0
    )

    X_test = pd.DataFrame(X_test, index=X_test[:, 0].astype(np.int32), columns=X.columns)
    # And we remove the column index
    X_test = X_test.drop(columns=["index"])
    return X_test, y_test
