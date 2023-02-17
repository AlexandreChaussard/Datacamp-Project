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
workflow = rw.workflows.Estimator()

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

def get_cgm_data(user_id, path='.'):
    """
    Fetch the CGM data of a given individual in the dataset
    """

    # Fetching the CGM data
    cgm_data = pd.read_csv(os.path.join(path, 'data', f'case  {user_id}.csv'), index_col=[0])
    # Interpolate the missing data (this is verified by smoothing techniques of Abbott & Dexcom CGMs)
    cgm_data = cgm_data.interpolate()
    return cgm_data.rename(columns={'hora': 'timestamp', 'glucemia': 'glycemia'})


def _read_clinical_data_and_labels(path):
    """
    Fetch the clinical data for every individual (this is meant to be used as a private shortcut function)
    """
    # Fetching the clinical data
    clinical_data = pd.read_csv(os.path.join(path, 'data', 'clinical_data.txt'), sep=" ")
    # We have an indexing issue on the clinical data as "79" has been skipped
    clinical_data = clinical_data.reset_index()
    # Dropping the old index and the "follow up" that isn't relevant in our study
    clinical_data = clinical_data.drop(columns=["index", "follow.up"])

    # Now we fetch the labels and seperate them from the original dataframe
    y = clinical_data["T2DM"].values.astype(np.int32)
    X = clinical_data.drop(columns=["T2DM"])
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
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=0
    )

    X_train = pd.DataFrame(X_train, columns=X.columns)
    return X_train, y_train


def get_test_data(path='.'):
    """
    Fetch the test dataset (X_test, y_test)
    X_test is a pandas Dataframe
    y_test is a numpy vector of labels (1 if turned out to be diabetic, 0 otherwise)
    """
    X, y = _read_clinical_data_and_labels(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=0
    )

    X_test = pd.DataFrame(X_train, columns=X.columns)
    return X_test, y_test
