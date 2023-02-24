import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import RepeatedStratifiedKFold

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

def _load_data(path, fileName):
    """
    Helper function to load a data file
    """
    df = pd.read_csv(os.path.join(path, 'data', f'{fileName}.csv'))
    df = df.set_index('patient_id')
    # We output X (pandas dataframe) samples, y (np array) labels
    return df.drop(columns=['label']), df['label'].values


def get_train_data(path='.'):
    """
    Fetch the training dataset (X_train, y_train)
    X_train is a pandas Dataframe
    y_train is a numpy vector of labels (1 if turned out to be diabetic, 0 otherwise)
    """
    return _load_data(path, 'train')


def get_test_data(path='.'):
    """
    Fetch the test dataset (X_test, y_test)
    X_test is a pandas Dataframe
    y_test is a numpy vector of labels (1 if turned out to be diabetic, 0 otherwise)
    """
    return _load_data(path, 'test')
