import os
import pandas as pd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score, precision_score

problem_title = 'Types 2 Diabetes prediction'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1]
)
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()


# -----------------------------------------------------------------------------
# Scores
# -----------------------------------------------------------------------------

class Recall(ClassifierBaseScoreType):
    def __init__(self,
                 name : str ='Recall_0',
                 precision : int =2,
                 pos_label : int = 0):
        self.name = name
        self.precision = precision
        self.pos_label = pos_label
    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(y_true_label_index, y_pred_label_index, pos_label=self.pos_label)
        return score


class Precision(ClassifierBaseScoreType):
    def __init__(self,
                 name : str ='Precision_0',
                 precision : int =2,
                 pos_label : int = 0):
        self.name = name
        self.precision = precision
        self.pos_label = pos_label

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(y_true_label_index, y_pred_label_index, pos_label=self.pos_label)
        return score

score_types = [
    Recall(name='recall_1', pos_label=1),
    Recall(name='recall_0', pos_label=0),
    Precision(name='precision_1', pos_label=1),
    Precision(name='precision_0', pos_label=0),

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
