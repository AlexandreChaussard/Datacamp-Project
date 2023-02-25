"""
This script enables to do some preprocessing on the data,
following the procedure proposed in the `DT2_starting_kit.ipynb`.

It will generate the `external_data.csv` and the `data/train.csv`, `data/test.csv` files
to be used in the `problem.py` and the workflow.
"""
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



def get_cgm_data(user_id, path='.'):
    """
    Fetch the CGM data of a given individual in the dataset
    """

    # Fetching the CGM data
    cgm_data = pd.read_csv(os.path.join(path, 'data', 'raw', f'case  {user_id}.csv'), index_col=[0])
    # Interpolate the missing data (this is verified by smoothing techniques of Abbott & Dexcom CGMs)
    cgm_data = cgm_data.interpolate()
    return cgm_data.rename(columns={'hora': 'timestamp', 'glucemia': 'glycemia'})


def get_48h_cgm_data(path='.'):
    """
    Fetch the clinical data for every individual (this is meant to be used as a private shortcut function)
    putting it into a dataframe for `external_data.csv`
    """
    user_id_list = [i for i in range(1, 209)]

    def sample_index_to_time(sample_index):
        # We introduce an intermediate function to turn the 5 minute sampled indexes of the CGM into HH:MM format
        # We know that every one is starting the study around midnight + or - 4 minutes
        minute_value = sample_index * 5

        hour_value = str(minute_value // 60)
        if len(hour_value) == 1:
            hour_value = "0" + hour_value

        minute_value = str(minute_value % 60)
        if len(minute_value) == 1:
            minute_value = "0" + minute_value

        return hour_value + ":" + minute_value

    # Building the dataframe
    df = pd.DataFrame(columns=[sample_index_to_time(i) for i in range(0, 576)])
    for user_id in user_id_list:
        cgm_data = get_cgm_data(user_id, path)["glycemia"].values.tolist()
        # We have 2 possible situations, since the data are sampled at 5 minutes rate:
        #  * Either the patient has come through a 48h monitoring, in which case the time serie is 576 long
        #  * Either the patient has come through a 24h monitoring, which case the time serie is 288 long, and we fill
        while len(cgm_data) < 576:
            cgm_data += [np.NaN]
        cgm_data = np.array(cgm_data)

        df.loc[user_id] = cgm_data
    return df


def read_raw_clinical_data(path='.'):
    """
    Helper function to simply fetch the raw clinical data
    """
    # Fetching the clinical data
    clinical_data = pd.read_csv(os.path.join(path, 'data', 'raw', 'clinical_data.txt'), sep=" ")
    # We have an indexing issue on the clinical data as "79" has been skipped
    clinical_data = clinical_data.reset_index()
    clinical_data.index += 1
    # Dropping the old index and the "follow up" that isn't relevant in our study
    clinical_data = clinical_data.drop(columns=["index", "follow.up"])
    return clinical_data


def read_clinical_data_and_labels(path='.'):
    """
    Fetch the clinical data for every individual (this is meant to be used as a private shortcut function)
    """
    # We fetch the raw clinical data first
    clinical_data = read_raw_clinical_data(path)

    # We also drop the "HbA1c" feature as it is part of our study to try to infer with blood measurement
    clinical_data = clinical_data.drop(columns=["HbA1c"])

    # Now we fetch the labels and seperate them from the original dataframe
    y = clinical_data["T2DM"].values.astype(np.int32)
    X = clinical_data.drop(columns=["T2DM"])

    # Filling the one missing value of the BIM using the iterative imputer (cf notebook for origin of that procedure)
    imp = IterativeImputer(max_iter=10, random_state=0)
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    return X, y


def build_datasets(test_size):
    # Build the `external_data.csv`
    print("[*] Collecting CGM data")
    external_data = get_48h_cgm_data()
    external_data.to_csv('external_data.csv', index_label='patient_id')

    print("  * external_data.csv built")

    # Build the `train.csv` and `test.csv`
    print("[*] Collecting clinical data")

    def build_set_helper(clinical_data, X_array, y_labels, fileName):
        # helper function to store the data into a csv file
        df = pd.DataFrame(X_array, index=X_array[:, 0].astype(np.int32), columns=clinical_data.columns)
        df = df.drop(columns=["index"])
        df['label'] = y_labels
        path = os.path.join('.', 'data', f'{fileName}.csv')
        df.to_csv(path, index_label='patient_id')
        print(f"  * {path} built")

    X, y = read_clinical_data_and_labels()
    # For now we put back the indexes in X so we can put them in the final result (this is temporary)
    X = X.reset_index()
    # We make stratified train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=0
    )

    build_set_helper(X, X_train, y_train, fileName='train')
    build_set_helper(X, X_test, y_test, fileName='test')

    print("[*] Data downloaded successfully!")


if __name__ == '__main__':
    # One can pick the test size they want in the end
    build_datasets(test_size=0.33)
