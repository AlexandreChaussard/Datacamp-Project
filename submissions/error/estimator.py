from sklearn.ensemble import RandomForestClassifier


# test failing submission
def get_estimator():
    clf = RandomForestClassifier(n_estimators=1, max_leaf_nodes=2,
                                 random_state=61)
    raise ValueError("Random failure")
    return clf
