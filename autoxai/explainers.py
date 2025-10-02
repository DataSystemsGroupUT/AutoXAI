import numpy as np

def get_explanations(model, X, method):
    np.random.seed(0)
    feature_importances = dict(zip(X.columns, np.random.rand(X.shape[1])))
    return feature_importances
