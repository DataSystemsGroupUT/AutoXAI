from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_sample_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y
