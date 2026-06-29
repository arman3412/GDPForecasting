import csv
import pandas as pd
import numpy as np
import datetime as dt
import torch
from data import make_windows



# Performs train/validation/test split and creates 8-quarter windows 

def split_data():
    data = pd.read_csv('oecd_panel.csv', parse_dates=['date'])

    X, y, groups, dates = make_windows(data, lookback=8, target_col='GDP_target', feature_cols=None)
    _, numerical_labels = np.unique(groups, return_inverse=True)

    train_val_cutoff = pd.Timestamp(2022, 1, 1)
    val_test_cutoff = pd.Timestamp(2024, 1, 1)

    X_train, y_train, groups_train, numerical_labels_train = [], [], [], []
    X_val, y_val, groups_val, numerical_labels_val = [], [], [], []
    X_test, y_test, groups_test, numerical_labels_test = [], [], [], []

    
    for feature, target, group, numerical_label, date in zip(X, y, groups, numerical_labels, dates):
        if date < train_val_cutoff:
            X_train.append(feature)
            y_train.append(target)
            groups_train.append(group)
            numerical_labels_train.append(numerical_label)

        elif date < val_test_cutoff:
            X_val.append(feature)
            y_val.append(target)
            groups_val.append(group)
            numerical_labels_val.append(numerical_label)
        
        else:
            X_test.append(feature)
            y_test.append(target)
            groups_test.append(group)
            numerical_labels_test.append(numerical_label)
    
    num_countries = len(set(groups))

    return {"X_train": np.array(X_train), "y_train": np.array(y_train), "groups_train": np.array(groups_train), "numerical_labels_train": np.array(numerical_labels_train),
            "X_val": np.array(X_val), "y_val": np.array(y_val), "groups_val": np.array(groups_val), "numerical_labels_val": np.array(numerical_labels_val),
            "X_test": np.array(X_test), "y_test": np.array(y_test), "groups_test": np.array(groups_test), "numerical_labels_test": np.array(numerical_labels_test), 
            "num_countries": num_countries}

if __name__ == "__main__":
    data_splits = split_data()
    for key, value in data_splits.items():
        if key != "num_countries":
            print(f"{key}: {value.shape}")

                                            





