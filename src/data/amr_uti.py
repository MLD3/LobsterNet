import os, random
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def generate_amr_uti_nc(data_dir, rep, prescriptions, nc_rate=0.5,
                        nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # check data path
    assert os.path.isdir(data_dir)
    p_path = os.path.join(data_dir, "all_prescriptions.csv")
    f_path = os.path.join(data_dir, "all_uti_features.csv")
    l_path = os.path.join(data_dir, "all_uti_resist_labels.csv")
    assert all(os.path.isfile(p) for p in [p_path, f_path, l_path])
    # set random seed
    random.seed(rep)
    np.random.seed(rep)
    # Load raw sdata
    prescription_df = pd.read_csv(p_path)
    features_df = pd.read_csv(f_path)
    label_df = pd.read_csv(l_path)
    covariates = [c for c in features_df.columns if c not in ["example_id", "is_train", "uncomplicated"]]
    all_prescriptions = prescription_df.prescription.unique()
    if nc_type == "one-sided": assert len(set(prescriptions)) == 1
    elif nc_type == "two-sided": assert len(set(prescriptions)) == 2
    assert all(p in all_prescriptions for p in prescriptions)
    data_df = prescription_df.merge(features_df, on=["example_id", "is_train"]).merge(
        label_df, on=["example_id", "is_train", "uncomplicated"])
    data_df.drop(columns=["uncomplicated"], inplace=True)
    
    # two-sided non-compliance keeps only selected prescriptions
    if nc_type == "two-sided": data_df = data_df[data_df.prescription.isin(prescriptions)].copy()

    # standardize features
    X = data_df[covariates].values
    for i in range(X.shape[1]):
        if len(np.unique(X[:, i])) > 2: # avoid binary features
            X[:, i] = (X[:, i] - X[:, i].mean())/X[:, i].std()
    n, p = X.shape

    # generate treatment assignment
    T = np.int32(data_df.prescription==prescriptions[0])
    assert set(np.unique(T))=={0,1}

    # simulate treatment intake
    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if nc_type == "two-sided":
        w_t0 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.95, 0.02, 0.015, 0.01, 0.005], k=p))
        score_t0 = np.exp(X.dot(w_t0))
        p_t0 = score_t0 / score_t0.sum()
        idx_t0 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t0)
        A_t0[idx_t0] = 1
        w_t1 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.95, 0.02, 0.015, 0.01, 0.005], k=p))
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    if nc_type == "one-sided":
        w_t1 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.95, 0.02, 0.015, 0.01, 0.005], k=p))
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # generate outcome
    if nc_type == "one-sided":
        Y_a0 = np.zeros(n)
        Y_a1 = 1-np.float32(data_df[prescriptions[0]].values)
    elif nc_type == "two-sided":
        Y_a0 = 1-np.float32(data_df[prescriptions[1]].values)
        Y_a1 = 1-np.float32(data_df[prescriptions[0]].values)
    Y_a = np.stack((Y_a0, Y_a1)).T

    # ground truth counterfactuals
    Y_t0 = Y_a[range(n), A_t[:, 0]]
    Y_t1 = Y_a[range(n), A_t[:, 1]]
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = Y_t[range(n), T]

    # organize data into new df
    is_train = data_df.is_train.values.reshape(-1, 1)
    data_arr = np.hstack((X, np.vstack((T, A, Y)).T, A_t, Y_a, Y_t, is_train))
    columns =  covariates + ["T", "A", "Y", "A_T0", "A_T1", "Y_A0", "Y_A1", "Y_T0", "Y_T1", "train"]
    data_df = pd.DataFrame(data=data_arr, columns=columns)
    train_df = data_df[data_df.train==1].drop(columns=["train"]).reset_index(drop=True)
    test_df = data_df[data_df.train==0].drop(columns=["train"]).reset_index(drop=True)
    return train_df, test_df, covariates