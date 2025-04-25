import pyreadr, random
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

def generate_ihdp_nc(rdata_path, rep=0, rate=0.5, size=4, non_compliance_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    # load ihdp features
    ihdp_df = pyreadr.read_r(rdata_path)["ihdp"]
    drop_idx = ihdp_df[(ihdp_df.momwhite==0) & (ihdp_df.treat==1)].index
    ihdp_df.drop(index=drop_idx, inplace=True)
    ihdp_df.reset_index(drop=True, inplace=True)
    # standardize all continuous covariates:
    for col in ihdp_df.columns:
        if not ihdp_df[col].isin([0, 1]).all():
            ihdp_df[col] = (ihdp_df[col] - ihdp_df[col].mean()) / ihdp_df[col].std()

    # randomly select observed and unobserved covariates
    X_feat_names = ['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
                    'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 
                    'cig', 'first', 'booze', 'drugs', 'work.dur', 'prenatal', 
                    'ark', 'ein', 'har', 'mia', 'pen', 'tex', 'was']
    U_feat_names = ['momwhite', 'momblack', 'momhisp']
    X = ihdp_df[X_feat_names].values
    Z = ihdp_df[X_feat_names+U_feat_names].values
    T = ihdp_df["treat"].values
    n, p_x, p_z = len(ihdp_df), X.shape[1], Z.shape[1]

    # simulate treatment intake
    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if non_compliance_type == "two-sided":
        w_t0 = np.random.uniform(low=0, high=1, size=p_x)
        score_t0 = np.exp(X.dot(w_t0))
        p_t0 = score_t0 / score_t0.sum()
        idx_t0 = np.random.choice(n, size=int(np.round(n*rate)), replace=False, p=p_t0)
        A_t0[idx_t0] = 1
        w_t1 = np.random.uniform(low=0, high=1, size=p_x)
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    elif non_compliance_type == "one-sided":
        w_t1 = np.random.uniform(low=0, high=1, size=p_x)
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # simulate outcome
    w = np.array(random.choices([0.0, 0.1, 0.2, 0.3, 0.4], weights=[0.6, 0.1, 0.1, 0.1, 0.1], k=p_x))
    Y_a0 = np.exp((X+0.5).dot(w))
    Y_a1 = (X).dot(w)
    omega = (Y_a1[A==1] - Y_a0[A==1]).mean() - size
    Y_a1 = Y_a1 - omega
    Y_a = np.stack((Y_a0, Y_a1)).T

    # get counter factuals
    Y_t0 = Y_a[range(n), A_t[:, 0]] + np.random.normal(scale=1, size=n)
    Y_t1 = Y_a[range(n), A_t[:, 1]] + np.random.normal(scale=1, size=n)
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = Y_t[range(n), T] # factual outcome
    Y_a = Y_a + np.random.normal(scale=1, size=(n, 2))

    # organize data into df
    data_arr = np.hstack((X, np.vstack((T, A, Y)).T, A_t, Y_a, Y_t))
    X_cols = [f"X{i+1}" for i in range(X.shape[1])]
    columns =  X_cols + ["T", "A", "Y", "A_T0", "A_T1", "Y_A0", "Y_A1", "Y_T0", "Y_T1"]
    data_df = pd.DataFrame(data=data_arr, columns=columns)

    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    if non_compliance_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif non_compliance_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df

def generate_bernoli_response(X, rho, inter=True):
    n, p = X.shape
    alpha = np.random.binomial(n=1, p=rho, size=p)
    response = X.dot(alpha) + np.random.normal(size=n)
    if inter:
        beta = np.random.binomial(n=1, p=rho/p, size=p**2)
        X_inters = []
        for i in range(p):
            for j in range(p):
                X_inters.append(X[:, i]*X[:, j])
        X_inters = np.stack(X_inters).T
        response += X_inters.dot(beta)
    return response

sigmoid = lambda x: 1/(1 + np.exp(-x))

def generate_ihdp_share(rdata_path, rep=0, size=10, rho_shared=0.5, rho_separate=0.5,
        nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    # load ihdp features
    ihdp_df = pyreadr.read_r(rdata_path)["ihdp"]
    # drop_idx = ihdp_df[(ihdp_df.momwhite==0) & (ihdp_df.treat==1)].index
    # ihdp_df.drop(index=drop_idx, inplace=True)
    ihdp_df.reset_index(drop=True, inplace=True)

    # randomly select observed and unobserved covariates
    X_feat_names = ['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
                    'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 
                    'cig', 'first', 'booze', 'drugs', 'work.dur', 'prenatal', 
                    'ark', 'ein', 'har', 'mia', 'pen', 'tex', 'was']
    U_feat_names = ['momwhite', 'momblack', 'momhisp']
    # standardize all continuous covariates:
    # for col in X_feat_names + U_feat_names:
    #     if not ihdp_df[col].isin([0, 1]).all():
    #         ihdp_df[col] = (ihdp_df[col] - ihdp_df[col].mean()) / ihdp_df[col].std()
    X = StandardScaler().fit_transform(ihdp_df[X_feat_names].values)
    n, p_x = len(ihdp_df), X.shape[1]

    # T = ihdp_df["treat"].values
    p_T = sigmoid(generate_bernoli_response(X, 0.1, inter=True))
    T = np.random.binomial(1, p_T)

    # simulate shared responses
    all_shared_response = generate_bernoli_response(X, rho_shared, inter=True)
    a1_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    y0_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    a0_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    y1_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    
    # simulate treatment intake
    if nc_type == "two-sided":
        p_A_t0 = 1-sigmoid(all_shared_response+a0_reaponse)
        A_t0 = np.random.binomial(1, p_A_t0)
        p_A_t1 = sigmoid(all_shared_response+a1_reaponse)
        A_t1 = np.random.binomial(1, p_A_t1)
    if nc_type == "one-sided":
        A_t0 = np.zeros(n, dtype=int)
        p_A_t1 = sigmoid(all_shared_response+a1_reaponse)
        A_t1 = np.random.binomial(1, p_A_t1)
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]
    
    # simulate outcome
    Y_a0 = all_shared_response + y0_reaponse
    Y_a1 = all_shared_response + y1_reaponse + size
    # omega = (Y_a1[A==1] - Y_a0[A==1]).mean() - size
    # Y_a1 = Y_a1 - omega
    Y_a = np.stack((Y_a0, Y_a1)).T

    # ground counterfactuals
    Y_t0 = Y_a[range(n), A_t[:, 0]] + np.random.normal(scale=1, size=n)
    Y_t1 = Y_a[range(n), A_t[:, 1]] + np.random.normal(scale=1, size=n)
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = Y_t[range(n), T] # factual outcome
    Y_a = Y_a + np.random.normal(scale=1, size=(n, 2))

    # organize data into df
    data_arr = np.hstack((X, np.vstack((T, A, Y)).T, A_t, Y_a, Y_t))
    X_cols = [f"X{i+1}" for i in range(X.shape[1])]
    columns =  X_cols + ["T", "A", "Y", "A_T0", "A_T1", "Y_A0", "Y_A1", "Y_T0", "Y_T1"]
    data_df = pd.DataFrame(data=data_arr, columns=columns).sample(frac=1)

    # train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df
