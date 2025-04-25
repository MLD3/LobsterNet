import random, itertools
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sigmoid = lambda x: 1/(1 + np.exp(-x))


def generate_random_response(data, amplitude=10):
    n, p = data.shape
    rand_weight = np.random.uniform(low=-amplitude, high=amplitude, size=p)
    return data.dot(rand_weight)/np.sqrt(p) + np.random.normal(scale=amplitude/10, size=n)

#### Synthetic functions for controlling non-compliance rate & confounding strength ####
def generate_synthetic_nc_rate(n=1000, p=30, rep=0, nc_rate=0.5, num_u_ty=0, num_u_ay=0, amplitude=10,
                               nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)
    
    # generate features
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)
    if num_u_ty > 0:
        U_ty = np.random.multivariate_normal(mean=np.zeros(num_u_ty), cov=np.eye(num_u_ty), size=n)
    if num_u_ay > 0:
        U_ay = np.random.multivariate_normal(mean=np.zeros(num_u_ay), cov=np.eye(num_u_ay), size=n)

    # simulate treatment assignment
    if num_u_ty > 0: X_t = np.hstack((X, U_ty))
    else: X_t = X
    T_proba = sigmoid(generate_random_response(X_t, amplitude=amplitude))
    T = np.random.binomial(n=1, p=T_proba)

    # simulate treatment intake
    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if num_u_ay > 0: X_a = np.hstack((X, U_ay))
    else: X_a = X
    A_t1_score = sigmoid(generate_random_response(X_a, amplitude=amplitude))
    p_A_t1 = A_t1_score / A_t1_score.sum()
    nc_idx_A_t1 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_A_t1)
    A_t1[nc_idx_A_t1] = 0
    if nc_type == "two-sided":
        A_t0_score = sigmoid(generate_random_response(X_a, amplitude=amplitude))
        p_A_t0 = A_t0_score / A_t0_score.sum()
        nc_idx_A_t0 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_A_t0)
        A_t0[nc_idx_A_t0] = 1
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # simulate outcome
    X_y = X
    if num_u_ty > 0: X_y = np.hstack((X_y, U_ty))
    if num_u_ay > 0: X_y = np.hstack((X_y, U_ay))
    Y_a0_score = generate_random_response(X_y, amplitude=amplitude)
    Y_a0 = sigmoid(Y_a0_score + np.random.normal(scale=amplitude/10, size=n))
    Y_a1_score = generate_random_response(X_y, amplitude=amplitude)
    Y_a1 = sigmoid(Y_a1_score + np.random.normal(scale=amplitude/10, size=n))
    Y_a_score = np.stack((Y_a0_score, Y_a1_score)).T
    Y_a = np.stack((Y_a0, Y_a1)).T

    # get counter factuals
    Y_t0 = sigmoid(Y_a_score[range(n), A_t[:, 0]] + np.random.normal(scale=amplitude/10, size=n))
    Y_t1 = sigmoid(Y_a_score[range(n), A_t[:, 1]] + np.random.normal(scale=amplitude/10, size=n))
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = np.random.binomial(n=1, p=Y_t[range(n), T], size=n) # factual outcome

    # organize data into df
    data_arr = np.hstack((X, np.vstack((T, A, Y)).T, A_t, Y_a, Y_t))
    X_cols = [f"X{i+1}" for i in range(X.shape[1])]
    columns =  X_cols + ["T", "A", "Y", "A_T0", "A_T1", "Y_A0", "Y_A1", "Y_T0", "Y_T1"]
    data_df = pd.DataFrame(data=data_arr, columns=columns)
    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A", "Y"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]

    # validate assumption is met
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df
    

#### Synthetica functions for controlling amount of sharing ####
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

def generate_synthetic_nc_share(n=1000, p=30, rep=0, rho_shared=0.5, rho_separate=0.5, random_t=False,
                                nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)

    all_shared_response = generate_bernoli_response(X, rho_shared, inter=True)
    a0_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    a1_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    y0_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
    y1_reaponse = generate_bernoli_response(X, rho_separate, inter=True)

    if random_t:
        T = np.random.binomial(1, p=0.5, size=n)
    else:
        # t_reaponse = generate_bernoli_response(X, rho_separate, inter=True)
        # p_T = sigmoid(all_shared_response + t_reaponse)
        p_T = sigmoid(generate_bernoli_response(X, 0.1, inter=True))
        # p_T = sigmoid(X.dot(np.random.uniform(low=-1, high=1, size=p)))
        T = np.random.binomial(1, p_T)
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
    Y_a1 = all_shared_response + y1_reaponse + 10
    Y_a = np.stack((Y_a0, Y_a1)).T

    # get counterfactuals
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


def generate_synthetic_nc_effect(n=1000, p=30, rep=0, effect=10, amplitude=10,
                                nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)
    
    # generate features
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)

    # simulate treatment assignment
    T_proba = sigmoid(generate_random_response(X, amplitude=amplitude))
    T = np.random.binomial(n=1, p=T_proba)
    # simulate treatment intake
    if nc_type == "two-sided":
        p_A_t0 = sigmoid(generate_random_response(X, amplitude=amplitude))
        A_t0 = np.random.binomial(1, p_A_t0)
        p_A_t1 = sigmoid(generate_random_response(X, amplitude=amplitude))
        A_t1 = np.random.binomial(1, p_A_t1)
    if nc_type == "one-sided":
        A_t0 = np.zeros(n, dtype=int)
        p_A_t1 = sigmoid(generate_random_response(X, amplitude=amplitude))
        A_t1 = np.random.binomial(1, p_A_t1)
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # simulate outcome
    Y_a0_score = 0*np.ones(n) #generate_random_response(X, amplitude=10)
    Y_a0 = 0.1*np.ones(n) #sigmoid(Y_a0_score + np.random.normal(scale=1, size=n))
    Y_a1_score = generate_random_response(X, amplitude=amplitude) + effect
    Y_a1 = sigmoid(Y_a1_score + np.random.normal(scale=1, size=n))
    Y_a_score = np.stack((Y_a0_score, Y_a1_score)).T
    Y_a = np.stack((Y_a0, Y_a1)).T

    # get counter factuals
    Y_t0 = Y_a[range(n), A_t[:, 0]] #sigmoid(Y_a_score[range(n), A_t[:, 0]] + np.random.normal(scale=1, size=n))
    Y_t1 = Y_a[range(n), A_t[:, 1]] #sigmoid(Y_a_score[range(n), A_t[:, 1]] + np.random.normal(scale=1, size=n))
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = np.random.binomial(n=1, p=Y_t[range(n), T], size=n) # factual outcome

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

    # validate assumption is met
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df