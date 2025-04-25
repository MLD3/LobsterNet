import random, itertools
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sigmoid = lambda x: 1/(1 + np.exp(-x))

def generate_synthetic_nc_gap_old(n=1000, p=10, rep=0, nc_rate_gap=0,
                              nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    sigma = np.random.uniform(low=-1, high=1, size=(p,p))
    cov = (sigma + sigma.T)/2
    # X_t0 = np.random.multivariate_normal(mean=np.zeros(p)+(distance/2), cov=cov, size=round(n/1.1), check_valid="ignore")
    # X_t1 = np.random.multivariate_normal(mean=np.zeros(p)-(distance/2), cov=cov, size=n-round(n/1.1), check_valid="ignore")
    # X = np.vstack((X_t0, X_t1))
    # T = np.zeros(n, dtype=int)
    # T[n-round(n/2):] += 1

    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n, check_valid="ignore")
    T = np.zeros(n, dtype=int)
    w_t = np.random.uniform(-0.1, 0.1, size=p)
    T = np.random.binomial(1, sigmoid(X.dot(w_t)))
    idx_t0, idx_t1 = np.where(T==0)[0], np.where(T==1)[0]

    # # simulate treatment intake
    # if nc_type == "two-sided":
    #     w_t0 = np.random.uniform(low=-1, high=1, size=p)
    #     A_t0 = np.random.binomial(1, sigmoid(X.dot(w_t0)/np.sqrt(p)))
    #     w_t1 = np.random.uniform(low=-1, high=1, size=p)
    #     A_t1 = np.random.binomial(1, sigmoid(X.dot(w_t1)/np.sqrt(p)))
    # if nc_type == "one-sided":
    #     A_t0 = np.zeros(n+1000, dtype=int)
    #     w_t1 = np.random.uniform(low=-1, high=1, size=p)
    #     A_t1 = np.random.binomial(1, sigmoid(X.dot(w_t1)/np.sqrt(p)))
    # A_t = np.stack((A_t0, A_t1)).T
    # A = A_t[range(n+1000), T]

    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if nc_type == "two-sided":
        # nc for treatment not assigned
        w_t0 = np.random.uniform(low=0, high=1, size=p)
        score_t0 = np.exp(X.dot(w_t0))
        nc_idx_t0 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t0[idx_t0]/score_t0[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t0[idx_t1]/score_t0[idx_t1].sum())
        ))
        A_t0[nc_idx_t0] = 1
        # nc for treatment assigned
        w_t1 = np.random.uniform(low=0, high=1, size=p)
        score_t1 = np.exp(X.dot(w_t1))
        nc_idx_t1 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t0]/score_t1[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t1]/score_t1[idx_t1].sum())
        ))
        A_t1[nc_idx_t1] = 0
    if nc_type == "one-sided":
        # nc for treatment assigned
        w_t1 = np.random.uniform(low=0, high=1, size=p)
        score_t1 = np.exp(X.dot(w_t1))
        nc_idx_t1 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t0]/score_t1[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t1]/score_t1[idx_t1].sum())
        ))
        A_t1[nc_idx_t1] = 0
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]


    # simulate outcome
    # w = np.random.uniform(low=-1, high=1, size=p)
    # Y_a0 = np.exp(X.dot(w)/np.sqrt(p))
    # w = np.random.uniform(low=-1, high=1, size=p)
    # Y_a1 = np.exp(X.dot(w)/np.sqrt(p))
    Y_a0 = (1 * (2 + 0.5 * np.sin(np.pi * X[:,0]) - 0.5 * X[:, 1] + 0.75 * X[:, 2] * X[:, 8]))
    Y_a1 = Y_a0 + 1 + 2 * np.abs(X[:,3]) + (X[:, 9]) ** 2 # + np.sin(np.pi * X[:,4]*X[:,5]) + 2 * (X[:, 6]-0.5)**2 + 0.5*X[:, 7]
    # Y_a0 = np.sin(np.pi * X[:,0]*X[:,1]) + 2 * (X[:, 2]-0.5)**2 + X[:, 3] + 0.5*X[:, 4]
    # Y_a1 = Y_a0 + np.abs(X[:,0]) + (X[:, 1]) ** 2
    Y_a = np.stack((Y_a0, Y_a1)).T

    # ground counter factuals
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

    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df


def generate_synthetic_nc_gap(n=1000, p=10, rep=0, nc_rate_gap=0,
                              nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n, check_valid="ignore")
    T = np.zeros(n, dtype=int)
    T = np.random.binomial(1, sigmoid(X[:, 2]))
    idx_t0, idx_t1 = np.where(T==0)[0], np.where(T==1)[0]

    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if nc_type == "two-sided":
        # nc for treatment not assigned
        w_t0 = np.random.uniform(low=0, high=1, size=p)
        score_t0 = np.exp(X.dot(w_t0))
        nc_idx_t0 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t0[idx_t0]/score_t0[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t0[idx_t1]/score_t0[idx_t1].sum())
        ))
        A_t0[nc_idx_t0] = 1
        # nc for treatment assigned
        w_t1 = np.random.uniform(low=0, high=1, size=p)
        score_t1 = np.exp(X.dot(w_t1))
        nc_idx_t1 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t0]/score_t1[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t1]/score_t1[idx_t1].sum())
        ))
        A_t1[nc_idx_t1] = 0
    if nc_type == "one-sided":
        # nc for treatment assigned
        w_t1 = np.random.uniform(low=0, high=1, size=p)
        score_t1 = np.exp(X.dot(w_t1))
        nc_idx_t1 = np.concatenate((
            np.random.choice(idx_t0, size=int(np.round(len(idx_t0)*(0.5+nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t0]/score_t1[idx_t0].sum()),
            np.random.choice(idx_t1, size=int(np.round(len(idx_t1)*(0.5-nc_rate_gap/2))), 
                             replace=False, p=score_t1[idx_t1]/score_t1[idx_t1].sum())
        ))
        A_t1[nc_idx_t1] = 0
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    Y_a0 = (5 * (2 + 0.5 * np.sin(np.pi * X[:,0]) - 0.5 * X[:, 1] + 0.75 * X[:, 2] * X[:, 8]))
    Y_a1 = Y_a0 + 1 + 2 * np.abs(X[:,3]) + (X[:, 9]) ** 2 
    Y_a = np.stack((Y_a0, Y_a1)).T

    # ground counter factuals
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

    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
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

def generate_latent_dim(X, dim, inter=True, inter_rate=0.1):
    n, p = X.shape
    if inter:
        pair_inters = np.array(list(itertools.combinations(range(p), 2)))
        sel_inters = pair_inters[np.random.choice(len(pair_inters), size=round(p*inter_rate), replace=False)]
        X_inters = []
        for i, j in sel_inters:
            X_inters.append(X[:, i]*X[:, j])
        X_inters = np.stack(X_inters).T
        X = np.hstack((X, X_inters))
    weights = np.random.uniform(low=-1, high=1, size=(X.shape[1], dim))
    return X.dot(weights)

def generate_response(X):
    weights = np.random.uniform(low=-1, high=1, size=(X.shape[1]))
    return X.dot(weights)

def generate_synthetic_nc_share(n=1000, p=50, rep=0, rho_shared=0.5, rho_separate=0.5, random_t=False,
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

    # ground counter factuals
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


def cos_sim(u, v):
    return u.dot(v) / (np.sqrt(np.square(u).sum())*np.sqrt(np.square(v).sum()))

def rand_cos_sim(v, costheta, amplitude=1):
    # Form the unit vector parallel to v:
    u = v / np.linalg.norm(v)
    # Pick a random vector:
    r = np.random.uniform(low=-amplitude, high=amplitude, size=len(v))
    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u
    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)
    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta*u + np.sqrt(1 - costheta**2)*uperp
    # scale to the same as input
    return w*np.linalg.norm(v)

def generate_synthetic_nc_angle(n=1000, p=30, rep=0, angle=0.5, amplitude=1,
                                nc_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    # X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)
    sigma = np.random.uniform(low=-1, high=1, size=(p,p))
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=(sigma+sigma.T)/2, size=n, check_valid="ignore")

    sigmoid = lambda x: 1/(1 + np.exp(-x))
    w_t = np.random.uniform(-amplitude, amplitude, size=p)
    p_t = sigmoid(X.dot(w_t))
    T = np.random.binomial(1, p_t)

    # simulate treatment intake
    if nc_type == "two-sided":
        w_t0 = rand_cos_sim(w_t, costheta=angle, amplitude=amplitude)
        A_t0 = np.random.binomial(1, sigmoid(X.dot(w_t0)))
        w_t1 = rand_cos_sim(w_t, costheta=angle, amplitude=amplitude)
        A_t1 = np.random.binomial(1, sigmoid(X.dot(w_t1)))
    if nc_type == "one-sided":
        A_t0 = np.zeros(n, dtype=int)
        w_t1 = rand_cos_sim(w_t, costheta=angle, amplitude=amplitude)
        A_t1 = np.random.binomial(1, sigmoid(X.dot(w_t1)))
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # simulate outcome
    w = np.random.uniform(low=-1, high=1, size=p) #* np.array(random.choices([0., 1], weights=[0.5, 0.5], k=p))
    Y_a0 = X.dot(w)
    w = np.random.uniform(low=-1, high=1, size=p) #* np.array(random.choices([0., 1], weights=[0.5, 0.5], k=p))
    Y_a1 = X.dot(w)
    Y_a = np.stack((Y_a0, Y_a1)).T

    # ground counter factuals
    Y_t0 = Y_a[range(n), A_t[:, 0]] + np.random.normal(scale=0.1, size=n)
    Y_t1 = Y_a[range(n), A_t[:, 1]] + np.random.normal(scale=0.1, size=n)
    Y_t = np.stack((Y_t0, Y_t1)).T
    Y = Y_t[range(n), T] # factual outcome
    Y_a = Y_a + np.random.normal(scale=0.1, size=(n, 2))

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
    if nc_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif nc_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df
