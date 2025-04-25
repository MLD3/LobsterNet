import pyreadr, random
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

def load_pre_computed_ihdp(data_path, rep=0):
    data_npz = np.load(data_path)
    X, T = data_npz.f.x[:, :, rep].copy(), data_npz.f.t[:, rep].copy()
    YF, YCF = data_npz.f.yf[:, rep].copy(), data_npz.f.ycf[:, rep].copy()
    mu_0, mu_1 = data_npz.f.mu0[:, rep].copy(), data_npz.f.mu1[:, rep].copy()
    return X, T, YF, mu_0, mu_1

def generate_random_binary(data, get_proba=False, amplitude=10, num_zeros=5):
    n, p = data.shape
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    rand_weight = np.random.uniform(low=-amplitude, high=amplitude, size=p)
    proba = sigmoid(data.dot(rand_weight)/np.sqrt(p)+ np.random.normal(scale=amplitude/10, size=n))
    if get_proba: return proba
    else: return np.random.binomial(n=1, p=proba)


def generate_ihdp_nc_old(rdata_path, rep=0, amplitude=10, num_unobserved=3, verbose=True,
    non_compliance_type:Literal["none", 'one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    # load ihdp features
    ihdp_df = pyreadr.read_r(rdata_path)["ihdp"]
    # standardize all continuous covariates:
    for col in ihdp_df.columns:
        if not ihdp_df[col].isin([0, 1]).all():
            ihdp_df[col] = (ihdp_df[col] - ihdp_df[col].mean()) / ihdp_df[col].std()

    # randomly select observed and unobserved covariates
    feat_names = ['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
               'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 
               'cig', 'first', 'booze', 'drugs', 'work.dur', 'prenatal', 
               'ark', 'ein', 'har', 'mia', 'pen', 'tex', 'was', 'momwhite', 'momblack', 'momhisp']
    np.random.shuffle(feat_names)
    assert num_unobserved < len(feat_names)
    # non_white_treat_idx = ihdp_df[(ihdp_df["momwhite"]==0)&(ihdp_df["treat"]==1)].index
    # ihdp_df.drop(index=non_white_treat_idx, inplace=True)
    # T = ihdp_df["treat"].values
    X = ihdp_df[feat_names[num_unobserved:]].values
    Z = ihdp_df[feat_names].values
    n, p_x, p_z = len(ihdp_df), X.shape[1], Z.shape[1]
    if verbose:
        print(f"confounders: #all={p_z}, #observed={p_x}")

    # simulate treatment assignment
    T = generate_random_binary(Z, amplitude=amplitude).astype(int)
    n, num_treat, num_control = len(T), sum(T==1), sum(T==0)
    if verbose:
        print(f"#total={n}, #treat={num_treat}, #control={num_control}")

    # simulate action
    if non_compliance_type == "none":
        A_t0 = np.zeros(n).astype(int)
        A_t1 = np.ones(n).astype(int)
    elif non_compliance_type == "one-sided":
        A_t0 = np.zeros(n).astype(int)
        A_t1 = generate_random_binary(X, amplitude=amplitude).astype(int)
    elif non_compliance_type == "two-sided":
        A_t0 = generate_random_binary(X, amplitude=amplitude).astype(int)
        A_t1 = generate_random_binary(X, amplitude=amplitude).astype(int)
    else:
        raise Exception(f"{non_compliance_type} non-compliance is not defined")
    A_t = np.stack((A_t0, A_t1)).T
    num_comply_t0, num_comply_t1 = sum(A_t0==0), sum(A_t1==1)
    if verbose:
        print(f"#total={n}, #comply_t0={num_comply_t0}, #comply_t1={num_comply_t1}")

    # simulate potential outcome
    weight_0 = np.random.uniform(low=-10, high=amplitude, size=p_z)
    weight_1 = np.random.uniform(low=-10, high=amplitude, size=p_z)
    y_a0 = Z.dot(weight_0)/np.sqrt(p_z)
    y_a1 = Z.dot(weight_1)/np.sqrt(p_z)
    Y_a = np.stack((y_a0, y_a1)).T

    # ground truth outcome
    Y_t0 = Y_a[range(n), A_t0].copy() + np.random.normal(loc=0, scale=amplitude/10, size=n)
    Y_t1 = Y_a[range(n), A_t1].copy() + np.random.normal(loc=0, scale=amplitude/10, size=n)
    Y_t = np.stack((Y_t0, Y_t1)).T
    # observed compliance and outcome
    A_f = A_t[range(n), T].copy()
    Y_f = Y_t[range(n), T].copy()

    # organize data into df
    data_arr = np.hstack((X, np.vstack((T, A_f, Y_f)).T, Y_t))
    columns = [f"X{i+1}" for i in range(X.shape[1])] + ["T", "A", "Y", "Y0", "Y1"]
    data_df = pd.DataFrame(data=data_arr, columns=columns)
    return data_df

def cos_sim(u, v):
    return u.dot(v) / (np.sqrt(np.square(u).sum())*np.sqrt(np.square(v).sum()))

def rand_cos_sim(v, costheta):
    # Form the unit vector parallel to v:
    u = v / np.linalg.norm(v)
    # Pick a random vector:
    r = np.random.uniform(low=-10, high=10, size=len(v))
    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u
    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)
    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta*u + np.sqrt(1 - costheta**2)*uperp
    # scale to the same as input
    return w*np.linalg.norm(v)

def sigmoid(x): return 1/(1 + np.exp(-x))
def sigmoid_inv(y): return np.log(y) - np.log(1 - y)#return np.log(y / (1 - y))

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


def generate_ihdp_sim_nc(rep=0, t_rate=1.0, nc_rate=0.5,
                         non_compliance_type:Literal['one-sided', 'two-sided']="two-sided"):
    # set random seed
    random.seed(rep)
    np.random.seed(rep)

    Z = np.random.uniform(low=-1, high=1, size=(1000,30))
    T = np.random.binomial(1, 0.5, size=1000)
    w = np.random.uniform(0, 1, size=30)
    score = np.exp(Z[T==1, :].dot(w))
    p = score / score.sum()
    keep_idx = np.concatenate((np.random.choice(np.where(T==1)[0], size=round(T.sum()*t_rate), replace=False, p=p), np.where(T==0)[0]))
    Z, X, T = Z[keep_idx, :], Z[keep_idx, :25], T[keep_idx]
    n, p_x, p_z = len(X), X.shape[1], Z.shape[1]

    # simulate treatment intake
    A_t0 = np.zeros(n, dtype=int)
    A_t1 = np.ones(n, dtype=int)
    if non_compliance_type == "two-sided":
        w_t0 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.2, 0.15, 0.1, 0.05], k=p_x))
        score_t0 = np.exp(X.dot(w_t0))
        p_t0 = score_t0 / score_t0.sum()
        idx_t0 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t0)
        A_t0[idx_t0] = 1
        w_t1 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.2, 0.15, 0.1, 0.05], k=p_x))
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    if non_compliance_type == "one-sided":
        w_t1 = np.array(random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.2, 0.15, 0.1, 0.05], k=p_x))
        score_t1 = np.exp(X.dot(w_t1))
        p_t1 = score_t1 / score_t1.sum()
        idx_t1 = np.random.choice(n, size=int(np.round(n*nc_rate)), replace=False, p=p_t1)
        A_t1[idx_t1] = 0
    A_t = np.stack((A_t0, A_t1)).T
    A = A_t[range(n), T]

    # simulate outcome
    w = np.array(random.choices([0.0, 0.1, 0.2, 0.3, 0.4], weights=[0.6, 0.1, 0.1, 0.1, 0.1], k=p_z))
    Y_a0 = np.exp((Z+0.5).dot(w))
    w = np.array(random.choices([0.0, 0.1, 0.2, 0.3, 0.4], weights=[0.6, 0.1, 0.1, 0.1, 0.1], k=p_z))
    Y_a1 = np.exp((Z+0.5).dot(w))
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
    data_df = pd.DataFrame(data=data_arr, columns=columns)

    train_idx, test_idx = train_test_split(
        list(range(len(data_df))), test_size=0.2, train_size=0.8, 
        random_state=rep, stratify=data_df[["T", "A"]])
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    if non_compliance_type == "two-sided": assert train_df[["T", "A"]].value_counts().shape[0]==4
    elif non_compliance_type == "one-sided": assert train_df[["T", "A"]].value_counts().shape[0]==3
    return train_df, test_df

