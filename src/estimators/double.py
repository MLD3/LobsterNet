import os, itertools
import numpy as np
import tensorflow as tf
from .utils import train_estimator
from .standard import make_linear, make_mlp, slearner_ce_loss, slearner_accuracy, slearner_ce_loss, \
    make_slearner, make_tlearner, tlearner_ce_loss, tlearner_ce_loss, tlearner_accuracy, \
    slearner_mse_loss, slearner_mse_loss, tlearner_mse_loss, tlearner_mse_loss, \
    make_tarnet, tlearner_inference



def train_dr_tlearner_grid_search(X_train, T_train, Y_train, checkpoint_dir, overwrite=False, po_act=None, 
        verbose=False, early_stop=False, epochs=100, batch_size_list=[64, 128, 256], reg_l2_list=[1e-2, 1e-3, 1e-4], 
        learning_rate_list=[1e-3, 1e-4]):
    print(f"Training propensity score model")
    def loss(a, b): return slearner_ce_loss(a, b)
    def treatment_loss(a, b): return slearner_ce_loss(a, b)
    def treatment_accuracy(a, b): return slearner_accuracy(a, b)

    rand_idx = np.arange(len(X_train))
    np.random.shuffle(rand_idx)
    idx_A, idx_B = rand_idx[:round(len(rand_idx)/2)], rand_idx[round(len(rand_idx)/2):]
    y_target_train = np.concatenate([Y_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    crossfit_dict = {
        "SplitA": {"input": X_train[idx_A], "t_target": T_train[idx_A], "y_target": y_target_train[idx_A]},
        "SplitB": {"input": X_train[idx_B], "t_target": T_train[idx_B], "y_target": y_target_train[idx_B]}
    }
    crossfit_t_model_dict = {}
    for split in crossfit_dict:
        input_train = crossfit_dict[split]["input"]
        target_train = crossfit_dict[split]["t_target"]
        best_t_loss = np.inf
        best_t_model = None
        best_t_metric_dict = None
        best_t_hparam = None
        for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
            if verbose:
                print(f"Train T model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
            t_model = make_linear(X_train.shape[1], reg_l2=reg_l2, activation='sigmoid') 
            curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}", split)
            os.makedirs(curr_checkpoint_dir, exist_ok=True)
            t_model, t_metric_dict = train_estimator(loss=loss, model=t_model, learning_rate=learning_rate,
                y_concat_train=np.float32(target_train), x_train=input_train, batch_size=batch_size, epochs=epochs,
                metrics=[treatment_loss, treatment_accuracy], model_name="t_model", early_stop=early_stop,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose)
            if t_metric_dict["val_loss"][-1] < best_t_loss:
                best_t_loss = t_metric_dict["val_loss"][-1]
                best_t_model = t_model
                best_t_metric_dict = t_metric_dict
                best_t_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
        del best_t_metric_dict["loss"], best_t_metric_dict["val_loss"]
        crossfit_t_model_dict[split] = best_t_model

    print(f"Training output model")    
    y_input_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_accuracy(a, b): return tlearner_accuracy(a, b)
        y_metrics=[outcome_loss, outcome_accuracy]
    else:
        def loss(a, b): return tlearner_mse_loss(a, b)
        def outcome_loss(a, b): return tlearner_mse_loss(a, b)
        y_metrics=[outcome_loss]
    crossfit_y_model_dict = {}
    for split in crossfit_dict:
        input_train = crossfit_dict[split]["input"]
        target_train = crossfit_dict[split]["y_target"]
        best_y_loss = np.inf
        best_y_model = None
        best_y_metric_dict = None
        best_y_hparam_dict = None
        for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
            if verbose:
                print(f"Train Y model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
            y_model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
            curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}", split)
            os.makedirs(curr_checkpoint_dir, exist_ok=True)
            y_model, y_metric_dict = train_estimator(loss=loss, model=y_model, model_name="y_model",
                y_concat_train=target_train, x_train=input_train, metrics=y_metrics, early_stop=early_stop,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
                learning_rate=learning_rate, epochs=epochs)
        if y_metric_dict["val_loss"][-1] < best_y_loss:
            best_y_loss = y_metric_dict["val_loss"][-1]
            best_y_model = y_model
            best_y_metric_dict = y_metric_dict
            best_y_hparam_dict = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
        crossfit_y_model_dict[split] = best_y_model
    
    print(f"Training theta model")
    pi_train = np.vstack((crossfit_t_model_dict["SplitB"].predict(X_train[idx_A]), crossfit_t_model_dict["SplitA"].predict(X_train[idx_B]))).reshape(-1)
    mu_train = np.vstack((tlearner_inference(crossfit_y_model_dict["SplitB"], X_train[idx_A]), tlearner_inference(crossfit_y_model_dict["SplitA"], X_train[idx_B])))
    T_train = np.concatenate((T_train[idx_A], T_train[idx_B]))
    Y_train = np.concatenate((Y_train[idx_A], Y_train[idx_B]))
    X_train = np.vstack((X_train[idx_A], X_train[idx_B]))
    theta_target_train = T_train * (Y_train-mu_train[:, 1]) / pi_train - (1-T_train) * (Y_train-mu_train[:, 0]) / (1-pi_train) + (mu_train[:, 1]-mu_train[:, 0])
    theta_target_train = np.clip(theta_target_train, -1, 1)

    if po_act == "sigmoid":
        activation = lambda x: 2 * tf.nn.sigmoid(x) - 1
    else:
        activation = None
    def loss(a, b): return slearner_mse_loss(a, b)
    def effect_loss(a, b): return slearner_mse_loss(a, b)
    best_theta_loss = np.inf
    best_theta_model = None
    best_theta_metric_dict = None
    best_theta_hparam = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train Theta model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        theta_model = make_mlp(X_train.shape[1], reg_l2=reg_l2, activation=None) 
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}", "All")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        theta_model, theta_metric_dict = train_estimator(loss=loss, model=theta_model, learning_rate=learning_rate,
            y_concat_train=np.float32(theta_target_train), x_train=X_train, batch_size=batch_size, epochs=epochs,
            metrics=[effect_loss], model_name="theta_model", early_stop=early_stop,
            checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose)
        if theta_metric_dict["val_loss"][-1] < best_theta_loss:
            best_theta_loss = theta_metric_dict["val_loss"][-1]
            best_theta_model = theta_model
            best_theta_metric_dict = theta_metric_dict
            best_theta_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
    
    return best_theta_model, best_theta_metric_dict, best_theta_hparam