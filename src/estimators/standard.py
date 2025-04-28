import itertools, os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.metrics import binary_accuracy
from .utils import train_estimator


def make_linear(input_dim, reg_l2, activation=None):
    inputs = Input(shape=(input_dim,), name='input')
    y_pred = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    model = Model(inputs=inputs, outputs=y_pred)
    return model

def make_mlp(input_dim, reg_l2, activation=None):
    inputs = Input(shape=(input_dim,), name='input')
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y_pred = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(x)
    model = Model(inputs=inputs, outputs=y_pred)
    return model

def mlp_inference(model, X, batch_size=32):
    return model.predict(X, batch_size=batch_size)[:, 0]

def make_slearner(input_dim, reg_l2, activation=None, num_t=1):
    inputs = Input(shape=(input_dim+num_t,), name='input')
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y_pred = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(x)
    model = Model(inputs=inputs, outputs=y_pred)
    return model

def slearner_mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def slearner_ce_loss(y_true, y_pred):
    return tf.reduce_mean(K.binary_crossentropy(y_true, y_pred))

def slearner_accuracy(t_true, t_pred):
    return binary_accuracy(t_true, t_pred)


def slearner_inference(model, X, num_t=1, batch_size=32):
    y_pred_list = []
    for ts in itertools.product([0, 1], repeat=num_t):
        X_curr = [X]
        for t in ts: X_curr.append(np.ones((X.shape[0], 1))*t)
        X_curr = np.hstack(X_curr)
        y_pred_curr = model.predict(X_curr, batch_size=batch_size)[:, 0]
        y_pred_list.append(y_pred_curr)
    return np.stack(y_pred_list).T

def make_tlearner(input_dim, reg_l2, activation=None):
    inputs = Input(shape=(input_dim,), name='input')
    x0 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    x0 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x0)
    x0 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x0)
    y0_pred = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(x0)

    x1 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    x1 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x1)
    x1 = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x1)
    y1_pred = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(x1)

    concat_pred = Concatenate(1)([y0_pred, y1_pred])
    model = Model(inputs=inputs, outputs=concat_pred)
    return model

def tlearner_mse_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    return tf.reduce_mean(tf.square(y_true-y_pred))

def tlearner_ce_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    return tf.reduce_mean(K.binary_crossentropy(y_true, y_pred))

def tlearner_accuracy(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    return binary_accuracy(y_true, y_pred)

def tlearner_inference(model, X, batch_size=32):
    yt_hat = model.predict(X, batch_size=batch_size)
    y_t0_pred = yt_hat[:, 0]
    y_t1_pred = yt_hat[:, 1]
    return np.stack((y_t0_pred, y_t1_pred)).T

def make_tarnet(input_dim, reg_l2, activation=None):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(inputs)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=300, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_predictions = Dense(units=1, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model
    
def train_tlearner(X_train, T_train, Y_train, checkpoint_dir, overwrite=False, po_act=None, 
                   batch_size=64, verbose=False, early_stop=False, reg_l2=0.01, learning_rate=1e-3, epochs=100):
    model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
    target_train = np.concatenate([Y_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_accuracy(a, b): return tlearner_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy]
    else:
        def loss(a, b): return tlearner_mse_loss(a, b)
        def outcome_loss(a, b): return tlearner_mse_loss(a, b)
        metrics=[outcome_loss]
    model, metric_dict = train_estimator(loss=loss, model=model, model_name="model",
        y_concat_train=target_train, x_train=input_train, metrics=metrics, early_stop=early_stop,
        checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
        learning_rate=learning_rate, epochs=epochs)
    return model, metric_dict

def train_tlearner_grid_search(X_train, T_train, Y_train, checkpoint_dir, overwrite=False, po_act=None, 
        verbose=False, early_stop=False, epochs=100, batch_size_list=[64, 128, 256], reg_l2_list=[1e-2, 1e-3, 1e-4], 
        learning_rate_list=[1e-3, 1e-4]):
    target_train = np.concatenate([Y_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_accuracy(a, b): return tlearner_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy]
    else:
        def loss(a, b): return tlearner_mse_loss(a, b)
        def outcome_loss(a, b): return tlearner_mse_loss(a, b)
        metrics=[outcome_loss]
    best_loss = np.inf
    best_model = None
    best_metric_dict = None
    best_hparam_dict = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train T-learner with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        model, metric_dict = train_estimator(loss=loss, model=model, model_name="model",
            y_concat_train=target_train, x_train=input_train, metrics=metrics, early_stop=early_stop,
            checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
            learning_rate=learning_rate, epochs=epochs)
    if metric_dict["val_loss"][-1] < best_loss:
        best_loss = metric_dict["val_loss"][-1]
        best_model = model
        best_metric_dict = metric_dict
        best_hparam_dict = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
    
    return best_model, best_metric_dict, best_hparam_dict
