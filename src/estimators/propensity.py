import itertools, os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Concatenate
from .utils import train_estimator



def make_dragonnet(input_dim, reg_l2, activation=None):
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

    t_predictions = Dense(units=1, activation='sigmoid')(x)

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

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

def dragonnet_treatment_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    losst = K.binary_crossentropy(t_true, t_pred)
    return losst

def dragonnet_mse_outcome_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    loss = tf.square(y_true-y_pred)
    return loss

def dragonnet_ce_outcome_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    loss = K.binary_crossentropy(y_true, y_pred)
    return loss

def dragonnet_outcome_accuracy(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    return binary_accuracy(y_true, y_pred)
    

def dragonnet_mse_loss(concat_true, concat_pred, alpha=1.0):
    return tf.reduce_mean(dragonnet_mse_outcome_loss(concat_true, concat_pred) + \
        alpha*dragonnet_treatment_loss(concat_true, concat_pred))

def dragonnet_ce_loss(concat_true, concat_pred, alpha=1.0):
    return tf.reduce_mean(dragonnet_ce_outcome_loss(concat_true, concat_pred) + \
        alpha*dragonnet_treatment_loss(concat_true, concat_pred))

def dragonnet_outcome_accuracy(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pred = (1-t_true) * y0_pred + t_true * y1_pred
    return binary_accuracy(y_true, y_pred)

def dragonnet_treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)

def dragon_inference(model, X, batch_size=32):
    yt_hat = model.predict(X, batch_size=batch_size)
    y_t0_pred = yt_hat[:, 0].copy()
    y_t1_pred = yt_hat[:, 1].copy()
    t_pred = yt_hat[:, 2].copy()
    return np.stack((y_t0_pred, y_t1_pred)).T, t_pred


def train_dragon(X_train, T_train, Y_train, checkpoint_dir, overwrite=False, po_act=None, 
    batch_size=64, verbose=False, early_stop=False, reg_l2=0.01, learning_rate=1e-3, epochs=100):
    model = make_dragonnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
    target_train = np.concatenate([Y_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return dragonnet_ce_loss(a, b, alpha=1.0)
        def outcome_loss(a, b): return tf.reduce_mean(dragonnet_ce_outcome_loss(a, b))
        def outcome_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        def treatment_loss(a, b): return tf.reduce_mean(dragonnet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy, treatment_loss, treatment_accuracy]
    else:
        def loss(a, b): return dragonnet_mse_loss(a, b, alpha=1.0)
        def outcome_loss(a, b): return tf.reduce_mean(dragonnet_mse_outcome_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(dragonnet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        metrics=[outcome_loss, treatment_loss, treatment_accuracy]
    model, metric_dict = train_estimator(loss=loss, model=model, model_name="model",
            y_concat_train=target_train, x_train=input_train, metrics=metrics, batch_size=batch_size,
            checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose, early_stop=early_stop, 
            learning_rate=learning_rate, epochs=epochs)
    return model, metric_dict

def train_dragon_grid_search(X_train, T_train, Y_train, checkpoint_dir, overwrite=False, po_act=None, verbose=False,
        epochs=300, early_stop=True, batch_size_list=[64, 128, 256], reg_l2_list=[1e-2, 1e-3, 1e-4], 
        learning_rate_list=[1e-3, 1e-4]):
    target_train = np.concatenate([Y_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return dragonnet_ce_loss(a, b, alpha=1.0)
        def outcome_loss(a, b): return tf.reduce_mean(dragonnet_ce_outcome_loss(a, b))
        def outcome_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        def treatment_loss(a, b): return tf.reduce_mean(dragonnet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy, treatment_loss, treatment_accuracy]
    else:
        def loss(a, b): return dragonnet_mse_loss(a, b, alpha=1.0)
        def outcome_loss(a, b): return tf.reduce_mean(dragonnet_mse_outcome_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(dragonnet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return dragonnet_treatment_accuracy(a, b)
        metrics=[outcome_loss, treatment_loss, treatment_accuracy]
    
    best_loss = np.inf
    best_model = None
    best_metric_dict = None
    best_hparam_dict = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        model = make_dragonnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        model, metric_dict = train_estimator(loss=loss, model=model, model_name="model",
                y_concat_train=target_train, x_train=input_train, metrics=metrics, batch_size=batch_size,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=False, early_stop=early_stop, 
                learning_rate=learning_rate, epochs=epochs)
        if metric_dict["val_loss"][-1] < best_loss:
            best_loss = metric_dict["val_loss"][-1]
            best_model = model
            best_metric_dict = metric_dict
            best_hparam_dict = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}

    return best_model, best_metric_dict, best_hparam_dict
    