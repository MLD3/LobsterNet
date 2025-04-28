import pickle, os, time, itertools
from typing import Literal
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Concatenate
from .utils import train_estimator
from .standard import make_mlp, slearner_ce_loss, slearner_accuracy, slearner_ce_loss, \
    make_tlearner, tlearner_ce_loss, tlearner_ce_loss, tlearner_accuracy, \
    tlearner_mse_loss, tlearner_mse_loss, make_tarnet


def make_lobsternet(input_dim, reg_l2, activation=None, backbone_dim=300, head_dim=100, 
                    shared_backbone=True, shared_outcome=True):
    """
    Neural net predictive model. The lobster has two antennas, two claws, and one head.
    :param input_dim:
    :param reg:
    :return:
    """
    inputs = Input(shape=(input_dim,), name='input')

    if shared_backbone:
        # shared backbone
        x = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="shared_backbone1")(inputs)
        x = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="shared_backbone2")(x)
        x = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="shared_backbone3")(x)
        x_t = x_t0 = x_t1 = x
    else:
        # treatment assignment backbone
        x_t = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="t_backbone1")(inputs)
        x_t = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="t_backbone2")(x_t)
        x_t = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="t_backbone3")(x_t)
        # t0 backbone
        x_t0 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_backbone1")(inputs)
        x_t0 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_backbone2")(x_t0)
        x_t0 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_backbone3")(x_t0)
        # t1 backbone
        x_t1 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_backbone1")(inputs)
        x_t1 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_backbone2")(x_t1)
        x_t1 = Dense(units=backbone_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_backbone3")(x_t1)

    # treatment assignment prediction
    t_pred = Dense(units=1, activation='sigmoid', name="t_predict")(x_t)

    # treatment intake prediction
    a_t0_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_hidden1")(x_t0)
    a_t1_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_hidden1")(x_t1)
    a_t0_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_hidden2")(a_t0_hidden1)
    a_t1_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_hidden2")(a_t1_hidden1)
    a_t0_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t0_hidden3")(a_t0_hidden2)
    a_t1_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="a_t1_hidden3")(a_t1_hidden2)
    a_t0_pred = Dense(units=1, activation="sigmoid", name="a_t0_predict")(a_t0_hidden3)
    a_t1_pred = Dense(units=1, activation="sigmoid", name="a_t1_predict")(a_t1_hidden3)

    # Predicting outcome
    if shared_outcome:
        # first layer
        dense_a0_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_hidden1")
        dense_a1_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_hidden1")
        # second layer
        dense_a0_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_hidden2")
        dense_a1_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_hidden2")
        # third layer
        dense_a0_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_hidden3")
        dense_a1_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_hidden3")
        # output layer
        dense_a0_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a0_predict")
        dense_a1_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a1_predict")

        y_a0_t0_pred = dense_a0_output(dense_a0_hidden3(dense_a0_hidden2(dense_a0_hidden1(a_t0_hidden3))))
        y_a1_t0_pred = dense_a1_output(dense_a1_hidden3(dense_a1_hidden2(dense_a1_hidden1(a_t0_hidden3))))
        y_a0_t1_pred = dense_a0_output(dense_a0_hidden3(dense_a0_hidden2(dense_a0_hidden1(a_t1_hidden3))))
        y_a1_t1_pred = dense_a1_output(dense_a1_hidden3(dense_a1_hidden2(dense_a1_hidden1(a_t1_hidden3))))
    else:
        # first layer
        dense_a0_t0_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t0_hidden1")
        dense_a1_t0_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t0_hidden1")
        dense_a0_t1_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t1_hidden1")
        dense_a1_t1_hidden1 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t1_hidden1")
        # second layer
        dense_a0_t0_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t0_hidden2")
        dense_a1_t0_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t0_hidden2")
        dense_a0_t1_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t1_hidden2")
        dense_a1_t1_hidden2 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t1_hidden2")
        # third layer
        dense_a0_t0_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t0_hidden3")
        dense_a1_t0_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t0_hidden3")
        dense_a0_t1_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a0_t1_hidden3")
        dense_a1_t1_hidden3 = Dense(units=head_dim, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name="y_a1_t1_hidden3")
        # output layer
        dense_a0_t0_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a0_t0_predict")
        dense_a1_t0_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a1_t0_predict")
        dense_a0_t1_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a0_t1_predict")
        dense_a1_t1_output = Dense(units=1, kernel_regularizer=regularizers.l2(reg_l2), activation=activation, name="y_a1_t1_predict")
    
        y_a0_t0_pred = dense_a0_t0_output(dense_a0_t0_hidden3(dense_a0_t0_hidden2(dense_a0_t0_hidden1(a_t0_hidden3))))
        y_a1_t0_pred = dense_a1_t0_output(dense_a1_t0_hidden3(dense_a1_t0_hidden2(dense_a1_t0_hidden1(a_t0_hidden3))))
        y_a0_t1_pred = dense_a0_t1_output(dense_a0_t1_hidden3(dense_a0_t1_hidden2(dense_a0_t1_hidden1(a_t1_hidden3))))
        y_a1_t1_pred = dense_a1_t1_output(dense_a1_t1_hidden3(dense_a1_t1_hidden2(dense_a1_t1_hidden1(a_t1_hidden3))))

    concat_pred = Concatenate(1)([y_a0_t0_pred, y_a1_t0_pred, y_a0_t1_pred, y_a1_t1_pred, a_t0_pred, a_t1_pred, t_pred])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

def lobsternet_outcome_mse_loss(concat_true, concat_pred):
    Y_true, A, T = concat_true[:, 0], concat_true[:, 1], concat_true[:, 2]
    Y_a0_t0_pred, Y_a1_t0_pred = concat_pred[:, 0], concat_pred[:, 1] 
    Y_a0_t1_pred, Y_a1_t1_pred = concat_pred[:, 2], concat_pred[:, 3]

    Y_hat = Y_a0_t0_pred * (1-A) * (1-T) + \
            Y_a1_t0_pred * (A) * (1-T) + \
            Y_a0_t1_pred * (1-A) * (T) + \
            Y_a1_t1_pred * (A) * (T)
    loss = tf.square(Y_true-Y_hat)
    return loss

def lobsternet_outcome_ce_loss(concat_true, concat_pred):
    Y_true, A, T = concat_true[:, 0], concat_true[:, 1], concat_true[:, 2]
    Y_a0_t0_pred, Y_a1_t0_pred = concat_pred[:, 0], concat_pred[:, 1] 
    Y_a0_t1_pred, Y_a1_t1_pred = concat_pred[:, 2], concat_pred[:, 3]

    Y_hat = Y_a0_t0_pred * (1-A) * (1-T) + \
            Y_a1_t0_pred * (A) * (1-T) + \
            Y_a0_t1_pred * (1-A) * (T) + \
            Y_a1_t1_pred * (A) * (T)
    loss = K.binary_crossentropy(Y_true, Y_hat)
    return loss

def lobsternet_outcome_accuracy(concat_true, concat_pred):
    Y_true, A, T = concat_true[:, 0], concat_true[:, 1], concat_true[:, 2]
    Y_a0_t0_pred, Y_a1_t0_pred = concat_pred[:, 0], concat_pred[:, 1] 
    Y_a0_t1_pred, Y_a1_t1_pred = concat_pred[:, 2], concat_pred[:, 3]

    Y_hat = Y_a0_t0_pred * (1-A) * (1-T) + \
            Y_a1_t0_pred * (A) * (1-T) + \
            Y_a0_t1_pred * (1-A) * (T) + \
            Y_a1_t1_pred * (A) * (T)
    return binary_accuracy(Y_true, Y_hat)

def lobsternet_treatment_loss(concat_true, concat_pred):
    T = concat_true[:, 2]
    t_pred = concat_pred[:, 6]
    loss = K.binary_crossentropy(T, t_pred)
    return loss

def lobsternet_compliance_loss(concat_true, concat_pred):
    A, T = concat_true[:, 1], concat_true[:, 2]
    a_t0_pred, a_t1_pred = concat_pred[:, 4], concat_pred[:, 5]
    a_pred = a_t0_pred * (1-T) + a_t1_pred * (T)
    loss = K.binary_crossentropy(A, a_pred)
    return loss

def lobsternet_mse_loss(concat_true, concat_pred, alpha=1, beta=1):
    # - concat_true: [y, A, T]
    # - concat_pred: [y_a0_t0_pred, y_a1_t0_pred, y_a0_t1_pred, y_a1_t1_pred, a_t0_pred, a_t1_pred, t_pred]
    return tf.reduce_mean(
        lobsternet_outcome_mse_loss(concat_true, concat_pred) + \
        alpha*lobsternet_compliance_loss(concat_true, concat_pred) + \
        beta*lobsternet_treatment_loss(concat_true, concat_pred))

def lobsternet_ce_loss(concat_true, concat_pred, alpha=1, beta=1):
    # - concat_true: [y, A, T]
    # - concat_pred: [y_a0_t0_pred, y_a1_t0_pred, y_a0_t1_pred, y_a1_t1_pred, a_t0_pred, a_t1_pred, t_pred]
    return tf.reduce_mean(
        lobsternet_outcome_ce_loss(concat_true, concat_pred) + \
        alpha*lobsternet_compliance_loss(concat_true, concat_pred) + \
        beta*lobsternet_treatment_loss(concat_true, concat_pred))

def lobsternet_treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 2]
    t_pred = concat_pred[:, 6]
    return binary_accuracy(t_true, t_pred)

def lobsternet_compliance_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 2]
    a_true = concat_true[:, 1]
    a_t0_pred = concat_pred[:, 4] * (1-t_true)
    a_t1_pred = concat_pred[:, 5] * t_true
    a_pred = a_t0_pred + a_t1_pred
    return binary_accuracy(a_true, a_pred)

def lobster_inference(model, X, batch_size=32):
    yat_hat = model.predict(X, batch_size=batch_size)
    y_pred = yat_hat[:, :4].copy()
    a_pred = yat_hat[:, 4:6].copy()
    t_pred = yat_hat[:, 6].copy()
    return t_pred, a_pred, y_pred

def train_lobster(X_train, A_train, T_train, Y_train, checkpoint_dir, overwrite=False, backbone_dim=300, head_dim=100, 
                  po_act=None, alpha=1.0, beta=1.0, shared_backbone=True, shared_outcome=True,
                  batch_size=64, verbose=False, early_stop=False, reg_l2=0.01, learning_rate=1e-3, epochs=100):
    model = make_lobsternet(X_train.shape[1], reg_l2=reg_l2, activation=po_act, backbone_dim=backbone_dim, head_dim=head_dim, 
                            shared_backbone=shared_backbone, shared_outcome=shared_outcome)
    y_concat_train = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    x_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return lobsternet_ce_loss(a, b, alpha=alpha, beta=beta)
        def outcome_loss(a, b): return tf.reduce_mean(lobsternet_outcome_ce_loss(a, b))
        def outcome_accuracy(a, b): return lobsternet_outcome_accuracy(a, b)
        def compliance_loss(a, b): return tf.reduce_mean(lobsternet_compliance_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(lobsternet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return lobsternet_treatment_accuracy(a, b)
        def compliance_accuracy(a, b): return lobsternet_compliance_accuracy(a, b)
        metrics = [outcome_loss, outcome_accuracy, compliance_loss, treatment_loss, treatment_accuracy, compliance_accuracy]
    else:
        def loss(a, b): return lobsternet_mse_loss(a, b, alpha=alpha, beta=beta)
        def outcome_loss(a, b): return tf.reduce_mean(lobsternet_outcome_mse_loss(a, b))
        def compliance_loss(a, b): return tf.reduce_mean(lobsternet_compliance_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(lobsternet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return lobsternet_treatment_accuracy(a, b)
        def compliance_accuracy(a, b): return lobsternet_compliance_accuracy(a, b)
        metrics = [outcome_loss, compliance_loss, treatment_loss, treatment_accuracy, compliance_accuracy]
    model, metric_dict = train_estimator(loss=loss, model=model, batch_size=batch_size, learning_rate=learning_rate,
            y_concat_train=y_concat_train, x_train=x_train, metrics=metrics, early_stop=early_stop, epochs=epochs,
            checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose)
    return model, metric_dict

def train_lobster_grid_search(X_train, A_train, T_train, Y_train, checkpoint_dir, overwrite=False, 
    backbone_dim=300, head_dim=100, po_act=None, alpha=1.0, beta=1.0, shared_backbone=True, shared_outcome=True,
    verbose=False, early_stop=False, epochs=100, batch_size_list=[64, 128, 256], reg_l2_list=[1e-2, 1e-3, 1e-4], 
    learning_rate_list=[1e-3, 1e-4]):
    y_concat_train = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    x_train = X_train
    if po_act == "sigmoid":
        def loss(a, b): return lobsternet_ce_loss(a, b, alpha=alpha, beta=beta)
        def outcome_loss(a, b): return tf.reduce_mean(lobsternet_outcome_ce_loss(a, b))
        def outcome_accuracy(a, b): return lobsternet_outcome_accuracy(a, b)
        def compliance_loss(a, b): return tf.reduce_mean(lobsternet_compliance_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(lobsternet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return lobsternet_treatment_accuracy(a, b)
        def compliance_accuracy(a, b): return lobsternet_compliance_accuracy(a, b)
        metrics = [outcome_loss, outcome_accuracy, compliance_loss, treatment_loss, treatment_accuracy, compliance_accuracy]
    else:
        def loss(a, b): return lobsternet_mse_loss(a, b, alpha=alpha, beta=beta)
        def outcome_loss(a, b): return tf.reduce_mean(lobsternet_outcome_mse_loss(a, b))
        def compliance_loss(a, b): return tf.reduce_mean(lobsternet_compliance_loss(a, b))
        def treatment_loss(a, b): return tf.reduce_mean(lobsternet_treatment_loss(a, b))
        def treatment_accuracy(a, b): return lobsternet_treatment_accuracy(a, b)
        def compliance_accuracy(a, b): return lobsternet_compliance_accuracy(a, b)
        metrics = [outcome_loss, compliance_loss, treatment_loss, treatment_accuracy, compliance_accuracy]
    
    best_loss = np.inf
    best_model = None
    best_metric_dict = None
    best_hparam_dict = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        model = make_lobsternet(X_train.shape[1], reg_l2=reg_l2, activation=po_act, backbone_dim=backbone_dim, head_dim=head_dim, 
                                shared_backbone=shared_backbone, shared_outcome=shared_outcome)
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        model, metric_dict = train_estimator(loss=loss, model=model, batch_size=batch_size, learning_rate=learning_rate,
                y_concat_train=y_concat_train, x_train=x_train, metrics=metrics, early_stop=early_stop, epochs=epochs,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose)
        if metric_dict["val_loss"][-1] < best_loss:
            best_loss = metric_dict["val_loss"][-1]
            best_model = model
            best_metric_dict = metric_dict
            best_hparam_dict = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
    return best_model, best_metric_dict, best_hparam_dict

def train_cfd_tlearner(X_train, A_train, T_train, Y_train, checkpoint_dir, 
              shared_backbone=False, overwrite=False, po_act=None, batch_size=64, reg_l2=0.01, 
              verbose=False, early_stop=False, learning_rate=1e-3, epochs=100):
    print(f"Training propensity score model")
    model_t = make_mlp(X_train.shape[1], reg_l2=reg_l2, activation='sigmoid') 
    def loss(a, b): return slearner_ce_loss(a, b)
    def treatment_loss(a, b): return slearner_ce_loss(a, b)
    def treatment_accuracy(a, b): return slearner_accuracy(a, b)
    model_t, model_t_metric_dict = train_estimator(loss=loss, model=model_t, learning_rate=learning_rate,
        y_concat_train=np.float32(T_train), x_train=X_train, batch_size=batch_size, epochs=epochs,
        metrics=[treatment_loss, treatment_accuracy], model_name="model_t", early_stop=early_stop,
        checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose)
    del model_t_metric_dict["loss"], model_t_metric_dict["val_loss"]

    print(f"Training compliance model")
    if shared_backbone: model_a = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation="sigmoid")
    else: model_a = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation='sigmoid')
    target_train = np.concatenate([A_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    def loss(a, b): return tlearner_ce_loss(a, b)
    def compliance_loss(a, b): return tlearner_ce_loss(a, b)
    def compliance_accuracy(a, b): return tlearner_accuracy(a, b)
    metrics=[compliance_loss, compliance_accuracy]
    model_a, model_a_metric_dict = train_estimator(loss=loss, model=model_a, model_name="model_a", early_stop=early_stop,
        y_concat_train=np.float32(target_train), x_train=input_train, metrics=metrics, learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, epochs=epochs)
    del model_a_metric_dict["loss"], model_a_metric_dict["val_loss"]
        
    print(f"Training potential outcome model")
    if shared_backbone:
        model_y_t0 = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        model_y_t1 = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
    else:
        model_y_t0 = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        model_y_t1 = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        
    target_train_t0 = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1)], 1)[T_train==0]
    target_train_t1 = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1)], 1)[T_train==1]
    input_train_t0 = X_train[T_train==0]
    input_train_t1 = X_train[T_train==1]
    if po_act == "sigmoid":
        def loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_accuracy(a, b): return tlearner_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy]
    else:
        def loss(a, b): return tlearner_mse_loss(a, b)
        def outcome_loss(a, b): return tlearner_mse_loss(a, b)
        metrics=[outcome_loss]
    
    model_y_t0, model_y_t0_metric_dict = train_estimator(loss=loss, model=model_y_t0, model_name="model_y_t0",
            y_concat_train=target_train_t0, x_train=input_train_t0, metrics=metrics, early_stop=early_stop,
            checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
            learning_rate=learning_rate, epochs=epochs)
    model_y_t1, model_y_t1_metric_dict = train_estimator(loss=loss, model=model_y_t1, model_name="model_y_t1",
            y_concat_train=target_train_t1, x_train=input_train_t1, metrics=metrics, early_stop=early_stop,
            checkpoint_dir=checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
            learning_rate=learning_rate, epochs=epochs)
    del model_y_t0_metric_dict["loss"], model_y_t0_metric_dict["val_loss"]
    model_y_t0_metric_dict["t0_outcome_loss"] = model_y_t0_metric_dict.pop("outcome_loss")
    model_y_t0_metric_dict["val_t0_outcome_loss"] = model_y_t0_metric_dict.pop("val_outcome_loss")
    del model_y_t1_metric_dict["loss"], model_y_t1_metric_dict["val_loss"]
    model_y_t1_metric_dict["t1_outcome_loss"] = model_y_t1_metric_dict.pop("outcome_loss")
    model_y_t1_metric_dict["val_t1_outcome_loss"] = model_y_t1_metric_dict.pop("val_outcome_loss")
    if po_act == "sigmoid":
        model_y_t0_metric_dict["t0_outcome_accuracy"] = model_y_t0_metric_dict.pop("outcome_accuracy")
        model_y_t0_metric_dict["val_t0_outcome_accuracy"] = model_y_t0_metric_dict.pop("val_outcome_accuracy")
        model_y_t1_metric_dict["t1_outcome_accuracy"] = model_y_t1_metric_dict.pop("outcome_accuracy")
        model_y_t1_metric_dict["val_t1_outcome_accuracy"] = model_y_t1_metric_dict.pop("val_outcome_accuracy")
    metric_dict = {**model_t_metric_dict, **model_a_metric_dict, **model_y_t0_metric_dict, **model_y_t1_metric_dict}
    return model_t, model_a, model_y_t0, model_y_t1, metric_dict

def train_cfd_tlearner_grid_search(X_train, A_train, T_train, Y_train, checkpoint_dir, shared_backbone=False, 
        overwrite=False, po_act=None, verbose=False, early_stop=False, epochs=100, 
        batch_size_list=[64, 128, 256], reg_l2_list=[1e-2, 1e-3, 1e-4], learning_rate_list=[1e-3, 1e-4]):
    print(f"Training propensity score model")
    def loss(a, b): return slearner_ce_loss(a, b)
    def treatment_loss(a, b): return slearner_ce_loss(a, b)
    def treatment_accuracy(a, b): return slearner_accuracy(a, b)
    best_t_loss = np.inf
    best_t_model = None
    best_t_metric_dict = None
    best_t_hparam = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train T model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        t_model = make_mlp(X_train.shape[1], reg_l2=reg_l2, activation='sigmoid') 
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        t_model, t_metric_dict = train_estimator(loss=loss, model=t_model, learning_rate=learning_rate,
            y_concat_train=np.float32(T_train), x_train=X_train, batch_size=batch_size, epochs=epochs,
            metrics=[treatment_loss, treatment_accuracy], model_name="t_model", early_stop=early_stop,
            checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose)
        if t_metric_dict["val_loss"][-1] < best_t_loss:
            best_t_loss = t_metric_dict["val_loss"][-1]
            best_t_model = t_model
            best_t_metric_dict = t_metric_dict
            best_t_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
    del best_t_metric_dict["loss"], best_t_metric_dict["val_loss"]

    print(f"Training compliance model")
    target_train = np.concatenate([A_train.reshape(-1, 1), T_train.reshape(-1, 1)], 1)
    input_train = X_train
    def loss(a, b): return tlearner_ce_loss(a, b)
    def compliance_loss(a, b): return tlearner_ce_loss(a, b)
    def compliance_accuracy(a, b): return tlearner_accuracy(a, b)
    metrics=[compliance_loss, compliance_accuracy]
    best_a_loss = np.inf
    best_a_model = None
    best_a_metric_dict = None
    best_a_hparam = None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train A model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        if shared_backbone: a_model = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation="sigmoid")
        else: a_model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation='sigmoid')
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        a_model, a_metric_dict = train_estimator(loss=loss, model=a_model, model_name="a_model", early_stop=early_stop,
            y_concat_train=np.float32(target_train), x_train=input_train, metrics=metrics, learning_rate=learning_rate,
            checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, epochs=epochs)
        if a_metric_dict["val_loss"][-1] < best_a_loss:
            best_a_loss = a_metric_dict["val_loss"][-1]
            best_a_model = a_model
            best_a_metric_dict = a_metric_dict
            best_a_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
    del best_a_metric_dict["loss"], best_a_metric_dict["val_loss"]
        
    print(f"Training potential outcome model")        
    target_train_t0 = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1)], 1)[T_train==0]
    target_train_t1 = np.concatenate([Y_train.reshape(-1, 1), A_train.reshape(-1, 1)], 1)[T_train==1]
    input_train_t0 = X_train[T_train==0]
    input_train_t1 = X_train[T_train==1]
    if po_act == "sigmoid":
        def loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_loss(a, b): return tlearner_ce_loss(a, b)
        def outcome_accuracy(a, b): return tlearner_accuracy(a, b)
        metrics=[outcome_loss, outcome_accuracy]
    else:
        def loss(a, b): return tlearner_mse_loss(a, b)
        def outcome_loss(a, b): return tlearner_mse_loss(a, b)
        metrics=[outcome_loss]
    best_y_t0_loss, best_y_t1_loss = np.inf, np.inf
    best_y_t0_model, best_y_t1_model = None, None
    best_y_t0_metric_dict, best_y_t1_metric_dict = None, None
    best_y_t0_hparam, best_y_t1_hparam = None, None
    for batch_size, reg_l2, learning_rate in itertools.product(batch_size_list, reg_l2_list, learning_rate_list):
        if verbose:
            print(f"Train Y model with hparam: batch_size={batch_size}, reg_l2={reg_l2}, learning_rate={learning_rate}")
        if shared_backbone:
            y_t0_model = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
            y_t1_model = make_tarnet(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        else:
            y_t0_model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
            y_t1_model = make_tlearner(X_train.shape[1], reg_l2=reg_l2, activation=po_act)
        curr_checkpoint_dir = os.path.join(checkpoint_dir, f"batch={batch_size}_l2={reg_l2}_lr={learning_rate}")
        os.makedirs(curr_checkpoint_dir, exist_ok=True)
        y_t0_model, y_t0_metric_dict = train_estimator(loss=loss, model=y_t0_model, model_name="y_t0_model",
                y_concat_train=target_train_t0, x_train=input_train_t0, metrics=metrics, early_stop=early_stop,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
                learning_rate=learning_rate, epochs=epochs)
        y_t1_model, y_t1_metric_dict = train_estimator(loss=loss, model=y_t1_model, model_name="y_t1_model",
                y_concat_train=target_train_t1, x_train=input_train_t1, metrics=metrics, early_stop=early_stop,
                checkpoint_dir=curr_checkpoint_dir, overwrite=overwrite, verbose=verbose, batch_size=batch_size, 
                learning_rate=learning_rate, epochs=epochs)
        if y_t0_metric_dict["val_loss"][-1] < best_y_t0_loss:
            best_y_t0_loss = y_t0_metric_dict["val_loss"][-1]
            best_y_t0_model = y_t0_model
            best_y_t0_metric_dict = y_t0_metric_dict
            best_y_t0_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
        if y_t1_metric_dict["val_loss"][-1] < best_y_t1_loss:
            best_y_t1_loss = y_t1_metric_dict["val_loss"][-1]
            best_y_t1_model = y_t1_model
            best_y_t1_metric_dict = y_t1_metric_dict
            best_y_t1_hparam = {"batch_size": batch_size, "reg_l2": reg_l2, "learning_rate": learning_rate}
        

    del best_y_t0_metric_dict["loss"], best_y_t0_metric_dict["val_loss"]
    best_y_t0_metric_dict["t0_outcome_loss"] = best_y_t0_metric_dict.pop("outcome_loss")
    best_y_t0_metric_dict["val_t0_outcome_loss"] = best_y_t0_metric_dict.pop("val_outcome_loss")
    del best_y_t1_metric_dict["loss"], best_y_t1_metric_dict["val_loss"]
    best_y_t1_metric_dict["t1_outcome_loss"] = best_y_t1_metric_dict.pop("outcome_loss")
    best_y_t1_metric_dict["val_t1_outcome_loss"] = best_y_t1_metric_dict.pop("val_outcome_loss")
    if po_act == "sigmoid":
        best_y_t0_metric_dict["t0_outcome_accuracy"] = best_y_t0_metric_dict.pop("outcome_accuracy")
        best_y_t0_metric_dict["val_t0_outcome_accuracy"] = best_y_t0_metric_dict.pop("val_outcome_accuracy")
        best_y_t1_metric_dict["t1_outcome_accuracy"] = best_y_t1_metric_dict.pop("outcome_accuracy")
        best_y_t1_metric_dict["val_t1_outcome_accuracy"] = best_y_t1_metric_dict.pop("val_outcome_accuracy")
    best_metric_dict = {**best_t_metric_dict, **best_a_metric_dict, **best_y_t0_metric_dict, **best_y_t1_metric_dict}
    best_hparam_dict = {"best_t_model_hparam": best_t_hparam, "best_a_model_hparam": best_a_hparam,
                        "best_y_t0_model_hparam": best_y_t0_hparam, "best_y_t1_model_hparam": best_y_t1_hparam}
    return best_t_model, best_a_model, best_y_t0_model, best_y_t1_model, best_metric_dict, best_hparam_dict

def cfd_inference(t_pred, a_pred, y_pred, non_compliance_type:Literal["none", 'one-sided', 'two-sided']):
    assert a_pred.shape[1]==2 and y_pred.shape[1]==4
    a_t0_pred, a_t1_pred = a_pred[:, 0], a_pred[:, 1]
    y_a0_t0_pred, y_a1_t0_pred = y_pred[:, 0], y_pred[:, 1]
    y_a0_t1_pred, y_a1_t1_pred = y_pred[:, 2], y_pred[:, 3]

    if non_compliance_type == "none":
        Y_a0_hat, Y_a1_hat = y_a0_t0_pred, y_a1_t1_pred
        Y_t0_hat, Y_t1_hat = y_a0_t0_pred, y_a1_t1_pred
    elif non_compliance_type == "one-sided":
        Y_a0_hat = (y_a0_t0_pred * (1-t_pred) + y_a0_t1_pred * (t_pred))
        Y_a1_hat = y_a1_t1_pred
        Y_t0_hat = Y_a0_hat
        Y_t1_hat = (Y_a0_hat * (1-a_t1_pred) + Y_a1_hat * (a_t1_pred))
    elif non_compliance_type == "two-sided":
        Y_a0_hat = (y_a0_t0_pred * (1-t_pred) + y_a0_t1_pred * (t_pred))
        Y_a1_hat = (y_a1_t0_pred * (1-t_pred) + y_a1_t1_pred * (t_pred))
        Y_t0_hat = (Y_a0_hat * (1-a_t0_pred) + Y_a1_hat * (a_t0_pred))
        Y_t1_hat = (Y_a0_hat * (1-a_t1_pred) + Y_a1_hat * (a_t1_pred))

    Ya_pred = np.stack((Y_a0_hat, Y_a1_hat)).T
    Yt_pred = np.stack((Y_t0_hat, Y_t1_hat)).T
    return Ya_pred, Yt_pred


