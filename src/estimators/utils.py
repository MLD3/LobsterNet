import pickle, os, time
from tqdm.keras import TqdmCallback
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN


def train_estimator(y_concat_train, x_train, model, checkpoint_dir, loss, model_name="model",
    val_split=0.2, batch_size=64, learning_rate=1e-3, epochs=100,
    verbose=False, overwrite=False, metrics=None, early_stop=False):

    assert os.path.isdir(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"{model_name}.keras")
    metric_path = os.path.join(checkpoint_dir, f"{model_name}_metrics.pkl")
    if os.path.isfile(model_path) and os.path.isfile(metric_path) and not overwrite:
        metric_dict = pickle.load(open(metric_path, 'rb'))
        model.load_weights(model_path)
    else:
        metric_dict = {"loss": [], "val_loss": []}
        if metrics is not None:
            for m in metrics:
                metric_dict[m.__name__] = []
                metric_dict[f"val_{m.__name__}"] = []


        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)

        adam_callbacks = [TerminateOnNaN(), TqdmCallback(verbose=verbose),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, 
                verbose=verbose, mode='auto', min_delta=1e-8, cooldown=0, min_lr=0)]
        if early_stop: adam_callbacks.append(EarlyStopping(monitor='val_loss', patience=5, min_delta=0.))

        history = model.fit(
            x_train, y_concat_train, callbacks=adam_callbacks, verbose=verbose,
            validation_split=val_split, epochs=epochs, batch_size=batch_size)
        for m in metric_dict:
            metric_dict[m] += history.history[m]

        K.clear_session()
        model.save_weights(model_path)
        with open(metric_path, 'wb') as f:
            pickle.dump(metric_dict, f)

    return model, metric_dict