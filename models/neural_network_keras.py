"""Neural network (Keras), translated from Nueral_Net - Keras.ipynb.

Translates make_keras_neural_network_predictor with build_keras_regression_model, fitted once
with the parameters below (no tuning).
Defaults: the notebook's final neural_network_predictor settings and the builder's layer_settings.

How it differs from "Neural Network": the imputer, scaler, and rent scaling are fit on ALL training rows,
early stopping watches the last rows of the training data (Keras's validation_split), and every hidden
layer uses relu (fixed in build_keras_regression_model).
"""

import numpy as np

from models.shared import network_summary

NAME = "Neural Network - Keras"
DESCRIPTION = (
    "A feed-forward network with the hidden layers in the table, all using relu. Blank numbers are filled "
    "with the training median, predictors and rent are standardized using every training row, and early "
    "stopping watches the last rows of the training data."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the network fits its own StandardScaler.
SEED_USED_FOR = "the starting weights, shuffling, and dropout"

PARAMETERS = {
    "layer_settings": {
        "label": "Hidden layers (one row per layer, first to last)",
        "type": "table",
        "columns": {
            "neurons": {"label": "Neurons", "type": "int", "min": 1},
            "dropout_rate": {"label": "Dropout rate", "type": "float", "min": 0.0, "max": 0.99, "step": 0.05,
                             "format": "%.2f", "help": "0 means no dropout after this layer."},
        },
        # The notebook's layer_settings in build_keras_regression_model.
        "default": [
            {"neurons": 500, "dropout_rate": 0.20},
            {"neurons": 100, "dropout_rate": 0.20},
            {"neurons": 50, "dropout_rate": 0.10},
            {"neurons": 20, "dropout_rate": 0.00},
            {"neurons": 10, "dropout_rate": 0.00},
            {"neurons": 5, "dropout_rate": 0.00},
        ],
        "help": "Add or delete rows to add or remove layers. Notebook: 500, 100, 50, 20, 10, 5 neurons.",
    },
    "epochs": {
        "label": "Epochs (maximum)", "type": "int", "default": 400, "min": 1,
        "help": "Notebook: 400.",
    },
    "learning_rate": {
        "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.0, "step": 0.0001, "format": "%.5f",
        "help": "Adam optimizer step size. Notebook: 0.001.",
    },
    "batch_size": {
        "label": "Batch size", "type": "int", "default": 32, "min": 1,
        "help": "Training rows per weight update. Notebook: 32.",
    },
    "alpha": {
        "label": "Alpha (L2 penalty)", "type": "float", "default": 0.0001, "min": 0.0, "step": 0.0001, "format": "%.5f",
        "help": "Penalty on the hidden layers' weights. Notebook: 0.0001.",
    },
    "early_stopping": {
        "label": "Early stopping", "type": "bool", "default": True,
        "help": "Stop when the validation loss stops improving, and keep the best weights. Notebook: on.",
    },
    "validation_fraction": {
        "label": "Validation fraction", "type": "float", "default": 0.10, "min": 0.01, "max": 0.99, "step": 0.05,
        "format": "%.2f",
        "help": "Share of training rows (the last ones) used to watch for early stopping. Notebook: 0.10.",
    },
    "patience": {
        "label": "Patience (epochs)", "type": "int", "default": 30, "min": 1,
        "help": "Epochs without improvement before stopping. Notebook: 30.",
    },
    "tolerance": {
        "label": "Tolerance", "type": "float", "default": 1e-5, "min": 0.0, "step": 1e-6, "format": "%.1e",
        "help": "Smallest drop in validation loss that counts as improvement. Notebook: 1e-5.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, layer_settings, epochs, learning_rate, batch_size,
                    alpha, early_stopping, validation_fraction, patience, tolerance):
    """Returns (predictions, a summary of the layers and training)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    import keras
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The table's rows become the notebook's layer_settings: ((neurons, dropout rate), ...).
    layer_settings = tuple((row["neurons"], row["dropout_rate"]) for row in layer_settings)

    # From here on, the same steps in the same order as the notebook's neural_network_predictions.
    keras.backend.clear_session()
    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    # Fit feature preprocessing using all the training rows.
    feature_preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    X_train_processed = feature_preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_processed = feature_preprocessor.transform(X_test).astype(np.float32)

    # Scale the response for more stable training.
    target_scaler = StandardScaler()
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_train_scaled = target_scaler.fit_transform(y_train_array).astype(np.float32)

    # build_keras_regression_model: a new network with fresh weights.
    model = keras.Sequential([keras.Input(shape=(X_train_processed.shape[1],))])
    regularizer = keras.regularizers.L2(alpha)
    for neurons, dropout_rate in layer_settings:
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rates must be between 0 and less than 1.")
        model.add(keras.layers.Dense(neurons, activation="relu", kernel_regularizer=regularizer))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    # Standardized targets can be negative, so use a linear output.
    model.add(keras.layers.Dense(1, activation="linear"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error",
                  metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])

    callbacks = []
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, min_delta=tolerance, restore_best_weights=True, verbose=0,
        ))

    history = model.fit(X_train_processed, y_train_scaled, epochs=epochs, batch_size=batch_size,
                        validation_split=(validation_fraction if early_stopping else 0.0),
                        callbacks=callbacks, shuffle=True, verbose=0)

    predictions = target_scaler.inverse_transform(model.predict(X_test_processed, verbose=0))
    return np.asarray(predictions, dtype=float).reshape(-1), network_summary(layer_settings, history, early_stopping)
