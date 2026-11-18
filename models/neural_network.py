"""Neural network (Keras), translated from Nueral_Net.ipynb.

Translates make_neural_network_predictor, fitted once with the parameters below (no tuning).
Defaults: the notebook's final neural_network_predictor settings.

How it differs from "Neural Network - Keras": the early-stopping validation rows are a random
sample set aside BEFORE the imputer, scaler, and rent scaling are fit, and the activation is a setting.
"""

import numpy as np

from models.shared import network_summary

NAME = "Neural Network"
DESCRIPTION = (
    "A feed-forward network with the hidden layers in the table. Blank numbers are filled with the training "
    "median, predictors and rent are standardized, and early stopping watches a random validation sample "
    "that is set aside before anything is fit."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the network fits its own StandardScaler.
SEED_USED_FOR = "the starting weights, shuffling, dropout, and the random validation sample"

PARAMETERS = {
    "layer_settings": {
        "label": "Hidden layers (one row per layer, first to last)",
        "type": "table",
        "columns": {
            "neurons": {"label": "Neurons", "type": "int", "min": 1},
            "dropout_rate": {"label": "Dropout rate", "type": "float", "min": 0.0, "max": 0.99, "step": 0.05,
                             "format": "%.2f", "help": "0 means no dropout after this layer."},
        },
        # The notebook's layer_settings.
        "default": [
            {"neurons": 400, "dropout_rate": 0.20},
            {"neurons": 300, "dropout_rate": 0.20},
            {"neurons": 50, "dropout_rate": 0.10},
            {"neurons": 20, "dropout_rate": 0.00},
            {"neurons": 10, "dropout_rate": 0.00},
            {"neurons": 5, "dropout_rate": 0.00},
        ],
        "help": "Add or delete rows to add or remove layers. Notebook: 400, 300, 50, 20, 10, 5 neurons.",
    },
    "activation": {
        "label": "Activation", "type": "choice", "default": "relu",
        "options": ["relu", "elu", "selu", "gelu", "tanh", "sigmoid", "softplus", "swish", "linear"],
        "help": "Used by every hidden layer. Notebook: relu.",
    },
    "epochs": {
        "label": "Epochs (maximum)", "type": "int", "default": 600, "min": 1,
        "help": "Notebook: 500.",
    },
    "learning_rate": {
        "label": "Learning rate", "type": "float", "default": 0.0005, "min": 0.0, "step": 0.0001, "format": "%.5f",
        "help": "Adam optimizer step size. Notebook: 0.0001.",
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
        "help": "Share of training rows set aside to watch for early stopping. Notebook: 0.10.",
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


def fit_and_predict(X_train, y_train, X_test, random_state, layer_settings, activation, epochs, learning_rate,
                    batch_size, alpha, early_stopping, validation_fraction, patience, tolerance):
    """Returns (predictions, a summary of the layers and training)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    import keras
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The table's rows become the notebook's layer_settings: ((neurons, dropout rate), ...).
    layer_settings = tuple((row["neurons"], row["dropout_rate"]) for row in layer_settings)
    if any(not 0 <= rate < 1 for _, rate in layer_settings):
        raise ValueError("Dropout rates must be at least 0 and less than 1.")
    if early_stopping and not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    # From here on, the same steps in the same order as the notebook's neural_network_predictions.
    keras.backend.clear_session()
    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    y = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    if y.shape[0] != len(X_train) or not np.isfinite(y).all():
        raise ValueError("y_train needs one finite target value per training row.")

    # Reserve validation rows before fitting any preprocessing.
    X_fit, y_fit = X_train, y
    if early_stopping:
        X_fit, X_val, y_fit, y_val = train_test_split(X_train, y, test_size=validation_fraction, random_state=random_state)

    # Learn feature preprocessing from the fitting rows only.
    feature_preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    X_fit = feature_preprocessor.fit_transform(X_fit).astype(np.float32)
    X_test_processed = feature_preprocessor.transform(X_test).astype(np.float32)

    # Scale the target; predictions are converted back afterward.
    target_scaler = StandardScaler()
    y_fit = target_scaler.fit_transform(y_fit).astype(np.float32)

    validation_data = None
    callbacks = []
    if early_stopping:
        validation_data = (
            feature_preprocessor.transform(X_val).astype(np.float32),
            target_scaler.transform(y_val).astype(np.float32),
        )
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, min_delta=tolerance, restore_best_weights=True, verbose=0,
        ))

    # Build a new network with fresh weights.
    model = keras.Sequential([keras.Input(shape=(X_fit.shape[1],))])
    regularizer = keras.regularizers.L2(alpha)
    for neurons, dropout_rate in layer_settings:
        model.add(keras.layers.Dense(neurons, activation=activation, kernel_regularizer=regularizer))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    # Standardized targets can be negative.
    model.add(keras.layers.Dense(1, activation="linear"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    history = model.fit(X_fit, y_fit, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                        callbacks=callbacks, shuffle=True, verbose=0)

    predictions_scaled = model.predict(X_test_processed, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_scaled).reshape(-1)
    return predictions, network_summary(layer_settings, history, early_stopping)

