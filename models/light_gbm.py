"""LightGBM (lightgbm), translated from Light GBM.ipynb.

Translates lightgbm_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the notebook function's defaults. Your grid search results weren't saved in the notebook,
so change the defaults below to your lowest-RMSE settings once you have them.
"""

import numpy as np

from models.shared import importance_table

NAME = "LightGBM"
DESCRIPTION = (
    "Gradient boosting with leaf-wise trees: builds trees one after another, each correcting the errors the "
    "earlier trees still make, and grows each tree by splitting whichever leaf helps most. Fast and accurate, "
    "but can overfit small data. LightGBM handles blank numbers itself."
)
ONE_HOT_ENCODE = True  # As in the notebook's grid search.
SCALE_NUMERIC = False  # As in the notebook's grid search. Trees don't need scaling.
SEED_USED_FOR = "the random rows (row share) and predictors (predictor share) each tree uses"

PARAMETERS = {
    "num_leaves": {
        "label": "Maximum leaves per tree", "type": "int", "default": 31, "min": 2,
        "help": "LightGBM's main complexity setting. More leaves capture more detail but overfit more easily. "
                "Notebook: 31 (grid searched 15, 31, 63).",
    },
    "learning_rate": {
        "label": "Learning rate", "type": "float", "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.01,
        "format": "%.3f",
        "help": "How much each tree's correction counts. Smaller steps need more trees. "
                "Notebook: 0.05 (grid searched 0.03, 0.05, 0.1).",
    },
    "n_estimators": {
        "label": "Number of trees", "type": "int", "default": 1000, "min": 1,
        "help": "Works with the learning rate: a lower rate needs more trees. "
                "Notebook: 1,000 (grid searched 500 and 1,000).",
    },
    "min_child_samples": {
        "label": "Minimum rows per leaf", "type": "int", "default": 20, "min": 1,
        "help": "Stops leaf-wise trees from chasing tiny groups. Raise it for small or noisy data. "
                "Notebook: 20 (grid searched 5, 20, 50).",
    },
    "subsample": {
        "label": "Share of rows each tree sees", "type": "float", "default": 0.8, "min": 0.01, "max": 1.0,
        "step": 0.05, "format": "%.2f",
        "help": "Each tree trains on this random share of the rows. Notebook: 0.8.",
    },
    "colsample_bytree": {
        "label": "Share of predictors each tree sees", "type": "float", "default": 0.8, "min": 0.01, "max": 1.0,
        "step": 0.05, "format": "%.2f",
        "help": "Each tree uses this random share of the predictors. Counts each one-hot column separately. "
                "Notebook: 0.8.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, num_leaves, learning_rate, n_estimators,
                    min_child_samples, subsample, colsample_bytree):
    """Returns (predictions, each predictor's share of LightGBM's importance)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,   # needed, or LightGBM ignores subsample
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    # Plain numbers, not the DataFrame: LightGBM rejects column names with characters like , : [ ]
    # (one-hot columns such as "City-Zoning_M-1, light" can have them).
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))

    predictions = np.asarray(model.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    # Gain importance: how much each predictor's splits reduce the error, added up over all trees.
    return predictions, importance_table(X_train.columns, model.booster_.feature_importance(importance_type="gain"))
