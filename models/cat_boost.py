"""CatBoost (catboost), translated from Cat Boost.ipynb.

Translates catboost_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the notebook function's defaults. Your grid search results weren't saved in the notebook,
so change the defaults below to your lowest-RMSE settings once you have them.
"""

import numpy as np

from models.shared import importance_table

NAME = "CatBoost"
DESCRIPTION = (
    "Gradient boosting: builds trees one after another, each correcting the errors the earlier trees still make. "
    "CatBoost grows balanced (symmetric) trees, which makes it fairly resistant to overfitting. "
    "CatBoost handles blank numbers itself."
)
ONE_HOT_ENCODE = True  # As in the notebook's grid search.
SCALE_NUMERIC = False  # As in the notebook's grid search. Trees don't need scaling.
SEED_USED_FOR = "CatBoost's random row sampling and split scoring"

PARAMETERS = {
    "depth": {
        "label": "Tree depth", "type": "int", "default": 5, "min": 1, "max": 16,
        "help": "Deeper trees capture more interactions between predictors but overfit more easily. "
                "Notebook: 6 (grid searched 4 to 10).",
    },
    "learning_rate": {
        "label": "Learning rate", "type": "float", "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.01,
        "format": "%.3f",
        "help": "How much each tree's correction counts. Smaller steps need more iterations. "
                "Notebook: 0.05 (grid searched 0.03, 0.05, 0.1).",
    },
    "l2_leaf_reg": {
        "label": "L2 leaf regularization", "type": "float", "default": 3.0, "min": 0.0, "step": 0.5,
        "format": "%.2f",
        "help": "Shrinks each tree's predictions toward 0. Higher values are more conservative. "
                "Notebook: 3 (grid searched 1, 3, 10).",
    },
    "iterations": {
        "label": "Iterations (number of trees)", "type": "int", "default": 750, "min": 1,
        "help": "Works with the learning rate: a lower rate needs more trees. "
                "Notebook: 1,000 (grid searched 500 and 1,000).",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, depth, learning_rate, l2_leaf_reg, iterations):
    """Returns (predictions, each predictor's share of CatBoost's importance)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,   # no catboost_info folder on the server
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))

    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    # CatBoost's default importance: how much the predictions change, on average, when a predictor's value changes.
    return predictions, importance_table(X_train.columns, model.get_feature_importance())
