"""Random forest (scikit-learn), translated from Random Forest.ipynb.

Translates random_forest_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the notebook function's defaults. Your grid search results weren't saved in the notebook,
so change the defaults below to your lowest-RMSE settings once you have them.
"""

import numpy as np

from models.shared import importance_table

NAME = "Random Forest"
DESCRIPTION = (
    "Averages many decision trees. Each tree trains on a random sample of the rows and tries a random share of "
    "the predictors at each split, so the trees make different mistakes that cancel out. Blank numbers are "
    "filled with the training median."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search. Trees don't need scaling.
SEED_USED_FOR = "the random rows and predictors each tree uses"

PARAMETERS = {
    "n_estimators": {
        "label": "Number of trees", "type": "int", "default": 500, "min": 1,
        "help": "More trees give steadier predictions but take longer. More trees never overfit. "
                "Notebook: 500 (grid searched 50 to 1,000).",
    },
    "max_features": {
        "label": "Share of predictors tried at each split", "type": "float", "default": 1.0,
        "min": 0.01, "max": 1.0, "step": 0.01, "format": "%.2f",
        "help": "Lower values make the trees more different from each other; 1.0 tries every predictor. "
                "Counts each one-hot column separately. Notebook: 0.33 (grid searched 0.05 to 1.0).",
    },
    "min_samples_leaf": {
        "label": "Minimum rows per leaf", "type": "int", "default": 3, "min": 1,
        "help": "The smallest group a tree can end in. Higher values give smoother predictions. "
                "Notebook: 1 (grid searched 1 to 20).",
    },
    "max_samples": {
        "label": "Share of rows each tree trains on", "type": "float", "default": None,
        "min": 0.01, "max": 1.0, "step": 0.05, "format": "%.2f",
        "help": "1.0 draws as many rows as the training data has (the notebook's None). Lower values make the "
                "trees more different from each other. Notebook: None (grid searched 0.25 to 1.0).",
    },
    "max_depth": {
        "label": "Maximum tree depth (0 = no limit)", "type": "int", "default": 0, "min": 0,
        "help": "0 lets each tree grow until the minimum rows per leaf stops it. Notebook: None (no limit).",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, n_estimators, max_features, min_samples_leaf,
                    max_samples, max_depth):
    """Returns (predictions, each predictor's share of the forest's importance)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,       # 1.0 is the same as the notebook's None
            max_depth=max_depth or None,   # 0 means no limit (the notebook's None)
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))

    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    # Impurity importance: how much each predictor's splits reduce the trees' squared error.
    return predictions, importance_table(X_train.columns, model[-1].feature_importances_)
