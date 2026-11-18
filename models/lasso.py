"""Lasso regression (scikit-learn), translated from Lasso.ipynb.

Translates lasso_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the lowest-RMSE alpha from your grid search, and the notebook's values for the rest.
"""

import numpy as np

from models.shared import coefficient_table

NAME = "Lasso Regression"
DESCRIPTION = (
    "Linear regression with an L1 penalty (alpha), which shrinks coefficients and sets some to exactly 0. "
    "Blank numbers are filled with the training median, and every predictor is standardized first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the StandardScaler below scales every predictor.
SEED_USED_FOR = "the random coefficient update order"

PARAMETERS = {
    "alpha": {
        "label": "Alpha (L1 penalty)", "type": "float", "default": 505.07480483824537, "min": 0.0, "step": 1.0,
        "format": "%.4f",
        "help": "Higher values keep fewer predictors. Notebook: 505.07, the lowest RMSE in your grid search.",
    },
    "max_iter": {
        "label": "Maximum iterations", "type": "int", "default": 1_000_000, "min": 1,
        "help": "Notebook: 1,000,000.",
    },
    "tol": {
        "label": "Tolerance", "type": "float", "default": 1e-4, "min": 0.0, "step": 1e-5, "format": "%.1e",
        "help": "Stops when updates are smaller than this. Notebook: 1e-4.",
    },
    "selection": {
        "label": "Coefficient update order", "type": "choice", "options": ["random", "cyclic"], "default": "random",
        "help": "Notebook: random.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, alpha, max_iter, tol, selection):
    """Returns (predictions, the predictors kept with their coefficients)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state, selection=selection),
    )
    model.fit(X_train, y_train)

    fitted = model[-1]
    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    return predictions, coefficient_table(X_train.columns, fitted.coef_, fitted.intercept_)
