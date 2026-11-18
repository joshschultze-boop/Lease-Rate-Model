"""Ridge regression (scikit-learn), translated from Ridge.ipynb.

Translates ridge_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the lowest-RMSE alpha from your grid search, and the notebook's values for the rest.
"""

import numpy as np

from models.shared import coefficient_table

NAME = "Ridge Regression"
DESCRIPTION = (
    "Linear regression with an L2 penalty (alpha), which shrinks every coefficient toward 0. "
    "Blank numbers are filled with the training median, and every predictor is standardized first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the StandardScaler below scales every predictor.

PARAMETERS = {
    "alpha": {
        "label": "Alpha (L2 penalty)", "type": "float", "default": 17.851628990783755, "min": 0.0, "step": 0.1,
        "format": "%.4f",
        "help": "Higher values shrink coefficients more. Notebook: 17.85, the lowest RMSE in your grid search.",
    },
    "solver": {
        "label": "Solver", "type": "choice",
        "options": ["svd", "auto", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "default": "svd",
        "help": "How the coefficients are calculated. Notebook: svd.",
    },
}


def fit_and_predict(X_train, y_train, X_test, alpha, solver):
    """Returns (predictions, every predictor's coefficient)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha, solver=solver),
    )
    model.fit(X_train, y_train)

    fitted = model[-1]
    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    return predictions, coefficient_table(X_train.columns, fitted.coef_, fitted.intercept_, keep_zeros=True)
