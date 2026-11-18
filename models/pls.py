"""Partial least squares regression (scikit-learn), translated from PLS.ipynb.

Translates pls_regression_predictions, fitted once with the parameters below. The notebook's function
also chose the number of components by cross-validation inside every fit (from 1 up to max_components);
that's tuning, so here the number of components is a parameter instead.
Defaults: the lowest-RMSE setting from your grid search (at most 8 components), and the notebook's values
for the rest.
"""

import numpy as np

from models.shared import coefficient_table

NAME = "Partial Least Squares (PLS)"
DESCRIPTION = (
    "Combines the predictors into `n_components` components chosen to predict rent (unlike PCR, which ignores "
    "rent when building them) and fits on them. Blank numbers are filled with the training median first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: PLSRegression(scale=True) scales every predictor.
SEED_USED_FOR = None   # PLSRegression has no randomness.

PARAMETERS = {
    "n_components": {
        "label": "Number of components", "type": "int", "default": 8, "min": 1,
        "help": "At most the number of predictors after one-hot encoding, and at most the training rows. "
                "Notebook: chosen inside each fit by cross-validation, from 1 up to max_components; "
                "8 was the lowest-RMSE maximum in your grid search.",
    },
    "scale": {
        "label": "Standardize inside the model", "type": "bool", "default": True,
        "help": "Scale predictors and rent to unit variance before building the components. Notebook: on.",
    },
    "max_iter": {
        "label": "Maximum iterations", "type": "int", "default": 1_000, "min": 1,
        "help": "Notebook: 1,000.",
    },
    "tol": {
        "label": "Tolerance", "type": "float", "default": 1e-6, "min": 0.0, "step": 1e-7, "format": "%.1e",
        "help": "Notebook: 1e-6.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, n_components, scale, max_iter, tol):
    """Returns (predictions, each predictor's coefficient)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    maximum = min(X_train.shape[1], X_train.shape[0])
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        PLSRegression(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol),
    )
    model.fit(X_train, y_train)
    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    # Details: PLS reports coefficients per unit of each predictor; turn them into per 1 standard deviation
    # (like the other linear models), with the intercept as the prediction at average predictor values.
    X_fit = model[0].transform(X_train)
    coefficients = np.ravel(model[-1].coef_) * X_fit.std(axis=0)
    intercept = float(np.ravel(model[-1].predict(X_fit.mean(axis=0, keepdims=True)))[0])
    return predictions, coefficient_table(X_train.columns, coefficients, intercept, keep_zeros=True)
