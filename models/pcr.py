"""Principal component regression (scikit-learn), translated from PCR.ipynb.

Translates pcr_regression_predictions, fitted once with the parameters below (no tuning).
Defaults: the lowest-RMSE number of components from your grid search, and the notebook's values for the rest.
"""

import numpy as np

from models.shared import coefficient_table

NAME = "Principal Component Regression (PCR)"
DESCRIPTION = (
    "Combines the standardized predictors into `n_components` principal components (the directions with the "
    "most variation) and fits a linear regression on them. Blank numbers are filled with the training median first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the StandardScaler below scales every predictor.
SEED_USED_FOR = "the arpack and randomized solvers only (the default full solver has no randomness)"

PARAMETERS = {
    "n_components": {
        "label": "Number of components", "type": "int", "default": 69, "min": 1,
        "help": "At most the number of predictors after one-hot encoding, and at most the training rows minus 1. "
                "Notebook: 69, the lowest RMSE in your grid search.",
    },
    "svd_solver": {
        "label": "PCA solver", "type": "choice", "options": ["full", "auto", "covariance_eigh", "arpack", "randomized"],
        "default": "full",
        "help": "How the components are calculated. Notebook: full.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, n_components, svd_solver):
    """Returns (predictions, each predictor's coefficient)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Centering limits the number of nonzero components to at most rows minus one.
    maximum = min(X_train.shape[1], X_train.shape[0] - 1)
    if maximum < 1:
        raise ValueError("PCR requires at least two training rows and one feature.")
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state),
        LinearRegression(),
    )
    model.fit(X_train, y_train)

    # Details: the regression on the components, turned back into one coefficient per standardized predictor.
    pca, regression = model[-2], model[-1]
    coefficients = pca.components_.T @ regression.coef_
    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    return predictions, coefficient_table(X_train.columns, coefficients, regression.intercept_, keep_zeros=True)
