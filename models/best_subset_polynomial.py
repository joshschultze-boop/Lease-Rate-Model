"""Best subset polynomial regression (ABESS), translated from Best Subset - Polynomial.ipynb.

Translates polynomial_best_subset_predictions, fitted once with the parameters below. The notebook's
function also chose how many terms to keep by cross-validation inside every fit (from 0 up to
max_selected_terms); that's tuning, so here the number of terms kept is a parameter instead.
Defaults: the lowest-RMSE settings from your grid search (degree 1, at most 7 terms).
"""

import numpy as np

from models.shared import coefficient_table, linear_terms

NAME = "Best Subset Polynomial Regression"
DESCRIPTION = (
    "Adds polynomial terms up to `degree` (squares and products of predictors), then picks the best combination "
    "of `support_size` terms and fits a linear regression on them. Blank numbers are filled with the training "
    "median, and every term is standardized first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the StandardScaler below scales every term.
SEED_USED_FOR = None   # ABESS fits the one support size given with no randomness.

PARAMETERS = {
    "degree": {
        "label": "Degree", "type": "int", "default": 2, "min": 1, "max": 3,
        "help": "1 = the predictors themselves; 2 adds squares and every pair of predictors multiplied together; "
                "3 adds cubes and triples. 3 makes a very large number of terms and can be slow. "
                "Notebook: 2, the lowest RMSE in your grid search.",
    },
    "support_size": {
        "label": "Support size (number of terms kept)", "type": "int", "default": 10, "min": 0,
        "help": "Counts each polynomial term and one-hot column separately. Notebook: chosen inside each fit by "
                "cross-validation, from 0 up to max_selected_terms; 7 was the lowest-RMSE maximum in your grid search.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, degree, support_size):
    """Returns (predictions, the terms kept with their coefficients, what each column adds for the column weights)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from abess.linear import LinearRegression as ABESSLinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # Count the possible polynomial terms (the same way the notebook does) before fitting.
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    term_count = polynomial.fit(np.zeros((1, X_train.shape[1]))).n_output_features_
    if not 0 <= support_size <= term_count:
        raise ValueError(f"Support size {support_size} is more than the {term_count} polynomial terms available. "
                         "Lower it, raise the degree, or add predictor columns.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("polynomial", polynomial),
        ("scaler", StandardScaler()),
        # cv=1 and ic_type="loss" fit the one support size given, as in the notebook.
        ("model", ABESSLinearRegression(support_size=support_size, cv=1, ic_type="loss")),
    ])
    # The StandardScaler handles scaling, so ABESS's own normalization is turned off.
    model.fit(X_train, np.asarray(y_train, dtype=float), model__is_normal=False)

    fitted = model.named_steps["model"]
    terms = model.named_steps["polynomial"].get_feature_names_out(list(X_train.columns))
    return (np.asarray(model.predict(X_test), dtype=float).reshape(-1), coefficient_table(terms, fitted.coef_, fitted.intercept_),
            linear_terms(model, X_train))
