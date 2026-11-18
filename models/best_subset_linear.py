"""Best subset linear regression (ABESS), translated from Best Subset - Linear.ipynb.

Translates best_subset_grid_predictions, fitted once with the parameters below (no tuning).
Defaults: the lowest-RMSE settings from your grid search.
"""

import numpy as np

from models.shared import coefficient_table

NAME = "Best Subset Linear Regression"
DESCRIPTION = (
    "Picks the best combination of `support_size` predictors and fits a linear regression on them, "
    "with an optional ridge penalty. Blank numbers are filled with the training median, and every "
    "predictor is standardized first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # As in the notebook's grid search: the StandardScaler below scales every predictor.
SEED_USED_FOR = None   # ABESS fits the one support size given with no randomness.

PARAMETERS = {
    "support_size": {
        "label": "Support size (number of predictors kept)", "type": "int", "default": 6, "min": 0,
        "help": "Counts each one-hot column separately. Notebook: 6, the lowest RMSE in your grid search.",
    },
    "alpha": {
        "label": "Alpha (ridge penalty)", "type": "float", "default": 0.130045, "min": 0.0, "step": 0.01, "format": "%.4f",
        "help": "0 means no penalty. Notebook: 0.130045, the lowest RMSE in your grid search.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, support_size, alpha):
    """Returns (predictions, the predictors kept with their coefficients)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from abess.linear import LinearRegression as ABESSLinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not 0 <= support_size <= X_train.shape[1]:
        raise ValueError(f"Support size {support_size} is more than the {X_train.shape[1]} predictors available "
                         "after one-hot encoding. Lower it, or add predictor columns.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        # cv=1 and ic_type="loss" fit the one support size given; thread=1 is FIXED_PARAMS from the notebook.
        ("model", ABESSLinearRegression(support_size=support_size, alpha=alpha, cv=1, ic_type="loss", thread=1)),
    ])
    # The StandardScaler handles scaling, so ABESS's own normalization is turned off.
    model.fit(X_train, np.asarray(y_train, dtype=float), model__is_normal=False)

    fitted = model.named_steps["model"]
    return model.predict(X_test), coefficient_table(X_train.columns, fitted.coef_, fitted.intercept_)
