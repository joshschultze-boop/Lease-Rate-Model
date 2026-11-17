"""Best subset linear regression (ABESS), from Best Subset - Linear.ipynb.

Keeps the best `support_size` predictors and fits a linear regression on them, with an optional
ridge penalty (alpha). No tuning here: it fits the one set of parameters chosen on the page.
"""

import numpy as np
import pandas as pd
from abess.linear import LinearRegression as ABESSLinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Best Subset Linear Regression"
DESCRIPTION = (
    "Picks the best combination of `support_size` predictors and fits a linear regression on them. "
    "Blank numbers are filled with the training median, and every predictor is standardized first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = False  # The StandardScaler below already scales every predictor (same as the notebook).

PARAMETERS = {
    "support_size": {
        "label": "Support size (number of predictors kept)", "type": "int", "default": 27, "min": 0,
        "help": "Counts each one-hot column separately. Your lowest-RMSE setting in the notebook was 27.",
    },
    "alpha": {
        "label": "Alpha (ridge penalty)", "type": "float", "default": 1.0, "min": 0.0, "step": 0.01, "format": "%.4f",
        "help": "0 means no penalty. Your lowest-RMSE setting in the notebook was 1.0.",
    },
}


def fit_and_predict(X_train, y_train, X_test, support_size, alpha):
    """Returns (predictions, details table of the predictors kept)."""
    if not 0 <= support_size <= X_train.shape[1]:
        raise ValueError(
            f"Support size {support_size} is more than the {X_train.shape[1]} predictors available "
            "after one-hot encoding. Lower it, or add predictor columns."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        # thread=1 is FIXED_PARAMS from the notebook.
        ("model", ABESSLinearRegression(support_size=support_size, alpha=alpha, cv=1, ic_type="loss", thread=1)),
    ])
    # StandardScaler handles scaling, so ABESS's own normalization is turned off.
    model.fit(X_train, np.asarray(y_train, dtype=float), model__is_normal=False)

    # Details: the predictors kept, largest effect first. Coefficients are per 1 standard
    # deviation of each predictor, so they can be compared with each other.
    fitted = model.named_steps["model"]
    coefficient = "Coefficient (per 1 standard deviation)"
    kept = pd.DataFrame({"Predictor": X_train.columns, coefficient: fitted.coef_})
    kept = kept.loc[kept[coefficient] != 0].sort_values(coefficient, key=abs, ascending=False)
    details = pd.concat([pd.DataFrame({"Predictor": ["(intercept)"], coefficient: [float(np.ravel(fitted.intercept_)[0])]}), kept],
                        ignore_index=True)

    return model.predict(X_test), details
