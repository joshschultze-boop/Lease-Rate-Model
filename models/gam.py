"""Generalized additive model (pyGAM LinearGAM), translated from GAM.ipynb.

Translates GAMRegressor (with its median imputer), fitted once with the parameters below.
The notebook's gam_predictions also chose lam_scale from 0.1, 1, or 10 by cross-validation inside
every fit; that's tuning, so here lam_scale is a parameter instead.
Defaults: the notebook's gam_spline_terms and gam_prediction_function settings.
"""

import numpy as np
import pandas as pd

NAME = "GAM (Generalized Additive Model)"
DESCRIPTION = (
    "Fits a smooth curve (spline) for each predictor in the spline table, and a straight-line effect "
    "for every other predictor. Blank numbers are filled with the training median first."
)
ONE_HOT_ENCODE = True
SCALE_NUMERIC = True  # As in the notebook's validation of this model (normalize_numeric=True): numerical columns 0-1.

PARAMETERS = {
    "spline_terms": {
        "label": "Splines (one row per predictor; every other predictor gets a linear term)",
        "type": "table",
        "columns": {
            "column": {"label": "Predictor", "type": "predictor",
                       "help": "Any numerical or date predictor chosen in section 2."},
            "n_splines": {"label": "Number of splines", "type": "int", "min": 1,
                          "help": "More splines allow a wigglier curve. Must be more than the spline order."},
            "spline_order": {"label": "Spline order", "type": "int", "min": 0,
                             "help": "1 = straight pieces, 3 = smooth curves."},
            "lam": {"label": "Smoothing (lam)", "type": "float", "min": 0.0, "step": 0.1, "format": "%.4f",
                    "help": "Higher values make the curve smoother. Multiplied by the lam scale below."},
            "basis": {"label": "Basis", "type": "choice", "options": ["ps", "cp"],
                      "help": "ps = P-spline; cp = cyclic P-spline (the curve's ends meet)."},
        },
        # The notebook's gam_spline_terms.
        "default": [
            {"column": "Latitude", "n_splines": 20, "spline_order": 3, "lam": 5.0, "basis": "ps"},
            {"column": "Longitude", "n_splines": 20, "spline_order": 3, "lam": 5.0, "basis": "ps"},
            {"column": "Yard Size (AC)", "n_splines": 5, "spline_order": 3, "lam": 5.0, "basis": "ps"},
            {"column": "Lease Building (SF)", "n_splines": 5, "spline_order": 3, "lam": 5.0, "basis": "ps"},
            {"column": "Year Built", "n_splines": 5, "spline_order": 1, "lam": 5.0, "basis": "ps"},
            {"column": "Lease Start", "n_splines": 10, "spline_order": 1, "lam": 5.0, "basis": "ps"},
            {"column": "Lease Length (Months)", "n_splines": 10, "spline_order": 3, "lam": 5.0, "basis": "ps"},
            {"column": "Sale Price", "n_splines": 5, "spline_order": 1, "lam": 5.0, "basis": "ps"},
        ],
        "help": "Add a row to give any numerical or date predictor a spline, or delete a row to make it linear. "
                "Notebook: gam_spline_terms.",
    },
    "linear_lam": {
        "label": "Linear term penalty (linear_lam)", "type": "float", "default": 1e-6, "min": 0.0, "step": 1e-6,
        "format": "%.1e",
        "help": "Penalty on every predictor without a spline. 0 leaves them unpenalized. Notebook: 1e-6.",
    },
    "lam_scale": {
        "label": "Lam scale", "type": "float", "default": 1.0, "min": 0.0, "step": 0.1, "format": "%.4f",
        "help": "Multiplies every spline's lam. Notebook: chosen from 0.1, 1, or 10 by cross-validation inside "
                "each fit; 1.0 uses each spline's lam as entered.",
    },
    "max_iter": {
        "label": "Maximum iterations", "type": "int", "default": 100, "min": 1,
        "help": "Notebook: 100.",
    },
}


def fit_and_predict(X_train, y_train, X_test, spline_terms, linear_lam, lam_scale, max_iter):
    """Returns (predictions, the term used for each predictor)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    from pygam import LinearGAM, l, s
    from sklearn.impute import SimpleImputer

    # The table's rows become the notebook's gam_spline_terms: {predictor: {setting: value, ...}}.
    splines = {}
    for row in spline_terms:
        if row["column"] in splines:
            raise ValueError(f"{row['column']} has more than one row in the spline table.")
        splines[row["column"]] = {setting: value for setting, value in row.items() if setting != "column"}

    unknown = set(splines) - set(X_train.columns)
    if unknown:
        raise KeyError(f"Spline columns not found among the predictors: {sorted(unknown)}")

    # The notebook's pipeline: a median imputer, then GAMRegressor.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_fit = imputer.fit_transform(X_train)
    X_new = imputer.transform(X_test)

    # GAMRegressor.fit: a spline for each column in the table, a linear term for every other column.
    terms = None
    for index, name in enumerate(X_train.columns):
        if name in splines:
            settings = dict(splines[name])
            settings["lam"] = np.multiply(settings.get("lam", 0.6), lam_scale).tolist()
            term = s(index, **settings)
        else:
            # A zero penalty leaves linear terms unpenalized.
            term = l(index, lam=linear_lam or 0.0)
        terms = term if terms is None else terms + term

    gam = LinearGAM(terms=terms, max_iter=max_iter).fit(X_fit, np.asarray(y_train, dtype=float).reshape(-1))
    predictions = np.asarray(gam.predict(X_new), dtype=float).reshape(-1)

    # Details: each spline with the settings it used (lam after the lam scale), then the linear terms.
    rows = []
    for name, settings in splines.items():
        used = {**settings, "lam": float(np.multiply(settings.get("lam", 0.6), lam_scale))}
        rows.append({"Predictor": name, "Term": "Spline", "Settings": ", ".join(f"{k}={v}" for k, v in used.items())})
    rows.append({"Predictor": f"The other {X_train.shape[1] - len(splines)} predictors", "Term": "Linear",
                 "Settings": f"lam={linear_lam or 0.0:g}"})
    return predictions, pd.DataFrame(rows)
