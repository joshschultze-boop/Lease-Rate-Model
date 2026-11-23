"""BART: Bayesian Additive Regression Trees (pymc-bart), translated from BART.ipynb.

Translates bart_grid_predictions (the PyMC-BART version), fitted once with the parameters below (no tuning).
Defaults: the notebook function's defaults for the trees, and the notebook's FIXED_PARAMS for the sampler.
Your grid search results weren't saved in the notebook, so change the defaults below to your lowest-RMSE
settings once you have them.
Needs pymc-bart 0.12.0, the last pure-Python release (0.13 and later use a different backend).
"""

import numpy as np

from models.shared import importance_table

NAME = "BART (Bayesian Additive Regression Trees)"
DESCRIPTION = (
    "Adds up many small trees, each explaining a little of the target. A Bayesian prior keeps the trees small, "
    "and the prediction averages many sampled versions of the model (MCMC). Much slower than the other models: "
    "expect a minute or more per fit, longer the first time. Blank numbers are filled with the training median."
)
ONE_HOT_ENCODE = True  # As in the notebook's grid search.
SCALE_NUMERIC = False  # As in the notebook's grid search. Trees don't need scaling.
SEED_USED_FOR = "the MCMC sampler"

PARAMETERS = {
    "m": {
        "label": "Number of trees", "type": "int", "default": 50, "min": 1,
        "help": "More trees give smoother, more flexible fits but are slower. "
                "Notebook: 50 (grid searched 25 to 500).",
    },
    "alpha": {
        "label": "Alpha (tree prior: willingness to split)", "type": "float", "default": 0.95,
        "min": 0.01, "max": 0.99, "step": 0.01, "format": "%.2f",
        "help": "A node at depth d splits with probability alpha x (1 + d)^-beta. Higher alpha means more "
                "splitting. Notebook: 0.95 (grid searched 0.90 and 0.95).",
    },
    "beta": {
        "label": "Beta (tree prior: push toward shallow trees)", "type": "float", "default": 2.0,
        "min": 0.5, "step": 0.5, "format": "%.2f",
        "help": "Higher beta makes deep splits much less likely, so trees stay shallow. The minimum is 0.5: "
                "near 0 the trees never stop growing and the fit doesn't finish. "
                "Notebook: 2.0 (grid searched 1.0 and 2.0).",
    },
    "draws": {
        "label": "Draws (saved samples per chain)", "type": "int", "default": 250, "min": 1,
        "help": "The prediction averages this many sampled models per chain. More is steadier but slower. "
                "Notebook: 250.",
    },
    "tune": {
        "label": "Tune (warm-up samples per chain)", "type": "int", "default": 250, "min": 0,
        "help": "Warm-up samples that are thrown away while the trees settle. Notebook: 250.",
    },
    "chains": {
        "label": "Chains", "type": "int", "default": 2, "min": 1,
        "help": "Independent runs of the sampler, averaged together. Each one adds time. Notebook: 2.",
    },
}


def fit_and_predict(X_train, y_train, X_test, random_state, m, alpha, beta, draws, tune, chains):
    """Returns (predictions, each predictor's share of the trees' splits)."""
    # Model libraries load here, so a problem with one model's library can't affect the others.
    import logging

    import pandas as pd
    import pymc as pm
    import pymc_bart as pmb
    from numba import njit
    from sklearn.impute import SimpleImputer

    logging.getLogger("pymc").setLevel(logging.ERROR)   # keep the sampler's progress messages out of the logs

    # pymc-bart 0.12 grows its trees with NumPy's global random numbers (and Numba's for one step), which
    # pm.sample's random_seed doesn't reach. Seeding both is what makes a typed-in seed repeat a run exactly.
    @njit
    def seed_numba(seed):
        np.random.seed(seed)

    np.random.seed(random_state)
    seed_numba(random_state)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tr = imputer.fit_transform(np.asarray(X_train, dtype=float))
    X_te = imputer.transform(np.asarray(X_test, dtype=float))
    y_tr = np.asarray(y_train, dtype=float).ravel()

    with pm.Model():
        X_data = pm.Data("X", X_tr)
        mu = pmb.BART("mu", X_data, y_tr, m=m, alpha=alpha, beta=beta)
        sigma = pm.HalfNormal("sigma", sigma=y_tr.std())
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_tr, shape=mu.shape)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            var_names=["sigma"],   # the trees are kept by BART itself, so mu isn't stored
            random_seed=random_state,
            progressbar=False,
            compute_convergence_checks=False,
        )

        # Predict the test rows from the sampled trees, then average over every draw.
        pm.set_data({"X": X_te})
        posterior = pm.sample_posterior_predictive(idata, var_names=["mu"], random_seed=random_state, progressbar=False)

    predictions = np.asarray(posterior.posterior_predictive["mu"].mean(("chain", "draw")).values, dtype=float).reshape(-1)

    # Variable inclusion: each predictor's share of all the splits in the sampled trees.
    try:
        shares, names = pmb.get_variable_inclusion(idata, pd.DataFrame(X_tr, columns=list(X_train.columns)))
        details = importance_table(names, shares, label="Share of tree splits (%)")
    except Exception:   # the details table is optional; never let it stop the predictions
        details = None
    return predictions, details
