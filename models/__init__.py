"""The models the Predictions page can use.

The page builds its model dropdown and parameter boxes from MODEL_FILES,
so adding a model never changes the page.

HOW TO ADD A MODEL
  1. Make a new file in this folder (copy the outline below, or the closest existing model file).
  2. Import it below and add it to MODEL_FILES.
  3. Add any new packages it needs to requirements.txt.

OUTLINE: every model file has the same parts, in the same order

    \"\"\"<Model> (<library>), translated from <Notebook>.ipynb.

    Translates <the notebook's single-configuration function>, fitted once with the parameters below.
    Defaults: <where they came from, e.g. the lowest-RMSE settings from your grid search>.
    \"\"\"

    NAME = "Random Forest"                # shown in the dropdown
    DESCRIPTION = "What it does."         # shown under the dropdown
    ONE_HOT_ENCODE = True                 # False: categories arrive as pandas categories instead of
                                          #        one-hot columns (for models that handle categories themselves)
    SCALE_NUMERIC = False                 # True: numerical columns are scaled to 0-1 first
    PARAMETERS = {                        # one box on the page per entry, in this order
        "n_estimators": {"label": "Number of trees", "type": "int", "default": 500, "min": 1,
                         "help": "What it does. Notebook: 500."},
    }

    def fit_and_predict(X_train, y_train, X_test, n_estimators):
        from sklearn.ensemble import RandomForestRegressor   # model libraries load here
        ...                               # fit on X_train / y_train
        return predictions                # one per X_test row, same order
                                          # (or return predictions, details_table to show a table under "Model details")

PARAMETER TYPES
  "int", "float"   a number box. Optional: "min", "max", "step", "format" (e.g. "%.4f", "%.1e"), "help"
  "bool"           a checkbox
  "choice"         a dropdown. Needs "options": [...]
  "table"          an editable table with one row per item, where rows can be added and deleted
                   (e.g. GAM splines, network layers). Needs:
                     "columns": {name: {"label": ..., "type": "int" | "float" | "choice" | "predictor", ...}}
                         "predictor" is a dropdown of the numerical and date predictors chosen on the page
                     "default": [{name: value, ...}, ...]   one dictionary per starting row
                   fit_and_predict receives the rows as a list of dictionaries, e.g.
                   [{"neurons": 400, "dropout_rate": 0.2}, ...]

X_train and X_test arrive already prepared by models/shared.py, with identical columns.
"""

from models import best_subset_linear, gam, lasso, neural_network, neural_network_keras, ridge

MODEL_FILES = [
    best_subset_linear,
    lasso,
    ridge,
    gam,
    neural_network,
    neural_network_keras,
]

MODELS = {model.NAME: model for model in MODEL_FILES}
