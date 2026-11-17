"""The models the Predictions page can use.

The page builds its model dropdown and parameter boxes from MODEL_FILES,
so adding a model never changes the page.

HOW TO ADD A MODEL
  1. Make a new file in this folder (copy the outline below, or best_subset_linear.py).
  2. Import it below and add it to MODEL_FILES.
  3. Add any new packages it needs to requirements.txt.

OUTLINE: every model file has these six things

    NAME = "Random Forest"                # shown in the dropdown
    DESCRIPTION = "What it does."         # shown under the dropdown
    ONE_HOT_ENCODE = True                 # False: categories arrive as pandas categories instead of
                                          #        one-hot columns (for models that handle categories themselves)
    SCALE_NUMERIC = False                 # True: numerical columns are scaled to 0-1 first
    PARAMETERS = {                        # one box on the page per entry
        "n_estimators": {"label": "Number of trees", "type": "int", "default": 500, "min": 1},
    }                                     # types: "int", "float", "bool", "choice" (choice also needs "options": [...])
                                          # optional: "min", "max", "step", "format", "help"

    def fit_and_predict(X_train, y_train, X_test, n_estimators):
        ...                               # fit on X_train / y_train
        return predictions                # one per X_test row, same order
                                          # (or return predictions, details_table to show a table under "Model details")

X_train and X_test arrive already prepared by models/shared.py, with identical columns.
"""

from models import best_subset_linear

MODEL_FILES = [
    best_subset_linear,
]

MODELS = {model.NAME: model for model in MODEL_FILES}
