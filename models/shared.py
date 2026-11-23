"""What every model shares: reading the CSVs, preparing the data, fitting, measuring errors, and column weights.

The Predictions page calls run_model(), which prepares the data the same way for every model
and then hands it to the model you picked. Model files only say how they want the data
(ONE_HOT_ENCODE, SCALE_NUMERIC) and contain the model itself.
"""

from io import BytesIO
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler


# ---------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------

def read_csv_as_text(source):
    """
    Read a CSV keeping every value as the text written in the file (numbers are converted later),
    so a column like Zip can't be read as numbers in one file and as text in another.
    source can be an uploaded file, a path, or a DataFrame (written out as CSV first,
    so it's read exactly like a downloaded CSV would be).
    """
    if isinstance(source, pd.DataFrame):
        source = BytesIO(source.to_csv(index=False).encode("utf-8"))
    data = pd.read_csv(source, dtype=str)
    data.columns = [str(column).strip() for column in data.columns]
    return data


def numbers_where_possible(table):
    """Copy of a table where every column that is entirely numbers is stored as numbers (for display and sorting)."""
    table = table.copy()
    for column in table.columns:
        converted = pd.to_numeric(table[column], errors="coerce")
        if converted.notna().sum() == table[column].notna().sum():
            table[column] = converted
    return table


# ---------------------------------------------------------------------
# Preparing data (DataFramePreprocessor from Best Subset - Linear.ipynb)
# ---------------------------------------------------------------------
# Fit on the TRAINING rows, then transform both training and test rows. It learns:
#   - which categories appear at least min_frequency times (rarer ones become "unknown")
#   - the columns to create, so test rows always get exactly the same columns
#   - the 0-1 scaling of the numerical columns (only when normalize_numeric=True)
# Categories in the test data that weren't in training are treated as "unknown".
# one_hot=False keeps each categorical column as a pandas category (same category list for
# training and test) instead of one-hot columns, for models that handle categories themselves.

class DataFramePreprocessor:

    def __init__(self, categorical_columns, numerical_columns=None, normalize_numeric=True, min_frequency=3, one_hot=True):
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns or [])
        self.normalize_numeric = normalize_numeric
        self.min_frequency = min_frequency
        self.one_hot = one_hot

    def _clean_categories(self, df, fitting=False):
        cleaned = df[self.categorical_columns].copy()
        for column in self.categorical_columns:
            values = cleaned[column].fillna("unknown").astype(str)
            # "80202.0" and "80202" are the same category (Excel sometimes adds .0 to whole numbers).
            values = values.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
            if fitting:
                counts = values.value_counts()
                self.kept_categories_[column] = set(counts[counts >= self.min_frequency].index) | {"unknown"}
            cleaned[column] = values.where(values.isin(self.kept_categories_[column]), "unknown")
        return cleaned

    def fit(self, df):
        self.input_columns_ = list(df.columns)
        missing_columns = [c for c in self.categorical_columns + self.numerical_columns if c not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns: {missing_columns}")

        self.kept_categories_ = {}
        categorical_data = self._clean_categories(df, fitting=True)

        if self.one_hot:
            categories = [sorted(self.kept_categories_[column]) for column in self.categorical_columns]
            self.encoder_ = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
            self.encoder_.fit(categorical_data)
            self.encoded_columns_ = list(self.encoder_.get_feature_names_out(self.categorical_columns))
        else:
            self.encoded_columns_ = list(self.categorical_columns)

        self.other_columns_ = [column for column in self.input_columns_ if column not in self.categorical_columns]
        if self.normalize_numeric and self.numerical_columns:
            self.scaler_ = MinMaxScaler().fit(df[self.numerical_columns])

        # Where each prepared column came from: {prepared column: (predictor column, category or None)}.
        # The column weights use this to add a predictor's one-hot columns back together.
        self.column_sources_ = {column: (column, None) for column in self.other_columns_}
        if self.one_hot:
            sources = [(column, category) for column, categories in zip(self.categorical_columns, self.encoder_.categories_)
                       for category in categories]
            self.column_sources_.update(zip(self.encoded_columns_, sources))
        else:
            self.column_sources_.update({column: (column, None) for column in self.categorical_columns})

        self.output_columns_ = self.other_columns_ + self.encoded_columns_
        return self

    def transform(self, df):
        if not hasattr(self, "kept_categories_"):
            raise ValueError("Fit the preprocessor before transforming.")
        missing_columns = [column for column in self.input_columns_ if column not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns: {missing_columns}")

        df = df[self.input_columns_].copy()
        categorical_data = self._clean_categories(df)
        if self.one_hot:
            encoded_df = pd.DataFrame(self.encoder_.transform(categorical_data), columns=self.encoded_columns_, index=df.index)
        else:
            encoded_df = pd.DataFrame({
                column: pd.Categorical(categorical_data[column], categories=sorted(self.kept_categories_[column]))
                for column in self.categorical_columns
            }, index=df.index)

        processed_df = df[self.other_columns_].copy()
        if self.normalize_numeric and self.numerical_columns:
            processed_df[self.numerical_columns] = self.scaler_.transform(df[self.numerical_columns])

        processed_df = pd.concat([processed_df, encoded_df], axis=1)
        return processed_df.reindex(columns=self.output_columns_, fill_value=0.0)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------
# Fitting and predicting (the same for every model)
# ---------------------------------------------------------------------

def run_model(model, parameters, training, test, numerical_columns, date_columns, categorical_columns,
              target_column, min_frequency, random_state):
    """
    Everything between the CSVs and the model:
      1. Training rows need a target value.
      2. Keep the predictor columns. Number and date columns become numbers
         (dates are already "days since the start date" chosen on the Preprocess page).
      3. Prepare the data, learned from the training rows only, the way the model asks for it.
      4. Fit the model and predict the test rows, with the page's random seed (random_state).
      5. If the model returned linear_terms() (the linear models do), work out the column weights.
    Returns a dictionary with the predictions, the model's details table (or None), the column
    weights (or None), and counts.
    """
    def predictors(table):
        X = table[numerical_columns + date_columns + categorical_columns].copy()
        for column in numerical_columns + date_columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        return X

    target = pd.to_numeric(training[target_column], errors="coerce")
    has_target = target.notna()
    y_train = target[has_target].reset_index(drop=True)

    preprocessor = DataFramePreprocessor(
        categorical_columns, numerical_columns, normalize_numeric=model.SCALE_NUMERIC,
        min_frequency=min_frequency, one_hot=model.ONE_HOT_ENCODE,
    )
    X_train = preprocessor.fit_transform(predictors(training[has_target].reset_index(drop=True)))
    X_test = preprocessor.transform(predictors(test))

    result = model.fit_and_predict(X_train, y_train, X_test, random_state=random_state, **parameters)
    # A model returns predictions, or (predictions, details table), or (predictions, details table, linear_terms(...)).
    result = result if isinstance(result, tuple) else (result,)
    predictions, details, terms = (result + (None, None))[:3]
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(X_test):
        raise ValueError(f"The model returned {len(predictions)} predictions for {len(X_test)} test rows.")

    weights = None
    if terms is not None:
        weights = column_weights(terms, preprocessor, X_train, numerical_columns, date_columns, categorical_columns,
                                 scale_numeric=model.SCALE_NUMERIC)

    return {
        "predictions": predictions,
        "details": details,
        "column_weights": weights,
        "training_rows": len(y_train),
        "skipped_training_rows": int((~has_target).sum()),
        "model_predictors": X_train.shape[1],
    }


# ---------------------------------------------------------------------
# Model details (shared by the linear models, so their tables look the same)
# ---------------------------------------------------------------------

def coefficient_table(predictors, coefficients, intercept, keep_zeros=False):
    """
    The intercept, then each predictor's coefficient, largest effect first.
    Coefficients are per 1 standard deviation of each predictor (the models standardize first),
    so they can be compared with each other. Zero coefficients are left out unless keep_zeros=True.
    """
    coefficient = "Coefficient (per 1 standard deviation)"
    table = pd.DataFrame({"Predictor": list(predictors), coefficient: np.ravel(coefficients).astype(float)})
    if not keep_zeros:
        table = table.loc[table[coefficient] != 0]
    table = table.sort_values(coefficient, key=abs, ascending=False)
    intercept_row = pd.DataFrame({"Predictor": ["(intercept)"], coefficient: [float(np.ravel(intercept)[0])]})
    return pd.concat([intercept_row, table], ignore_index=True)


def importance_table(predictors, importances, label="Importance (% of total)"):
    """For the tree models: each predictor's share of the model's total importance, largest first (zeros left out)."""
    importances = np.ravel(importances).astype(float)
    total = importances.sum()
    shares = importances / total * 100 if total > 0 else np.zeros_like(importances)
    table = pd.DataFrame({"Predictor": list(predictors), label: shares})
    table = table.loc[table[label] > 0].sort_values(label, ascending=False, kind="stable")
    return table.reset_index(drop=True)


def network_summary(layer_settings, history, early_stopping):
    """For the neural networks: the layers used and how training went (from the Keras fit history)."""
    rows = [
        ("Hidden layers: neurons (dropout)", ", ".join(f"{n} ({d:g})" for n, d in layer_settings) or "none"),
        ("Epochs trained", len(history.history["loss"])),
    ]
    if early_stopping:
        rows.append(("Lowest validation loss at epoch", int(np.argmin(history.history["val_loss"])) + 1))
    return pd.DataFrame({"Setting": [name for name, _ in rows], "Value": [str(value) for _, value in rows]})


# ---------------------------------------------------------------------
# Column weights (the linear models: best subset linear and polynomial, lasso, ridge)
# ---------------------------------------------------------------------
# A linear model's prediction is the intercept plus, for every term, coefficient x standardized value.
# linear_terms() (called in the model file) splits each training prediction into what every prepared
# column adds. column_weights() (called by run_model) adds those back up into the predictor columns
# chosen on the page, so City's one-hot columns count together as City.

def _shapley_split(values, averages, powers):
    """
    Split a polynomial term (the product of values[:, j] ** powers[j]) between its columns, row by row:
    Shapley values measured from the average property. Each column gets the part of the term's change
    that comes from that column moving away from its average, so a column that barely varies gets
    little of the term. The pieces add up to term(row) - term(averages).
    """
    count = len(powers)

    def term_with(moved):   # the term with the columns in `moved` at the row's values, the rest at their averages
        return np.prod([values[:, j] ** powers[j] if j in moved else np.full(len(values), averages[j] ** powers[j])
                        for j in range(count)], axis=0)

    pieces = np.zeros((len(values), count))
    for j in range(count):
        others = [i for i in range(count) if i != j]
        for size in range(count):
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            for moved in combinations(others, size):
                pieces[:, j] += weight * (term_with(set(moved) | {j}) - term_with(set(moved)))
    return pieces


def linear_terms(model, X_train):
    """
    What each prepared column adds to each training prediction, for the column weights.
    model is the fitted Pipeline: imputer, (PolynomialFeatures), StandardScaler, then a linear model with coef_.
    A polynomial term made from several columns (e.g. Latitude x Longitude) is split between them with
    _shapley_split(), so each column gets the part of the term it actually moves.
    Returns {"contributions": a table with one row per training row and one column per prepared column
             (in target units; each row adds up to that row's prediction minus the intercept),
             "terms": every term with a nonzero coefficient, plus every single-column term}.
    """
    columns = list(X_train.columns)
    positions = {type(step): position for position, (_, step) in enumerate(model.steps[:-1])}
    polynomial = model[positions[PolynomialFeatures]] if PolynomialFeatures in positions else None
    scaler = model[positions[StandardScaler]]
    coefficients = np.ravel(model[-1].coef_).astype(float)
    if polynomial is None:
        names, powers = columns, np.eye(len(columns), dtype=int)
    else:
        names, powers = list(polynomial.get_feature_names_out(columns)), np.asarray(polynomial.powers_)

    standardized = np.asarray(model[:-1].transform(X_train), dtype=float)   # every term, as the model sees it
    contributions = np.zeros((len(X_train), len(columns)))
    if polynomial is not None:
        filled = np.asarray(model[:positions[PolynomialFeatures]].transform(X_train), dtype=float)  # blanks filled
        averages = filled.mean(axis=0)
    for term in np.flatnonzero(coefficients != 0):
        added = coefficients[term] * standardized[:, term]       # what this term adds to each prediction
        parts = np.flatnonzero(powers[term])
        if len(parts) == 1:
            contributions[:, parts[0]] += added
            continue
        pieces = coefficients[term] / scaler.scale_[term] * _shapley_split(filled[:, parts], averages[parts], powers[term, parts])
        # What's left is the same for every row (the term at the average property, minus the term's mean);
        # it's shared by powers so each row still adds up to its prediction. It doesn't change any weight.
        leftover = (added - pieces.sum(axis=1))[:, None] * (powers[term, parts] / powers[term, parts].sum())
        contributions[:, parts] += pieces + leftover

    used = coefficients != 0
    single = powers.sum(axis=1) == 1        # the term is one prepared column (to the power 1)
    keep = used | single
    return {
        "contributions": pd.DataFrame(contributions, columns=columns, index=X_train.index),
        "terms": pd.DataFrame({
            "Term": np.asarray(names, dtype=object)[keep],
            "Coefficient (per 1 standard deviation)": coefficients[keep],
            # Undo the standardizing: the change in the prediction for +1 of the column itself.
            "Effect of +1 unit": np.where(single, coefficients / scaler.scale_, np.nan)[keep],
            "Columns": [[columns[j] for j in np.flatnonzero(row)] for row in powers[keep]],
        }),
    }


def column_weights(terms, preprocessor, X_train, numerical_columns, date_columns, categorical_columns, scale_numeric):
    """
    The column weights shown on the Predictions page, from linear_terms(). Returns
      "overview": one row per predictor column, largest weight first
      "details":  {predictor column: {"type", "main": its own term(s), "other": polynomial terms that include it}}
    Weight = the standard deviation, across the training rows, of everything a predictor column adds to the
    prediction (its one-hot columns and its share of polynomial terms added together), in target units.
    For a numerical column used once, that's the size of its coefficient per 1 standard deviation.
    """
    sources = preprocessor.column_sources_
    contributions, table = terms["contributions"], terms["terms"]
    coefficient, per_unit = "Coefficient (per 1 standard deviation)", "Effect of +1 unit"
    kinds = {**{column: "Numerical" for column in numerical_columns}, **{column: "Date" for column in date_columns},
             **{column: "Categorical" for column in categorical_columns}}

    rows, details = [], {}
    for column, kind in kinds.items():
        prepared = [name for name, (source, _) in sources.items() if source == column]
        weight = float(contributions[prepared].sum(axis=1).std(ddof=0)) if prepared else 0.0
        mine = table.loc[table["Columns"].map(lambda names: any(sources[name][0] == column for name in names))]
        main = mine.loc[mine[per_unit].notna()]                  # the column's own term(s)
        other = mine.loc[mine[per_unit].isna()]                  # polynomial terms that include it
        is_used = bool((mine[coefficient] != 0).any())
        # Numbers scaled to 0-1 on the page (SCALE_NUMERIC) have no "+1 unit" in their own units.
        unit_known = kind == "Date" or (kind == "Numerical" and not scale_numeric)

        if kind == "Categorical":
            details[column] = {"type": kind, "main": pd.DataFrame({
                "Category": [sources[names[0]][1] for names in main["Columns"]],
                "Effect of being in this category": main[per_unit].to_numpy(),
                coefficient: main[coefficient].to_numpy(),
                "Training rows": [int((X_train[names[0]] == 1).sum()) for names in main["Columns"]],
            }).sort_values("Effect of being in this category", ascending=False, kind="stable").reset_index(drop=True)}
        else:
            details[column] = {"type": kind, "main": pd.DataFrame({
                "Term": main["Term"].to_numpy(),
                per_unit: main[per_unit].to_numpy() if unit_known else np.nan,
                coefficient: main[coefficient].to_numpy(),
            })}
        details[column]["other"] = (other[["Term", coefficient]]
                                    .rename(columns={"Term": "Polynomial term"})
                                    .sort_values(coefficient, key=abs, ascending=False, kind="stable")
                                    .reset_index(drop=True)) if len(other) else None

        simple = kind != "Categorical" and len(main) == 1 and other.empty
        if not is_used:
            direction = "Not used (coefficient 0)"
        elif kind == "Categorical":
            direction = "Depends on the category"
        elif not simple:
            direction = "Curved or combined with other columns"
        else:
            direction = "Higher value raises the prediction" if main[coefficient].iloc[0] > 0 else "Higher value lowers the prediction"
        rows.append({
            "Column": column, "Type": kind, "Weight": weight,
            per_unit: float(main[per_unit].iloc[0]) if simple and is_used and unit_known else np.nan,
            "Direction": direction,
        })

    overview = pd.DataFrame(rows)
    total = overview["Weight"].sum()
    overview.insert(3, "Share of total weight (%)", overview["Weight"] / total * 100 if total > 0 else 0.0)
    overview = overview.sort_values("Weight", ascending=False, kind="stable").reset_index(drop=True)
    return {"overview": overview, "details": details}


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

def calculate_metrics(actual, predicted):
    """
    RMSE and proportional error over the rows that have an actual value.
    Proportional error = average of |predicted - actual| / |actual| (rows with actual = 0 are skipped).
    """
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    has_actual = ~np.isnan(actual)
    errors = predicted[has_actual] - actual[has_actual]
    nonzero = ~np.isclose(actual[has_actual], 0)
    return {
        "Observations scored": int(has_actual.sum()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))) if has_actual.any() else np.nan,
        "Proportional Error": (float(np.mean(np.abs(errors[nonzero]) / np.abs(actual[has_actual][nonzero])))
                               if nonzero.any() else np.nan),
    }


def add_error_columns(results, actual_column, predicted_column):
    """Add Absolute Error and Proportional Error (%) for each row that has an actual value."""
    results = results.copy()
    actual = pd.to_numeric(results[actual_column], errors="coerce")
    results["Absolute Error"] = (results[predicted_column] - actual).abs()
    results["Proportional Error (%)"] = results["Absolute Error"] / actual.abs().replace(0, np.nan) * 100
    return results
