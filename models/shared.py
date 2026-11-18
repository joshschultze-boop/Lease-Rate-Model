"""What every model shares: reading the CSVs, preparing the data, fitting, and measuring errors.

The Predictions page calls run_model(), which prepares the data the same way for every model
and then hands it to the model you picked. Model files only say how they want the data
(ONE_HOT_ENCODE, SCALE_NUMERIC) and contain the model itself.
"""

from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


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
              target_column, min_frequency):
    """
    Everything between the CSVs and the model:
      1. Training rows need a target value.
      2. Keep the predictor columns. Number and date columns become numbers
         (dates are already "days since the start date" chosen on the Preprocess page).
      3. Prepare the data, learned from the training rows only, the way the model asks for it.
      4. Fit the model and predict the test rows.
    Returns a dictionary with the predictions, the model's details table (or None), and counts.
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

    result = model.fit_and_predict(X_train, y_train, X_test, **parameters)
    predictions, details = result if isinstance(result, tuple) else (result, None)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(X_test):
        raise ValueError(f"The model returned {len(predictions)} predictions for {len(X_test)} test rows.")

    return {
        "predictions": predictions,
        "details": details,
        "training_rows": len(y_train),
        "skipped_training_rows": int((~has_target).sum()),
        "model_predictors": X_train.shape[1],
    }


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
