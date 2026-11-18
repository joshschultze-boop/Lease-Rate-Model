# pages/4_Predictions.py
# Fits a model on the training CSV and predicts the test data:
#   1-2. Upload the training CSV and choose the predictor and target columns from its headers
#   3.   Choose the test data (from the Test Data page, or an uploaded CSV with or without rent)
#   4.   Choose a model and its parameters, then fit and predict
#   5.   Results: errors (when the test data has actual rent), column checkboxes, and a download
#
# Preparing the data and fitting happen in models/shared.py, the same way for every model.
# The models are listed in models/__init__.py. Adding a model doesn't change this page.

import pandas as pd
import streamlit as st

from models import MODELS
from models.shared import add_error_columns, calculate_metrics, numbers_where_possible, read_csv_as_text, run_model

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

# Starting column choices (the lists from Best Subset - Linear.ipynb). Only those in the CSV are pre-selected.
DEFAULT_NUMERICAL = [
    "Latitude", "Longitude", "Yard Size (AC)", "Lease Building (SF)", "yard / prop ratio (acres)",
    "yard / prop ratio (sf)", "Year Built", "Sale Price", "Distance to Interstate", "Sale Information Unknown",
    "Lease Length Unknown", "Lease Length (Months)", "Yard Size (AC)_log1p", "Lease Building (SF)_log1p",
]
DEFAULT_DATE = ["Lease Start", "Sale Date"]
DEFAULT_CATEGORICAL = ["City", "City-Zip", "County", "CBSA", "Lease Type", "County-Zoning", "City-Zoning", "NNN"]
DEFAULT_TARGET = "Monthly Rent"

# Columns checked to start with in the results (others can be checked below the results).
SHOWN_TO_START = ["Full Address", "Predicted {target}", "Actual {target}"]

FROM_TEST_DATA_PAGE = "Test data from the Test Data page"
UPLOAD_TEST_CSV = "Upload a test CSV"


def parameter_box(model_name, name, spec):
    """One parameter box, built from an entry in the model file's PARAMETERS."""
    key, label, help_text = f"{model_name}: {name}", spec["label"], spec.get("help")  # each model keeps its own values
    if spec["type"] == "int":
        return int(st.number_input(label, value=int(spec["default"]), step=int(spec.get("step", 1)),
                                   min_value=spec.get("min"), max_value=spec.get("max"), help=help_text, key=key))
    if spec["type"] == "float":
        as_float = lambda value: None if value is None else float(value)
        return float(st.number_input(label, value=float(spec["default"]), step=float(spec.get("step", 0.01)),
                                     min_value=as_float(spec.get("min")), max_value=as_float(spec.get("max")),
                                     format=spec.get("format", "%.4f"), help=help_text, key=key))
    if spec["type"] == "bool":
        return st.checkbox(label, value=bool(spec["default"]), help=help_text, key=key)
    if spec["type"] == "choice":
        return st.selectbox(label, spec["options"], index=spec["options"].index(spec["default"]), help=help_text, key=key)
    raise ValueError(f"Unknown parameter type for {name}: {spec['type']}")


def choose_data_and_model():
    """Sections 1-4. Returns the choices, or None (with a message) if something is still missing."""

    st.header("1. Training data")
    training_file = st.file_uploader("Preprocessed training CSV (from the Preprocess page)", type=["csv"])
    if training_file is None:
        st.info("Upload the training CSV to continue.")
        return None
    training = read_csv_as_text(training_file)
    headers = list(training.columns)
    st.caption(f"{len(training)} rows and {len(headers)} columns.")

    st.header("2. Columns")
    st.write("Choose the columns the model uses, from the training CSV's headers. Each column can be in only one list.")
    left, right = st.columns(2)
    defaults = lambda columns: [column for column in columns if column in headers]
    numerical = left.multiselect("Numerical predictors", headers, default=defaults(DEFAULT_NUMERICAL))
    dates = left.multiselect("Date predictors (days since 1950-01-01)", headers, default=defaults(DEFAULT_DATE))
    categorical = right.multiselect("Categorical predictors", headers, default=defaults(DEFAULT_CATEGORICAL))
    target = right.selectbox("Target column (what to predict)", headers, placeholder="Choose the target",
                             index=headers.index(DEFAULT_TARGET) if DEFAULT_TARGET in headers else None)
    predictors = numerical + dates + categorical

    in_two_lists = sorted({column for column in predictors if predictors.count(column) > 1})
    problems = [message for failed, message in [
        (not predictors, "Choose at least one predictor."),
        (target is None, "Choose a target column."),
        (in_two_lists, "These columns are in more than one list: " + ", ".join(in_two_lists)),
        (target in predictors, f"The target ({target}) is also chosen as a predictor."),
    ] if failed]
    for problem in problems:
        st.error(problem)
    if problems:
        return None
    made_from_target = [column for column in predictors if target in column]
    if made_from_target:
        st.warning("These predictors look like they're made from the target, which would give the model the answer: "
                   + ", ".join(made_from_target))
    st.caption(f"{len(numerical)} numerical, {len(dates)} date, and {len(categorical)} categorical predictors.")

    st.header("3. Test data")
    sources = ([FROM_TEST_DATA_PAGE] if "test_data" in st.session_state else []) + [UPLOAD_TEST_CSV]
    if st.radio("Test data", sources, horizontal=True) == FROM_TEST_DATA_PAGE:
        test = read_csv_as_text(st.session_state["test_data"])
        properties = st.session_state["test_data_properties"].reset_index(drop=True)
        test_name = st.session_state["test_data_run"]["file_name"]
    else:
        test_file = st.file_uploader("Preprocessed test CSV (with or without the target column)", type=["csv"])
        if test_file is None:
            st.info("Upload a test CSV to continue, or make test data on the Test Data page.")
            return None
        test, properties, test_name = read_csv_as_text(test_file), None, test_file.name

    missing = [column for column in predictors if column not in test.columns]
    if missing:
        st.error("The test data is missing these predictor columns: " + ", ".join(missing))
        return None
    has_actual = target in test.columns and pd.to_numeric(test[target], errors="coerce").notna().any()
    st.caption(f"{len(test)} rows. " + (f"The test data has actual {target}, so errors will be calculated." if has_actual
                                        else f"There's no actual {target} in the test data, so only predictions are made."))

    st.header("4. Model")
    model_name = st.selectbox("Model", list(MODELS))
    model = MODELS[model_name]
    st.caption(model.DESCRIPTION)
    st.write("**Parameters**")
    boxes = st.columns(3)
    parameters = {}
    for position, (name, spec) in enumerate(model.PARAMETERS.items()):
        with boxes[position % 3]:
            parameters[name] = parameter_box(model_name, name, spec)
    with st.expander("Encoding settings (used by every model)"):
        min_frequency = int(st.number_input("Group categories seen fewer than this many times in the training data "
                                            "into \"unknown\"", min_value=1, value=3, step=1))

    return {
        "run_clicked": st.button("Fit model and predict", type="primary"),
        "training": training, "test": test, "test_name": test_name, "properties": properties, "has_actual": has_actual,
        "numerical": numerical, "dates": dates, "categorical": categorical, "target": target,
        "model_name": model_name, "model": model, "parameters": parameters, "min_frequency": min_frequency,
    }


def fit_and_save(choices):
    """Fit the model, predict the test data, and keep the results in memory."""
    model, target, test = choices["model"], choices["target"], choices["test"]
    try:
        with st.spinner(f"Fitting {choices['model_name']}..."):
            fit = run_model(model, choices["parameters"], choices["training"], test, choices["numerical"],
                            choices["dates"], choices["categorical"], target, choices["min_frequency"])
    except Exception as error:
        st.error(f"The model couldn't be fit: {error}")
        return

    # Results table: property info (if any), prediction, actual and errors (if any), then the test data.
    predicted = f"Predicted {target}"
    results = pd.DataFrame({predicted: fit["predictions"]})
    result_columns = [predicted]
    if choices["properties"] is not None:
        results = pd.concat([numbers_where_possible(choices["properties"]), results], axis=1)
        result_columns = list(choices["properties"].columns) + result_columns
    metrics = None
    if choices["has_actual"]:
        actual = f"Actual {target}"
        results[actual] = pd.to_numeric(test[target], errors="coerce")
        results = add_error_columns(results, actual, predicted)
        result_columns += [actual, "Absolute Error", "Proportional Error (%)"]
        metrics = calculate_metrics(results[actual], fit["predictions"])
    test_columns = [column for column in test.columns if column != target]  # the target is shown as "Actual"
    results = pd.concat([results, numbers_where_possible(test[test_columns])], axis=1)

    predictors = choices["numerical"] + choices["dates"] + choices["categorical"]
    st.session_state["prediction_result"] = {
        "results": results,
        "column_groups": {
            "Results": result_columns,
            "Predictors used": predictors,
            "Other columns in the test data": [column for column in test_columns if column not in predictors],
        },
        "shown_to_start": [column.format(target=target) for column in SHOWN_TO_START],
        "model_name": choices["model_name"],
        "metrics": metrics,
        "details": fit["details"],
        "summary": (
            f"From the last run: **{choices['model_name']}** ("
            + ", ".join(f"{model.PARAMETERS[name]['label']} = {value}" for name, value in choices["parameters"].items())
            + f"). Trained on {fit['training_rows']} rows"
            + (f" ({fit['skipped_training_rows']} rows with no {target} were skipped)" if fit["skipped_training_rows"] else "")
            + f", using {fit['model_predictors']} predictors after preparing the data. "
            f"Predicted {len(results)} rows of {choices['test_name']}."
        ),
    }


def show_results():
    """Section 5, from the last run (stays visible after switching pages)."""
    result = st.session_state.get("prediction_result")
    if result is None:
        return
    st.header("5. Results")
    st.caption(result["summary"])

    if result["metrics"] is not None:
        rmse_box, proportional_box, count_box = st.columns(3)
        rmse_box.metric("RMSE", f"{result['metrics']['RMSE']:,.2f}")
        proportional_box.metric("Proportional Error", f"{result['metrics']['Proportional Error']:.2%}")
        count_box.metric("Observations scored", result["metrics"]["Observations scored"])
    if result["details"] is not None:
        with st.expander("Model details"):
            st.dataframe(result["details"], hide_index=True)

    st.subheader("Columns to show")
    shown_columns = []
    for group, columns in result["column_groups"].items():
        if columns:
            st.write(f"**{group}**")
            boxes = st.columns(4)
            for position, column in enumerate(columns):
                if boxes[position % 4].checkbox(column, value=column in result["shown_to_start"], key=f"show column: {column}"):
                    shown_columns.append(column)
    if not shown_columns:
        st.info("Check at least one column to see the results.")
        return

    shown = result["results"][shown_columns]
    st.dataframe(shown, height=500, hide_index=True)
    file_name = st.text_input("Name for the download", value=f"predictions - {result['model_name']}").strip()
    st.download_button("Download these results (CSV)", data=shown.to_csv(index=False).encode("utf-8"),
                       file_name=file_name.removesuffix(".csv") + ".csv", mime="text/csv", type="primary",
                       disabled=not file_name)
    st.caption("The download has the checked columns, in the order shown.")


st.title("🔮 Predictions")
st.write("Fit a model on your training data and predict the test data.")
choices = choose_data_and_model()
if choices is not None and choices["run_clicked"]:
    fit_and_save(choices)
show_results()
