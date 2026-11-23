# pages/4_Predictions.py
# Fits a model on the training data and predicts the test data:
#   1-2. Choose the training data (from the Preprocess page, or an uploaded CSV) and its predictor
#        and target columns
#   3.   Choose the test data (from the Test Data page, or an uploaded CSV with or without rent)
#   4.   Choose the random seed (one for every model), a model, and its parameters, then fit and predict
#   5.   Results: each property's Full Address with its predicted rent (in $), errors (when the test data
#        has actual rent), column weights (linear models), column checkboxes, and a download
#
# Preparing the data and fitting happen in models/shared.py, the same way for every model.
# The models are listed in models/__init__.py. Adding a model doesn't change this page.

import hashlib
import inspect
import secrets

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

# Columns that identify each property. They come from the Test Data page, or from any uploaded test CSV
# that has them (the Test Data and Preprocess downloads do). They're shown, never used as model inputs.
PROPERTY_COLUMNS = ["Sheet Row", "Street Address", "Full Address"]

# Columns checked to start with in the results (others can be checked below the results).
SHOWN_TO_START = ["Full Address", "Predicted {target}", "Actual {target}"]

# The random seed is one setting for every model. Each model file says what it uses it for (SEED_USED_FOR).
NEW_SEED = "New random seed each fit"
TYPED_SEED = "Use this seed"
SEED_SETTINGS = {
    "seed_choice": {"label": "Random seed (all models)", "type": "choice", "options": [NEW_SEED, TYPED_SEED],
                    "default": NEW_SEED,
                    "help": "A new seed each fit is like your notebooks' random seed per run. The seed used is shown "
                            "with the results, so you can type it in to repeat a run exactly."},
    "seed": {"label": "Seed", "type": "int", "default": 42, "min": 0, "max": 2**32 - 1,
             "help": "Any whole number from 0 to 4,294,967,295. The same seed gives the same results."},
}

FROM_PREPROCESS_PAGE = "Training data from the Preprocess page"
UPLOAD_TRAINING_CSV = "Upload a training CSV"
FROM_TEST_DATA_PAGE = "Test data from the Test Data page"
UPLOAD_TEST_CSV = "Upload a test CSV"


# Streamlit forgets a box's value while the box isn't shown (another model is picked, or another page is
# open), so every parameter value is also kept in st.session_state["saved_parameters"], keyed by
# "<model>: <parameter>", and put back when the box is shown again.

def saved_parameters():
    return st.session_state.setdefault("saved_parameters", {})


def parameter_box(model_name, name, spec):
    """One parameter box, built from an entry in the model file's PARAMETERS."""
    key, label, help_text, kind = f"{model_name}: {name}", spec["label"], spec.get("help"), spec["type"]
    if key not in st.session_state and key in saved_parameters():
        st.session_state[key] = saved_parameters()[key]
    # A box with a saved value starts from it; otherwise it starts from the model file's default.
    start = {} if key in st.session_state else (
        {"index": spec["options"].index(spec["default"])} if kind == "choice" else {"value": spec["default"]})

    if kind == "int":
        value = int(st.number_input(label, step=int(spec.get("step", 1)), min_value=spec.get("min"),
                                    max_value=spec.get("max"), help=help_text, key=key, **start))
    elif kind == "float":
        as_float = lambda number: None if number is None else float(number)
        start = {"value": float(start["value"])} if start else {}
        value = float(st.number_input(label, step=float(spec.get("step", 0.01)), min_value=as_float(spec.get("min")),
                                      max_value=as_float(spec.get("max")), format=spec.get("format", "%.4f"),
                                      help=help_text, key=key, **start))
    elif kind == "bool":
        value = st.checkbox(label, help=help_text, key=key, **start)
    elif kind == "choice":
        value = st.selectbox(label, spec["options"], help=help_text, key=key, **start)
    else:
        raise ValueError(f"Unknown parameter type for {name}: {kind}")
    saved_parameters()[key] = value
    return value


def table_box(model_name, name, spec, predictor_columns):
    """
    An editable table parameter (one row per item, e.g. a spline or a layer), built from the model
    file's PARAMETERS entry. Returns (rows as a list of dictionaries, a problem message or None).
    predictor_columns are the choices for "predictor" columns (the numerical and date predictors).
    """
    columns = spec["columns"]
    config = {}
    for column, column_spec in columns.items():
        kind, label, help_text = column_spec["type"], column_spec["label"], column_spec.get("help")
        if kind in ("int", "float"):
            config[column] = st.column_config.NumberColumn(
                label, help=help_text, required=True, min_value=column_spec.get("min"), max_value=column_spec.get("max"),
                step=1 if kind == "int" else column_spec.get("step"),
                format=column_spec.get("format", "%d" if kind == "int" else None))
        elif kind in ("choice", "predictor"):
            options = column_spec["options"] if kind == "choice" else predictor_columns
            config[column] = st.column_config.SelectboxColumn(label, options=options, help=help_text, required=True)

    # Start from the saved rows (or the model file's defaults), leaving out any that name a predictor
    # not chosen in section 2. Those rows come back if the predictor is chosen again.
    saved_key = f"{model_name}: {name}"
    starting = saved_parameters().get(saved_key, spec["default"])
    predictor_keys = [column for column, column_spec in columns.items() if column_spec["type"] == "predictor"]
    rows = [row for row in starting if all(row[key] in predictor_columns for key in predictor_keys)]
    left_out = [row for row in starting if row not in rows]

    st.write(f"**{spec['label']}**")
    if spec.get("help"):
        st.caption(spec["help"])
    if left_out:
        st.caption("Starting rows left out because those columns aren't predictors in section 2: "
                   + ", ".join(str(row[predictor_keys[0]]) for row in left_out))
    # The key changes when the starting rows change, so edits are never applied to the wrong rows.
    starting_rows = hashlib.md5(repr(rows).encode()).hexdigest()[:8]
    edited = st.data_editor(pd.DataFrame(rows, columns=list(columns)), column_config=config, num_rows="dynamic",
                            hide_index=True, key=f"{saved_key}: {starting_rows}")

    # Each row as a dictionary with the right types. A new row left completely empty is ignored.
    result, incomplete = [], 0
    for row in edited.to_dict("records"):
        filled = [not pd.isna(value) for value in row.values()]
        if not any(filled):
            continue
        if not all(filled):
            incomplete += 1
            continue
        convert = {"int": int, "float": float}
        result.append({column: convert.get(columns[column]["type"], str)(value) for column, value in row.items()})
    if incomplete:
        return result, f"{spec['label']}: fill in every cell of the incomplete row(s), or delete them."
    saved_parameters()[saved_key] = result + left_out   # saved only when complete, so half-typed rows stay put
    return result, None


def choose_data_and_model():
    """Sections 1-4. Returns the choices, or None (with a message) if something is still missing."""

    st.header("1. Training data")
    preprocess_result = st.session_state.get("preprocess_result")
    sources = ([FROM_PREPROCESS_PAGE] if preprocess_result else []) + [UPLOAD_TRAINING_CSV]
    if st.radio("Training data", sources, horizontal=True) == FROM_PREPROCESS_PAGE:
        training, training_name = read_csv_as_text(preprocess_result["data"]), preprocess_result["file_name"]
        st.caption(f"Using **{training_name}**, made on the Preprocess page in this session.")
    else:
        training_file = st.file_uploader("Preprocessed training CSV (from the Preprocess page)", type=["csv"])
        if training_file is None:
            st.info("Run the Preprocess page first (its result is used here automatically), "
                    "or upload a training CSV you downloaded earlier.")
            return None
        training, training_name = read_csv_as_text(training_file), training_file.name
    headers = list(training.columns)
    st.caption(f"{len(training)} rows and {len(headers)} columns.")

    st.header("2. Columns")
    st.write("Choose the columns the model uses, from the training CSV's headers. Each column can be in only one list.")
    left, right = st.columns(2)
    defaults = lambda columns: [column for column in columns if column in headers]
    numerical = left.multiselect("Numerical predictors", headers, default=defaults(DEFAULT_NUMERICAL))
    dates = left.multiselect("Date predictors (days since the Preprocess start date)", headers, default=defaults(DEFAULT_DATE))
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
            st.info("Make test data on the Test Data page (it's used here automatically), "
                    "or upload a test CSV you downloaded earlier.")
            return None
        test, test_name = read_csv_as_text(test_file), test_file.name
        found = [column for column in PROPERTY_COLUMNS if column in test.columns]
        properties = test[found].reset_index(drop=True) if found else None

    missing = [column for column in predictors if column not in test.columns]
    if missing:
        st.error("The test data is missing these predictor columns: " + ", ".join(missing))
        return None
    actual_count = int(pd.to_numeric(test[target], errors="coerce").notna().sum()) if target in test.columns else 0
    if actual_count:
        st.caption(f"{len(test)} rows. {actual_count} of them have an actual {target} to compare the predictions with.")
        has_actual = st.checkbox(f"Report errors against the actual {target}", value=True,
                                 help="RMSE and proportional error for the whole set, plus absolute and proportional "
                                      "error for each row that has an actual value.")
    else:
        st.caption(f"{len(test)} rows. There's no actual {target} in the test data, so only predictions are made.")
        has_actual = False

    st.header("4. Model")
    seed_box, seed_number_box, _ = st.columns([1, 1, 1])
    with seed_box:
        seed_choice = parameter_box("All models", "seed_choice", SEED_SETTINGS["seed_choice"])
    with seed_number_box:
        typed_seed = parameter_box("All models", "seed", SEED_SETTINGS["seed"]) if seed_choice == TYPED_SEED else None

    model_name = st.selectbox("Model", list(MODELS))
    model = MODELS[model_name]
    st.caption(model.DESCRIPTION)
    st.caption(f"The random seed sets {model.SEED_USED_FOR}." if model.SEED_USED_FOR
               else "This model has no randomness, so the seed doesn't change its results.")
    title, reset = st.columns([4, 1])
    title.write("**Parameters**")
    if reset.button("Reset to defaults", help="Put this model's parameters back to the values in its model file."):
        for key in [key for key in saved_parameters() if key.startswith(f"{model_name}: ")]:
            del saved_parameters()[key]
        for key in [key for key in st.session_state if isinstance(key, str) and key.startswith(f"{model_name}: ")]:
            del st.session_state[key]
        st.rerun()
    parameters, table_problems, position = {}, [], 0
    for name, spec in model.PARAMETERS.items():   # in the model file's order
        if spec["type"] == "table":               # tables use the full width
            parameters[name], problem = table_box(model_name, name, spec, numerical + dates)
            table_problems += [problem] if problem else []
            position = 0
        else:                                     # other boxes go three to a row
            if position % 3 == 0:
                boxes = st.columns(3)
            with boxes[position % 3]:
                parameters[name] = parameter_box(model_name, name, spec)
            position += 1
    with st.expander("Encoding settings (used by every model)"):
        min_frequency = int(st.number_input("Group categories seen fewer than this many times in the training data "
                                            "into \"unknown\"", min_value=1, value=3, step=1))
    for problem in table_problems:
        st.error(problem)
    if table_problems:
        return None

    return {
        "run_clicked": st.button("Fit model and predict", type="primary"),
        "training": training, "training_name": training_name, "test": test, "test_name": test_name, "properties": properties, "has_actual": has_actual,
        "numerical": numerical, "dates": dates, "categorical": categorical, "target": target,
        "model_name": model_name, "model": model, "parameters": parameters, "min_frequency": min_frequency,
        "typed_seed": typed_seed,
    }


def fit_and_save(choices):
    """Fit the model, predict the test data, and keep the results in memory."""
    model, target, test = choices["model"], choices["target"], choices["test"]
    try:
        with st.spinner(f"Fitting {choices['model_name']}..."):
            # A new random seed comes from the operating system, so it isn't affected by any seed set earlier.
            seed = choices["typed_seed"] if choices["typed_seed"] is not None else secrets.randbelow(2**32)
            fit = run_model(model, choices["parameters"], choices["training"], test, choices["numerical"],
                            choices["dates"], choices["categorical"], target, choices["min_frequency"], seed)
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
    # The target is shown as "Actual", and the property columns are shown first, so leave them out here.
    test_columns = [column for column in test.columns if column != target and column not in PROPERTY_COLUMNS]
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
        "money_columns": [predicted] + ([f"Actual {target}", "Absolute Error"] if metrics is not None else []),
        "model_name": choices["model_name"],
        "metrics": metrics,
        "details": fit["details"],
        "column_weights": fit["column_weights"],
        "target": target,
        "run_id": secrets.token_hex(4),   # gives this run's widgets their own keys
        "min_frequency": choices["min_frequency"],
        "summary": (
            f"From the last run: **{choices['model_name']}** ("
            + ", ".join(f"{model.PARAMETERS[name]['label']} = "
                        + (f"{len(value)} rows" if isinstance(value, list) else f"{value}")
                        for name, value in choices["parameters"].items())
            + f"). Trained on {fit['training_rows']} rows of {choices['training_name']}"
            + (f" ({fit['skipped_training_rows']} rows with no {target} were skipped)" if fit["skipped_training_rows"] else "")
            + f", using {fit['model_predictors']} predictors after preparing the data. "
            f"Predicted {len(results)} rows of {choices['test_name']}. "
            f"Random seed: {seed} ({'typed in' if choices['typed_seed'] is not None else 'new random seed'})."
        ),
    }


def show_column_weights(result):
    """
    Column weights (linear models only; models/shared.py works them out): how much each predictor column
    moves the prediction, as a chart and table, then a closer look inside any one column.
    """
    import altair as alt

    weights, target = result["column_weights"], result["target"]
    overview = weights["overview"]
    # Blank cells show a dash instead of "None" (the placeholder option only exists in newer Streamlit versions).
    blank = {"placeholder": "—"} if "placeholder" in inspect.signature(st.dataframe).parameters else {}
    coefficient = {"Coefficient (per 1 standard deviation)": st.column_config.NumberColumn(format="%.2f")}
    st.subheader("Column weights")
    st.caption(
        f"How much each predictor column moves the predicted {target}. **Weight** is the typical size of everything "
        "a column adds to a prediction: the standard deviation of its contribution across the training rows, in $. "
        "For a numerical column that's its coefficient per 1 standard deviation; a categorical column's one-hot "
        "columns are added together. Weights show what the model leans on, not cause and effect, and related "
        "columns (like Yard Size and its log) can share or trade weight."
    )

    used = overview.loc[overview["Weight"] > 0]
    if used.empty:
        st.info("Every coefficient is 0, so no column moves the prediction.")
    else:
        # One series: bars in a single color (the Streamlit theme's light or dark step), largest first.
        theme = getattr(getattr(st, "context", None), "theme", None)
        color = "#3987e5" if getattr(theme, "type", None) == "dark" else "#2a78d6"
        chart_data = used.rename(columns={"Share of total weight (%)": "Share"})
        chart = alt.Chart(chart_data).mark_bar(color=color, size=18, cornerRadiusEnd=4).encode(
            x=alt.X("Weight:Q", title=f"Weight (typical effect on {target}, $)", axis=alt.Axis(format="$,.0f", tickCount=6)),
            y=alt.Y("Column:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Column:N"), alt.Tooltip("Weight:Q", format="$,.2f"),
                     alt.Tooltip("Share:Q", title="Share of total weight (%)", format=".1f"), alt.Tooltip("Direction:N")],
        ).properties(height=max(120, 32 * len(chart_data)))
        st.altair_chart(chart)

    st.dataframe(overview, hide_index=True, **blank, column_config={
        "Weight": st.column_config.NumberColumn(
            format="dollar", help=f"Typical effect on the predicted {target}: the standard deviation of everything "
                                  "this column adds, across the training rows."),
        "Share of total weight (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Effect of +1 unit": st.column_config.NumberColumn(
            format="dollar", help=f"For a numerical or date column used once: how much the predicted {target} changes "
                                  "when the column goes up by 1 (1 acre, 1 square foot, 1 day, ...), everything else "
                                  "the same. Blank for categorical columns and columns in polynomial terms."),
    })
    st.download_button("Download column weights (CSV)", data=overview.round(4).to_csv(index=False).encode("utf-8"),
                       file_name=f"column weights - {result['model_name']}.csv", mime="text/csv")

    column = st.selectbox("Look inside one column", list(overview["Column"]), key=f"column weights: {result['run_id']}")
    detail = weights["details"][column]
    row = overview.loc[overview["Column"] == column].iloc[0]
    if row["Direction"].startswith("Not used"):
        st.info(f"The model doesn't use {column}: all of its coefficients are 0.")
        return
    if detail["type"] == "Categorical":
        st.caption(
            f"Each category's effect on the predicted {target}, everything else the same. Compare categories with "
            f"each other: the gap between two categories is the predicted difference between them. Categories seen "
            f"fewer than {result.get('min_frequency', 3)} times in training, and new categories in the test data, "
            "count as \"unknown\"."
        )
        st.dataframe(detail["main"], hide_index=True, column_config={
            **coefficient, "Effect of being in this category": st.column_config.NumberColumn(format="dollar")})
    else:
        if detail["other"] is not None:
            st.caption(f"{column}'s own straight-line term. It's only part of the effect: {column} is also in the "
                       "polynomial terms below.")
        else:
            st.caption(f"How much the predicted {target} changes when {column} goes up by 1, everything else the same"
                       + (" (blank: numbers are scaled to 0-1 for this model)." if detail["main"]["Effect of +1 unit"].isna().all()
                          else "."))
        st.dataframe(detail["main"], hide_index=True, **blank, column_config={
            **coefficient, "Effect of +1 unit": st.column_config.NumberColumn(format="dollar")})
    if detail["other"] is not None:
        st.write(f"**Polynomial terms that include {column}**")
        st.caption("With polynomial terms, the column's effect also depends on the other columns in each term, so "
                   "its own coefficient isn't the whole story. In the weight, each term is split between its columns "
                   "by how much each one moves it away from the average property (Shapley values).")
        st.dataframe(detail["other"], hide_index=True, column_config=coefficient)


def show_results():
    """Section 5, from the last run (stays visible after switching pages)."""
    result = st.session_state.get("prediction_result")
    if result is None:
        return
    st.header("5. Results")
    st.caption(result["summary"])

    if result["metrics"] is not None:
        rmse_box, proportional_box, count_box = st.columns(3)
        rmse_box.metric("RMSE", f"${result['metrics']['RMSE']:,.2f}")
        proportional_box.metric("Proportional Error", f"{result['metrics']['Proportional Error']:.2%}")
        count_box.metric("Observations scored", result["metrics"]["Observations scored"])
    if result["details"] is not None:
        with st.expander("Model details"):
            st.dataframe(result["details"], hide_index=True)
    if result.get("column_weights") is not None:
        show_column_weights(result)

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
    money = [column for column in result["money_columns"] if column in shown_columns]
    percent = [column for column in ["Proportional Error (%)"] if column in shown_columns]

    # Rent and errors in dollars, proportional error as a percent, and Full Address kept in view.
    column_config = {column: st.column_config.NumberColumn(format="dollar") for column in money}
    column_config.update({column: st.column_config.NumberColumn(format="%.1f%%") for column in percent})
    if "Full Address" in shown_columns:
        column_config["Full Address"] = st.column_config.TextColumn(pinned=True)
    st.dataframe(shown, column_config=column_config, height=500, hide_index=True)

    file_name = st.text_input("Name for the download", value=f"predictions - {result['model_name']}").strip()
    dollar_download = st.checkbox("Use $ formatting in the download", value=True,
                                  help="On: rent and errors like $12,345.67, which Excel reads as currency. "
                                       "Off: plain numbers, easier to reuse in Python.")
    download = shown.copy()
    if dollar_download:
        for column in money:
            download[column] = download[column].map(lambda value: "" if pd.isna(value) else f"${value:,.2f}")
        for column in percent:
            download[column] = download[column].map(lambda value: "" if pd.isna(value) else f"{value:.1f}%")
    else:
        download[money] = download[money].round(2)      # dollars and cents
        download[percent] = download[percent].round(2)
    st.download_button("Download these results (CSV)", data=download.to_csv(index=False).encode("utf-8"),
                       file_name=file_name.removesuffix(".csv") + ".csv", mime="text/csv", type="primary",
                       disabled=not file_name)
    st.caption("The download has the checked columns, in the order shown.")


st.title("🔮 Predictions")
st.write("Fit a model on your training data and predict the test data.")
choices = choose_data_and_model()
if choices is not None and choices["run_clicked"]:
    fit_and_save(choices)
show_results()
