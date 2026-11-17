# pages/2_Preprocess.py
# This page:
#   1. Uploads a cleaned comp sheet
#   2. Runs the chosen preprocessing steps IN A FIXED ORDER when you click Run,
#      so the data is ready for the model
#   3. Downloads the result as a CSV, with a preview of exactly which columns are saved
#
# Tick "Test data with no Monthly Rent" for new properties you want to predict rent for.
# The same steps run, minus the rent parts, so the CSV matches the training CSV's format.

from io import BytesIO

import pandas as pd
import streamlit as st
from preprocessing_steps import (
    ALL_PREPROCESSING_STEPS,
    ALL_STATES,
    CLEAN_TARGET,
    LOG_METHOD_NAMES,
    MODEL_COLUMNS,
    PREDICTIVE_COLUMNS,
    REMOVE_ROWS,
    describe_column,
    run_preprocessing_steps,
    table_for_display,
)

st.set_page_config(page_title="Preprocess", page_icon="📊", layout="wide")

# The preprocessing steps, their settings, and their order are in preprocessing_steps.py,
# which the Test Data page also uses. This file is just the page itself.


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("📊 Preprocess for the model")
st.write("Turn a cleaned comp sheet into a CSV that's ready for the predictive model.")


# --- 1. Upload ---

st.header("1. Upload the cleaned file")

no_rent = st.checkbox("Test data with no Monthly Rent (new properties to predict rent for)")
has_rent = not no_rent

if no_rent:
    st.caption(
        "Monthly Rent isn't needed. The rent conversion step is skipped, rows aren't removed for "
        "blank rent, and rounding and log columns leave Monthly Rent out. Everything else runs "
        "exactly as it does for training data, so the CSV has the same columns minus the rent columns."
    )
else:
    st.caption("Leave this unticked for comps that have rent, including a test set you hold back to score a model.")

uploaded_file = st.file_uploader("Cleaned Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload a cleaned file to continue. You can make one on the Clean Data page.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet_names = pd.ExcelFile(BytesIO(file_bytes)).sheet_names
sheet = st.selectbox("Sheet to use", sheet_names)

data = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
data.columns = [str(column).strip() for column in data.columns]

# The file must have every column the model uses (Monthly Rent only when there is rent).
columns_needed = MODEL_COLUMNS if has_rent else PREDICTIVE_COLUMNS
missing_columns = [column for column in columns_needed if column not in data.columns]
if missing_columns:
    st.error(
        "This file is missing columns the preprocessing needs: " + ", ".join(missing_columns)
        + ". Run it through the Clean Data page first, or add these columns."
    )
    st.stop()

# Keep only the model columns, in the fixed order from SETTINGS.
data = data[columns_needed]
st.caption(f"{len(data)} rows. The {len(columns_needed)} columns the model uses will be loaded.")


# --- 2. Choose steps ---

st.header("2. Choose preprocessing steps")
chosen_steps = st.multiselect("Preprocessing steps", ALL_PREPROCESSING_STEPS, default=ALL_PREPROCESSING_STEPS)

# Put the chosen steps in run order, no matter what order they were picked in.
steps_in_order = [step for step in ALL_PREPROCESSING_STEPS if step in chosen_steps]
if no_rent:
    steps_in_order = [step for step in steps_in_order if step != CLEAN_TARGET]

if steps_in_order:
    st.write("**These steps will run in this order:**")
    st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(steps_in_order, start=1)))
else:
    st.info("Choose at least one step to run.")

with st.expander("Step settings"):
    states_to_keep = st.multiselect("Keep rows from these states", ALL_STATES, default=["CO"])
    lease_bcr_maximum = st.number_input("Remove rows with Lease BCR above", min_value=0.0, value=0.25, step=0.05)
    log_method = st.radio(
        "Log columns",
        options=list(LOG_METHOD_NAMES),
        format_func=lambda method: LOG_METHOD_NAMES[method],
        horizontal=True,
    )

if REMOVE_ROWS in chosen_steps and not states_to_keep:
    st.warning("No states are selected to keep, so every row will be removed.")


# --- 3. Run and download ---

st.header("3. Run and download")
default_csv_name = "preprocessed_test" if no_rent else "preprocessed"
csv_name = st.text_input("Name for the CSV file", value=default_csv_name).strip()

if st.button("Run preprocessing", type="primary", disabled=not steps_in_order):
    if not csv_name:
        st.error("Type a name for the CSV file first.")
        st.stop()

    with st.spinner("Preprocessing..."):
        output, report = run_preprocessing_steps(
            data, chosen_steps, states_to_keep, lease_bcr_maximum, log_method, has_rent
        )

    # Save the result in memory so the download and preview stay after the page refreshes.
    st.session_state["preprocessed_data"] = output
    st.session_state["preprocessed_csv_name"] = csv_name.removesuffix(".csv") + ".csv"
    st.session_state["preprocess_report"] = report
    st.session_state["preprocess_mode"] = "test data, no rent" if no_rent else "data with rent"

if "preprocessed_data" in st.session_state:
    output = st.session_state["preprocessed_data"]

    st.write("**Step results, in the order they ran:**")
    st.dataframe(
        pd.DataFrame(st.session_state["preprocess_report"], columns=["", "Step", "Rows after", "Notes"]),
        hide_index=True,
    )

    st.download_button(
        "Download CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=st.session_state["preprocessed_csv_name"],
        mime="text/csv",
        type="primary",
    )


    # --- 4. Preview ---

    st.header("4. Preview of the CSV")
    st.caption(
        f"{len(output)} rows and {len(output.columns)} columns, from the last run "
        f"({st.session_state['preprocess_mode']}). "
        "Change a step or setting? Click Run again to update."
    )

    st.write("**Columns in the CSV**")
    column_summary = pd.DataFrame({
        "Column": output.columns,
        "Type": [describe_column(output, column) for column in output.columns],
        "Filled": output.notna().sum().values,
        "Blank": output.isna().sum().values,
    })
    st.dataframe(column_summary, hide_index=True)

    st.write("**Data**")
    st.dataframe(table_for_display(output), height=400)
