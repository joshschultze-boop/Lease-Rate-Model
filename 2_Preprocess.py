# pages/2_Preprocess.py
# This page:
#   1. Uploads a cleaned comp sheet
#   2. Runs the chosen preprocessing steps IN A FIXED ORDER when you click Run,
#      so the data is ready for the model
#   3. Downloads the result as a CSV, with a preview of exactly which columns are saved

from io import BytesIO

import pandas as pd
import streamlit as st

from cleaning_script_functions import STATE_FIPS_TO_ABBR
from preprocessing_functions import (
    add_log_transforms,
    add_unknown_indicator,
    add_yard_property_ratios,
    clean_dataframe,
    clean_target_columns,
    combine_categorical_columns,
    convert_data_types,
    drop_excluded_columns,
    fill_blanks_with_zero,
    remove_rows_above_maximums,
    round_columns,
)

st.set_page_config(page_title="Preprocess", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------
# SETTINGS (same as the Inputs section of preprocessing_script.py)
# ---------------------------------------------------------------------

CATEGORICAL_COLUMNS = ["City", "State", "Zip", "County", "CBSA", "Lease Type", "Zoning", "NNN"]

NUMERICAL_COLUMNS = [
    "Latitude",
    "Longitude",
    "Yard Size (AC)",
    "Lease Building (SF)",
    "Property (AC)",
    "Property Building (SF)",
    "Year Built",
    "Sale Price",
    "Lease BCR",
    "Distance to Interstate",
    "Lease Length (Months)",
]

DATE_COLUMNS = ["Lease Start", "Sale Date"]
EPOCH = pd.Timestamp("1950-01-01")  # Dates become "days since" this date.

TARGET_COLUMNS = ["Monthly Rent"]

# Only these columns are read from the file.
MODEL_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + DATE_COLUMNS + TARGET_COLUMNS

# Rows that are blank (or "unknown") in any of these are removed.
REQUIRED_COLUMNS = [
    "CBSA", "County", "State", "Zip", "City",
    "Latitude", "Longitude", "Yard Size (AC)", "Lease Building (SF)", "NNN",
]

YEAR_BUILT_MAXIMUM = 9999
EXCLUDE_COLUMNS = ["Lease BCR"]  # Used to filter rows, then left out of the CSV.
ROUND_COLUMNS = ["Monthly Rent", "Distance to Interstate"]
ZERO_FILL_COLUMNS = ["Lease Building (SF)", "Property Building (SF)"]
LOG_COLUMNS = ["Lease Building (SF)", "Yard Size (AC)", "Property Building (SF)", "Property (AC)", "Monthly Rent"]

ALL_STATES = sorted(set(STATE_FIPS_TO_ABBR.values()))

LOG_METHOD_NAMES = {
    "log1p": "log1p: ln(1 + x), zeros stay 0",
    "log": "log: ln(x), zeros become blank",
}


# ---------------------------------------------------------------------
# PREPROCESSING STEPS, listed in the order they run
# ---------------------------------------------------------------------

CLEAN_TARGET = "Convert Monthly Rent to numbers"
COMBINE_CITY_ZIP = "Combine City + Zip (City-Zip)"
COMBINE_CITY_ZONING = "Combine City + Zoning (City-Zoning)"
COMBINE_COUNTY_ZONING = "Combine County + Zoning (County-Zoning)"
CONVERT_TYPES = "Convert data types (categories, numbers, dates as days)"
SALE_UNKNOWN = "Flag unknown sale info (Sale Information Unknown)"
LEASE_LENGTH_UNKNOWN = "Flag unknown lease length (Lease Length Unknown)"
REMOVE_ROWS = "Remove excluded and incomplete rows"
REMOVE_ABOVE_MAXIMUMS = "Remove rows above maximums"
DROP_COLUMNS = "Leave Lease BCR out of the CSV"
ROUND = "Round Monthly Rent and Distance to Interstate"
RATIOS = "Add yard / property ratios"
ZERO_FILL = "Fill blank building SF with 0"
LOG_TRANSFORMS = "Add log columns"

ALL_STEPS = [
    CLEAN_TARGET,
    COMBINE_CITY_ZIP,
    COMBINE_CITY_ZONING,
    COMBINE_COUNTY_ZONING,
    CONVERT_TYPES,
    SALE_UNKNOWN,
    LEASE_LENGTH_UNKNOWN,
    REMOVE_ROWS,
    REMOVE_ABOVE_MAXIMUMS,
    DROP_COLUMNS,
    ROUND,
    RATIOS,
    ZERO_FILL,
    LOG_TRANSFORMS,
]


# ---------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------

report = []  # One line per step: (icon, step name, rows after the step, note)


def run_step(step_name, df, step_function, **arguments):
    """
    Run one step on the data and record the result.
    If a step fails, the error is recorded and the data is passed on unchanged,
    so the remaining steps still run.
    """
    try:
        new_df = step_function(df, **arguments)
    except Exception as error:
        report.append(("⚠️", step_name, len(df), f"Skipped: {error}"))
        return df

    rows_removed = len(df) - len(new_df)
    note = f"{rows_removed} rows removed" if rows_removed else ""
    report.append(("✅", step_name, len(new_df), note))
    return new_df


def run_chosen_steps(df, chosen_steps, states_to_keep, lease_bcr_maximum, log_method):
    """Check each step in order. If the user chose it, run it. Returns the finished data."""

    # 1. Rent must be a number before rows with $0 rent can be removed (step 8).
    if CLEAN_TARGET in chosen_steps:
        df = run_step(CLEAN_TARGET, df, clean_target_columns, target_columns=TARGET_COLUMNS)

    # 2-4. Combine columns. This runs BEFORE step 5 (same as your script), so a combined
    #      value is blank when either part is blank, rather than containing "unknown".
    if COMBINE_CITY_ZIP in chosen_steps:
        df = run_step(COMBINE_CITY_ZIP, df, combine_categorical_columns,
                      columns=["City", "Zip"], new_column="City-Zip")

    if COMBINE_CITY_ZONING in chosen_steps:
        df = run_step(COMBINE_CITY_ZONING, df, combine_categorical_columns,
                      columns=["City", "Zoning"], new_column="City-Zoning")

    if COMBINE_COUNTY_ZONING in chosen_steps:
        df = run_step(COMBINE_COUNTY_ZONING, df, combine_categorical_columns,
                      columns=["County", "Zoning"], new_column="County-Zoning")

    # 5. Blank categories become "unknown", text in number columns becomes blank,
    #    and dates become days since EPOCH.
    if CONVERT_TYPES in chosen_steps:
        df = run_step(CONVERT_TYPES, df, convert_data_types,
                      categorical_columns=CATEGORICAL_COLUMNS,
                      numerical_columns=NUMERICAL_COLUMNS,
                      date_columns=DATE_COLUMNS,
                      epoch=EPOCH)

    # 6-7. 0/1 flag columns: 1 when the value is blank or "unknown".
    if SALE_UNKNOWN in chosen_steps:
        df = run_step(SALE_UNKNOWN, df, add_unknown_indicator,
                      columns=["Sale Price", "Sale Date"], new_column="Sale Information Unknown")

    if LEASE_LENGTH_UNKNOWN in chosen_steps:
        df = run_step(LEASE_LENGTH_UNKNOWN, df, add_unknown_indicator,
                      columns="Lease Length (Months)", new_column="Lease Length Unknown")

    # 8. Remove rows with $0 or blank rent, rows from states not being kept,
    #    and rows that are blank or "unknown" in a required column.
    if REMOVE_ROWS in chosen_steps:
        exclude_values = {
            "Monthly Rent": [0, None],
            "State": [state for state in ALL_STATES if state not in states_to_keep],
        }
        df = run_step(REMOVE_ROWS, df, clean_dataframe,
                      exclude_values=exclude_values,
                      required_columns=REQUIRED_COLUMNS)

    # 9. Needs numbers (step 5) and needs Lease BCR, so it runs before step 10.
    if REMOVE_ABOVE_MAXIMUMS in chosen_steps:
        maximum_values = {"Lease BCR": lease_bcr_maximum, "Year Built": YEAR_BUILT_MAXIMUM}
        df = run_step(REMOVE_ABOVE_MAXIMUMS, df, remove_rows_above_maximums,
                      maximum_values=maximum_values)

    # 10. Lease BCR was only needed for step 9.
    if DROP_COLUMNS in chosen_steps:
        df = run_step(DROP_COLUMNS, df, drop_excluded_columns, columns_to_exclude=EXCLUDE_COLUMNS)

    # 11. Round to 2 decimals.
    if ROUND in chosen_steps:
        df = run_step(ROUND, df, round_columns, columns=ROUND_COLUMNS, decimals=2)

    # 12. Ratios run BEFORE step 13, so an unknown building size gives a blank ratio, not 0.
    if RATIOS in chosen_steps:
        df = run_step(RATIOS, df, add_yard_property_ratios)

    # 13. Zero fill runs BEFORE step 14, so filled zeros get a log value instead of staying blank.
    if ZERO_FILL in chosen_steps:
        df = run_step(ZERO_FILL, df, fill_blanks_with_zero, columns=ZERO_FILL_COLUMNS)

    # 14. Adds new columns named like "Monthly Rent_log1p". The original columns stay.
    if LOG_TRANSFORMS in chosen_steps:
        df = run_step(LOG_TRANSFORMS, df, add_log_transforms, columns=LOG_COLUMNS, method=log_method)

    return df


def describe_column(df, column):
    """Plain-language type for the preview table."""
    if column in DATE_COLUMNS and pd.api.types.is_numeric_dtype(df[column]):
        return f"Days since {EPOCH:%Y-%m-%d}"
    if isinstance(df[column].dtype, pd.CategoricalDtype):
        return "Category"
    if pd.api.types.is_numeric_dtype(df[column]):
        return "Number"
    return "Text"


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("📊 Preprocess for the model")
st.write("Turn a cleaned comp sheet into a CSV that's ready for the predictive model.")


# --- 1. Upload ---

st.header("1. Upload the cleaned file")

uploaded_file = st.file_uploader("Cleaned Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload a cleaned file to continue. You can make one on the Clean Data page.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet_names = pd.ExcelFile(BytesIO(file_bytes)).sheet_names
sheet = st.selectbox("Sheet to use", sheet_names)

data = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
data.columns = [str(column).strip() for column in data.columns]

# The file must have every column the model uses.
missing_columns = [column for column in MODEL_COLUMNS if column not in data.columns]
if missing_columns:
    st.error(
        "This file is missing columns the preprocessing needs: " + ", ".join(missing_columns)
        + ". Run it through the Clean Data page first, or add these columns."
    )
    st.stop()

# Keep only the model columns, in the same order as the file (like usecols in your script).
data = data[[column for column in data.columns if column in MODEL_COLUMNS]]
st.caption(f"{len(data)} rows. The {len(MODEL_COLUMNS)} columns the model uses will be loaded.")


# --- 2. Choose steps ---

st.header("2. Choose preprocessing steps")
chosen_steps = st.multiselect("Preprocessing steps", ALL_STEPS, default=ALL_STEPS)

# Put the chosen steps in run order, no matter what order they were picked in.
steps_in_order = [step for step in ALL_STEPS if step in chosen_steps]

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
csv_name = st.text_input("Name for the CSV file", value="preprocessed").strip()

if st.button("Run preprocessing", type="primary", disabled=not steps_in_order):
    if not csv_name:
        st.error("Type a name for the CSV file first.")
        st.stop()

    with st.spinner("Preprocessing..."):
        output = run_chosen_steps(data, chosen_steps, states_to_keep, lease_bcr_maximum, log_method)

    # Save the result in memory so the download and preview stay after the page refreshes.
    st.session_state["preprocessed_data"] = output
    st.session_state["preprocessed_csv_name"] = csv_name.removesuffix(".csv") + ".csv"
    st.session_state["preprocess_report"] = report

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
        f"{len(output)} rows and {len(output.columns)} columns, from the last run. "
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
    st.dataframe(output, height=400)
