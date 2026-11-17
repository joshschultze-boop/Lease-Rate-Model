# pages/3_Test_Data.py
# This page:
#   1. Uploads a sheet of new properties to predict rent for (no Monthly Rent needed)
#   2. Runs the SAME cleaning steps as the Clean Data page, then the SAME preprocessing
#      steps as the Preprocess page with the "no Monthly Rent" option on
#   3. Keeps the result in memory (st.session_state) so model pages can use it,
#      with a CSV download as a backup
#
# The steps come from cleaning_steps.py and preprocessing_steps.py, the same files the
# other two pages use, so test data always matches the training data's format.

import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from app_helpers import add_entries_to_excel
from cleaning_steps import (
    ADJUST_YARD,
    ALL_CLEANING_STEPS,
    CAP_SIZES,
    HIGHWAY_TYPE_NAMES,
    run_cleaning_steps,
)
from preprocessing_steps import (
    ALL_PREPROCESSING_STEPS,
    ALL_STATES,
    CLEAN_TARGET,
    LOG_METHOD_NAMES,
    PREDICTIVE_COLUMNS,
    REMOVE_ROWS,
    describe_column,
    run_preprocessing_steps,
    table_for_display,
)

st.set_page_config(page_title="Test Data", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------------
# What gets kept in memory (st.session_state) after a run
# ---------------------------------------------------------------------
#   "test_data"            -> the model inputs: same columns as the training CSV, minus rent
#   "test_data_properties" -> Sheet Row and address for each row of test_data, in the same order,
#                             so predictions can be matched back to properties
#   "test_data_info"       -> file name and number of rows and columns
#
# These stay while this browser tab is open. Refreshing the page or closing the tab clears
# them, so download the CSV if you want to keep a copy.

# Columns kept alongside the test data to identify each property. They are not model inputs.
PROPERTY_COLUMNS = ["Sheet Row", "Street Address", "Full Address"]

TEST_DATA_KEYS = [
    "test_data",
    "test_data_properties",
    "test_data_info",
    "test_data_removed",
    "test_data_cleaning_report",
    "test_data_preprocessing_report",
    "test_data_cleaned_bytes",
    "test_data_csv_name",
]


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("🧪 Test Data")
st.write(
    "Clean and preprocess new properties in one step, so they're ready for the models. "
    "Monthly Rent isn't needed."
)

# Show what's already in memory, with a way to clear it.
if "test_data" in st.session_state:
    info = st.session_state["test_data_info"]
    st.success(
        f"Test data in memory: **{info['file_name']}**, {info['rows']} rows and {info['columns']} columns."
    )
    if st.button("Clear test data"):
        for key in TEST_DATA_KEYS:
            st.session_state.pop(key, None)
        st.rerun()


# --- 1. Upload ---

st.header("1. Upload the test file")

uploaded_file = st.file_uploader("Excel file of properties to predict (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload a sheet of properties to continue. It needs the same columns as your comp sheet, except Monthly Rent.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet_names = pd.ExcelFile(BytesIO(file_bytes)).sheet_names
sheet = st.selectbox("Sheet to use", sheet_names)

uploaded_rows = len(pd.read_excel(BytesIO(file_bytes), sheet_name=sheet).dropna(how="all"))
st.caption(f"{uploaded_rows} properties in this sheet.")


# --- 2. Steps ---

st.header("2. Steps")
st.write(
    "These are the same steps and settings as the Clean Data and Preprocess pages. "
    "If you changed any of them for your training data, make the same changes here so the formats match."
)

with st.expander("Cleaning steps and settings"):
    cleaning_steps = st.multiselect("Cleaning steps", ALL_CLEANING_STEPS, default=ALL_CLEANING_STEPS)
    generate_script_address = st.checkbox("Geocoding: also fill Script Address and County", value=True)
    highway_types = st.multiselect(
        "Distance to highway: road types to measure to",
        options=list(HIGHWAY_TYPE_NAMES),
        default=["I"],
        format_func=lambda code: HIGHWAY_TYPE_NAMES[code],
    )
    zoning_character = st.text_input("Zoning: character to standardize codes with", value="-")

with st.expander("Preprocessing steps and settings"):
    st.caption(
        "The \"no Monthly Rent\" option is always on here: the rent conversion step is skipped, "
        "rows aren't removed for blank rent, and rounding and log columns leave Monthly Rent out."
    )
    preprocessing_steps = st.multiselect(
        "Preprocessing steps", ALL_PREPROCESSING_STEPS, default=ALL_PREPROCESSING_STEPS
    )
    states_to_keep = st.multiselect("Keep rows from these states", ALL_STATES, default=["CO"])
    lease_bcr_maximum = st.number_input("Remove rows with Lease BCR above", min_value=0.0, value=0.25, step=0.05)
    log_method = st.radio(
        "Log columns",
        options=list(LOG_METHOD_NAMES),
        format_func=lambda method: LOG_METHOD_NAMES[method],
        horizontal=True,
    )

# Count the steps that will actually run (the rent conversion step never runs here).
cleaning_count = len(cleaning_steps)
preprocessing_count = len([step for step in preprocessing_steps if step != CLEAN_TARGET])
st.write(f"**{cleaning_count} cleaning steps and {preprocessing_count} preprocessing steps will run.**")

if ADJUST_YARD in cleaning_steps and CAP_SIZES not in cleaning_steps:
    st.warning(f"“{ADJUST_YARD}” is meant to run after “{CAP_SIZES}”. Consider adding it.")
if REMOVE_ROWS in preprocessing_steps and not states_to_keep:
    st.warning("No states are selected to keep, so every row will be removed.")


# --- 3. Run ---

st.header("3. Run")
default_csv_name = Path(uploaded_file.name).stem + " - test data"
csv_name = st.text_input("Name for the CSV download", value=default_csv_name).strip()

if st.button("Run cleaning and preprocessing", type="primary"):
    if not csv_name:
        st.error("Type a name for the CSV first.")
        st.stop()

    # Step A: clean a copy of the uploaded file, exactly like the Clean Data page.
    with tempfile.TemporaryDirectory() as temp_folder:
        working_file = Path(temp_folder) / "working_copy.xlsx"
        working_file.write_bytes(file_bytes)

        # Add any template columns the sheet is missing (no new rows are added).
        add_entries_to_excel(working_file, sheet, [])

        cleaning_report = run_cleaning_steps(
            path=working_file,
            sheet=sheet,
            chosen_steps=cleaning_steps,
            generate_script_address=generate_script_address,
            highway_types=highway_types,
            zoning_character=zoning_character,
        )
        cleaned_bytes = working_file.read_bytes()

    cleaned = pd.read_excel(BytesIO(cleaned_bytes), sheet_name=sheet)
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    # The cleaned sheet must have every column the model uses.
    missing_columns = [column for column in PREDICTIVE_COLUMNS if column not in cleaned.columns]
    if missing_columns:
        st.error(
            "After cleaning, the sheet is still missing these columns: " + ", ".join(missing_columns)
            + ". They're usually added by cleaning steps, so check that those steps are selected."
        )
        st.stop()

    # Step B: preprocess, exactly like the Preprocess page with "no Monthly Rent" ticked.
    # Sheet Row and the address columns ride along so each row can be matched to its property.
    # No preprocessing step reads or changes them.
    cleaned["Sheet Row"] = cleaned.index + 2  # +2: Excel rows start at 1, and row 1 is the header
    property_columns = [column for column in PROPERTY_COLUMNS if column in cleaned.columns]
    data = cleaned[property_columns + PREDICTIVE_COLUMNS]

    with st.spinner("Preprocessing..."):
        output, preprocessing_report = run_preprocessing_steps(
            data, preprocessing_steps, states_to_keep, lease_bcr_maximum, log_method, has_rent=False
        )

    # Split the property columns off, so test_data holds only the model inputs.
    properties = output[property_columns].reset_index(drop=True)
    test_data = output.drop(columns=property_columns).reset_index(drop=True)

    # Keep exactly what the CSV holds, read back the same way your training CSV will be read.
    # (Before this, a category column can mix numbers and text, which models can't use.)
    test_data = pd.read_csv(BytesIO(test_data.to_csv(index=False).encode("utf-8")))

    # Properties removed by preprocessing (they won't get a prediction). Empty sheet rows aren't listed.
    is_empty_row = cleaned.drop(columns="Sheet Row").isna().all(axis=1)
    was_removed = ~data["Sheet Row"].isin(properties["Sheet Row"])
    removed = data.loc[was_removed & ~is_empty_row, property_columns]

    # Keep everything in memory for the model pages.
    st.session_state["test_data"] = test_data
    st.session_state["test_data_properties"] = properties
    st.session_state["test_data_info"] = {
        "file_name": uploaded_file.name,
        "rows": len(test_data),
        "columns": len(test_data.columns),
    }
    st.session_state["test_data_removed"] = removed
    st.session_state["test_data_cleaning_report"] = cleaning_report
    st.session_state["test_data_preprocessing_report"] = preprocessing_report
    st.session_state["test_data_cleaned_bytes"] = cleaned_bytes
    st.session_state["test_data_csv_name"] = csv_name.removesuffix(".csv") + ".csv"

    st.rerun()  # Refresh so the "Test data in memory" message at the top shows the new data.


# --- Results (from the last run) ---

if "test_data" in st.session_state:
    test_data = st.session_state["test_data"]
    properties = st.session_state["test_data_properties"]
    removed = st.session_state["test_data_removed"]
    cleaning_report = st.session_state["test_data_cleaning_report"]
    preprocessing_report = st.session_state["test_data_preprocessing_report"]

    st.header("Results")

    # Step results. Open automatically if any step had a problem.
    cleaning_problems = any(icon == "⚠️" for icon, _, _ in cleaning_report)
    with st.expander("Cleaning step results", expanded=cleaning_problems):
        for icon, step_name, error_message in cleaning_report:
            if error_message:
                # "  \n" keeps the error's line breaks when Streamlit shows it.
                st.warning(f"**{step_name}** didn't finish:  \n" + error_message.replace("\n", "  \n"), icon=icon)
            else:
                st.write(f"{icon} {step_name}")

    preprocessing_problems = any(row[0] == "⚠️" for row in preprocessing_report)
    with st.expander("Preprocessing step results", expanded=preprocessing_problems):
        st.dataframe(
            pd.DataFrame(preprocessing_report, columns=["", "Step", "Rows after", "Notes"]),
            hide_index=True,
        )

    if len(removed):
        st.warning(f"{len(removed)} properties were removed during preprocessing and won't get a prediction:")
        st.dataframe(removed, hide_index=True)

    download_left, download_right, _ = st.columns([1, 1, 2])
    download_left.download_button(
        "Download test CSV",
        data=test_data.to_csv(index=False).encode("utf-8"),
        file_name=st.session_state["test_data_csv_name"],
        mime="text/csv",
        type="primary",
    )
    download_right.download_button(
        "Download cleaned Excel",
        data=st.session_state["test_data_cleaned_bytes"],
        file_name=st.session_state["test_data_csv_name"].removesuffix(".csv") + " (cleaned).xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Preview.
    st.header("Preview of the test data")
    st.caption(f"{len(test_data)} rows and {len(test_data.columns)} columns.")

    st.write("**Columns in the test data**")
    column_summary = pd.DataFrame({
        "Column": test_data.columns,
        "Type": [describe_column(test_data, column) for column in test_data.columns],
        "Filled": test_data.notna().sum().values,
        "Blank": test_data.isna().sum().values,
    })
    st.dataframe(column_summary, hide_index=True)

    st.write("**Data**")
    st.caption(
        f"The first {len(properties.columns)} columns ({', '.join(properties.columns)}) are shown so you can "
        "tell which property each row is. They're kept separately and aren't part of the test data or the CSV."
    )
    st.dataframe(table_for_display(pd.concat([properties, test_data], axis=1)), height=400, hide_index=True)
