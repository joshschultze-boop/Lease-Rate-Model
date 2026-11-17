# pages/3_Test_Data.py
# Cleans and preprocesses new properties (no Monthly Rent needed) in one run, using the same
# steps as the Clean Data and Preprocess pages, and keeps the result in memory for the models.
#
# Kept in memory (st.session_state) until the tab is closed, refreshed, or Clear is clicked:
#   "test_data"            -> the model inputs: same columns as the training CSV, minus rent
#   "test_data_properties" -> Sheet Row and address for each test_data row, in the same order
#   "test_data_run"        -> the file name, step reports, removed rows, and downloads from the run

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

import cleaning.steps as cleaning_steps
import preprocessing.steps as preprocessing_steps

st.set_page_config(page_title="Test Data", page_icon="🧪", layout="wide")

# Kept alongside the test data to identify each property. They are not model inputs.
PROPERTY_COLUMNS = ["Sheet Row", "Street Address", "Full Address"]

st.title("🧪 Test Data")
st.write("Clean and preprocess new properties in one step, so they're ready for the models. Monthly Rent isn't needed.")

if "test_data" in st.session_state:
    run = st.session_state["test_data_run"]
    st.success(f"Test data in memory: **{run['file_name']}**, {len(st.session_state['test_data'])} rows.")
    if st.button("Clear test data"):
        for key in ["test_data", "test_data_properties", "test_data_run"]:
            st.session_state.pop(key)
        st.rerun()


# --- 1. Upload ---

st.header("1. Upload the test file")
uploaded_file = st.file_uploader("Excel file of properties to predict (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload a sheet of properties to continue. It needs the same columns as your comp sheet, except Monthly Rent.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet = st.selectbox("Sheet to use", pd.ExcelFile(BytesIO(file_bytes)).sheet_names)
st.caption(f"{len(pd.read_excel(BytesIO(file_bytes), sheet_name=sheet).dropna(how='all'))} properties in this sheet.")


# --- 2. Steps (the same choices as the Clean Data and Preprocess pages) ---

st.header("2. Steps")
st.write("The same steps and settings as the Clean Data and Preprocess pages. If you changed any of them "
         "for your training data, make the same changes here so the formats match.")
with st.expander("Cleaning steps and settings"):
    cleaning_choices, cleaning_settings = cleaning_steps.choose_steps()
with st.expander("Preprocessing steps and settings"):
    st.caption("Monthly Rent is left out here: the rent conversion step is skipped, rows aren't removed "
               "for blank rent, and rounding and log columns leave Monthly Rent out.")
    preprocessing_choices, preprocessing_settings = preprocessing_steps.choose_steps(has_rent=False)


# --- 3. Run ---

st.header("3. Run")
csv_name = st.text_input("Name for the CSV download", value=Path(uploaded_file.name).stem + " - test data").strip()

if st.button("Run cleaning and preprocessing", type="primary", disabled=not csv_name):
    # A. Clean, exactly like the Clean Data page (no new entries are added).
    cleaned_bytes, cleaning_report = cleaning_steps.clean_excel(file_bytes, sheet, [], cleaning_choices, cleaning_settings)
    cleaned = pd.read_excel(BytesIO(cleaned_bytes), sheet_name=sheet)
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned["Sheet Row"] = cleaned.index + 2  # Excel rows start at 1, and row 1 is the header.

    # B. Preprocess, exactly like the Preprocess page but with no rent.
    #    Sheet Row and the address ride along so each row can be matched to its property.
    data, missing = preprocessing_steps.select_model_columns(cleaned, has_rent=False, extra_columns=PROPERTY_COLUMNS)
    if missing:
        st.error("After cleaning, the sheet is still missing these columns: " + ", ".join(missing)
                 + ". They're usually added by cleaning steps, so check that those steps are selected.")
        st.stop()
    with st.spinner("Preprocessing..."):
        output, preprocessing_report = preprocessing_steps.run_steps(
            data, preprocessing_choices, preprocessing_settings, has_rent=False
        )

    # C. Split the property columns off, so test_data holds only the model inputs, stored exactly
    #    as the CSV reads back (a category column can mix numbers and text before this).
    property_columns = [column for column in PROPERTY_COLUMNS if column in output.columns]
    properties = output[property_columns].reset_index(drop=True)
    test_data = output.drop(columns=property_columns).reset_index(drop=True)
    test_data = pd.read_csv(BytesIO(test_data.to_csv(index=False).encode("utf-8")))

    # Properties removed during preprocessing (they won't get a prediction). Empty sheet rows aren't listed.
    is_empty_row = cleaned.drop(columns="Sheet Row").isna().all(axis=1)
    removed = data.loc[~data["Sheet Row"].isin(properties["Sheet Row"]) & ~is_empty_row, property_columns]

    st.session_state["test_data"] = test_data
    st.session_state["test_data_properties"] = properties
    st.session_state["test_data_run"] = {
        "file_name": uploaded_file.name,
        "csv_name": csv_name.removesuffix(".csv") + ".csv",
        "cleaned_bytes": cleaned_bytes,
        "cleaning_report": cleaning_report,
        "preprocessing_report": preprocessing_report,
        "removed": removed,
    }
    st.rerun()  # Refresh so the "Test data in memory" message at the top shows the new data.


# --- Results (from the last run) ---

if "test_data" in st.session_state:
    test_data, run = st.session_state["test_data"], st.session_state["test_data_run"]
    st.header("Results")

    with st.expander("Cleaning step results", expanded=any(icon == "⚠️" for icon, *_ in run["cleaning_report"])):
        cleaning_steps.show_report(run["cleaning_report"])
    with st.expander("Preprocessing step results", expanded=any(row[0] == "⚠️" for row in run["preprocessing_report"])):
        preprocessing_steps.show_report(run["preprocessing_report"])

    if len(run["removed"]):
        st.warning(f"{len(run['removed'])} properties were removed during preprocessing and won't get a prediction:")
        st.dataframe(run["removed"], hide_index=True)

    left, right, _ = st.columns([1, 1, 2])
    left.download_button("Download test CSV", data=test_data.to_csv(index=False).encode("utf-8"),
                         file_name=run["csv_name"], mime="text/csv", type="primary")
    right.download_button("Download cleaned Excel", data=run["cleaned_bytes"],
                          file_name=run["csv_name"].removesuffix(".csv") + " (cleaned).xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.header("Preview of the test data")
    preprocessing_steps.show_preview(test_data, reference_columns=st.session_state["test_data_properties"])
