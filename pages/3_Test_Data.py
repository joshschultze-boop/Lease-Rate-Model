# pages/3_Test_Data.py
# Cleans and preprocesses properties to predict (no Monthly Rent needed) in one run, using the same
# steps as the Clean Data and Preprocess pages, and keeps the result in memory for the models.
# Properties can come from an uploaded sheet, be typed in on this page, or both.
#
# The steps and settings start from your last Clean Data and Preprocess runs (including the start
# date for dates), so test data comes out in the same format as your training data.
#
# The training data's row filters (Lease BCR, state, ...) don't run here. Rows are removed only by
# the rules you choose in section 3 (plus Year Built above 9999, always), each listed with the reason.
#
# Kept in memory (st.session_state) until the tab is closed or refreshed:
#   "test_entries"         -> properties typed in on this page
#   "test_data"            -> the model inputs: same columns as the training CSV, minus rent
#   "test_data_properties" -> Sheet Row and address for each test_data row, in the same order
#   "test_data_run"        -> where the data came from, step reports, removed/flagged rows, downloads

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

import cleaning.steps as cleaning_steps
import preprocessing.steps as preprocessing_steps
from cleaning.template import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, blank_sheet, entry_form, show_entries

st.set_page_config(page_title="Test Data", page_icon="🧪", layout="wide")
st.session_state.setdefault("test_entries", [])

# Kept alongside the test data to identify each property. They are not model inputs.
PROPERTY_COLUMNS = ["Sheet Row", "Street Address", "Full Address"]

# The entry form asks for everything a comp needs except Monthly Rent.
TEST_REQUIRED_COLUMNS = {column: kind for column, kind in REQUIRED_COLUMNS.items() if column != "Monthly Rent"}

st.title("🧪 Test Data")
st.write("Clean and preprocess properties to predict, so they're ready for the models. Monthly Rent isn't needed.")

if "test_data" in st.session_state:
    st.success(f"Test data in memory: **{st.session_state['test_data_run']['file_name']}**, "
               f"{len(st.session_state['test_data'])} rows. The Predictions page uses it while this tab is open.")
    if st.button("Clear test data"):
        for key in ["test_data", "test_data_properties", "test_data_run"]:
            st.session_state.pop(key)
        st.rerun()


# --- 1. Properties ---

st.header("1. Properties to predict")
st.write("Upload a sheet, type properties in below, or both.")

uploaded_file = st.file_uploader("Excel file of properties (optional)", type=["xlsx"])
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    sheet = st.selectbox("Sheet to use", pd.ExcelFile(BytesIO(file_bytes)).sheet_names)
    uploaded_count = len(pd.read_excel(BytesIO(file_bytes), sheet_name=sheet).dropna(how="all"))
else:
    file_bytes, sheet, uploaded_count = blank_sheet(), "Sheet1", 0  # typed-in properties go into an empty sheet

st.subheader("Type in properties")
entries = st.session_state["test_entries"]
entry_form("test_entry_form", TEST_REQUIRED_COLUMNS, OPTIONAL_COLUMNS, entries)
show_entries(entries, "added after any uploaded rows when you run.")

property_count = uploaded_count + len(entries)
st.caption(f"{property_count} properties to run: {uploaded_count} uploaded and {len(entries)} typed in.")


# --- 2. Steps (the same choices as the Clean Data and Preprocess pages) ---

st.header("2. Steps")
st.write("The same steps and settings as the Clean Data and Preprocess pages. If you changed any of them "
         "for your training data, make the same changes here so the formats match.")
cleaning_start, preprocessing_start = st.session_state.get("cleaning_result"), st.session_state.get("preprocess_result")
with st.expander("Cleaning steps and settings"):
    st.caption("Starting from your Clean Data run in this session." if cleaning_start else
               "No Clean Data run in this session, so these are the defaults.")
    cleaning_choices, cleaning_settings = cleaning_steps.choose_steps(start=cleaning_start)
with st.expander("Preprocessing steps and settings"):
    st.caption(("Starting from your Preprocess run in this session, including its start date for dates. "
                if preprocessing_start else
                "No Preprocess run in this session, so these are the defaults. Set the start date for dates "
                "to the one your training data used. ")
               + "Without Monthly Rent, the rent conversion step doesn't run, and rounding and log columns leave "
                 "Monthly Rent out. The training data's row filters don't run either: choose row rules in section 3.")
    preprocessing_choices, preprocessing_settings = preprocessing_steps.choose_steps(has_rent=False, start=preprocessing_start)


# --- 3. Rows to remove ---

st.header("3. Rows to remove")
st.write(f"Every property is kept unless you choose a rule here (rows with Year Built above "
         f"{preprocessing_steps.YEAR_BUILT_MAXIMUM} are always removed). After the run, removed rows are listed "
         "with the reason, and so are kept rows that break a rule you didn't choose.")
remove_rules = st.multiselect("Remove test rows that…", preprocessing_steps.ROW_RULES, default=[], placeholder="Keep every row")
left, right = st.columns(2)
states_to_keep = left.multiselect("States to keep (state rule)", preprocessing_steps.ALL_STATES, default=["CO"])
lease_bcr_maximum = right.number_input("Lease BCR limit (BCR rule)", min_value=0.0, value=0.25, step=0.05)


# --- 4. Run ---

st.header("4. Run")
source_name = Path(uploaded_file.name).stem if uploaded_file is not None else "properties"
csv_name = st.text_input("Name for the CSV download", value=f"{source_name} - test data").strip()

if st.button("Run cleaning and preprocessing", type="primary", disabled=property_count == 0 or not csv_name):
    # A. Clean the uploaded rows, then the typed-in rows, exactly like the Clean Data page.
    cleaned_bytes, cleaning_report = cleaning_steps.clean_excel(file_bytes, sheet, entries, cleaning_choices, cleaning_settings)
    cleaned = pd.read_excel(BytesIO(cleaned_bytes), sheet_name=sheet)
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned["Sheet Row"] = cleaned.index + 2  # The row in the cleaned Excel download (row 1 is the header).
    cleaned = cleaned.loc[~cleaned.drop(columns="Sheet Row").isna().all(axis=1)]  # Skip empty rows.

    data, missing = preprocessing_steps.select_model_columns(cleaned, has_rent=False, extra_columns=PROPERTY_COLUMNS)
    if missing:
        st.error("After cleaning, the sheet is still missing these columns: " + ", ".join(missing)
                 + ". They're usually added by cleaning steps, so check that those steps are selected.")
        st.stop()

    # B. Remove only the rows caught by the rules you chose, and record why for every caught row.
    problems = preprocessing_steps.find_row_problems(data, states_to_keep, lease_bcr_maximum)
    other_rules = [rule for rule in preprocessing_steps.ROW_RULES if rule not in remove_rules]
    property_columns = [column for column in PROPERTY_COLUMNS if column in data.columns]

    def reasons(rows, rules):
        """Each row's reasons under these rules, e.g. "Missing CBSA; Lease BCR 0.40 is above 0.25"."""
        return ["; ".join(problems.at[row, rule] for rule in rules if problems.at[row, rule]) for row in rows]

    applied_rules = remove_rules + preprocessing_steps.ALWAYS_APPLIED_RULES
    is_removed = problems[applied_rules].ne("").any(axis=1)
    is_flagged = ~is_removed & problems[other_rules].ne("").any(axis=1)
    removed = data.loc[is_removed, property_columns].assign(Reason=reasons(data.index[is_removed], applied_rules))
    flagged = data.loc[is_flagged, property_columns].assign(**{"Kept, but": reasons(data.index[is_flagged], other_rules)})
    if is_removed.all():
        st.error("Every property was removed:")
        st.dataframe(removed, hide_index=True)
        st.stop()

    # C. Preprocess the rest, exactly like the Preprocess page but without rent.
    #    Sheet Row and the address ride along so each row can be matched to its property.
    with st.spinner("Preprocessing..."):
        output, preprocessing_report = preprocessing_steps.run_steps(
            data.loc[~is_removed], preprocessing_choices, preprocessing_settings, has_rent=False
        )

    # D. Split the property columns off, so test_data holds only the model inputs, stored exactly
    #    as the CSV reads back (a category column can mix numbers and text before this).
    properties = output[property_columns].reset_index(drop=True)
    test_data = output.drop(columns=property_columns).reset_index(drop=True)
    test_data = pd.read_csv(BytesIO(test_data.to_csv(index=False).encode("utf-8")))

    sources = ([uploaded_file.name] if uploaded_file is not None else []) + (
        [f"{len(entries)} typed-in {'property' if len(entries) == 1 else 'properties'}"] if entries else [])
    st.session_state["test_data"] = test_data
    st.session_state["test_data_properties"] = properties
    st.session_state["test_data_run"] = {
        "file_name": " and ".join(sources),
        "csv_name": csv_name.removesuffix(".csv") + ".csv",
        "cleaned_bytes": cleaned_bytes,
        "cleaning_report": cleaning_report,
        "preprocessing_report": preprocessing_report,
        "removed": removed,
        "flagged": flagged,
        "start_date": preprocessing_settings["start_date"],
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
        st.warning(f"{len(run['removed'])} properties were removed and won't get a prediction:")
        st.dataframe(run["removed"], hide_index=True)
    if len(run["flagged"]):
        st.info(f"{len(run['flagged'])} properties break a rule you didn't choose, so they were kept:")
        st.dataframe(run["flagged"], hide_index=True)

    st.caption("Kept in memory for the Predictions page. Download a copy only if you want to keep it.")
    left, right, _ = st.columns([1, 1, 2])
    left.download_button("Download test CSV", data=test_data.to_csv(index=False).encode("utf-8"),
                         file_name=run["csv_name"], mime="text/csv")
    right.download_button("Download cleaned Excel", data=run["cleaned_bytes"],
                          file_name=run["csv_name"].removesuffix(".csv") + " (cleaned).xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.page_link("pages/4_Predictions.py", label="Next: Predictions", icon="🔮")

    st.header("Preview of the test data")
    preprocessing_steps.show_preview(test_data, run["start_date"], reference_columns=st.session_state["test_data_properties"])
