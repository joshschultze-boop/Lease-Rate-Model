# Home.py  (start the app with:  streamlit run Home.py)
# Upload a comp sheet, add new entries, and go to the other pages.
#
# Kept in memory (st.session_state) for the Clean Data page:
#   "file_bytes", "file_name", "sheet_name"  -> the uploaded Excel file and the sheet to use
#   "new_entries"                            -> new rows typed in below (each row is a dictionary)

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st

from cleaning.functions import STATE_FIPS_TO_ABBR
from cleaning.template import ALL_TEMPLATE_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS

st.set_page_config(page_title="Lease Comps", page_icon="🏗️", layout="wide")
st.session_state.setdefault("new_entries", [])

STATE_OPTIONS = sorted(set(STATE_FIPS_TO_ABBR.values()))

PAGES = [
    ("pages/1_Clean_Data.py", "Clean Data", "🧹",
     "Adds your new entries, runs the cleaning steps you choose, and downloads the cleaned Excel file."),
    ("pages/2_Preprocess.py", "Preprocess", "📊",
     "Upload a cleaned sheet, choose the preprocessing steps, and download a CSV for the model."),
    ("pages/3_Test_Data.py", "Test Data", "🧪",
     "Upload new properties with no rent. They're cleaned and preprocessed the same way as your training data "
     "and kept in memory for the models."),
    ("pages/4_Predictions.py", "Predictions", "🔮",
     "Fit a model on your training CSV and predict rent for your test data, with errors when it has actual rent."),
]


def entry_box(label, kind):
    """Show the right kind of input box for a column and return what was entered (None if blank)."""
    if kind == "text":
        return st.text_input(label).strip() or None
    if kind == "state":
        return st.selectbox(label, STATE_OPTIONS, index=None, placeholder="Choose a state")
    if kind == "number":
        return st.number_input(label, min_value=0.0, value=None)
    if kind == "whole number":
        return st.number_input(label, min_value=0, step=1, value=None)
    if kind == "coordinate":
        return st.number_input(label, value=None, format="%.6f")
    if kind == "date":
        return st.date_input(label, value=None, min_value=date(1950, 1, 1), max_value=date(2100, 12, 31))
    raise ValueError(f"Unknown input kind: {kind}")


def entry_boxes(columns, required):
    """One box per column, spread across three side-by-side columns."""
    page_columns = st.columns(3)
    values = {}
    for position, (column_name, kind) in enumerate(columns.items()):
        with page_columns[position % 3]:
            values[column_name] = entry_box(column_name + (" *" if required else ""), kind)
    return values


st.title("🏗️ Lease Comps")
st.write("Upload your comp sheet and add any new entries here, then use the pages below.")


# --- 1. Upload ---

st.header("1. Upload your Excel file")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    st.session_state["file_bytes"] = uploaded_file.getvalue()
    st.session_state["file_name"] = uploaded_file.name

if "file_bytes" not in st.session_state:
    st.info("Upload a file to see a preview here. You can start adding new entries below at any time.")
else:
    st.success(f"Current file: **{st.session_state['file_name']}**")
    sheet_names = pd.ExcelFile(BytesIO(st.session_state["file_bytes"])).sheet_names
    previous = st.session_state.get("sheet_name")
    st.session_state["sheet_name"] = st.selectbox(
        "Sheet to use", sheet_names, index=sheet_names.index(previous) if previous in sheet_names else 0
    )

    data = pd.read_excel(BytesIO(st.session_state["file_bytes"]), sheet_name=st.session_state["sheet_name"])
    st.caption(f"{len(data)} rows, {len(data.columns)} columns")
    st.dataframe(data, height=300)

    missing = [name for name in ALL_TEMPLATE_COLUMNS if name not in [str(c).strip() for c in data.columns]]
    if missing:
        st.warning("These columns aren't in your file. They'll be added as blank columns when you clean it: "
                   + ", ".join(missing))


# --- 2. New entries ---

st.header("2. Add new entries (optional)")
st.caption("Fields marked * are required. The form keeps your values after you add an entry, "
           "so entering several leases at the same property is quick.")

with st.form("new_entry_form"):
    st.subheader("Required")
    new_entry = entry_boxes(REQUIRED_COLUMNS, required=True)
    with st.expander("Optional, but recommended"):
        new_entry.update(entry_boxes(OPTIONAL_COLUMNS, required=False))
    add_clicked = st.form_submit_button("Add entry", type="primary")

if add_clicked:
    missing_required = [name for name in REQUIRED_COLUMNS if new_entry[name] is None]
    if missing_required:
        st.error("Fill in these required fields first: " + ", ".join(missing_required))
    else:
        st.session_state["new_entries"].append(new_entry)
        st.success(f"Added {new_entry['Street Address']}, {new_entry['City']}.")

new_entries = st.session_state["new_entries"]
if new_entries:
    st.write(f"**New entries ({len(new_entries)})**: added to the bottom of your sheet when you run the cleaning.")
    st.dataframe(pd.DataFrame(new_entries), hide_index=True)
    remove_column, clear_column, _ = st.columns([1, 1, 4])
    if remove_column.button("Remove last entry"):
        new_entries.pop()
        st.rerun()
    if clear_column.button("Clear all entries"):
        new_entries.clear()
        st.rerun()


# --- 3. Pages ---

st.header("3. Next steps")
for page, label, icon, description in PAGES:
    st.page_link(page, label=label, icon=icon)
    st.caption(description)
