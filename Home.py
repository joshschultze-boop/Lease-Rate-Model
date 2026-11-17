# Home.py
# Start the app from this folder with:   streamlit run Home.py
#
# This page:
#   1. Uploads an Excel file and shows a preview
#   2. Lets the user type in new entries
#   3. Links to the Clean Data page (pages/1_Clean_Data.py)

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st

from app_helpers import ALL_TEMPLATE_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from cleaning_script_functions import STATE_FIPS_TO_ABBR

st.set_page_config(page_title="Lease Comps Cleaner", page_icon="🏗️", layout="wide")


# ---------------------------------------------------------------------
# Memory that survives switching pages
# ---------------------------------------------------------------------
# st.session_state is a dictionary Streamlit keeps while the app is open.
# We store:
#   "file_bytes"  -> the uploaded Excel file
#   "file_name"   -> its name
#   "sheet_name"  -> which sheet to use
#   "new_entries" -> list of new rows typed in below (each row is a dictionary)

if "new_entries" not in st.session_state:
    st.session_state["new_entries"] = []

STATE_OPTIONS = sorted(set(STATE_FIPS_TO_ABBR.values()))
EARLIEST_DATE = date(1950, 1, 1)
LATEST_DATE = date(2100, 12, 31)


# ---------------------------------------------------------------------
# Input boxes for the new-entry form
# ---------------------------------------------------------------------

def entry_box(label, kind):
    """Show the right kind of input box and return what the user entered (None if blank)."""
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
        return st.date_input(label, value=None, min_value=EARLIEST_DATE, max_value=LATEST_DATE)
    raise ValueError(f"Unknown input kind: {kind}")


def entry_boxes(columns, required):
    """Show one box per column, spread across three side-by-side columns."""
    values = {}
    page_columns = st.columns(3)

    for position, (column_name, kind) in enumerate(columns.items()):
        label = column_name + " *" if required else column_name
        with page_columns[position % 3]:  # 0, 1, 2, 0, 1, 2, ...
            values[column_name] = entry_box(label, kind)

    return values


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("🏗️ Lease Comps Cleaner")
st.write("Upload your comps file, add any new entries, then open **Clean Data** to clean and download it.")


# ---------------------------------------------------------------------
# 1. Upload an Excel file
# ---------------------------------------------------------------------

st.header("1. Upload your Excel file")

uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.session_state["file_bytes"] = uploaded_file.getvalue()
    st.session_state["file_name"] = uploaded_file.name

if "file_bytes" not in st.session_state:
    st.info("Upload a file to see a preview here. You can start adding new entries below at any time.")
else:
    st.success(f"Current file: **{st.session_state['file_name']}**")

    # Pick which sheet to use (remember the last choice when coming back to this page).
    sheet_names = pd.ExcelFile(BytesIO(st.session_state["file_bytes"])).sheet_names
    previous_sheet = st.session_state.get("sheet_name")
    starting_index = sheet_names.index(previous_sheet) if previous_sheet in sheet_names else 0
    st.session_state["sheet_name"] = st.selectbox("Sheet to use", sheet_names, index=starting_index)

    # Preview the data.
    data = pd.read_excel(BytesIO(st.session_state["file_bytes"]), sheet_name=st.session_state["sheet_name"])
    st.caption(f"{len(data)} rows, {len(data.columns)} columns")
    st.dataframe(data, height=300)

    # Point out template columns the file doesn't have yet.
    file_columns = [str(name).strip() for name in data.columns]
    missing_columns = [name for name in ALL_TEMPLATE_COLUMNS if name not in file_columns]
    if missing_columns:
        st.warning(
            "These columns aren't in your file. They'll be added as blank columns when you clean it: "
            + ", ".join(missing_columns)
        )


# ---------------------------------------------------------------------
# 2. Add new entries
# ---------------------------------------------------------------------

st.header("2. Add new entries (optional)")
st.caption(
    "Fields marked * are required. The form keeps your values after you add an entry, "
    "so entering several leases at the same property is quick."
)

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

# Show the entries waiting to be added.
new_entries = st.session_state["new_entries"]

if new_entries:
    st.write(f"**New entries ({len(new_entries)})** — these are added to the bottom of your sheet when you run the cleaning.")
    st.dataframe(pd.DataFrame(new_entries), hide_index=True)

    remove_column, clear_column, _ = st.columns([1, 1, 4])
    if remove_column.button("Remove last entry"):
        new_entries.pop()
        st.rerun()
    if clear_column.button("Clear all entries"):
        new_entries.clear()
        st.rerun()


# ---------------------------------------------------------------------
# 3. Go to the cleaning page
# ---------------------------------------------------------------------

st.header("3. Clean the data")
st.page_link("pages/1_Clean_Data.py", label="Open Clean Data", icon="🧹")
