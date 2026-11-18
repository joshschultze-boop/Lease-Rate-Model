"""The comp sheet template: its columns, the form for typing in entries, and adding entries to a sheet.

The Home page (comps) and the Test Data page (properties to predict) both use the entry form.
Before cleaning, add_entries_to_excel adds any missing template columns and the typed-in entries.
"""

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from openpyxl import Workbook, load_workbook

from cleaning.functions import STATE_FIPS_TO_ABBR

# Each column is paired with the kind of input box the entry form shows for it:
#   "text", "state" (dropdown), "number" (0 or more), "whole number",
#   "coordinate" (can be negative, 6 decimals), or "date".
# Names must match the cleaning functions exactly (e.g. "Street Address", capital A).

REQUIRED_COLUMNS = {
    "Street Address": "text",
    "City": "text",
    "Zip": "text",  # text, so leading zeros like 02108 are kept
    "State": "state",
    "Yard Size (AC)": "number",
    "Lease Building (SF)": "number",
    "Lease Type": "text",
    "Lease Start": "date",
    "Monthly Rent": "number",
    "NNN": "text",  # the model treats NNN as a category, not a number
}

OPTIONAL_COLUMNS = {
    "Sale Date": "date",
    "Sale Price": "number",
    "Zoning": "text",
    "Year Built": "whole number",
    "Lease Length (Months)": "whole number",
    "Property (AC)": "number",
    "Property Building (SF)": "number",
    "Latitude": "coordinate",
    "Longitude": "coordinate",
}

ALL_TEMPLATE_COLUMNS = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
STATE_OPTIONS = sorted(set(STATE_FIPS_TO_ABBR.values()))


# ---------------------------------------------------------------------
# The entry form (shown on the page)
# ---------------------------------------------------------------------

def _entry_box(label, kind):
    """The right kind of input box for a column. Returns what was entered (None if blank)."""
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


def _entry_boxes(columns, required):
    """One box per column, spread across three side-by-side columns."""
    page_columns = st.columns(3)
    values = {}
    for position, (column_name, kind) in enumerate(columns.items()):
        with page_columns[position % 3]:
            values[column_name] = _entry_box(column_name + (" *" if required else ""), kind)
    return values


def entry_form(form_key, required_columns, optional_columns, entries):
    """
    A form with one box per column. When "Add entry" is clicked and every required field
    is filled in, the entry (a dictionary of column -> value) is added to `entries`.
    The form keeps its values afterwards, so entering several similar rows is quick.
    """
    with st.form(form_key):
        st.subheader("Required")
        entry = _entry_boxes(required_columns, required=True)
        with st.expander("Optional, but recommended"):
            entry.update(_entry_boxes(optional_columns, required=False))
        clicked = st.form_submit_button("Add entry", type="primary")

    if clicked:
        missing = [column for column in required_columns if entry[column] is None]
        if missing:
            st.error("Fill in these required fields first: " + ", ".join(missing))
        else:
            entries.append(entry)
            st.success(f"Added {entry['Street Address']}, {entry['City']}.")


def show_entries(entries, note):
    """The entries added so far, with buttons to remove the last one or clear them all."""
    if not entries:
        return
    st.write(f"**Entries ({len(entries)})**: {note}")
    st.dataframe(pd.DataFrame(entries), hide_index=True)
    remove_column, clear_column, _ = st.columns([1, 1, 4])
    if remove_column.button("Remove last entry"):
        entries.pop()
        st.rerun()
    if clear_column.button("Clear all entries"):
        entries.clear()
        st.rerun()


# ---------------------------------------------------------------------
# Excel files
# ---------------------------------------------------------------------

def blank_sheet():
    """An empty comp sheet with just the template headers (sheet "Sheet1"), as Excel file bytes."""
    workbook = Workbook()
    workbook.active.title = "Sheet1"
    workbook.active.append(ALL_TEMPLATE_COLUMNS)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def add_entries_to_excel(excel_path, sheet_name, new_entries):
    """
    1. Add any template column the sheet is missing (blank), so every cleaning step finds its columns.
    2. Add each new entry (a dictionary of column -> value) as a row at the bottom of the sheet.
    """
    workbook = load_workbook(excel_path)
    worksheet = workbook[sheet_name]

    # Header name -> column number, e.g. {"City": 3}
    header_columns = {str(cell.value).strip(): cell.column for cell in worksheet[1] if cell.value is not None}

    for column_name in ALL_TEMPLATE_COLUMNS:
        if column_name not in header_columns:
            header_columns[column_name] = worksheet.max_column + 1
            worksheet.cell(row=1, column=header_columns[column_name], value=column_name)

    # The last row with data (Excel can count empty-but-formatted rows, which would leave a gap).
    last_row = worksheet.max_row
    while last_row > 1 and all(cell.value is None for cell in worksheet[last_row]):
        last_row -= 1

    for entry in new_entries:
        last_row += 1
        for column_name, value in entry.items():
            if value is not None:
                worksheet.cell(row=last_row, column=header_columns[column_name], value=value)

    workbook.save(excel_path)
    workbook.close()
