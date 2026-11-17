# app_helpers.py
# Things both pages need:
#   1. The new-entry template (which columns exist and what kind of box each one gets)
#   2. A function that adds the new entries to the bottom of an Excel sheet

from openpyxl import load_workbook


# ---------------------------------------------------------------------
# New-entry template
# ---------------------------------------------------------------------
# Each column name is paired with the kind of input box the user types into:
#   "text"         -> free text
#   "state"        -> dropdown of state abbreviations
#   "number"       -> decimal number (0 or more)
#   "whole number" -> whole number (0 or more)
#   "coordinate"   -> latitude / longitude (can be negative, 6 decimals)
#   "date"         -> calendar date picker
#
# Column names must match the names the cleaning functions look for exactly
# (for example "Street Address" with a capital A).

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


# ---------------------------------------------------------------------
# Add new entries to an Excel file
# ---------------------------------------------------------------------

def add_entries_to_excel(excel_path, sheet_name, new_entries):
    """
    Step 1: Add any template column the sheet is missing (as a blank column),
            so every cleaning step can find the columns it needs.
    Step 2: Add each new entry as a new row at the bottom of the sheet.

    new_entries is a list of dictionaries, e.g. [{"City": "Denver", "State": "CO", ...}]
    """
    workbook = load_workbook(excel_path)
    worksheet = workbook[sheet_name]

    # Match each header in row 1 with its column number, e.g. {"City": 3}
    header_columns = {}
    for cell in worksheet[1]:
        if cell.value is not None:
            header_columns[str(cell.value).strip()] = cell.column

    # Step 1: add missing template columns to the end of the header row.
    for column_name in ALL_TEMPLATE_COLUMNS:
        if column_name not in header_columns:
            new_column_number = worksheet.max_column + 1
            worksheet.cell(row=1, column=new_column_number, value=column_name)
            header_columns[column_name] = new_column_number

    # Step 2: find the last row that actually has data.
    # (Excel sometimes counts empty-but-formatted rows, which would leave a gap.)
    last_row = worksheet.max_row
    while last_row > 1 and all(cell.value is None for cell in worksheet[last_row]):
        last_row -= 1

    # Write each new entry on the next empty row. Blank fields are skipped.
    for entry in new_entries:
        last_row += 1
        for column_name, value in entry.items():
            if value is not None:
                worksheet.cell(row=last_row, column=header_columns[column_name], value=value)

    workbook.save(excel_path)
    workbook.close()
