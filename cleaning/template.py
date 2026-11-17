"""The comp sheet template: its columns, and adding new entries to a sheet.

The Home page builds its new-entry form from these columns. Before cleaning,
add_entries_to_excel adds any missing template columns and the new entries.
"""

from openpyxl import load_workbook

# Each column is paired with the kind of input box the Home page shows for it:
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
