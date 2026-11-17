# preprocessing_steps.py
# The preprocessing steps, their settings, and the ORDER they run in.
# The Preprocess page and the Test Data page both use this file,
# so training data and test data always come out in exactly the same format.
#
#   preprocessing_functions.py  -> WHAT each step does
#   preprocessing_steps.py (this file) -> WHICH steps exist, their settings, and their ORDER

import pandas as pd

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

# Only these columns are read from the file, and they always come out in this order,
# no matter how the file is arranged. That keeps training and test CSVs lined up.
PREDICTIVE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + DATE_COLUMNS
MODEL_COLUMNS = PREDICTIVE_COLUMNS + TARGET_COLUMNS

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

ALL_PREPROCESSING_STEPS = [
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

def run_preprocessing_steps(df, chosen_steps, states_to_keep, lease_bcr_maximum, log_method, has_rent):
    """
    Check each step in order. If the user chose it, run it.
    has_rent=False is for test data with no Monthly Rent: every step runs the same way,
    except the parts that use Monthly Rent.

    Returns (finished data, report). The report has one line per step:
    (icon, step name, rows after the step, note).
    """
    report = []

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

    # Test data with no rent: leave Monthly Rent out of rounding (step 11) and logs (step 14).
    if has_rent:
        columns_to_round = ROUND_COLUMNS
        columns_to_log = LOG_COLUMNS
    else:
        columns_to_round = [column for column in ROUND_COLUMNS if column not in TARGET_COLUMNS]
        columns_to_log = [column for column in LOG_COLUMNS if column not in TARGET_COLUMNS]

    # 1. Rent must be a number before rows with $0 rent can be removed (step 8).
    #    Skipped for test data with no rent.
    if CLEAN_TARGET in chosen_steps and has_rent:
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

    # 8. Remove rows with $0 or blank rent (only when there is rent), rows from states
    #    not being kept, and rows that are blank or "unknown" in a required column.
    if REMOVE_ROWS in chosen_steps:
        exclude_values = {}
        if has_rent:
            exclude_values["Monthly Rent"] = [0, None]
        exclude_values["State"] = [state for state in ALL_STATES if state not in states_to_keep]
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
        df = run_step(ROUND, df, round_columns, columns=columns_to_round, decimals=2)

    # 12. Ratios run BEFORE step 13, so an unknown building size gives a blank ratio, not 0.
    if RATIOS in chosen_steps:
        df = run_step(RATIOS, df, add_yard_property_ratios)

    # 13. Zero fill runs BEFORE step 14, so filled zeros get a log value instead of staying blank.
    if ZERO_FILL in chosen_steps:
        df = run_step(ZERO_FILL, df, fill_blanks_with_zero, columns=ZERO_FILL_COLUMNS)

    # 14. Adds new columns named like "Monthly Rent_log1p". The original columns stay.
    if LOG_TRANSFORMS in chosen_steps:
        df = run_step(LOG_TRANSFORMS, df, add_log_transforms, columns=columns_to_log, method=log_method)

    return df, report


def describe_column(df, column):
    """Plain-language type for the preview table."""
    if column in DATE_COLUMNS and pd.api.types.is_numeric_dtype(df[column]):
        return f"Days since {EPOCH:%Y-%m-%d}"
    if column in CATEGORICAL_COLUMNS or isinstance(df[column].dtype, pd.CategoricalDtype):
        return "Category"
    if pd.api.types.is_numeric_dtype(df[column]):
        return "Number"
    return "Text"


def table_for_display(df):
    """
    Copy of the data that Streamlit can show in a table.
    A category column can hold numbers and text together (a Zip column with 80202 and
    "unknown"), which Streamlit can't display, so category columns are shown as text.
    The data itself is not changed.
    """
    df = df.copy()
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype("string")
    return df
