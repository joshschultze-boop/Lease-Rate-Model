"""The preprocessing steps: which exist, their settings, the ORDER they run in, and running them.

The Preprocess and Test Data pages both use this file, so training data and test data
always come out in exactly the same format. Each page calls:
    select_model_columns() -> keeps the columns the model uses, in a fixed order
    choose_steps()         -> shows the step list and settings, returns your choices
                              (start=... begins from earlier choices, e.g. the last Preprocess run)
    run_steps()            -> runs them
    show_report() / show_preview() -> shows how each step went, and the columns that will be saved

has_rent=False is for test data with no Monthly Rent: every step runs the same way, except the
rent parts and the training data's row filters (TRAINING_ONLY_STEPS). The Test Data page has its
own row rules instead (find_row_problems), so it removes only the rows you choose and says why.
Rows with Year Built above YEAR_BUILT_MAXIMUM are always removed, from training and test data alike.
"""

from datetime import date

import pandas as pd
import streamlit as st

from cleaning.functions import STATE_FIPS_TO_ABBR
from preprocessing.functions import (
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
# SETTINGS (the Inputs section of preprocessing_script.py)
# ---------------------------------------------------------------------

CATEGORICAL_COLUMNS = ["City", "State", "Zip", "County", "CBSA", "Lease Type", "Zoning", "NNN"]
NUMERICAL_COLUMNS = [
    "Latitude", "Longitude", "Yard Size (AC)", "Lease Building (SF)", "Property (AC)", "Property Building (SF)",
    "Year Built", "Sale Price", "Lease BCR", "Distance to Interstate", "Lease Length (Months)",
]
DATE_COLUMNS = ["Lease Start", "Sale Date"]
DEFAULT_START_DATE = date(1950, 1, 1)  # Dates become "days since" this date (changeable on the page).
TARGET_COLUMNS = ["Monthly Rent"]

# Only these columns are kept, always in this order (so training and test CSVs line up).
PREDICTIVE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + DATE_COLUMNS
MODEL_COLUMNS = PREDICTIVE_COLUMNS + TARGET_COLUMNS

# Rows that are blank (or "unknown") in any of these are removed.
REQUIRED_COLUMNS = [
    "CBSA", "County", "State", "Zip", "City", "Latitude", "Longitude", "Yard Size (AC)", "Lease Building (SF)", "NNN",
]

YEAR_BUILT_MAXIMUM = 9999
EXCLUDE_COLUMNS = ["Lease BCR"]  # Used to filter rows, then left out of the CSV.
ROUND_COLUMNS = ["Monthly Rent", "Distance to Interstate"]
ZERO_FILL_COLUMNS = ["Lease Building (SF)", "Property Building (SF)"]
# Log columns: the default choice (your preprocessing script's log_columns), and every column that can be chosen.
# Each chosen column gets a new "<column>_log1p" (or "_log") column; the original stays.
LOG_COLUMNS = ["Lease Building (SF)", "Yard Size (AC)", "Property Building (SF)", "Property (AC)", "Monthly Rent"]
RATIO_COLUMNS = ["yard / prop ratio (acres)", "yard / prop ratio (sf)"]  # made by the ratios step (step 12)
LOG_COLUMN_OPTIONS = LOG_COLUMNS + [  # every number column except Lease BCR (left out of the CSV), plus the ratios
    column for column in NUMERICAL_COLUMNS + RATIO_COLUMNS if column not in LOG_COLUMNS + EXCLUDE_COLUMNS
]

ALL_STATES = sorted(set(STATE_FIPS_TO_ABBR.values()))
LOG_METHOD_NAMES = {"log1p": "log1p: ln(1 + x), zeros stay 0", "log": "log: ln(x), zeros become blank"}


# ---------------------------------------------------------------------
# STEPS, listed in the order they run
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
    CLEAN_TARGET, COMBINE_CITY_ZIP, COMBINE_CITY_ZONING, COMBINE_COUNTY_ZONING, CONVERT_TYPES, SALE_UNKNOWN,
    LEASE_LENGTH_UNKNOWN, REMOVE_ROWS, REMOVE_ABOVE_MAXIMUMS, DROP_COLUMNS, ROUND, RATIOS, ZERO_FILL, LOG_TRANSFORMS,
]

# Never run on test data (no rent): the rent conversion, and the training data's row filters.
TRAINING_ONLY_STEPS = [CLEAN_TARGET, REMOVE_ROWS, REMOVE_ABOVE_MAXIMUMS]


# ---------------------------------------------------------------------
# Row rules for test data (the Test Data page chooses which to apply)
# ---------------------------------------------------------------------
# Judged the same way as the training filters in steps 8-9, but with the Test Data page's own limits.

MISSING_REQUIRED = "Are missing required information"
STATE_NOT_KEPT = "Are in a state not in the list"
BCR_ABOVE_LIMIT = "Have Lease BCR above the limit"
YEAR_BUILT_TOO_HIGH = f"Have Year Built above {YEAR_BUILT_MAXIMUM}"
ROW_RULES = [MISSING_REQUIRED, STATE_NOT_KEPT, BCR_ABOVE_LIMIT]   # the ones you can choose
ALWAYS_APPLIED_RULES = [YEAR_BUILT_TOO_HIGH]                      # always removed, not a choice


def find_row_problems(table, states_to_keep, lease_bcr_maximum):
    """
    For each row of cleaned data, why each rule would remove it ("" when the row passes).
    One column per rule in ROW_RULES + ALWAYS_APPLIED_RULES, e.g. "Missing CBSA, County" or
    "Lease BCR 0.40 is above 0.25".
    """
    problems = pd.DataFrame("", index=table.index, columns=ROW_RULES + ALWAYS_APPLIED_RULES)

    # Required info: number columns must hold a number; other columns can't be blank or "unknown".
    def is_missing(column):
        if column in NUMERICAL_COLUMNS:
            return pd.to_numeric(table[column], errors="coerce").isna()
        return table[column].isna() | table[column].astype("string").str.strip().str.casefold().eq("unknown").fillna(False)

    missing = pd.DataFrame({column: is_missing(column) for column in REQUIRED_COLUMNS})
    for row in table.index[missing.any(axis=1)]:
        problems.at[row, MISSING_REQUIRED] = "Missing " + ", ".join(missing.columns[missing.loc[row]])

    state_out = table["State"].isin([state for state in ALL_STATES if state not in states_to_keep])
    problems.loc[state_out, STATE_NOT_KEPT] = "State is " + table.loc[state_out, "State"].astype(str)

    bcr = pd.to_numeric(table["Lease BCR"], errors="coerce")
    over = bcr > lease_bcr_maximum
    problems.loc[over, BCR_ABOVE_LIMIT] = bcr[over].map(lambda value: f"Lease BCR {value:.2f} is above {lease_bcr_maximum:g}")

    year_built = pd.to_numeric(table["Year Built"], errors="coerce")
    too_high = year_built > YEAR_BUILT_MAXIMUM
    problems.loc[too_high, YEAR_BUILT_TOO_HIGH] = year_built[too_high].map(
        lambda value: f"Year Built {value:g} is above {YEAR_BUILT_MAXIMUM}")

    return problems


# ---------------------------------------------------------------------
# Choosing columns and steps (shown on the page)
# ---------------------------------------------------------------------

def select_model_columns(table, has_rent, extra_columns=()):
    """
    Keep the model columns, in the fixed order above, after any extra columns that are present
    (the Test Data page keeps Sheet Row and the address alongside; no step touches them).
    Returns (data, missing model columns). data is None if any are missing.
    """
    needed = MODEL_COLUMNS if has_rent else PREDICTIVE_COLUMNS
    missing = [column for column in needed if column not in table.columns]
    extra = [column for column in extra_columns if column in table.columns]
    return (None if missing else table[extra + needed]), missing


def choose_steps(has_rent, start=None):
    """
    Show the step list, the order they'll run in, and the settings. Returns (chosen steps, settings).
    start: earlier choices to begin from ({"steps": [...], "settings": {...}}), or None for the defaults.
    """
    options = ALL_STEPS if has_rent else [step for step in ALL_STEPS if step not in TRAINING_ONLY_STEPS]
    start_steps = [step for step in (start or {}).get("steps", options) if step in options]
    start_settings = (start or {}).get("settings", {})

    chosen = st.multiselect("Preprocessing steps", options, default=start_steps)
    in_order = [step for step in options if step in chosen]
    if in_order:
        st.markdown("**These steps will run in this order:**\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(in_order, 1)))

    st.write("**Settings**")
    settings = {}
    if has_rent:  # Row filters are for training data only.
        settings["states_to_keep"] = st.multiselect("Keep rows from these states", ALL_STATES,
                                                    default=start_settings.get("states_to_keep", ["CO"]))
        settings["lease_bcr_maximum"] = st.number_input("Remove rows with Lease BCR above", min_value=0.0, step=0.05,
                                                        value=start_settings.get("lease_bcr_maximum", 0.25))
        if REMOVE_ROWS in chosen and not settings["states_to_keep"]:
            st.warning("No states are selected to keep, so every row will be removed.")
    # Log columns. Test data (no rent) can't log Monthly Rent, so it isn't offered there.
    log_options = [column for column in LOG_COLUMN_OPTIONS if has_rent or column not in TARGET_COLUMNS]
    settings["log_columns"] = st.multiselect(
        "Log columns: add a log version of these columns", log_options,
        default=[column for column in start_settings.get("log_columns", LOG_COLUMNS) if column in log_options],
        help="Each one gets a new column (e.g. \"Yard Size (AC)_log1p\"); the original column stays. "
             "Values that can't be logged (0 or less for log, -1 or less for log1p) become blank. "
             "Default: your preprocessing script's log_columns.",
    )
    if LOG_TRANSFORMS in chosen and not settings["log_columns"]:
        st.caption("No log columns are chosen, so the log step won't add any.")
    if LOG_TRANSFORMS in chosen and RATIOS not in chosen and set(settings["log_columns"]) & set(RATIO_COLUMNS):
        st.warning(f"The ratio columns only exist when “{RATIOS}” runs. Add that step, or remove the ratios "
                   "from the log columns.")
    settings["log_method"] = st.radio("Log method", list(LOG_METHOD_NAMES), format_func=LOG_METHOD_NAMES.get,
                                      index=list(LOG_METHOD_NAMES).index(start_settings.get("log_method", "log1p")),
                                      horizontal=True)
    settings["start_date"] = st.date_input(
        "Dates: count days from", value=start_settings.get("start_date", DEFAULT_START_DATE),
        min_value=date(1900, 1, 1), max_value=date(2100, 12, 31), format="YYYY-MM-DD",
        help="Lease Start and Sale Date become the number of days since this date. "
             "Training and test data must use the same date.",
    )
    return chosen, settings


# ---------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------

def run_steps(df, chosen_steps, settings, has_rent):
    """
    Check each step in order. If it was chosen, run it.
    Returns (finished data, report). The report has one (icon, step, rows after, note) per step.
    """
    report = []
    if not has_rent:
        chosen_steps = [step for step in chosen_steps if step not in TRAINING_ONLY_STEPS]

    def run_step(step_name, df, step_function, **arguments):
        # A failed step is recorded and the data passes on unchanged, so the rest still run.
        try:
            new_df = step_function(df, **arguments)
        except Exception as error:
            report.append(("⚠️", step_name, len(df), f"Skipped: {error}"))
            return df
        rows_removed = len(df) - len(new_df)
        report.append(("✅", step_name, len(new_df), f"{rows_removed} rows removed" if rows_removed else ""))
        return new_df

    # With no rent, Monthly Rent is left out of rounding (step 11) and logs (step 14).
    not_rent = lambda columns: columns if has_rent else [c for c in columns if c not in TARGET_COLUMNS]

    # 1. Rent must be a number before rows with $0 rent can be removed (step 8).
    if CLEAN_TARGET in chosen_steps:
        df = run_step(CLEAN_TARGET, df, clean_target_columns, target_columns=TARGET_COLUMNS)

    # 2-4. Combine columns. BEFORE step 5 (same as your script), so a combined value is blank
    #      when either part is blank, rather than containing "unknown".
    for step, columns, new_column in [
        (COMBINE_CITY_ZIP, ["City", "Zip"], "City-Zip"),
        (COMBINE_CITY_ZONING, ["City", "Zoning"], "City-Zoning"),
        (COMBINE_COUNTY_ZONING, ["County", "Zoning"], "County-Zoning"),
    ]:
        if step in chosen_steps:
            df = run_step(step, df, combine_categorical_columns, columns=columns, new_column=new_column)

    # 5. Blank categories become "unknown", text in number columns becomes blank,
    #    and dates become days since the chosen start date.
    if CONVERT_TYPES in chosen_steps:
        df = run_step(CONVERT_TYPES, df, convert_data_types, categorical_columns=CATEGORICAL_COLUMNS,
                      numerical_columns=NUMERICAL_COLUMNS, date_columns=DATE_COLUMNS,
                      epoch=pd.Timestamp(settings["start_date"]))

    # 6-7. 0/1 flag columns: 1 when the value is blank or "unknown".
    if SALE_UNKNOWN in chosen_steps:
        df = run_step(SALE_UNKNOWN, df, add_unknown_indicator,
                      columns=["Sale Price", "Sale Date"], new_column="Sale Information Unknown")
    if LEASE_LENGTH_UNKNOWN in chosen_steps:
        df = run_step(LEASE_LENGTH_UNKNOWN, df, add_unknown_indicator,
                      columns="Lease Length (Months)", new_column="Lease Length Unknown")

    # 8. Remove rows with $0 or blank rent, rows from states not kept,
    #    and rows that are blank or "unknown" in a required column.
    if REMOVE_ROWS in chosen_steps:
        exclude_values = {
            "Monthly Rent": [0, None],
            "State": [state for state in ALL_STATES if state not in settings["states_to_keep"]],
        }
        df = run_step(REMOVE_ROWS, df, clean_dataframe, exclude_values=exclude_values, required_columns=REQUIRED_COLUMNS)

    # 9. Needs numbers (step 5) and needs Lease BCR, so it runs before step 10.
    if REMOVE_ABOVE_MAXIMUMS in chosen_steps:
        df = run_step(REMOVE_ABOVE_MAXIMUMS, df, remove_rows_above_maximums,
                      maximum_values={"Lease BCR": settings["lease_bcr_maximum"], "Year Built": YEAR_BUILT_MAXIMUM})

    # 10. Lease BCR was only needed for step 9.
    if DROP_COLUMNS in chosen_steps:
        df = run_step(DROP_COLUMNS, df, drop_excluded_columns, columns_to_exclude=EXCLUDE_COLUMNS)

    # 11. Round to 2 decimals.
    if ROUND in chosen_steps:
        df = run_step(ROUND, df, round_columns, columns=not_rent(ROUND_COLUMNS), decimals=2)

    # 12. Ratios BEFORE step 13, so an unknown building size gives a blank ratio, not 0.
    if RATIOS in chosen_steps:
        df = run_step(RATIOS, df, add_yard_property_ratios)

    # 13. Zero fill BEFORE step 14, so filled zeros get a log value instead of staying blank.
    if ZERO_FILL in chosen_steps:
        df = run_step(ZERO_FILL, df, fill_blanks_with_zero, columns=ZERO_FILL_COLUMNS)

    # 14. New columns named like "Monthly Rent_log1p". The original columns stay.
    if LOG_TRANSFORMS in chosen_steps:
        # The chosen columns, always in the order of LOG_COLUMN_OPTIONS (however they were clicked), so the new
        # log columns line up between training and test data.
        chosen_log_columns = settings.get("log_columns", LOG_COLUMNS)
        columns_to_log = not_rent([column for column in LOG_COLUMN_OPTIONS if column in chosen_log_columns])
        if columns_to_log:
            df = run_step(LOG_TRANSFORMS, df, add_log_transforms, columns=columns_to_log, method=settings["log_method"])

    return df, report


# ---------------------------------------------------------------------
# Showing the results
# ---------------------------------------------------------------------

def show_report(report):
    st.dataframe(pd.DataFrame(report, columns=["", "Step", "Rows after", "Notes"]), hide_index=True)


def show_preview(data, start_date, reference_columns=None):
    """
    The columns that will be saved (type, filled, blank), then the data itself.
    start_date is the date the date columns count days from.
    reference_columns (optional) are shown on the left of the data so you can tell rows apart;
    they aren't part of the saved data.
    """
    st.caption(f"{len(data)} rows and {len(data.columns)} columns.")

    def describe(column):
        if column in DATE_COLUMNS and pd.api.types.is_numeric_dtype(data[column]):
            return f"Days since {start_date:%Y-%m-%d}"
        if column in CATEGORICAL_COLUMNS or isinstance(data[column].dtype, pd.CategoricalDtype):
            return "Category"
        return "Number" if pd.api.types.is_numeric_dtype(data[column]) else "Text"

    st.write("**Columns**")
    st.dataframe(pd.DataFrame({
        "Column": data.columns,
        "Type": [describe(column) for column in data.columns],
        "Filled": data.notna().sum().values,
        "Blank": data.isna().sum().values,
    }), hide_index=True)

    st.write("**Data**")
    shown = data
    if reference_columns is not None:
        st.caption(f"{', '.join(reference_columns.columns)} are shown for reference and aren't part of the saved data.")
        shown = pd.concat([reference_columns, data], axis=1)

    # A category column can mix numbers and text (Zip: 80202 and "unknown"), which the table
    # can't display, so category columns are shown as text. The data itself isn't changed.
    shown = shown.apply(lambda col: col.astype("string") if isinstance(col.dtype, pd.CategoricalDtype) else col)
    st.dataframe(shown, height=400, hide_index=True)
