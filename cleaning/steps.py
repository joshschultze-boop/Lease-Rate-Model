"""The cleaning steps: which exist, their settings, the ORDER they run in, and running them.

The Clean Data and Test Data pages both use this file, so training data and test data
are always cleaned exactly the same way. Each page calls:
    choose_steps()   -> shows the step list and settings, returns your choices
                        (start=... begins from earlier choices, e.g. the last Clean Data run)
    clean_excel()    -> cleans a copy of an Excel file with those choices
    show_report()    -> shows how each step went
"""

import tempfile
from pathlib import Path

import streamlit as st

from cleaning.functions import (
    add_distance_to_highway,
    address_abbreviator,
    adjust_yard_size_for_building,
    cap_sizes_to_property,
    cbsa_addition_by_city,
    cbsa_addition_by_county,
    clean_column,
    fill_addresses,
    fill_building_coverage,
    fill_missing_values_for_duplicate_keys,
    geocode_address,
    rent_fill,
    standardize_zoning_character,
)
from cleaning.template import add_entries_to_excel


# ---------------------------------------------------------------------
# SETTINGS (the INPUTS section of your original cleaning script)
# ---------------------------------------------------------------------

# Lookup files in the app's data folder (this file is in cleaning/, so go up one level).
DATA_FOLDER = Path(__file__).parent.parent / "data"
CBSA_FILE_CITY = DATA_FOLDER / "list2_2023.xlsx"
CBSA_FILE_COUNTY = DATA_FOLDER / "list1_2023.xlsx"
HIGHWAY_FILE = DATA_FOLDER / "tl_2025_us_primaryroads" / "tl_2025_us_primaryroads.shp"

CAPITALIZATION_COLUMNS = ["Street Address", "City", "State", "County", "Full Address", "Script Address", "CBSA", "Submarket"]
ABBREVIATION_COLUMNS = ["Street Address", "Full Address", "Script Address"]

# Rows with the same Full Address + County are treated as the same property.
PROPERTY_MATCH_COLUMNS = ["Full Address", "County"]
PROPERTY_FILL_COLUMNS = ["Property (AC)", "Property Building (SF)", "Year Built", "Zoning", "Sale Price", "Sale Date"]

HIGHWAY_TYPE_NAMES = {"I": "Interstates", "U": "U.S. routes", "S": "State highways"}


# ---------------------------------------------------------------------
# STEPS, listed in the order they run
# ---------------------------------------------------------------------

CLEAN_TEXT = "Clean text (abbreviate street types, uppercase)"
BUILD_FULL_ADDRESS = "Build Full Address"
SHARE_COORDINATES = "Copy Latitude/Longitude to rows with the same address"
GEOCODE = "Geocode addresses (Latitude, Longitude, County, Script Address)"
CBSA_BY_CITY = "Add CBSA by City"
CBSA_BY_COUNTY = "Add CBSA by County"
ANNUAL_RENT = "Fill Annual Rent (Monthly Rent x 12)"
HIGHWAY_DISTANCE = "Distance to highway"
SHARE_PROPERTY_INFO = "Copy property info to rows of the same property"
CAP_SIZES = "Cap yard and building sizes at property sizes"
ADJUST_YARD = "Adjust yard size for building footprint"
LEASE_BCR = "Lease building coverage ratio (Lease BCR)"
PROPERTY_BCR = "Property building coverage ratio (Property BCR)"
STANDARDIZE_ZONING = "Standardize zoning codes"

ALL_STEPS = [
    CLEAN_TEXT, BUILD_FULL_ADDRESS, SHARE_COORDINATES, GEOCODE, CBSA_BY_CITY, CBSA_BY_COUNTY, ANNUAL_RENT,
    HIGHWAY_DISTANCE, SHARE_PROPERTY_INFO, CAP_SIZES, ADJUST_YARD, LEASE_BCR, PROPERTY_BCR, STANDARDIZE_ZONING,
]


# ---------------------------------------------------------------------
# Choosing steps (shown on the page)
# ---------------------------------------------------------------------

def choose_steps(start=None):
    """
    Show the step list, the order they'll run in, and the settings. Returns (chosen steps, settings).
    start: earlier choices to begin from ({"steps": [...], "settings": {...}}), or None for the defaults.
    """
    start_steps = (start or {}).get("steps", ALL_STEPS)
    start_settings = (start or {}).get("settings", {})

    chosen = st.multiselect("Cleaning steps", ALL_STEPS, default=start_steps)
    in_order = [step for step in ALL_STEPS if step in chosen]
    if in_order:
        st.markdown("**These steps will run in this order:**\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(in_order, 1)))
    if ADJUST_YARD in chosen and CAP_SIZES not in chosen:
        st.warning(f"“{ADJUST_YARD}” is meant to run after “{CAP_SIZES}”. Consider adding it.")

    st.write("**Settings**")
    settings = {
        "generate_script_address": st.checkbox("Geocoding: also fill Script Address and County",
                                               value=start_settings.get("generate_script_address", True)),
        "highway_types": st.multiselect(
            "Distance to highway: road types to measure to", list(HIGHWAY_TYPE_NAMES),
            default=start_settings.get("highway_types", ["I"]), format_func=HIGHWAY_TYPE_NAMES.get,
        ),
        "zoning_character": st.text_input("Zoning: character to standardize codes with",
                                          value=start_settings.get("zoning_character", "-")),
    }
    return chosen, settings


# ---------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------

def clean_excel(file_bytes, sheet, new_entries, chosen_steps, settings):
    """
    Clean a copy of an Excel file (the original is never changed).
    Adds missing template columns and any new entries first.
    Returns (cleaned file as bytes, report).
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "working_copy.xlsx"
        path.write_bytes(file_bytes)
        add_entries_to_excel(path, sheet, new_entries)
        report = run_steps(path, sheet, chosen_steps, settings)
        return path.read_bytes(), report


def run_steps(path, sheet, chosen_steps, settings):
    """
    Check each step in order. If it was chosen, run it.
    Every function reads the Excel file at `path`, changes it, and saves it back.
    Returns a report: one (icon, step name, error message or "") per step.
    """
    report = []

    def run_step(step_name, cleaning_function, **arguments):
        # A failed step is recorded and the rest still run (like rent_fill in your script).
        with st.spinner(f"Running: {step_name}"):
            try:
                cleaning_function(**arguments)
                report.append(("✅", step_name, ""))
            except Exception as error:
                report.append(("⚠️", step_name, str(error)))

    # The three arguments almost every function needs. **file_args fills them in at once.
    file_args = {"input_excel_file": path, "output_excel_file": path, "sheet_name": sheet}

    def clean_text(step_name):
        run_step(step_name + ": abbreviate", address_abbreviator, **file_args, column_names=ABBREVIATION_COLUMNS)
        run_step(step_name + ": uppercase", clean_column, **file_args, column_names=CAPITALIZATION_COLUMNS)

    # 1. Clean text first, so Full Address is built from tidy pieces.
    if CLEAN_TEXT in chosen_steps:
        clean_text(CLEAN_TEXT)

    # 2. Full Address is used by geocoding and by both "copy to matching rows" steps.
    if BUILD_FULL_ADDRESS in chosen_steps:
        run_step(BUILD_FULL_ADDRESS, fill_addresses, **file_args)

    # 3. Copy coordinates BEFORE geocoding, so repeat addresses aren't looked up again.
    if SHARE_COORDINATES in chosen_steps:
        run_step(SHARE_COORDINATES, fill_missing_values_for_duplicate_keys, **file_args,
                 key_columns=["Full Address"], fill_columns=["Latitude", "Longitude"])

    # 4. Geocode: Latitude/Longitude, and optionally Script Address and County (about 1 second per address).
    if GEOCODE in chosen_steps:
        run_step(GEOCODE, geocode_address, **file_args, generate_script_address=settings["generate_script_address"])

    # 5-6. CBSA. County runs AFTER geocoding (which fills County) and AFTER the city match,
    #      so a county match replaces a city match when both are found.
    if CBSA_BY_CITY in chosen_steps:
        run_step(CBSA_BY_CITY, cbsa_addition_by_city, **file_args, cbsa_file_city=CBSA_FILE_CITY)
    if CBSA_BY_COUNTY in chosen_steps:
        run_step(CBSA_BY_COUNTY, cbsa_addition_by_county, **file_args, cbsa_file_county=CBSA_FILE_COUNTY)

    # Clean text again so the new County, Script Address, and CBSA values match the old rows
    # (step 9 matches on County, and step 14 groups by County).
    if CLEAN_TEXT in chosen_steps:
        clean_text(CLEAN_TEXT + " (second pass)")

    # 7. Annual Rent. Bad Monthly Rent rows show up as a warning in the report.
    if ANNUAL_RENT in chosen_steps:
        run_step(ANNUAL_RENT, rent_fill, **file_args)

    # 8. Distance to highway. Needs coordinates, so it runs after geocoding.
    #    (This function names its file arguments differently, so they're written out.)
    if HIGHWAY_DISTANCE in chosen_steps:
        run_step(HIGHWAY_DISTANCE, add_distance_to_highway, input_file=path, output_file=path, sheet_name=sheet,
                 highway_file=HIGHWAY_FILE, latitude_col="Latitude", longitude_col="Longitude",
                 road_types=tuple(settings["highway_types"]), column_name="Distance to Interstate")

    # 9. Copy property info between rows of the same property. BEFORE steps 10-13, which use property sizes.
    if SHARE_PROPERTY_INFO in chosen_steps:
        run_step(SHARE_PROPERTY_INFO, fill_missing_values_for_duplicate_keys, **file_args,
                 key_columns=PROPERTY_MATCH_COLUMNS, fill_columns=PROPERTY_FILL_COLUMNS)

    # 10. Cap yard and lease building sizes at the property sizes.
    if CAP_SIZES in chosen_steps:
        run_step(CAP_SIZES, cap_sizes_to_property, **file_args)

    # 11. Adjust yard size. Must come after step 10 (the function's own instructions).
    if ADJUST_YARD in chosen_steps:
        run_step(ADJUST_YARD, adjust_yard_size_for_building, **file_args)

    # 12-13. Coverage ratios, after step 10 so they use the capped sizes
    #        (they only fill blank cells, so running them earlier would lock in old sizes).
    if LEASE_BCR in chosen_steps:
        run_step(LEASE_BCR, fill_building_coverage, **file_args, building_sf_column="Lease Building (SF)",
                 land_ac_column="Yard Size (AC)", output_column="Lease BCR")
    if PROPERTY_BCR in chosen_steps:
        run_step(PROPERTY_BCR, fill_building_coverage, **file_args, building_sf_column="Property Building (SF)",
                 land_ac_column="Property (AC)", output_column="Property BCR")

    # 14. Zoning last, so zoning codes copied in step 9 get standardized too.
    if STANDARDIZE_ZONING in chosen_steps:
        run_step(STANDARDIZE_ZONING, standardize_zoning_character, **file_args,
                 character=settings["zoning_character"])

    return report


# ---------------------------------------------------------------------
# Showing the report
# ---------------------------------------------------------------------

def show_report(report):
    """One line per step. Problems are shown as warnings, keeping the error's line breaks."""
    for icon, step_name, error_message in report:
        if error_message:
            st.warning(f"**{step_name}** didn't finish:  \n" + error_message.replace("\n", "  \n"), icon=icon)
        else:
            st.write(f"{icon} {step_name}")
