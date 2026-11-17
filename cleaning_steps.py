# cleaning_steps.py
# The cleaning steps, their settings, and the ORDER they run in.
# The Clean Data page and the Test Data page both use this file,
# so training data and test data are always cleaned exactly the same way.
#
#   cleaning_script_functions.py -> WHAT each step does
#   cleaning_steps.py (this file) -> WHICH steps exist, their settings, and their ORDER

from pathlib import Path

import streamlit as st

from cleaning_script_functions import (
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


# ---------------------------------------------------------------------
# SETTINGS (same as the INPUTS section of the original script)
# ---------------------------------------------------------------------

# The lookup files live in the app's own "data" folder, so the app can find them
# both on your computer and when it's hosted online.
# Path(__file__) is this file; .parent is the app folder, where the data folder is.
DATA_FOLDER = Path(__file__).parent / "data"

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
# CLEANING STEPS, listed in the order they run
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

ALL_CLEANING_STEPS = [
    CLEAN_TEXT,
    BUILD_FULL_ADDRESS,
    SHARE_COORDINATES,
    GEOCODE,
    CBSA_BY_CITY,
    CBSA_BY_COUNTY,
    ANNUAL_RENT,
    HIGHWAY_DISTANCE,
    SHARE_PROPERTY_INFO,
    CAP_SIZES,
    ADJUST_YARD,
    LEASE_BCR,
    PROPERTY_BCR,
    STANDARDIZE_ZONING,
]



# ---------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------

def run_cleaning_steps(path, sheet, chosen_steps, generate_script_address, highway_types, zoning_character):
    """
    Check each step in order. If the user chose it, run it.
    Every function reads the Excel file at `path`, changes it, and saves it back to `path`.

    Returns a report with one line per step: (icon, step name, error message or "").
    """
    report = []

    def run_step(step_name, cleaning_function, **arguments):
        """
        Run one cleaning function and record whether it worked.
        If a step fails, the error is recorded and the remaining steps still run
        (the same way the original script handled rent_fill errors).
        """
        with st.spinner(f"Running: {step_name}"):
            try:
                cleaning_function(**arguments)
                report.append(("✅", step_name, ""))
            except Exception as error:
                report.append(("⚠️", step_name, str(error)))

    def clean_text(step_name):
        """Abbreviate street types, then strip and uppercase the text columns."""
        run_step(step_name + ": abbreviate", address_abbreviator,
                 input_excel_file=path, output_excel_file=path, sheet_name=sheet,
                 column_names=ABBREVIATION_COLUMNS)
        run_step(step_name + ": uppercase", clean_column,
                 input_excel_file=path, output_excel_file=path, sheet_name=sheet,
                 column_names=CAPITALIZATION_COLUMNS)

    # The three arguments almost every cleaning function needs.
    # Writing **file_args inside a function call fills them all in at once.
    file_args = {"input_excel_file": path, "output_excel_file": path, "sheet_name": sheet}

    # 1. Clean text first, so Full Address is built from tidy pieces.
    if CLEAN_TEXT in chosen_steps:
        clean_text(CLEAN_TEXT)

    # 2. Full Address is used by geocoding and by both "copy to matching rows" steps.
    if BUILD_FULL_ADDRESS in chosen_steps:
        run_step(BUILD_FULL_ADDRESS, fill_addresses, **file_args)

    # 3. Copy coordinates BEFORE geocoding, so repeat addresses aren't looked up again.
    if SHARE_COORDINATES in chosen_steps:
        run_step(SHARE_COORDINATES, fill_missing_values_for_duplicate_keys, **file_args,
                 key_columns=["Full Address"],
                 fill_columns=["Latitude", "Longitude"])

    # 4. Geocode: fills Latitude/Longitude, and optionally Script Address and County.
    #    About 1 second per address (the free geocoder's speed limit).
    if GEOCODE in chosen_steps:
        run_step(GEOCODE, geocode_address, **file_args,
                 generate_script_address=generate_script_address)

    # 5. CBSA by City.
    if CBSA_BY_CITY in chosen_steps:
        run_step(CBSA_BY_CITY, cbsa_addition_by_city, **file_args,
                 cbsa_file_city=CBSA_FILE_CITY)

    # 6. CBSA by County. Runs AFTER geocoding (which fills County) and AFTER the city
    #    match, so a county match replaces a city match when both are found.
    if CBSA_BY_COUNTY in chosen_steps:
        run_step(CBSA_BY_COUNTY, cbsa_addition_by_county, **file_args,
                 cbsa_file_county=CBSA_FILE_COUNTY)

    # Clean text again so the new County, Script Address, and CBSA values match the
    # old rows. This matters for step 9 (matches on County) and step 14 (groups by County).
    if CLEAN_TEXT in chosen_steps:
        clean_text(CLEAN_TEXT + " (second pass)")

    # 7. Annual Rent. Bad Monthly Rent rows show up as a warning in the report.
    if ANNUAL_RENT in chosen_steps:
        run_step(ANNUAL_RENT, rent_fill, **file_args)

    # 8. Distance to highway. Needs Latitude/Longitude, so it runs after geocoding.
    #    (This function names its file arguments differently, so they're written out.)
    if HIGHWAY_DISTANCE in chosen_steps:
        run_step(HIGHWAY_DISTANCE, add_distance_to_highway,
                 input_file=path,
                 output_file=path,
                 sheet_name=sheet,
                 highway_file=HIGHWAY_FILE,
                 latitude_col="Latitude",
                 longitude_col="Longitude",
                 road_types=tuple(highway_types),
                 column_name="Distance to Interstate")

    # 9. Copy property info (acres, building SF, zoning, sale info) between rows of
    #    the same property. Runs BEFORE steps 10-13, which use those property sizes.
    if SHARE_PROPERTY_INFO in chosen_steps:
        run_step(SHARE_PROPERTY_INFO, fill_missing_values_for_duplicate_keys, **file_args,
                 key_columns=PROPERTY_MATCH_COLUMNS,
                 fill_columns=PROPERTY_FILL_COLUMNS)

    # 10. Cap yard and lease building sizes at the property sizes.
    if CAP_SIZES in chosen_steps:
        run_step(CAP_SIZES, cap_sizes_to_property, **file_args)

    # 11. Adjust yard size. Must come after step 10 (the function's own instructions).
    if ADJUST_YARD in chosen_steps:
        run_step(ADJUST_YARD, adjust_yard_size_for_building, **file_args)

    # 12-13. Coverage ratios. After step 10, so they use the capped sizes.
    #        (These only fill blank cells, so running them before capping would lock in old sizes.)
    if LEASE_BCR in chosen_steps:
        run_step(LEASE_BCR, fill_building_coverage, **file_args,
                 building_sf_column="Lease Building (SF)",
                 land_ac_column="Yard Size (AC)",
                 output_column="Lease BCR")

    if PROPERTY_BCR in chosen_steps:
        run_step(PROPERTY_BCR, fill_building_coverage, **file_args,
                 building_sf_column="Property Building (SF)",
                 land_ac_column="Property (AC)",
                 output_column="Property BCR")

    # 14. Zoning goes last, so zoning codes copied in step 9 get standardized too.
    if STANDARDIZE_ZONING in chosen_steps:
        run_step(STANDARDIZE_ZONING, standardize_zoning_character, **file_args,
                 character=zoning_character)

    return report
