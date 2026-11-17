# pages/1_Clean_Data.py
# This page:
#   1. Lets the user choose which cleaning steps to run
#   2. Runs the chosen steps IN A FIXED ORDER on a copy of the uploaded file
#      (plus any new entries typed in on the Home page)
#   3. Lets the user download the cleaned file under whatever name they type

import tempfile
from pathlib import Path

import streamlit as st

from app_helpers import add_entries_to_excel
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

st.set_page_config(page_title="Clean Data", page_icon="🧹", layout="wide")


# ---------------------------------------------------------------------
# SETTINGS (same as the INPUTS section of the original script)
# ---------------------------------------------------------------------

FOLDER = "C:/Users/joshu/Dropbox/Hampton Yards/Comp List/Leasing Model Comps"

CBSA_FILE_CITY = FOLDER + "/Metropolitan Information Sheets/list2_2023.xlsx"
CBSA_FILE_COUNTY = FOLDER + "/Metropolitan Information Sheets/list1_2023.xlsx"
HIGHWAY_FILE = FOLDER + "/highway data/tl_2025_us_primaryroads/tl_2025_us_primaryroads.shp"

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

ALL_STEPS = [
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

report = []  # One line per step: (icon, step name, error message or "")


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


def clean_text(path, sheet, step_name):
    """Abbreviate street types, then strip and uppercase the text columns."""
    run_step(step_name + ": abbreviate", address_abbreviator,
             input_excel_file=path, output_excel_file=path, sheet_name=sheet,
             column_names=ABBREVIATION_COLUMNS)
    run_step(step_name + ": uppercase", clean_column,
             input_excel_file=path, output_excel_file=path, sheet_name=sheet,
             column_names=CAPITALIZATION_COLUMNS)


def run_chosen_steps(path, sheet, chosen_steps, generate_script_address, highway_types, zoning_character):
    """
    Check each step in order. If the user chose it, run it.
    Every function reads the Excel file at `path`, changes it, and saves it back to `path`.
    """
    # The three arguments almost every cleaning function needs.
    # Writing **file_args inside a function call fills them all in at once.
    file_args = {"input_excel_file": path, "output_excel_file": path, "sheet_name": sheet}

    # 1. Clean text first, so Full Address is built from tidy pieces.
    if CLEAN_TEXT in chosen_steps:
        clean_text(path, sheet, CLEAN_TEXT)

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
        clean_text(path, sheet, CLEAN_TEXT + " (second pass)")

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


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("🧹 Clean Data")

# Stop here if no file has been uploaded on the Home page yet.
if "file_bytes" not in st.session_state:
    st.warning("Upload an Excel file on the Home page first.")
    st.page_link("Home.py", label="Go to Home", icon="🏗️")
    st.stop()

sheet = st.session_state["sheet_name"]
new_entries = st.session_state.get("new_entries", [])

st.write(f"File: **{st.session_state['file_name']}** (sheet: **{sheet}**)")
if new_entries:
    st.write(f"{len(new_entries)} new entries from the Home page will be added to the bottom of the sheet before cleaning.")


# --- Choose steps ---

st.header("1. Choose cleaning steps")
chosen_steps = st.multiselect("Cleaning steps", ALL_STEPS, default=ALL_STEPS)

# Put the chosen steps in run order, no matter what order they were picked in.
steps_in_order = [step for step in ALL_STEPS if step in chosen_steps]

if steps_in_order:
    st.write("**These steps will run in this order:**")
    st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(steps_in_order, start=1)))
else:
    st.info("Choose at least one step to run.")

if ADJUST_YARD in chosen_steps and CAP_SIZES not in chosen_steps:
    st.warning(f"“{ADJUST_YARD}” is meant to run after “{CAP_SIZES}”. Consider adding it.")


# --- Step settings ---

with st.expander("Step settings"):
    generate_script_address = st.checkbox("Geocoding: also fill Script Address and County", value=True)
    highway_types = st.multiselect(
        "Distance to highway: road types to measure to",
        options=list(HIGHWAY_TYPE_NAMES),
        default=["I"],
        format_func=lambda code: HIGHWAY_TYPE_NAMES[code],
    )
    zoning_character = st.text_input("Zoning: character to standardize codes with", value="-")


# --- Name the output and run ---

st.header("2. Run and download")

default_name = Path(st.session_state["file_name"]).stem + " - Cleaned"
output_name = st.text_input("Name for the cleaned file", value=default_name).strip()

if st.button("Run cleaning", type="primary", disabled=not steps_in_order):
    if not output_name:
        st.error("Type a name for the cleaned file first.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_folder:
        # Work on a copy of the uploaded file, so the original is never changed.
        working_file = Path(temp_folder) / "working_copy.xlsx"
        working_file.write_bytes(st.session_state["file_bytes"])

        # Add missing template columns and the new entries from the Home page.
        add_entries_to_excel(working_file, sheet, new_entries)

        run_chosen_steps(
            path=working_file,
            sheet=sheet,
            chosen_steps=chosen_steps,
            generate_script_address=generate_script_address,
            highway_types=highway_types,
            zoning_character=zoning_character,
        )

        # Save the result in memory so the download button keeps working after the page refreshes.
        st.session_state["cleaned_file_bytes"] = working_file.read_bytes()

    st.session_state["cleaned_file_name"] = output_name.removesuffix(".xlsx") + ".xlsx"
    st.session_state["cleaning_report"] = report


# --- Results ---

if "cleaned_file_bytes" in st.session_state:
    st.subheader("Results")

    for icon, step_name, error_message in st.session_state["cleaning_report"]:
        if error_message:
            # "  \n" keeps the error's line breaks when Streamlit shows it.
            st.warning(f"**{step_name}** didn't finish:  \n" + error_message.replace("\n", "  \n"), icon=icon)
        else:
            st.write(f"{icon} {step_name}")

    st.download_button(
        "Download cleaned file",
        data=st.session_state["cleaned_file_bytes"],
        file_name=st.session_state["cleaned_file_name"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
