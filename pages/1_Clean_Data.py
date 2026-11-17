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
from cleaning_steps import (
    ADJUST_YARD,
    ALL_CLEANING_STEPS,
    CAP_SIZES,
    HIGHWAY_TYPE_NAMES,
    run_cleaning_steps,
)

st.set_page_config(page_title="Clean Data", page_icon="🧹", layout="wide")

# The cleaning steps, their settings, and their order are in cleaning_steps.py,
# which the Test Data page also uses. This file is just the page itself.


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
chosen_steps = st.multiselect("Cleaning steps", ALL_CLEANING_STEPS, default=ALL_CLEANING_STEPS)

# Put the chosen steps in run order, no matter what order they were picked in.
steps_in_order = [step for step in ALL_CLEANING_STEPS if step in chosen_steps]

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

        report = run_cleaning_steps(
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

    st.page_link("pages/2_Preprocess.py", label="Next: upload this file on the Preprocess page", icon="📊")
