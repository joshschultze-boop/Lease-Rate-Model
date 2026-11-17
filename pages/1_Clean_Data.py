# pages/1_Clean_Data.py
# Cleans the file uploaded on the Home page (plus any new entries) with the steps you choose,
# and downloads it. The steps themselves are in cleaning/steps.py.

from pathlib import Path

import streamlit as st

import cleaning.steps as cleaning_steps

st.set_page_config(page_title="Clean Data", page_icon="🧹", layout="wide")
st.title("🧹 Clean Data")

if "file_bytes" not in st.session_state:
    st.warning("Upload an Excel file on the Home page first.")
    st.page_link("Home.py", label="Go to Home", icon="🏗️")
    st.stop()

sheet = st.session_state["sheet_name"]
new_entries = st.session_state.get("new_entries", [])
st.write(f"File: **{st.session_state['file_name']}** (sheet: **{sheet}**)")
if new_entries:
    st.write(f"{len(new_entries)} new entries from the Home page will be added to the bottom of the sheet before cleaning.")

st.header("1. Choose cleaning steps")
chosen_steps, settings = cleaning_steps.choose_steps()

st.header("2. Run and download")
default_name = Path(st.session_state["file_name"]).stem + " - Cleaned"
output_name = st.text_input("Name for the cleaned file", value=default_name).strip()

if st.button("Run cleaning", type="primary", disabled=not chosen_steps or not output_name):
    cleaned_bytes, report = cleaning_steps.clean_excel(
        st.session_state["file_bytes"], sheet, new_entries, chosen_steps, settings
    )
    # Kept in memory so the download button still works after the page refreshes.
    st.session_state["cleaning_result"] = {
        "file_bytes": cleaned_bytes,
        "file_name": output_name.removesuffix(".xlsx") + ".xlsx",
        "report": report,
    }

result = st.session_state.get("cleaning_result")
if result:
    st.subheader("Results")
    cleaning_steps.show_report(result["report"])
    st.download_button("Download cleaned file", data=result["file_bytes"], file_name=result["file_name"],
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.page_link("pages/2_Preprocess.py", label="Next: upload this file on the Preprocess page", icon="📊")
