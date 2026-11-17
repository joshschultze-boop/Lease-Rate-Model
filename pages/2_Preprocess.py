# pages/2_Preprocess.py
# Turns a cleaned comp sheet (with rent) into a CSV for the models, with the steps you choose.
# The steps themselves are in preprocessing/steps.py. For new properties with no rent, use Test Data.

from io import BytesIO

import pandas as pd
import streamlit as st

import preprocessing.steps as preprocessing_steps

st.set_page_config(page_title="Preprocess", page_icon="📊", layout="wide")
st.title("📊 Preprocess for the model")
st.write("Turn a cleaned comp sheet into a CSV that's ready for the models. "
         "For new properties with no rent, use the Test Data page instead.")

st.header("1. Upload the cleaned file")
uploaded_file = st.file_uploader("Cleaned Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload a cleaned file to continue. You can make one on the Clean Data page.")
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet = st.selectbox("Sheet to use", pd.ExcelFile(BytesIO(file_bytes)).sheet_names)
table = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
table.columns = [str(column).strip() for column in table.columns]

data, missing = preprocessing_steps.select_model_columns(table, has_rent=True)
if missing:
    st.error("This file is missing columns the preprocessing needs: " + ", ".join(missing)
             + ". Run it through the Clean Data page first, or add these columns.")
    st.stop()
st.caption(f"{len(data)} rows. The {len(data.columns)} columns the model uses will be loaded.")

st.header("2. Choose preprocessing steps")
chosen_steps, settings = preprocessing_steps.choose_steps(has_rent=True)

st.header("3. Run and download")
csv_name = st.text_input("Name for the CSV file", value="preprocessed").strip()

if st.button("Run preprocessing", type="primary", disabled=not chosen_steps or not csv_name):
    with st.spinner("Preprocessing..."):
        output, report = preprocessing_steps.run_steps(data, chosen_steps, settings, has_rent=True)
    # Kept in memory so the download and preview stay after the page refreshes.
    st.session_state["preprocess_result"] = {
        "data": output, "file_name": csv_name.removesuffix(".csv") + ".csv", "report": report,
    }

result = st.session_state.get("preprocess_result")
if result:
    st.write("**Step results, in the order they ran:**")
    preprocessing_steps.show_report(result["report"])
    st.download_button("Download CSV", data=result["data"].to_csv(index=False).encode("utf-8"),
                       file_name=result["file_name"], mime="text/csv", type="primary")

    st.header("4. Preview of the CSV")
    st.caption("From the last run. Change a step or setting? Click Run again to update.")
    preprocessing_steps.show_preview(result["data"])
