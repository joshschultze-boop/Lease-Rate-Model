# pages/2_Preprocess.py
# Turns a cleaned comp sheet (with rent) into training data for the models, with the steps you choose.
# Uses the cleaned file from the Clean Data page (kept in memory), or one you upload.
# The result is kept in memory ("preprocess_result") for the Test Data and Predictions pages,
# so downloading it is optional. The steps themselves are in preprocessing/steps.py.

from io import BytesIO

import pandas as pd
import streamlit as st

import preprocessing.steps as preprocessing_steps

st.set_page_config(page_title="Preprocess", page_icon="📊", layout="wide")

FROM_CLEAN_DATA = "Cleaned file from the Clean Data page"
UPLOAD = "Upload a cleaned Excel file"


def choose_cleaned_data():
    """Section 1. Returns the model columns of the cleaned data, or None (with a message) if there isn't any yet."""
    st.header("1. Choose the cleaned data")
    cleaning_result = st.session_state.get("cleaning_result")
    sources = ([FROM_CLEAN_DATA] if cleaning_result else []) + [UPLOAD]

    if st.radio("Cleaned data", sources, horizontal=True) == FROM_CLEAN_DATA:
        file_bytes, sheet = cleaning_result["file_bytes"], cleaning_result["sheet"]
        st.caption(f"Using **{cleaning_result['file_name']}**, cleaned on the Clean Data page in this session.")
    else:
        uploaded_file = st.file_uploader("Cleaned Excel file (.xlsx)", type=["xlsx"])
        if uploaded_file is None:
            st.info("Run the Clean Data page first (its result is used here automatically), "
                    "or upload a cleaned file you downloaded earlier.")
            return None
        file_bytes = uploaded_file.getvalue()
        sheet = st.selectbox("Sheet to use", pd.ExcelFile(BytesIO(file_bytes)).sheet_names)

    table = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    table.columns = [str(column).strip() for column in table.columns]
    data, missing = preprocessing_steps.select_model_columns(table, has_rent=True)
    if missing:
        st.error("This file is missing columns the preprocessing needs: " + ", ".join(missing)
                 + ". Run it through the Clean Data page first, or add these columns.")
        return None
    st.caption(f"{len(data)} rows. The {len(data.columns)} columns the model uses will be loaded.")
    return data


st.title("📊 Preprocess for the model")
st.write("Turn a cleaned comp sheet into training data for the models. For properties with no rent, use Test Data.")

result = st.session_state.get("preprocess_result")
if result:
    st.success(f"Training data in memory: **{result['file_name']}**, {len(result['data'])} rows. "
               "The Test Data and Predictions pages use it while this tab is open.")

data = choose_cleaned_data()
if data is not None:
    st.header("2. Choose preprocessing steps")
    chosen_steps, settings = preprocessing_steps.choose_steps(has_rent=True, start=result)

    st.header("3. Run")
    csv_name = st.text_input("Name for the training data (and its CSV download)", value="preprocessed").strip()
    if st.button("Run preprocessing", type="primary", disabled=not chosen_steps or not csv_name):
        with st.spinner("Preprocessing..."):
            output, report = preprocessing_steps.run_steps(data, chosen_steps, settings, has_rent=True)
        st.session_state["preprocess_result"] = {
            "data": output, "file_name": csv_name.removesuffix(".csv") + ".csv",
            "steps": chosen_steps, "settings": settings, "report": report,
        }
        st.rerun()  # Refresh so the "Training data in memory" message at the top shows the new data.

if result:
    st.header("Results")
    st.write("**Step results, in the order they ran:**")
    preprocessing_steps.show_report(result["report"])
    st.caption("Kept in memory for the next pages. Download a copy to keep it, or to upload on the "
               "Predictions page in a later session.")
    st.download_button("Download CSV", data=result["data"].to_csv(index=False).encode("utf-8"),
                       file_name=result["file_name"], mime="text/csv")
    st.page_link("pages/3_Test_Data.py", label="Next: Test Data", icon="🧪")
    st.page_link("pages/4_Predictions.py", label="Or go straight to Predictions", icon="🔮")

    st.header("Preview of the training data")
    st.caption("From the last run. Change a step or setting? Click Run again to update.")
    preprocessing_steps.show_preview(result["data"], result["settings"]["start_date"])
