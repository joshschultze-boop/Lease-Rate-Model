# Home.py  (start the app with:  streamlit run Home.py)
# Upload a comp sheet, add new entries, and go to the other pages.
#
# Kept in memory (st.session_state) for the Clean Data page:
#   "file_bytes", "file_name", "sheet_name"  -> the uploaded Excel file and the sheet to use
#   "new_entries"                            -> new rows typed in below (each row is a dictionary)

from io import BytesIO

import pandas as pd
import streamlit as st

from cleaning.template import ALL_TEMPLATE_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, entry_form, show_entries

st.set_page_config(page_title="Lease Comps", page_icon="🏗️", layout="wide")
st.session_state.setdefault("new_entries", [])

PAGES = [
    ("pages/1_Clean_Data.py", "Clean Data", "🧹",
     "Adds your new entries, runs the cleaning steps you choose, and downloads the cleaned Excel file."),
    ("pages/2_Preprocess.py", "Preprocess", "📊",
     "Upload a cleaned sheet, choose the preprocessing steps, and download a CSV for the model."),
    ("pages/3_Test_Data.py", "Test Data", "🧪",
     "Upload or type in properties to predict (no rent needed). They're cleaned and preprocessed the same way "
     "as your training data and kept in memory for the models."),
    ("pages/4_Predictions.py", "Predictions", "🔮",
     "Fit a model on your training CSV and predict rent for your test data, with errors when it has actual rent."),
]


st.title("🏗️ Lease Comps")
st.write("Upload your comp sheet and add any new entries here, then use the pages below.")


# --- 1. Upload ---

st.header("1. Upload your Excel file")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    st.session_state["file_bytes"] = uploaded_file.getvalue()
    st.session_state["file_name"] = uploaded_file.name

if "file_bytes" not in st.session_state:
    st.info("Upload a file to see a preview here. You can start adding new entries below at any time.")
else:
    st.success(f"Current file: **{st.session_state['file_name']}**")
    sheet_names = pd.ExcelFile(BytesIO(st.session_state["file_bytes"])).sheet_names
    previous = st.session_state.get("sheet_name")
    st.session_state["sheet_name"] = st.selectbox(
        "Sheet to use", sheet_names, index=sheet_names.index(previous) if previous in sheet_names else 0
    )

    data = pd.read_excel(BytesIO(st.session_state["file_bytes"]), sheet_name=st.session_state["sheet_name"])
    st.caption(f"{len(data)} rows, {len(data.columns)} columns")
    st.dataframe(data, height=300)

    missing = [name for name in ALL_TEMPLATE_COLUMNS if name not in [str(c).strip() for c in data.columns]]
    if missing:
        st.warning("These columns aren't in your file. They'll be added as blank columns when you clean it: "
                   + ", ".join(missing))


# --- 2. New entries ---

st.header("2. Add new entries (optional)")
st.caption("Fields marked * are required. The form keeps your values after you add an entry, "
           "so entering several leases at the same property is quick.")

entry_form("new_entry_form", REQUIRED_COLUMNS, OPTIONAL_COLUMNS, st.session_state["new_entries"])
show_entries(st.session_state["new_entries"], "added to the bottom of your sheet when you run the cleaning.")


# --- 3. Pages ---

st.header("3. Next steps")
for page, label, icon, description in PAGES:
    st.page_link(page, label=label, icon=icon)
    st.caption(description)
