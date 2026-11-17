# Lease Comps Cleaner

A Streamlit app that cleans lease comp spreadsheets.

Comp sheets are uploaded in the browser each time the app is used. They are only
held in memory and in a temporary copy while cleaning, and are never saved to this
repository. The `.gitignore` file blocks spreadsheets from being committed by accident.

## Files

| File | What it is |
| --- | --- |
| `Home.py` | Home page: upload a comp sheet and add new entries |
| `pages/1_Clean_Data.py` | Choose cleaning steps, run them, and download the result |
| `app_helpers.py` | New-entry template columns and the function that adds entries to the sheet |
| `cleaning_script_functions.py` | The cleaning functions |
| `data/list1_2023.xlsx` | Census CBSA delineation file (matched by county) |
| `data/list2_2023.xlsx` | Census principal cities file (matched by city) |
| `data/tl_2025_us_primaryroads/` | Census 2025 TIGER/Line primary roads, for distance to highway |

The road lines were simplified to about 2 meters so the `.shp` file fits under
GitHub's 25 MB upload limit. Distances measured from 200 test points changed by
at most 5.4 feet compared with the original file. All road types are kept.

## Run on your own computer

    pip install -r requirements.txt
    streamlit run Home.py

## Deploy on Streamlit Community Cloud

1. Upload everything in this folder to a private GitHub repository, keeping the
   `pages` and `data` folders.
2. Sign in at https://share.streamlit.io with GitHub and create a new app.
3. Choose this repository and set the main file path to `Home.py`.
