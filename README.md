# Lease Comps

A Streamlit app that cleans lease comp spreadsheets, prepares them for predictive models,
and predicts rent for new properties.

Comp sheets are uploaded in the browser each time the app is used. They are only held in
memory, and in a temporary copy while cleaning, and are never saved to this repository.
The `.gitignore` file blocks spreadsheets from being committed by accident.

## How it's organized

Pages only show things on screen. The work happens in one folder per stage, and each stage
keeps *what* a step does separate from *which* steps run and in what order.

| Path | What it is |
| --- | --- |
| `Home.py` | Upload a comp sheet and add new entries |
| `pages/1_Clean_Data.py` | Clean the comp sheet and download it |
| `pages/2_Preprocess.py` | Turn a cleaned sheet (with rent) into a CSV for the models |
| `pages/3_Test_Data.py` | Clean and preprocess properties to predict and keep them in memory (any actual rents are kept for comparing) |
| `pages/4_Predictions.py` | Choose columns, a random seed, and a model, predict the test data, report errors where actual rent is known, inspect and download the results |
| `cleaning/functions.py` | What each cleaning step does (your cleaning functions) |
| `cleaning/steps.py` | Which cleaning steps exist, their settings and order, and running them (pages 1 and 3) |
| `cleaning/template.py` | The comp sheet's columns, and adding new entries to a sheet |
| `preprocessing/functions.py` | What each preprocessing step does (your preprocessing functions) |
| `preprocessing/steps.py` | Which preprocessing steps exist, their settings and order, running them, and the CSV preview (pages 2 and 3) |
| `models/__init__.py` | The list of models, how to add one, and the parameter types (including tables) |
| `models/shared.py` | What every model shares: reading CSVs, preparing the data, fitting, model details, and errors |
| `models/best_subset_linear.py` | Best subset linear regression (ABESS), from Best Subset - Linear.ipynb |
| `models/lasso.py` | Lasso regression, from Lasso.ipynb |
| `models/ridge.py` | Ridge regression, from Ridge.ipynb |
| `models/gam.py` | GAM with a spline table (pyGAM), from GAM.ipynb |
| `models/neural_network.py` | Neural network with a layer table (Keras), from Nueral_Net.ipynb |
| `models/neural_network_keras.py` | Neural network with a layer table (Keras), from Nueral_Net - Keras.ipynb |
| `data/` | Census CBSA files and 2025 TIGER/Line primary roads |

The road lines were simplified to about 2 meters so the `.shp` file fits under GitHub's 25 MB
upload limit. Distances measured from 200 test points changed by at most 5.4 feet.

## Adding a model

See `models/__init__.py`: add a file to `models/`, add one line to the list there, and add
any new packages to `requirements.txt`. The Predictions page doesn't change.

Every model file has the same parts in the same order, and every setting its notebook passes to
the model is a parameter on the page, with the notebook's value as the default (or, where the
notebook tuned it, the lowest-RMSE value from that tuning). Each parameter's help text says which.
The random seed is one setting on the Predictions page for every model (a new seed each fit, or
one you type in); each model file receives it as `random_state` and says what it uses it for.

## Run on your own computer

    pip install -r requirements.txt
    streamlit run Home.py

## Deploy on Streamlit Community Cloud

1. Upload everything in this folder to a private GitHub repository, keeping the folders.
2. Sign in at https://share.streamlit.io with GitHub and create a new app.
3. Choose this repository and set the main file path to `Home.py`.
