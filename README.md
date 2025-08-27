# Diabetes Prediction (Streamlit + scikit-learn)

A production-clean Streamlit app that trains a **RandomForestClassifier** to predict diabetes using the **Pima Indians Diabetes** dataset. It includes a dataset preview, train/evaluate controls with metrics and feature importances, and a live prediction form that returns the probability of diabetes.

## Tech Stack
- Python 3.10+
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib

# Diabetes Prediction (Streamlit + scikit-learn)

A production-clean Streamlit app that trains a **RandomForestClassifier** to predict diabetes using the **Pima Indians Diabetes** dataset. It includes a dataset preview, train/evaluate controls with metrics and feature importances, and a live prediction form that returns the probability of diabetes.

## Tech Stack
- Python 3.10+
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib

## Quick Start
```bash
# 1) (Recommended) create and activate a virtual environment
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch the app
streamlit run app.py


## Features
- **Dataset preview & summary**
  - Shows shape and **class balance** (bar chart)
- **Train & Evaluate** (RandomForest)
  - User controls: **test size** slider, **random state** input  
  - Metrics displayed via `st.metric`: **Accuracy**, **ROC AUC**
  - **Feature importance** chart using `model.feature_importances_` (matplotlib)
- **Live Prediction**
  - Numeric inputs for all features
  - Returns **Probability of Diabetes: 0.xxx** to three decimals
- **Reproducibility & Speed**
  - consistent `random_state` defaults (42)
  - `st.cache_data` for data loading and precomputation
- **No scaling required** (tree-based model)

## Acceptance Tests
1. `streamlit run app.py` launches without errors.
2. After training, **Accuracy** and **ROC AUC** render as `st.metric`.
3. **Feature importance** chart renders using `model.feature_importances_`.
4. Submitting the live form shows **Probability of Diabetes: 0.xxx**.
5. `app.py` stays under 250 lines and uses clear function boundaries (`load_data`, `train_model`, `infer`) with docstrings.

## Screenshots
> Replace the placeholders below with actual screenshots after you run the app.

- **A) Dataset Preview & Class Balance**  
  _[screenshot-dataset.png]_

- **B) Train & Evaluate (Metrics + Feature Importances)**  
  _[screenshot-train.png]_

- **C) Live Prediction**  
  _[screenshot-predict.png]_

## Notes
- Dataset is loaded from the Plotly raw CSV:  
  `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`
- Target column is **Outcome** (0 = No Diabetes, 1 = Diabetes).
- ROC AUC requires both classes present in the test split; with stratification and typical test sizes this holds. If not, the app gracefully shows **N/A**.
- Extra polish: class balance visualization uses normalized `value_counts`.

## License
MIT License © 2025 Jehad AlAzzeh
