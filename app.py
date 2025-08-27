# app.py
# Diabetes Prediction App (Pima Indians Dataset)
# Author: Your Name
# Python 3.10+

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Keep app logs clean
warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
TARGET_COL = "Outcome"
DEFAULT_RANDOM_STATE = 42

# ------------------------------- Data --------------------------------- #
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from remote CSV."""
    df = pd.read_csv(DATA_URL)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


@st.cache_data(show_spinner=False)
def class_balance(df: pd.DataFrame) -> pd.Series:
    """Return normalized class distribution for Outcome."""
    return df[TARGET_COL].value_counts(normalize=True).sort_index()


@st.cache_data(show_spinner=False)
def feature_defaults() -> pd.Series:
    """Compute median defaults for each feature to seed the prediction form."""
    df = load_data()
    feats = [c for c in df.columns if c != TARGET_COL]
    return df[feats].median(numeric_only=True)


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on Outcome."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ------------------------------ Model --------------------------------- #
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> RandomForestClassifier:
    """Train a RandomForestClassifier with sensible defaults."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def infer(model: RandomForestClassifier, row_df: pd.DataFrame) -> float:
    """Return probability (class=1) for a single-row DataFrame."""
    return float(model.predict_proba(row_df)[0, 1])


@st.cache_data(show_spinner=False)
def default_inference_model(
    random_state: int,
) -> Tuple[RandomForestClassifier, list[str]]:
    """
    Provide a ready-to-use model trained on the full dataset (for live form
    when the user hasn't trained yet). This does NOT affect evaluation.
    """
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model, list(X.columns)


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float | None]:
    """Compute Accuracy and ROC AUC (handles single-class test edge case)."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = None
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    return acc, roc


# ------------------------------ UI ------------------------------------ #
def section_dataset_preview(df: pd.DataFrame) -> None:
    st.header("A) Dataset preview & quick summary")
    st.caption(
        "Pima Indians Diabetes dataset from Plotly. "
        "Target column: **Outcome** (0 = No Diabetes, 1 = Diabetes)."
    )

    with st.expander("How to read this section", expanded=False):
        st.write(
            "- **Shape** shows rows × columns.\n"
            "- **Class balance** helps check for imbalance in Outcome.\n"
            "- A mild imbalance is normal here; we stratify the split to preserve it."
        )

    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Class balance")
    balance = class_balance(df)
    st.write(
        pd.DataFrame(
            {"Outcome": balance.index, "Proportion (%)": (balance.values * 100).round(2)}
        )
    )
    st.bar_chart(balance.rename(index={0: "No Diabetes", 1: "Diabetes"}))


def section_train_evaluate(df: pd.DataFrame) -> None:
    st.header("B) Train & evaluate")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            test_size = st.slider(
                "Test size (fraction)",
                min_value=0.10,
                max_value=0.40,
                step=0.05,
                value=0.20,
                help="Portion of data held out for testing (stratified on Outcome).",
            )
        with c2:
            rnd_state = st.number_input(
                "Random state",
                min_value=0,
                max_value=1_000_000,
                value=DEFAULT_RANDOM_STATE,
                step=1,
                help="Used everywhere to keep results reproducible.",
            )
        with c3:
            train_btn = st.button("Train & Evaluate", use_container_width=True)

        if train_btn:
            with st.spinner("Training RandomForest and evaluating..."):
                X_train, X_test, y_train, y_test = split_data(
                    df, test_size=test_size, random_state=int(rnd_state)
                )
                model = train_model(X_train, y_train, random_state=int(rnd_state))
                acc, roc = evaluate_model(model, X_test, y_test)

            # Save to session state for the live prediction section
            st.session_state["trained_model"] = model
            st.session_state["feature_names"] = list(X_train.columns)

            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("ROC AUC", f"{roc:.3f}" if roc is not None else "N/A")

            st.subheader("Feature importance")
            importances = pd.Series(
                model.feature_importances_, index=st.session_state["feature_names"]
            ).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(importances.index, importances.values)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("RandomForest Feature Importances")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "Set parameters then click **Train & Evaluate** to compute metrics and show feature importances."
            )


def section_live_prediction() -> None:
    st.header("C) Live prediction")
    st.caption(
        "Enter patient metrics to predict the **probability of diabetes** using the current model."
    )
    with st.expander("Tips for reliable predictions", expanded=False):
        st.write(
            "- If you trained a model above, this form uses that.\n"
            "- Otherwise, it uses a default model trained on the full dataset.\n"
            "- Inputs are clipped to reasonable ranges to avoid typos."
        )

    # Ensure a model is available for the form
    if "trained_model" in st.session_state and "feature_names" in st.session_state:
        model = st.session_state["trained_model"]
        feature_names = st.session_state["feature_names"]
    else:
        model, feature_names = default_inference_model(DEFAULT_RANDOM_STATE)

    defaults = feature_defaults()
    with st.form(key="prediction_form", clear_on_submit=False):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            default_val = float(defaults.get(feat, 0.0))
            step = 0.1 if feat in ("BMI", "DiabetesPedigreeFunction") else 1.0
            min_val = 0.0
            max_val = 300.0 if feat in ("Glucose", "Insulin") else 200.0
            if feat == "BMI":
                max_val = 80.0
            if feat == "DiabetesPedigreeFunction":
                max_val = 3.0
            with col:
                values[feat] = st.number_input(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=float(np.clip(default_val, min_val, max_val)),
                    step=step,
                    format="%.3f" if step == 0.1 else "%.0f",
                    help="Median value prefilled. Adjust as needed.",
                )

        submitted = st.form_submit_button("Predict")
        if submitted:
            row = pd.DataFrame([values], columns=feature_names)
            proba = infer(model, row)
            st.success(f"Probability of Diabetes: {proba:.3f}")


# ---------------------------- Entrypoint ------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

    # Subtle cosmetic polish
    st.markdown(
        """
        <style>
        .stMetric { background: #f7f9fc; padding: 8px 12px; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: quick guide
    with st.sidebar:
        st.markdown("## How to use")
        st.write(
            "1. **Preview** the dataset.\n"
            "2. **Train & Evaluate**: pick test size and random state, then train.\n"
            "3. **Live Prediction**: enter metrics and click **Predict**."
        )
        st.markdown("—")
        st.caption(
            "Reproducible with `random_state=42`. No scaling needed (tree model). "
            "Metrics: Accuracy & ROC AUC."
        )

    st.title("Diabetes Prediction (Random Forest)")
    st.write(
        "A production-clean Streamlit app using the Pima Indians Diabetes dataset. "
        "Built with scikit-learn, pandas, numpy, and matplotlib."
    )

    df = load_data()
    section_dataset_preview(df)
    st.divider()
    section_train_evaluate(df)
    st.divider()
    section_live_prediction()

    st.caption(
        "© 2025 · Streamlit · scikit-learn · pandas · numpy · matplotlib · All rights reserved. Jehad AlAzzeh"
    )


if __name__ == "__main__":
    main()
