# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(layout="wide")

st.title("📊 ML Classification Model Analysis Dashboard")

DATA_PATH = "data/adult.csv"
MODEL_FOLDER = "model"
TARGET_COLUMN = "income"

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset empty.")
    return df

# =====================================================
# LOAD PREPROCESSOR + MODELS
# =====================================================

@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_FOLDER, "preprocessor.joblib"))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def convert_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

df = load_dataset()
preprocessor = load_preprocessor()

# =====================================================
# METRICS
# =====================================================

def calculate_metrics(y_true, y_pred, prob=None):

    scores = {}

    scores["Accuracy"] = accuracy_score(y_true, y_pred)
    scores["Precision"] = precision_score(y_true, y_pred, average="macro")
    scores["Recall"] = recall_score(y_true, y_pred, average="macro")
    scores["F1"] = f1_score(y_true, y_pred, average="macro")
    scores["MCC"] = matthews_corrcoef(y_true, y_pred)

    try:
        if prob is not None:
            if len(np.unique(y_true)) == 2:
                scores["AUC"] = roc_auc_score(y_true, prob[:,1])
            else:
                scores["AUC"] = roc_auc_score(y_true, prob, multi_class="ovr")
    except:
        scores["AUC"] = None

    return scores

# =====================================================
# LOAD AVAILABLE MODELS
# =====================================================

available_models = {
    f.replace(".joblib",""): os.path.join(MODEL_FOLDER, f)
    for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".joblib") and f != "preprocessor.joblib"
}

if not available_models:
    st.error("No models found in model folder.")
    st.stop()

# =====================================================
# PREPROCESS DATA
# =====================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

if y.dtype == "object":
    y = y.astype("category").cat.codes

X_transformed = preprocessor.transform(X)
X_transformed = X_transformed.astype(np.float32)

# =====================================================
# UI NAVIGATION
# =====================================================

tab_eval, tab_compare = st.tabs(["🔎 Evaluate Model", "🏆 Compare Models"])

# =====================================================
# MODEL COMPARISON TAB
# =====================================================

with tab_compare:

    st.header("🏆 Model Benchmark Leaderboard")

    comp_path = os.path.join(MODEL_FOLDER, "comparison_table.csv")
    comp_df = pd.read_csv(comp_path, index_col=0)

    metric_choice = st.selectbox(
        "Sort models by metric",
        comp_df.columns
    )

    comp_df = comp_df.sort_values(metric_choice, ascending=False)

    st.dataframe(comp_df.style.background_gradient(cmap="Greens"))

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=comp_df.reset_index())
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.success(f"🥇 Best model: {comp_df.index[0]}")

# =====================================================
# MODEL EVALUATION TAB
# =====================================================

with tab_eval:

    st.info("Target column fixed as 'income'. Auto evaluation enabled.")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys())
    )

    st.header(f"🔎 {model_choice} Model Analysis")

    model = load_model(available_models[model_choice])

    # GaussianNB requires dense input
    if model_choice == "NaiveBayes":
        X_input = X_transformed.toarray()
    else:
        X_input = X_transformed

    y_pred = model.predict(X_input)

    try:
        y_prob = model.predict_proba(X_input)
    except:
        y_prob = None

    scores = calculate_metrics(y, y_pred, y_prob)

    cols = st.columns(6)

    for i, key in enumerate(["Accuracy","AUC","Precision","Recall","F1","MCC"]):
        val = scores.get(key)
        cols[i].metric(key, f"{val:.3f}" if isinstance(val,float) else "N/A")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y,y_pred), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y,y_pred,output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    pred_df = X.copy()
    pred_df["Actual"] = y
    pred_df["Predicted"] = y_pred

    st.dataframe(pred_df.head(500))

    st.download_button(
        "⬇ Download Predictions CSV",
        convert_to_csv_bytes(pred_df),
        file_name=f"{model_choice}_predictions.csv",
        mime="text/csv"
    )
