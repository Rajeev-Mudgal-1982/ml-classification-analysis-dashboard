# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from concurrent.futures import ThreadPoolExecutor
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

# -----------------------------------------------------
# Model URLs (Kept on HugginFace due to size limitations on GitHub)
# -----------------------------------------------------

HF_MODEL_URLS = {
    "LogisticRegression.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/LogisticRegression.joblib",
    "DecisionTree.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/DecisionTree.joblib",
    "KNN.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/KNN.joblib",
    "NaiveBayes.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/NaiveBayes.joblib",
    "RandomForest.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/RandomForest.joblib",
    "XGBoost.joblib": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/XGBoost.joblib",
    "comparison_table.csv": "https://huggingface.co/Rajeev-Mudgal-1982/ml-classification-analysis-models/resolve/main/comparison_table.csv"
}

# =====================================================
# ENTERPRISE HUGGINGFACE DOWNLOADER
# =====================================================

def download_single_file(filename, url):

    local_path = os.path.join(MODEL_FOLDER, filename)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


@st.cache_resource
def download_models_parallel():

    st.info("🚀 Models Retrived")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for filename, url in HF_MODEL_URLS.items():
            futures.append(executor.submit(download_single_file, filename, url))

        for future in futures:
            future.result()

    return True

# Run once
download_models_parallel()

# =====================================================
# GLOBAL DATA LOADING
# =====================================================

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset empty.")
    return df

@st.cache_resource
def load_model_safe(path):

    try:
        return joblib.load(path)
    except Exception as e:
        st.error("❌ Model load failed (likely sklearn version mismatch)")
        st.code(str(e))
        st.stop()

@st.cache_data
def convert_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

DATAFRAME = load_dataset()

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
# LOAD MODELS
# =====================================================

available_models = {
    f.replace(".joblib",""): os.path.join(MODEL_FOLDER, f)
    for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".joblib")
}

df = DATAFRAME

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

    # Dynamic header based on selected model
    st.header(f"🔎 {model_choice} Model Analysis")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    model = load_model_safe(available_models[model_choice])

    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)
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
