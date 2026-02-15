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
# GLOBAL DATA LOADING (DEPLOYMENT SAFE)
# =====================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

@st.cache_resource
def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("❌ Model failed to load (likely sklearn version mismatch).")
        st.code(str(e))
        st.stop()

@st.cache_data
def convert_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# Load dataset once globally
try:
    DATAFRAME = load_dataset()
except Exception as e:
    st.error(f"Dataset failed to load: {e}")
    st.stop()

# =====================================================
# METRIC COMPUTATION
# =====================================================

def calculate_metrics(y_true, y_pred, prob=None):

    scores = {}

    scores["Accuracy"] = accuracy_score(y_true, y_pred)
    scores["Precision"] = precision_score(y_true, y_pred, average="macro")
    scores["Recall"] = recall_score(y_true, y_pred, average="macro")
    scores["F1"] = f1_score(y_true, y_pred, average="macro")
    scores["MCC"] = matthews_corrcoef(y_true, y_pred)

    auc = None

    if prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, prob[:,1])
            else:
                auc = roc_auc_score(y_true, prob, multi_class="ovr")
        except:
            auc = None

    scores["AUC"] = auc

    return scores

# =====================================================
# LOAD MODELS
# =====================================================

available_models = {
    f.replace(".joblib",""): os.path.join(MODEL_FOLDER, f)
    for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".joblib")
}

if not available_models:
    st.error("No trained models detected.")
    st.stop()

# =====================================================
# NAVIGATION
# =====================================================

tab_eval, tab_compare = st.tabs(["🔎 Evaluate Model", "🏆 Compare Models"])

df = DATAFRAME

# =====================================================
# MODEL COMPARISON TAB
# =====================================================

with tab_compare:

    st.header("🏆 Model Benchmark Leaderboard")

    comp_path = os.path.join(MODEL_FOLDER, "comparison_table.csv")

    if os.path.exists(comp_path):

        comp_df = pd.read_csv(comp_path, index_col=0)

        metric_choice = st.selectbox(
            "Sort models by metric",
            comp_df.columns
        )

        comp_df = comp_df.sort_values(metric_choice, ascending=False)

        st.dataframe(comp_df.style.background_gradient(cmap="Greens"))

        st.subheader("Visual Benchmark Chart")

        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=comp_df.reset_index())
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.success(f"🥇 Best model based on {metric_choice}: {comp_df.index[0]}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100))

# =====================================================
# MODEL EVALUATION TAB (AUTO EXECUTION)
# =====================================================

with tab_eval:

    st.header("🔎 Single Model Analysis")

    st.info("Target column is fixed as 'income'. Evaluation runs automatically.")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys())
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100))

    # -------------------------------------------------
    # AUTO EXECUTION
    # -------------------------------------------------

    with st.spinner("Running model evaluation..."):

        if TARGET_COLUMN not in df.columns:
            st.error(f"Target column '{TARGET_COLUMN}' not found.")
            st.stop()

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

        # Metrics cards
        cols = st.columns(6)

        for i, key in enumerate(["Accuracy","AUC","Precision","Recall","F1","MCC"]):
            val = scores[key]
            cols[i].metric(key, f"{val:.3f}" if isinstance(val,float) else "N/A")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y,y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # ROC Curve
        if y_prob is not None and len(np.unique(y)) == 2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y, y_prob[:,1])
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr,tpr)
            ax2.plot([0,1],[0,1],'--')
            st.pyplot(fig2)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y,y_pred,output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3))

        # Feature Importance (if supported)
        if hasattr(model.named_steps["clf"], "feature_importances_"):

            st.subheader("Feature Importance")

            imp = model.named_steps["clf"].feature_importances_
            names = model.named_steps["preproc"].get_feature_names_out()

            imp_df = pd.DataFrame({
                "Feature": names,
                "Importance": imp
            }).sort_values("Importance", ascending=False).head(20)

            st.bar_chart(imp_df.set_index("Feature"))

        # Predictions
        st.subheader("Predictions")

        pred_df = X.copy()
        pred_df["Actual"] = y
        pred_df["Predicted"] = y_pred

        if y_prob is not None:
            prob_df = pd.DataFrame(y_prob)
            prob_df.columns = [f"Prob_{i}" for i in range(prob_df.shape[1])]
            pred_df = pd.concat([pred_df, prob_df], axis=1)

        st.dataframe(pred_df.head(500))

        st.download_button(
            "⬇ Download Predictions CSV",
            convert_to_csv_bytes(pred_df),
            file_name=f"{model_choice}_predictions.csv",
            mime="text/csv"
        )
