# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
# LOAD DATA
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

df = load_dataset()

# =====================================================
# LOAD PREPROCESS CONFIG (VERSION-PROOF)
# =====================================================

with open(os.path.join(MODEL_FOLDER, "preprocessing_config.json")) as f:
    config = json.load(f)

num_cols = config["numeric_cols"]
cat_cols = config["categorical_cols"]

# =====================================================
# REBUILD PREPROCESSOR (SAFE)
# =====================================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        min_frequency=0.02
    ))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =====================================================
# PREPROCESS DATA
# =====================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

if y.dtype == "object":
    y = y.astype("category").cat.codes

# Fit preprocessor dynamically (version-proof)
preprocessor.fit(X)

X_t = preprocessor.transform(X).astype(np.float32)

# =====================================================
# LOAD MODELS (EXCLUDE PREPROCESSOR)
# =====================================================

MODEL_NAMES = [
    "LogisticRegression",
    "DecisionTree",
    "KNN",
    "NaiveBayes",
    "RandomForest",
    "XGBoost"
]

available_models = {
    name: os.path.join(MODEL_FOLDER, f"{name}.joblib")
    for name in MODEL_NAMES
}

# =====================================================
# SIDEBAR MODEL SELECTION
# =====================================================

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(available_models.keys())
)

st.header(f"🔎 {model_choice} Model Analysis")

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(available_models[model_choice])

# NaiveBayes requires dense matrix
if model_choice == "NaiveBayes":
    X_input = X_t.toarray()
else:
    X_input = X_t

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_input)

# =====================================================
# METRICS DISPLAY
# =====================================================

cols = st.columns(5)

cols[0].metric("Accuracy", f"{accuracy_score(y,y_pred):.3f}")
cols[1].metric("Precision", f"{precision_score(y,y_pred,average='macro'):.3f}")
cols[2].metric("Recall", f"{recall_score(y,y_pred,average='macro'):.3f}")
cols[3].metric("F1", f"{f1_score(y,y_pred,average='macro'):.3f}")
cols[4].metric("MCC", f"{matthews_corrcoef(y,y_pred):.3f}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y,y_pred), annot=True, fmt="d", ax=ax)
st.pyplot(fig)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

st.subheader("Classification Report")

report = classification_report(y,y_pred,output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(3))

# =====================================================
# SHOW PREDICTIONS
# =====================================================

st.subheader("Predictions Preview")

pred_df = X.copy()
pred_df["Actual"] = y
pred_df["Predicted"] = y_pred

st.dataframe(pred_df.head(500))

# =====================================================
# DOWNLOAD BUTTON
# =====================================================

@st.cache_data
def convert_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Predictions CSV",
    convert_to_csv(pred_df),
    file_name=f"{model_choice}_predictions.csv",
    mime="text/csv"
)
