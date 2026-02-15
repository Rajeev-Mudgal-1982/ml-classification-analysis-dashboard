# train_models.py

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================
# SETTINGS
# =====================================================

DATA_PATH = "data/adult.csv"
TARGET_COL = "income"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

if y.dtype == "object":
    y = y.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# DETERMINE COLUMN TYPES
# =====================================================

num_cols = list(X.select_dtypes(include=["int64","float64"]).columns)
cat_cols = list(X.select_dtypes(include=["object","category"]).columns)

# SAVE PREPROCESSING CONFIG (VERSION PROOF)
config = {
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols
}

with open(f"{MODEL_DIR}/preprocessing_config.json","w") as f:
    json.dump(config,f)

# =====================================================
# BUILD PREPROCESSOR (TRAINING ONLY)
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

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

X_train_t = X_train_t.astype(np.float32)
X_test_t = X_test_t.astype(np.float32)

# =====================================================
# MODELS
# =====================================================

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=5
    ),
    "XGBoost": XGBClassifier(
        n_estimators=60,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss"
    )
}

for name, model in models.items():

    print(f"Training {name}")

    if name == "NaiveBayes":
        model.fit(X_train_t.toarray(), y_train)
    else:
        model.fit(X_train_t, y_train)

    joblib.dump(model, f"{MODEL_DIR}/{name}.joblib", compress=("lzma",6))

print("Training complete.")
