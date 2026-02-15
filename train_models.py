# train_models.py

import os
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

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ---------------------------
# SETTINGS
# ---------------------------

DATA_PATH = "data/adult.csv"
TARGET_COL = "income"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

if y.dtype == "object":
    y = y.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# PREPROCESSING
# ---------------------------

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ---------------------------
# MODELS
# ---------------------------

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# ---------------------------
# EVALUATION
# ---------------------------

def evaluate(pipe):

    y_pred = pipe.predict(X_test)

    try:
        y_proba = pipe.predict_proba(X_test)
    except:
        y_proba = None

    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred, average="macro")
    metrics["Recall"] = recall_score(y_test, y_pred, average="macro")
    metrics["F1"] = f1_score(y_test, y_pred, average="macro")
    metrics["MCC"] = matthews_corrcoef(y_test, y_pred)

    if y_proba is not None:
        try:
            if len(np.unique(y_test)) == 2:
                metrics["AUC"] = roc_auc_score(y_test, y_proba[:,1])
            else:
                metrics["AUC"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan

    return metrics


# ---------------------------
# TRAIN LOOP
# ---------------------------

results = []

for name, model in models.items():

    print(f"\nTraining {name}")

    pipe = Pipeline([
        ("preproc", preprocessor),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)

    joblib.dump(pipe, f"{MODEL_DIR}/{name}.joblib")

    metrics = evaluate(pipe)

    row = {"Model": name}
    row.update(metrics)
    results.append(row)

df_results = pd.DataFrame(results).set_index("Model")
df_results.to_csv(f"{MODEL_DIR}/comparison_table.csv")

print("\nTraining complete.")
print(df_results)
