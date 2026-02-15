# ML Classification Analysis — Assignment 2

> Repository for ML Assignment 2: classification models, Streamlit app, and deployment.

---

## a. Problem statement

Predict whether an individual's annual income exceeds a threshold (the **income** target) using demographic and employment features. The goal is to implement multiple classification models, evaluate them using standard metrics, compare model performance, and provide an interactive Streamlit dashboard for demonstration and download of predictions.

(Assignment instructions followed as per the course PDF.) :contentReference[oaicite:2]{index=2}

---

## b. Dataset description  [ 1 mark ]

**Dataset:** Adult Income dataset (UCI / public), stored at `data/adult.csv` in this repository.

**Key details**
- Task type: Binary classification (`income` target)
- Number of features: ≥ 12 (mix of numerical and categorical)
- Instances: ≥ 500
- Missing values handled with `SimpleImputer` (median for numerical, most_frequent for categorical)
- Preprocessing: standard scaling for numeric features; One-Hot encoding for categorical features (rare categories grouped)

---

## c. Models used & Evaluation metrics [ 6 marks ]

All models were trained and evaluated on the *same* dataset and split (train/test). The following models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

**Evaluation metrics computed for each model**:
- Accuracy
- AUC Score
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Matthews Correlation Coefficient (MCC)

### Comparison table (results extracted from `comparison_table.csv`) :contentReference[oaicite:3]{index=3}

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|:--------:|:-------:|:---------:|:-------:|:-------:|:-------:|
| LogisticRegression  | 0.854    | 0.905   | 0.811     | 0.766   | 0.785   | 0.576   |
| DecisionTree        | 0.818    | 0.756   | 0.750     | 0.756   | 0.753   | 0.506   |
| KNN                 | 0.835    | 0.855   | 0.776     | 0.754   | 0.764   | 0.530   |
| NaiveBayes          | 0.759    | 0.875   | 0.718     | 0.790   | 0.725   | 0.503   |
| RandomForest        | 0.861    | 0.913   | 0.839     | 0.757   | 0.785   | 0.590   |
| XGBoost             | 0.871    | 0.924   | 0.845     | 0.782   | 0.806   | 0.624   |

> *Notes:* numbers rounded to 3 decimal places for display. Source CSV used: `model/comparison_table.csv`. :contentReference[oaicite:4]{index=4}

---

## Observations on model performance [ 3 marks ]

Below are concise observations about each model's behaviour on this dataset (why metrics look the way they do, plausible causes).

- **Logistic Regression**  
  - Strong baseline performance (Accuracy ≈ 0.85, AUC ≈ 0.905).  
  - Good precision and balanced recall show linear decision boundary captures much of signal.  
  - Interpretable coefficients make it useful as a reference model.

- **Decision Tree**  
  - Lower AUC (≈ 0.756) and modest F1 — suggests a single tree overfit small parts of space but struggled to generalize.  
  - Trees can be sensitive to categorical splits (especially with many categories). Pruning / depth limit can help.

- **KNN**  
  - Competitive (Accuracy ≈ 0.835, AUC ≈ 0.855).  
  - Works well when similar instances have similar labels; performance depends on choice of K and feature scaling (we used StandardScaler).

- **Naive Bayes (Gaussian)**  
  - Lower overall accuracy relative to ensembles but surprisingly high AUC (≈ 0.875).  
  - NB assumes feature independence; despite that, ranking (probability ordering) remains good (hence higher AUC). Performance in classification thresholds is weaker vs ensembles.

- **Random Forest (Ensemble)**  
  - One of the top performers (Accuracy ≈ 0.861, AUC ≈ 0.913).  
  - Ensembles reduce variance vs single trees; feature importance can be inspected for interpretability.

- **XGBoost (Ensemble)**  
  - Best performing model overall (Accuracy ≈ 0.871, AUC ≈ 0.924, MCC ≈ 0.624).  
  - Gradient boosting captures complex non-linear relationships and provides excellent ranking & classification performance with tuned hyperparameters.

**High-level takeaway:** Ensembles (RandomForest and XGBoost) outperform single models and linear methods on this dataset — XGBoost yields the best combination of AUC and thresholded metrics (F1, Accuracy). Naive Bayes surprisingly achieves good AUC (probability ranking), but classification thresholds reduce its final F1/Accuracy vs ensemble methods.

---

## Additional implementation notes

- Preprocessing pipeline:
  - Numeric: `SimpleImputer(strategy='median')` → `StandardScaler`.
  - Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=0.02)` (rare categories grouped).
  - Converted transformed arrays to `float32` to reduce model size for GitHub (and compressed with LZMA for joblib save).
  - For GaussianNB (which requires dense input), the pipeline converts sparse matrix to dense when training/predicting.

- Model saving:
  - Preprocessor saved separately as `model/preprocessor.joblib`
  - Each trained model saved under `model/<ModelName>.joblib`
  - Comparison table saved as `model/comparison_table.csv` (used here for README).

- Streamlit app features (present in `app.py`):
  - Loads `data/adult.csv` from the repo `data/` directory.
  - Model selection dropdown and automatic evaluation for the selected model.
  - Displays evaluation metrics, confusion matrix, classification report, and predictions table.
  - Download button to obtain predictions CSV.

---

## How to run (local)

1. Create a Python environment (recommended: Python 3.10)  
2. Install requirements:

```bash
pip install -r requirements.txt
