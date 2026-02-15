# ML Classification Analysis — Assignment 2

> End-to-end classification analysis, model training, and Streamlit dashboard for the Adult Income dataset.

---

## a. Problem statement

Predict whether an individual's annual income exceeds \$50K (target column: `income`) using demographic and employment features. The goal is to implement, evaluate and compare six classification models, create a Streamlit dashboard showing model metrics and predictions, and deploy the app on Streamlit Community Cloud following the assignment instructions (see course PDF). 

---

## b. Dataset description  

**Dataset file:** `data/adult.csv` (included in this repository, downloaded from Kaggle)

**Key facts (computed from `data/adult.csv`):**
- Total instances (rows): **48,842**
- Number of columns (features + target): **15**
- Features: mixture of numerical and categorical attributes (e.g., age, workclass, education, occupation, hours-per-week, etc.)
- Target: `income` — two classes:
  - `<=50K`: **37,155** instances
  - `>50K` : **11,687** instances
- Class balance: approx **76.1%** `<=50K`, **23.9%** `>50K` (class imbalance should be considered when evaluating metrics).

**Missing values & preprocessing**
- Missing values handled with `SimpleImputer`:
  - Numeric columns → `median`
  - Categorical columns → `most_frequent`
- Numeric features scaled using `StandardScaler`.
- Categorical features encoded with `OneHotEncoder(handle_unknown='ignore')` — rare categories were grouped (min frequency threshold used during training to reduce feature explosion).
- After transformation we convert feature arrays to `float32` for smaller model artifacts and efficient serialization.

---

## c. Models used & Evaluation metrics 

**Models trained (all on same dataset and split):**
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

**Metrics computed for each model:**
- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision (macro)  
- Recall (macro)  
- F1 Score (macro)  
- Matthews Correlation Coefficient (MCC)

> The evaluation numbers below were read from `model/comparison_table.csv`.

### Comparison table (results taken from `model/comparison_table.csv`)

| ML Model Name       | Accuracy  | AUC       | Precision | Recall    | F1        | MCC       |
|---------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| LogisticRegression  | 0.854028  | 0.904990  | 0.811412  | 0.766273  | 0.784570  | 0.575919  |
| DecisionTree        | 0.817893  | 0.756000  | 0.750062  | 0.757178  | 0.753465  | 0.507190  |
| KNN                 | 0.833145  | 0.855434  | 0.776661  | 0.753719  | 0.762536  | 0.529795  |
| NaiveBayes          | 0.758829  | 0.875288  | 0.718341  | 0.789735  | 0.724623  | 0.503034  |
| RandomForest        | 0.861398  | 0.913163  | 0.838570  | 0.756753  | 0.785339  | 0.589674  |
| XGBoost             | 0.871486  | 0.924600  | 0.844223  | 0.781808  | 0.811310  | 0.624688  |

*(Values shown as in `model/comparison_table.csv`; rounded to six decimal places for precision)*

---

## Observations on model performance 

Below are concise observations, one per model, based on the comparison table and dataset characteristics:

- **Logistic Regression**  
  - Strong baseline: high Accuracy (≈ 0.854) and AUC (≈ 0.905). The linear decision boundary captures much of the signal; robust and interpretable.

- **Decision Tree**  
  - Lower AUC (≈ 0.756) and modest F1: suggests single tree overfits certain splits and generalizes less well compared to ensembles. Hyperparameter tuning / pruning could improve performance.

- **KNN**  
  - Competitive (Accuracy ≈ 0.833, AUC ≈ 0.855). KNN benefits from proper scaling (we used StandardScaler). Performance depends on K and feature sparsity induced by encoding.

- **Naive Bayes (Gaussian)**  
  - Lower Accuracy (≈ 0.759) but good ranking ability (AUC ≈ 0.875). Independence assumption reduces thresholded classification performance while still providing useful probability rankings.

- **Random Forest**  
  - One of the top performers (Accuracy ≈ 0.861, AUC ≈ 0.913). Ensembles reduce variance and handle categorical splits well after preprocessing.

- **XGBoost**  
  - Best overall performer (Accuracy ≈ 0.871, AUC ≈ 0.925). Gradient boosting captures complex non-linear relationships and yields highest MCC (≈ 0.625), indicating strong class-wise discrimination.

**High-level takeaway:** Ensemble methods (Random Forest and XGBoost) outperform single-model baselines on this dataset. Naive Bayes is competitive in ranking (AUC) but weaker at thresholded classification. Class imbalance (~24% positive) means AUC and MCC are important complementary metrics alongside accuracy.

---

## Additional implementation notes

- Preprocessing:
  - Numeric: `SimpleImputer(strategy='median')` → `StandardScaler()`
  - Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=0.02)`
  - After transform: convert to `float32` for smaller artifacts.
  - **Important:** For GaussianNB (which requires dense input), the code converts the sparse matrix to dense only at model-training / model-prediction time (not globally), to retain the storage benefit.

- Model saving & version-proof approach:
  - Saved each trained model to `model/<ModelName>.joblib` (only the raw model object — no pickled preprocessing pipeline).
  - Saved preprocessing configuration to `model/preprocessing_config.json` (lists numeric and categorical columns). The Streamlit app rebuilds the preprocessing pipeline at runtime using the current sklearn version — this avoids sklearn-version-related unpickle errors.
  - Models were compressed using LZMA (`joblib.dump(..., compress=('lzma', 6))`) after training and float32 casting to keep sizes within GitHub limits.

- Streamlit app features (in `app.py`):
  - Loads `data/adult.csv` from `data/`.
  - Reconstructs preprocessor from `model/preprocessing_config.json`, fits it on the dataset (safe at inference time), transforms features, and loads selected model from `model/`.
  - Model selection dropdown (only model names — `preprocessor` excluded).
  - Shows evaluation metrics (Accuracy, Precision, Recall, F1, MCC), confusion matrix, classification report, and a predictions preview table.
  - Download predictions CSV button.

---

## How to run (local)

1. Create a Python environment (recommended: Python **3.10**).  
2. Install packages pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
