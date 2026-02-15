# 📊 Machine Learning Classification Analysis

---

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether an individual's annual income exceeds a certain threshold using structured demographic and employment-related features.

The project involves:

- Training multiple supervised classification models.
- Evaluating models using multiple performance metrics.
- Comparing model performance quantitatively and qualitatively.
- Developing an interactive Streamlit application for model evaluation and visualization.

The goal is to determine which model provides the best balance between predictive performance, generalization capability, and robustness.

---

## b. Dataset Description

The dataset used for this project is the **Adult Income Dataset**, a well-known classification dataset containing demographic and employment attributes of individuals.

### Dataset Characteristics:

- Task Type: Binary Classification
- Target Variable: `income`
- Objective: Predict whether income is greater than 50K or less/equal to 50K.
- Number of samples: Large-scale structured dataset.
- Number of features: Includes both numerical and categorical variables.

### Feature Categories:

- Demographic attributes (age, education, marital status)
- Employment-related features (occupation, work class)
- Economic indicators (capital gain/loss, hours per week)

The dataset contains mixed data types, requiring preprocessing steps such as:

- Handling categorical variables using encoding.
- Scaling numerical features.
- Managing missing values.

---

## c. Models Used

The following six classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble Method)
6. XGBoost (Gradient Boosting Ensemble)

Evaluation metrics used:

- Accuracy
- Area Under ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression       | 0.8543   | 0.9057 | 0.8119    | 0.7668 | 0.7851 | 0.5769 |
| Decision Tree             | 0.8179   | 0.7572 | 0.7501    | 0.7572 | 0.7535 | 0.5072 |
| kNN                       | 0.8331   | 0.8552 | 0.7733    | 0.7537 | 0.7625 | 0.5267 |
| Naive Bayes               | 0.6240   | 0.8268 | 0.6684    | 0.7257 | 0.6110 | 0.3900 |
| Random Forest (Ensemble)  | 0.8571   | 0.9055 | 0.8121    | 0.7787 | 0.7930 | 0.5898 |
| XGBoost (Ensemble)        | 0.8741   | 0.9298 | 0.8402    | 0.7988 | 0.8163 | 0.6377 |

---

## Model Performance Observations

| ML Model Name               | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Provided strong baseline performance with high AUC and stable precision. Its linear nature allows good generalization but may limit capturing complex nonlinear patterns. |
| Decision Tree               | Achieved reasonable performance but showed comparatively lower MCC, indicating potential overfitting or sensitivity to training data variations. |
| kNN                         | Demonstrated balanced performance across metrics. However, sensitivity to feature scaling and computational cost for large datasets can affect scalability. |
| Naive Bayes                 | Produced lower accuracy due to strong independence assumptions between features, which may not hold true for this dataset. Still maintained reasonable recall. |
| Random Forest (Ensemble)    | Improved performance over single tree models by reducing variance through ensemble averaging. Provided strong overall stability and generalization. |
| XGBoost (Ensemble)          | Achieved the best overall performance across most metrics. Gradient boosting allowed capturing complex patterns and feature interactions effectively. |

---

## Streamlit Application

An interactive Streamlit dashboard was developed to:

- Upload datasets dynamically.
- Select trained models for evaluation.
- Visualize evaluation metrics.
- Display confusion matrices and ROC curves.
- Inspect predictions interactively.
- Download prediction results.

---

## Conclusion

The comparative analysis demonstrates that ensemble-based models, particularly XGBoost, outperform simpler models in terms of predictive capability and robustness for the Adult Income classification task. Linear models provide strong baselines, while probabilistic models like Naive Bayes may struggle when feature independence assumptions are violated.

---

