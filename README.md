# Loan Payback Prediction with Stacking Ensemble

This project tackles a binary classification problem: predicting whether a loan
will be paid back based on borrower financial and demographic information.
The solution follows a Kaggle-style machine learning pipeline with robust
cross-validation and stacking ensemble learning.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Overall Pipeline](#overall-pipeline)
- [Feature Processing](#feature-processing)
- [Modeling Strategy](#modeling-strategy)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [Future Work](#future-work)

---

## Problem Statement
Given historical loan records, the goal is to predict whether a loan will be
successfully paid back. This is formulated as a **binary classification problem**
with imbalanced classes, where ROC-AUC is used as the primary evaluation metric.

---

## Dataset
- Source: Kaggle Playground Series (Season 5, Episode 11)
- Target variable: `loan_paid_back`
- Feature types:
  - Numerical: income, debt ratio, loan amount, interest rate, etc.
  - Categorical: employment status, education level, home ownership, etc.

Raw data files are not included in this repository. See `data/README.md` for
details on how to obtain the dataset.

---

## Overall Pipeline
1. Data loading and concatenation (train + test)
2. Missing value handling
3. Categorical encoding
4. Stratified K-Fold cross-validation
5. Training multiple gradient boosting models
6. Out-of-Fold (OOF) prediction generation
7. Stacking ensemble with a meta-model
8. Performance evaluation using ROC-AUC

---

## Feature Processing
- Numerical features:
  - Missing values filled using median statistics
- Categorical features:
  - Missing values filled with `"missing"`
  - Encoded using `LabelEncoder`
- Identifier column (`id`) is dropped before modeling

All preprocessing steps are applied consistently to both training and test sets.

---

## Modeling Strategy
### Base Models
The following tree-based models are trained using **Stratified K-Fold (5 folds)**
cross-validation with multiple random seeds:

- LightGBM
- XGBoost
- CatBoost

For each model, Out-of-Fold (OOF) predictions are collected to ensure an unbiased
estimate of generalization performance.

### Stacking Ensemble
OOF predictions from all base models are used as meta-features to train a
Logistic Regression meta-model. This stacking approach leverages the strengths
of different learners and improves overall performance stability.

---

## Evaluation
- Metric: ROC-AUC
- Validation strategy: Out-of-Fold (OOF) predictions
- Visualization: ROC Curve plotted using OOF probabilities

This evaluation strategy avoids information leakage and provides a reliable
estimate of real-world performance.

---

## Results
| Model | OOF ROC-AUC |
|------|-------------:|
| LightGBM | 0.9227 |
| XGBoost | 0.9219 |
| CatBoost | 0.9232 |
| Stacking Ensemble | **0.9234** |

The stacking ensemble achieves the best performance, demonstrating the benefit
of combining multiple gradient boosting models.

**Kaggle Leaderboard Performance**

- Public Score: 0.92343
- Private Score: 0.92425

The leaderboard scores are highly consistent with the out-of-fold (OOF) ROC-AUC,
indicating strong generalization and a reliable validation strategy.


---

## Project Structure
```text
.
├── predicting-loan-payback.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── images/
```

---

## How to Reproduce
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Download the dataset from Kaggle and place train.csv and test.csv
inside the data/ directory.

3. Run the notebook:
```bash
jupyter notebook predicting-loan-payback.ipynb
```

## Future Work

- Replace label encoding with target encoding for categorical variables

- Perform feature importance analysis

- Experiment with additional meta-models for stacking

---
