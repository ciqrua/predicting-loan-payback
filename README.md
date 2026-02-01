
# Loan Payback Prediction (Kaggle Playground S5E11)

Predict whether a loan will be paid back using tabular financial and demographic features.
This project implements a Kaggle-style ML pipeline with unified preprocessing, out-of-fold (OOF) validation, and stacking ensemble.

## Overview

- **Task**: Binary classification (`loan_paid_back`)
- **Challenge**: Imbalanced target → use **ROC-AUC** as the primary metric
- **Approach**:
  - Unified preprocessing across train/test
  - **Stratified K-Fold OOF** training (5 folds) with multiple random seeds
  - Base learners: **LightGBM**, **XGBoost**, **CatBoost**
  - **Stacking** with Logistic Regression as meta-model
  - ROC Curve plotted using OOF predictions

## Methods

### 1) Preprocessing

- Fill missing values:
  - Numerical features → median
  - Categorical features → `"missing"` then `LabelEncoder`
- Drop `id` column (not used as feature)
- Train/test are concatenated for consistent preprocessing, then split back

### 2) Out-of-Fold Training (OOF)

- `StratifiedKFold(n_splits=5, shuffle=True)`
- Seeds used: `42`, `2023`
- For each model:
  - Train on 4 folds, validate on 1 fold
  - Collect OOF predictions for robust ROC-AUC evaluation
  - Average predictions across folds & seeds

### 3) Stacking Ensemble

- Train base models and collect:
  - `oof_lgb`, `oof_xgb`, `oof_cat`
- Meta features:
  - `meta_X = [oof_lgb, oof_xgb, oof_cat]`
- Meta-model:
  - Logistic Regression (`max_iter=500`) trained on OOF meta features

## Results (OOF ROC-AUC)

| Model                  |       OOF ROC-AUC |
| ---------------------- | ----------------: |
| LightGBM               |           0.92272 |
| XGBoost                |           0.92192 |
| CatBoost               |           0.92315 |
| Stacking (LogReg meta) | **0.92335** |

> Note: Scores are computed from out-of-fold predictions, which better reflect generalization performance than a single split.

## Repository Structure

├── predicting-loan-payback.ipynb

├── README.md

├── data

├── images

└── requirements.txt

## How to Run

### Option A: Run on Kaggle (recommended)

1. Upload the notebook to :contentReference[oaicite:0]{index=0}
2. Attach the competition dataset
3. Run all cells

### Option B: Run locally

1. Install dependencies

```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn lightgbm xgboost catboost
```

2. Download the Kaggle dataset and update file paths in the notebook:

* `train.csv`
* `test.csv`

3. Run the notebook:

```bash
jupyter notebook predicting-loan-payback.ipynb
```

## Notes / Next Improvements

* Replace `LabelEncoder` with `OneHotEncoder` / target encoding for potentially better performance
* Add feature importance (LGB/XGB) and error analysis
