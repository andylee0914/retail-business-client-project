# Census Income Classification & Segmentation

## Project Overview

This project addresses two tasks:

- **Task 1** (`train_income_classifier.py`): Train and evaluate a classification model that predicts whether an individual earns more or less than $50,000 per year.
- **Task 2** (`segmentation_model.py`): Build a customer segmentation model that groups the adult population into distinct marketing segments.

---

## Repository Structure

```
├── report.pdf: Report for this project
├── train_income_classifier.py: Main training and evaluation script for task 1
├── segmentation_model.py: Main training and evaluation script for task 2
├── census-bureau.data: Raw census dataset
├── census-bureau.columns: Column names file
└── README.md: README file
```

Both scripts must be run from the same directory as `census-bureau.data` and `census-bureau.columns`.

---

## Requirements

### Python Version

Python 3.10 or higher is recommended. The project was developed and tested on **Python 3.10**.

### Dependencies

Install all required packages using the command below.

```bash
pip install \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scikit-learn==1.7.2 \
  joblib==1.4.2 \
  kmodes==0.12.2
```

### Conda Environment (Recommended)

If you are using Anaconda or Miniconda, create an environment to avoid dependency conflicts:

```bash
conda create -n census python=3.10
conda activate census
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.7.2 joblib==1.4.2 kmodes==0.12.2
```

---

## Running Task 1 — Income Classification

Run the below command in your terminal with necessary dependencies installed as mentioned above:

```bash
python train_income_classifier.py
```

### What it does

1. Loads `census-bureau.data` and `census-bureau.columns`
2. Cleans the data and engineers features
3. Conducts feature selection
4. Splits data into 80% training / 20% validation (stratified, with population weights)
5. Trains three models: Logistic Regression, Random Forest, HistGradientBoosting
6. Evaluates each model at the default 0.50 threshold and an F1-optimal tuned threshold
7. Prints a full model comparison table and the final classification report
8. Saves the best model and results

### Expected runtime

Around **~10–15 minutes** in total.

### Outputs

- `best_income_model.joblib`: Best fitted model, optimal threshold, and feature list
- `model_comparison_results.csv`: Metrics for all model/threshold combinations

---

## Running Task 2 — Customer Segmentation

Run the below command in your terminal with necessary dependencies installed as mentioned above:

```bash
python segmentation_model.py
```

### What it does

1. Loads `census-bureau.data` and `census-bureau.columns`
2. Cleans the data, engineers features, and filters to adults (age >= 18)
3. Defines features for segmentation
4. Runs K-Means elbow analysis (k=2–8) on numeric features as a baseline
5. Runs K-Prototypes elbow analysis (k=2–7) on the full mixed feature set
6. Fits the final K-Prototypes model with k=5
7. Prints full cluster profiles with income validation
8. Saves the model and cluster assignments

### Expected runtime

Around **~25–35 minutes** in total.

### Outputs

- `segmentation_model.joblib`: Fitted K-Prototypes model, scaler, imputer, and feature metadata
- `segmentation_results.csv`: Full adult dataset with K-Means and K-Prototypes cluster assignments

---

## Loading Saved Models

### Task 1 — Scoring new individuals

```python
import joblib
import pandas as pd

artifact = joblib.load("best_income_model.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# new_data must be a DataFrame with the same columns as features
new_data = pd.DataFrame([...])
proba = model.predict_proba(new_data[features])[:, 1]
predictions = (proba >= threshold).astype(int)
print(predictions)
```

`predictions` will return `1` if the predicted income is more than $50,000 and `0` if the predicted income is $50,000 or less.

### Task 2 — Assigning new individuals to segments

```python
import joblib
import numpy as np
import pandas as pd

artifact = joblib.load("segmentation_model.joblib")
model = artifact["model"]
scaler = artifact["scaler"]
imputer = artifact["imputer"]
num_feats = artifact["numeric_features"]
cat_feats = artifact["categorical_features"]
cat_idx = artifact["cat_indices"]

# Prepare new data with the same features
new_data = pd.DataFrame([...])
X_num = scaler.transform(imputer.transform(new_data[num_feats]))
X_cat = new_data[cat_feats].values
X_kp = np.concatenate([X_num, X_cat], axis=1)

cluster_assignments = model.predict(X_kp, categorical=cat_idx)
print(cluster_assignments)
```

`cluster_assignments` returns an integer array which indicates which cluster each row is assigned to.

Each value corresponds to the following cluster:

- `0`: Retired / non-working Women
- `1`: Foreign-born Working Adults
- `2`: Self-employed / Small Business
- `3`: Wealthy Investor Households
- `4`: Core Working Adults

---

## Potential Troubleshooting

**`ModuleNotFoundError: No module named 'kmodes'`**
Install kmodes: `pip install kmodes==0.12.2`

**`ConvergenceWarning` for Logistic Regression**
This warning indicates the optimizer did not fully converge. It does not affect the validity of predictions. The warning is suppressed in the script. If you want to eliminate it entirely, increase `max_iter` in the `LogisticRegression` definition.

**Memory errors on Task 2**
K-Prototypes on 143K rows requires approximately 4–6 GB of RAM. Close other applications before running if memory is limited.

**Results differ slightly between runs**
Both models use `random_state=42` for reproducibility. Results should be identical across runs on the same machine and Python/library version. Minor floating-point differences across operating systems or library versions are normal.

---
