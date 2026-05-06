"""
=============================================
Census Income Classification Pipeline
---------------------------------------------
Pipeline summary:
  1. Load raw census data
  2. Clean, engineer features, handle missing values
  3. Drop 18 features with justification
  4. Stratified 80/20 train/validation split with population sample weights
  5. Fit three models: Logistic Regression, Random Forest, HistGradientBoosting
     - sample_weight passed to all fits (population-representative training)
     - Default 0.50 and F1-optimal thresholds evaluated for each
  6. Best model saved with threshold and feature list

Outputs:
    - model_comparison_results.csv : metrics for every model/threshold combo
    - best_income_model.joblib : best fitted model + threshold + feature list
"""

import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide="ignore", over="ignore", invalid="ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

RANDOM_STATE = 42

# Function to load the data

def load_data(data_path: str, columns_path: str) -> pd.DataFrame:
    """
    Args:
        data_path:Path to census-bureau.data
        columns_path: Path to census-bureau.columns

    Returns:
        Raw dataframe with named columns.
    """
    cols = pd.read_csv(columns_path, header=None)[0].tolist()
    df = pd.read_csv(data_path, header=None, names=cols)
    return df


# Function to clean and engineer features from the raw data for modeling

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: Raw dataframe from load_data().

    Returns:
        Cleaned dataframe with all engineered features added.
    """
    df = df.copy()

    # Clean string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Replace missing value sentinels
    df = df.replace("?", np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Create binary target
    # Have 1 as an indicator for income over 50000
    # Have 0 as an indicator for income under 50000 or missing label
    df["income_over_50k"] = df["label"].apply(
        lambda x: 1 if "50000+" in str(x) else 0
    )

    # Create hourly wage indicator
    # Using the raw value directly would introduce a skewed numeric distribution
    df["has_hourly_wage"] = (df["wage per hour"] > 0).astype(int)

    # Net investment income
    df["overall_gains"] = (
        df["capital gains"]
        - df["capital losses"]
        + df["dividends from stocks"]
    )

    # Log-compressed version of overall_gains for use in logistic regression.
    # overall_gains has a long-tailed distribution with large outliers
    df["overall_gains_signed_log"] = (
        np.sign(df["overall_gains"]) * np.log1p(np.abs(df["overall_gains"]))
    )
    lower = df["overall_gains_signed_log"].quantile(0.001)
    upper = df["overall_gains_signed_log"].quantile(0.999)
    df["overall_gains_signed_log"] = df["overall_gains_signed_log"].clip(lower, upper)

    # High-value investment flag
    df["overall_gains_flag"] = (df["overall_gains"] >= 50_000).astype(int)

    # Any investment activity (binary)
    df["has_investment_activity"] = (
        (df["capital gains"] > 0)
        | (df["capital losses"] > 0)
        | (df["dividends from stocks"] > 0)
    ).astype(int)

    # Hispanic origin indicator
    # The raw column has 10+ categories. We collapse to binary:
    # 0 = "All other" or missing, 1 = a specific Hispanic origin is recorded.
    if "hispanic origin" in df.columns:
        df["hispanic_origin_indicator"] = np.where(
            df["hispanic origin"].isin(["Missing", "All other"])
            | df["hispanic origin"].isna(),
            0,
            1,
        )

    # Explicit missing value handling
    # We fill missing values with "Unknown" rather than imputation
    if "country of birth self" in df.columns:
        df["country of birth self"] = df["country of birth self"].fillna("Unknown")

    return df


# Function to build the sklearn ColumnTransformer for preprocessing

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Args:
        X: Feature dataframe used to infer column types.
        scale_numeric: If True, apply StandardScaler on numeric columns.
                       Set False for Random Forest and HistGradientBoosting.

    Returns:
        A fitted-ready ColumnTransformer.
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Transform numeric variables
    if scale_numeric:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    # Transform categorical variables
    categorical_transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=100,
        sparse_output=False,
    )

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


# Function to fine the optimal threshold

def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray):
    """
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        best_threshold: The threshold value that maximises F1.
        threshold_df: DataFrame of all thresholds with precision/recall/F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
    })

    threshold_df["f1"] = (
        2 * threshold_df["precision"] * threshold_df["recall"]
        / (threshold_df["precision"] + threshold_df["recall"])
    ).fillna(0)

    best_threshold = threshold_df.sort_values("f1", ascending=False).iloc[0]["threshold"]

    return best_threshold, threshold_df


# Function to evaluate model at a given threshold

def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray = None,
) -> dict:
    """
    Metrics:
        - accuracy
        - precision_>50K
        - recall_>50K 
        - f1_>50K
        - roc_auc
        - pr_auc
        - TP/FP/TN/FN

    Args:
        model_name: String label for this configuration.
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold to apply.
        sample_weight: Population weights for weighted metric computation.

    Returns:
        Dictionary of metric name to value.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "precision_>50K": precision_score(y_true, y_pred, zero_division=0,
                                          sample_weight=sample_weight),
        "recall_>50K": recall_score(y_true, y_pred, zero_division=0,
                                    sample_weight=sample_weight),
        "f1_>50K": f1_score(y_true, y_pred, zero_division=0,
                            sample_weight=sample_weight),
        "roc_auc": roc_auc_score(y_true, y_proba, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, y_proba,
                                          sample_weight=sample_weight),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


# Main function to run the entire pipeline

def main():

    # Load data
    print("Loading data")
    df = load_data(
        data_path="census-bureau.data",
        columns_path="census-bureau.columns",
    )

    # Clean and engineer features
    print("Engineering features")
    df = clean_and_engineer_features(df)

    # Identify features that will not be included in the model.
    FEATURES_TO_DROP = [
        "detailed industry recode",
        "detailed occupation recode",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "migration prev res in sunbelt",
        "region of previous residence",
        "state of previous residence",
        "country of birth father",
        "country of birth mother",
        "fill inc questionnaire for veteran's admin",
        "weight",
        "label",
        "income_over_50k",
        "target",
        "hispanic origin",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "overall_gains",
    ]

    existing_drop_cols = [c for c in FEATURES_TO_DROP if c in df.columns]

    # Preserve sample weights BEFORE dropping the weight column
    sample_weight = df["weight"].copy()

    X = df.drop(columns=existing_drop_cols)
    y = df["income_over_50k"]

    # Spliting of the training and validation dataset
    # Population weights is also split alongside X and y
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain size: {X_train.shape[0]:,} | Validation size: {X_valid.shape[0]:,}")
    print(f"Positive rate — Train: {y_train.mean():.4f} | Valid: {y_valid.mean():.4f}")
    print(f"Weight sum — Train: {w_train.sum():,.0f} | Valid: {w_valid.sum():,.0f}")

    # Models that will be used in the comparison
    models = {
        # Logistic Regression
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
            ("classifier", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=0.01,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )),
        ]),

        # Random Forest
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        # HistGradientBoosting: gradient boosted trees with histogram binning.
        "HistGradientBoosting": Pipeline([
            ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=80,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    # Fit models on full training set
    print("\nFitting models on full training set with sample weights...")
    fitted_models = {}
    thresholds = {}
    results = []

    for model_name, model in models.items():
        print(f"Fitting {model_name}...")
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=w_train,
        )

        y_proba = model.predict_proba(X_valid)[:, 1]

        # Evaluate at default threshold and use sample_weight for population-level metrics
        results.append(evaluate_model(
            f"{model_name} - Default 0.50",
            y_valid, y_proba,
            threshold=0.50,
            sample_weight=w_valid,
        ))

        # Find F1-optimal threshold and evaluate
        best_threshold, _ = find_best_threshold(y_valid, y_proba)
        thresholds[model_name] = best_threshold

        results.append(evaluate_model(
            f"{model_name} - Tuned",
            y_valid, y_proba,
            threshold=best_threshold,
            sample_weight=w_valid,
        ))

        fitted_models[model_name] = (model, best_threshold)

    # Comparison of all models
    results_df = pd.DataFrame(results).sort_values("f1_>50K", ascending=False)

    print("\n" + "=" * 70)
    print("VALIDATION SET COMPARISON (sorted by weighted F1 for >$50K class)")
    print("=" * 70)
    print(results_df[[
        "model", "threshold", "precision_>50K", "recall_>50K",
        "f1_>50K", "roc_auc", "pr_auc"
    ]].to_string(index=False))

    # Final evaluation of the best model
    best_row = results_df.iloc[0]
    best_model_name = (
        best_row["model"]
        .replace(" - Tuned", "")
        .replace(" - Default 0.50", "")
    )
    best_model, best_threshold = fitted_models[best_model_name]

    y_proba_best = best_model.predict_proba(X_valid)[:, 1]
    y_pred_best = (y_proba_best >= best_threshold).astype(int)

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_row['model']}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"{'='*70}")
    print("\nFinal Classification Report (unweighted):")
    print(classification_report(y_valid, y_pred_best, target_names=["<=50K", ">50K"]))
    print("Final Classification Report (population-weighted):")
    print(classification_report(y_valid, y_pred_best, target_names=["<=50K", ">50K"],
                                sample_weight=w_valid))
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred_best))

    # Save output for future
    results_df.to_csv("model_comparison_results.csv", index=False)

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "features": X.columns.tolist(),
        },
        "best_income_model.joblib",
    )

    print("\nSaved model to best_income_model.joblib")
    print("Saved results to model_comparison_results.csv")


if __name__ == "__main__":
    main()