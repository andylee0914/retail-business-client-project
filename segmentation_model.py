"""
=======================================================
Census Income — Customer Segmentation Pipeline
--------------------------------------------------------
Pipeline summary:
  1. Load the raw dataset
  2. Clean data, engineer features, filter to adults (age >= 18)
  3. Define segmentation feature set
  4. K-Means baseline
  5. K-Prototypes primary model
  6. Print cluster profiles and save results

Outputs:
    - segmentation_results.csv : full dataset with cluster assignments
    - segmentation_model.joblib : Best model for cluster assignment
"""

import warnings
import numpy as np
warnings.filterwarnings("ignore")
np.seterr(divide="ignore", over="ignore", invalid="ignore")

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kmodes.kprototypes import KPrototypes

RANDOM_STATE = 42

# Function to load the data

def load_data(data_path: str, columns_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: Path to census-bureau.data
        columns_path: Path to census-bureau.columns

    Returns:
        Raw dataframe with named columns.
    """
    cols = pd.read_csv(columns_path, header=None)[0].tolist()
    df = pd.read_csv(data_path, header=None, names=cols)
    return df


# Function to clean, engineer features, and filter to adults

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: Raw dataframe from load_data().

    Returns:
        Cleaned, filtered dataframe with engineered features.
    """
    df = df.copy()

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Replace sentinel missing values
    df = df.replace("?", np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Keep income label for post-hoc validation only
    df["income_over_50k"] = df["label"].apply(
        lambda x: 1 if "50000+" in str(x) else 0
    )

    # Filter to adults
    # Minors form an uninformative cluster and are not a valid retail target.
    n_before = len(df)
    df = df[df["age"] >= 18].copy()
    n_removed = n_before - len(df)
    print(f"Adult filter: removed {n_removed:,} minors ({n_removed/n_before*100:.1f}%)")
    print(f"Adult dataset: {len(df):,} rows | Income >$50K rate: "
          f"{df['income_over_50k'].mean()*100:.2f}%")

    # Net investment income which is log-compressed
    df["overall_gains"] = (
        df["capital gains"]
        - df["capital losses"]
        + df["dividends from stocks"]
    )
    df["overall_gains_log"] = (
        np.sign(df["overall_gains"]) * np.log1p(np.abs(df["overall_gains"]))
    )

    # Create hourly wage indicator
    df["has_hourly_wage"] = (df["wage per hour"] > 0).astype(int)

    # US-born indicator
    df["is_us_born"] = (df["country of birth self"] == "United-States").astype(int)

    # Hispanic origin indicator
    df["hispanic_origin_indicator"] = np.where(
        df["hispanic origin"].isin(["Missing", "All other"])
        | df["hispanic origin"].isna(),
        0, 1
    )

    # Explicit missing value handling
    # Fill with "Unknown" instead of imputation
    df["country of birth self"] = df["country of birth self"].fillna("Unknown")

    return df


# Defining the numeric and categorical features for segmentation

NUMERIC_FEATURES = [
    "age",
    "weeks worked in year",
    "overall_gains_log",
    "has_hourly_wage",
    "own business or self employed",
    "veterans benefits",
    "is_us_born",
    "hispanic_origin_indicator",
]

CATEGORICAL_FEATURES = [
    "education",
    "marital stat",
    "sex",
    "race",
    "major occupation code",
    "major industry code",
    "class of worker",
    "full or part time employment stat",
    "tax filer stat",
    "citizenship",
    "enroll in edu inst last wk",
    "detailed household and family stat",
    "family members under 18",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# helper functions for cluster evaluation and description

def choose_k_kmeans(X_scaled: np.ndarray, k_range: range) -> None:
    """
    Args:
        X_scaled: Scaled numeric feature matrix.
        k_range: Range of k values to evaluate.
    """
    print("\nK-Means elbow analysis (numeric features only):")
    print(f"{'k':>4}  {'inertia':>12}  {'silhouette':>10}  {'davies-bouldin':>14}")
    print("-" * 46)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=10_000,
                               random_state=RANDOM_STATE)
        db = davies_bouldin_score(X_scaled, labels)
        print(f"{k:>4}  {km.inertia_:>12.0f}  {sil:>10.4f}  {db:>14.4f}")


def describe_cluster(cluster_id: int, df_seg: pd.DataFrame) -> None:
    """
    Args:
        cluster_id: Cluster label to describe.
        df_seg: Segmentation dataframe with 'kproto_cluster' column.
    """
    mask = df_seg["kproto_cluster"] == cluster_id
    subset = df_seg[mask]

    n = len(subset)
    pct = n / len(df_seg) * 100
    income_rate = subset["income_over_50k"].mean() * 100

    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id} | n={n:,} ({pct:.1f}% of adult population)")
    print(f"Income >$50K: {income_rate:.1f}%  "
          f"(overall avg: {df_seg['income_over_50k'].mean()*100:.1f}%)")
    print(f"{'-'*60}")

    print("Numeric averages:")
    for feat in NUMERIC_FEATURES:
        print(f"  {feat:35s}: {subset[feat].mean():.2f}")

    print("\nMost common categorical values:")
    for feat in CATEGORICAL_FEATURES:
        top_val = subset[feat].value_counts().index[0]
        top_pct = subset[feat].value_counts().iloc[0] / n * 100
        print(f"{feat:35s}: {top_val} ({top_pct:.0f}%)")


# Main function to run the entire pipeline

def main():

    # Load data
    print("Loading data...")
    df = load_data(
        data_path="census-bureau.data",
        columns_path="census-bureau.columns",
    )

    # Clean, engineer features, filter to adults
    print("\nCleaning and engineering features...")
    df = clean_and_prepare(df)

    # Build segmentation dataset
    df_seg = df[ALL_FEATURES + ["income_over_50k"]].copy()

    print(f"\nSegmentation dataset: {df_seg.shape[0]:,} rows x {len(ALL_FEATURES)} features")
    print(f"Numeric features({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"Categorical features({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

    # Scale numeric features
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_num = imputer.fit_transform(df_seg[NUMERIC_FEATURES])
    X_num_scaled = scaler.fit_transform(X_num)

    # K-means model
    print("\n" + "="*60)
    print("MODEL 1 — K-MEANS BASELINE (numeric features only)")
    print("="*60)

    choose_k_kmeans(X_num_scaled, range(2, 9))

    # Fit K-Means with chosen k = 5
    KMEANS_K = 5
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=10)
    df_seg["kmeans_cluster"] = kmeans.fit_predict(X_num_scaled)

    print(f"\nK-Means (k={KMEANS_K}) cluster sizes:")
    print(df_seg["kmeans_cluster"].value_counts().sort_index().to_string())

    print("\nK-Means income >$50K rate per cluster (post-hoc validation):")
    km_income = (
        df_seg.groupby("kmeans_cluster")["income_over_50k"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pct_over_50k", "count": "n"})
    )
    km_income["pct_over_50k"] = (km_income["pct_over_50k"] * 100).round(1)
    print(km_income.to_string())

    # K-prototypes model
    print("\n" + "="*60)
    print("MODEL 2 — K-PROTOTYPES (primary model, mixed data)")
    print("="*60)

    X_cat = df_seg[CATEGORICAL_FEATURES].values
    X_kp = np.concatenate([X_num_scaled, X_cat], axis=1)
    cat_idx = list(range(len(NUMERIC_FEATURES),
                         len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

    # Elbow analysis for K-Prototypes
    print("\nK-Prototypes elbow analysis (mixed distance):")
    print(f"{'k':>4}  {'cost':>14}")
    print("-" * 20)
    kp_costs = []
    for k in range(2, 8):
        kp = KPrototypes(n_clusters=k, init="Cao", n_init=3,
                         random_state=RANDOM_STATE)
        kp.fit(X_kp, categorical=cat_idx)
        kp_costs.append(kp.cost_)
        print(f"{k:>4} {kp.cost_:>14.2f}")

    # Fit final K-Prototypes model with chosen k = 5
    KP_K = 5
    print(f"\nFitting final K-Prototypes model (k={KP_K})...")
    kp_final = KPrototypes(n_clusters=KP_K, init="Cao", n_init=5,
                           random_state=RANDOM_STATE)
    df_seg["kproto_cluster"] = kp_final.fit_predict(X_kp, categorical=cat_idx)

    print(f"\nK-Prototypes (k={KP_K}) cluster sizes:")
    print(df_seg["kproto_cluster"].value_counts().sort_index().to_string())

    # Cluster profiles
    print("\n" + "="*60)
    print("CLUSTER PROFILES — K-PROTOTYPES")
    print("="*60)

    print("\nIncome >$50K rate per cluster (post-hoc validation):")
    kp_income = (
        df_seg.groupby("kproto_cluster")["income_over_50k"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pct_over_50k", "count": "n"})
    )
    kp_income["pct_over_50k"] = (kp_income["pct_over_50k"] * 100).round(1)
    kp_income["pct_of_adults"] = (kp_income["n"] / len(df_seg) * 100).round(1)
    print(kp_income.to_string())

    for cid in range(KP_K):
        describe_cluster(cid, df_seg)

    # Save results and model
    df_output = df[ALL_FEATURES + ["income_over_50k"]].copy()
    df_output["kmeans_cluster"] = df_seg["kmeans_cluster"].values
    df_output["kproto_cluster"] = df_seg["kproto_cluster"].values
    df_output.to_csv("segmentation_results.csv", index=False)

    joblib.dump(
        {
            "model": kp_final,
            "cat_indices": cat_idx,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "scaler": scaler,
            "imputer": imputer,
            "n_clusters": KP_K,
        },
        "segmentation_model.joblib",
    )

    print("\nSaved model to segmentation_model.joblib")
    print("Saved results to segmentation_results.csv")


if __name__ == "__main__":
    main()