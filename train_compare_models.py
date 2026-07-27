"""
Train and compare phishing detection models on the PhiUSIIL dataset.

Compares Decision Tree, Logistic Regression, Random Forest, and XGBoost
on (1) the full numeric reference feature set and (2) the deployable-only
feature set that a Chrome extension can extract from URL + DOM.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

# Relative to this script's directory (project folder).
CSV_PATH = Path(
    r"..\..\Fall_2024\COMP_3260\Proof Of Concept\Data\phishing_url_website.csv"
    r"\PhiUSIIL_Phishing_URL_Dataset (Full).csv"
)

# Features a Chrome extension can compute from the active tab (URL + DOM).
DEPLOYABLE_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "HasFavicon",
    "IsResponsive",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

# Numeric features that need dataset-side / external signals not available
# from a single browser tab snapshot.
REFERENCE_ONLY = [
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "Robots",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
]

NON_FEATURE_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]

RANDOM_STATE = 42
N_ESTIMATORS_DEFAULT = 200
CV_FOLDS = 5
TEST_SIZE = 0.2


def resolve_csv_path() -> Path:
    base = Path(__file__).resolve().parent
    candidate = (base / CSV_PATH).resolve()
    if candidate.is_file():
        return candidate
    # Fallback: absolute path provided by the project owner.
    fallback = Path(
        r"C:\Users\m_ray\OneDrive\Документы\Raiyan\Fall_2024\COMP_3260"
        r"\Proof Of Concept\Data\phishing_url_website.csv"
        r"\PhiUSIIL_Phishing_URL_Dataset (Full).csv"
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Dataset not found at relative path {candidate} or fallback {fallback}"
    )


def build_models(n_estimators: int = N_ESTIMATORS_DEFAULT) -> dict:
    # Keep n_jobs low: full-dataset CV + ensembles OOMs on modest machines.
    models = {
        "DecisionTree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=12,
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=16,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        print("WARNING: xgboost not installed; skipping XGBoost.")
    return models


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs=1 avoids joblib memmap / paging-file failures on Windows.
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # label 0 = Phishing, label 1 = Legitimate
    phishing_recall = recall_score(y_test, y_pred, pos_label=0)
    phishing_f1 = f1_score(y_test, y_pred, pos_label=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Phishing", "Legitimate"]
    )

    print(f"\n{'=' * 72}")
    print(f"Model: {name}")
    print(f"{'=' * 72}")
    print(
        f"5-fold CV macro-F1 (train): "
        f"{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )
    print(f"Holdout macro-F1:          {macro_f1:.4f}")
    print(f"Phishing (0) recall:       {phishing_recall:.4f}")
    print(f"Phishing (0) F1:           {phishing_f1:.4f}")
    print("Confusion matrix [rows=true 0,1; cols=pred 0,1]:")
    print(cm)
    print("\nClassification report:")
    print(report)

    return {
        "name": name,
        "model": model,
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "holdout_macro_f1": float(macro_f1),
        "phishing_recall": float(phishing_recall),
        "phishing_f1": float(phishing_f1),
        "confusion_matrix": cm,
    }


def run_feature_set(label: str, features: list[str], df: pd.DataFrame) -> list[dict]:
    print(f"\n{'#' * 72}")
    print(f"# FEATURE SET: {label} ({len(features)} features)")
    print(f"{'#' * 72}")
    print("Features:", features)

    X = df[features].to_numpy(dtype=np.float32, copy=False)
    y = df["label"].to_numpy(dtype=np.int32, copy=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    for name, model in build_models().items():
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))
    return results


def pick_best_deployable(results: list[dict]) -> dict:
    """
    Priority:
      1) Phishing-class recall (label 0)
      2) Phishing-class F1
      3) Prefer smaller models (DT / LR) if ensemble lead is marginal (~1-2 F1 pts)
    """
    ranked = sorted(
        results,
        key=lambda r: (r["phishing_recall"], r["phishing_f1"], r["holdout_macro_f1"]),
        reverse=True,
    )
    best = ranked[0]
    small = [r for r in results if r["name"] in ("DecisionTree", "LogisticRegression")]
    if best["name"] in ("RandomForest", "XGBoost") and small:
        best_small = max(small, key=lambda r: (r["phishing_recall"], r["phishing_f1"]))
        f1_gap = best["phishing_f1"] - best_small["phishing_f1"]
        recall_gap = best["phishing_recall"] - best_small["phishing_recall"]
        print(
            f"\nEnsemble '{best['name']}' vs best small '{best_small['name']}': "
            f"phishing F1 gap={f1_gap:.4f}, recall gap={recall_gap:.4f}"
        )
        if f1_gap <= 0.02 and recall_gap <= 0.02:
            print(
                "Ensemble lead is marginal (<=2 pts). Preferring smaller deployable model."
            )
            return best_small
    return best


def main() -> None:
    csv_path = resolve_csv_path()
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print("label value_counts:")
    print(df["label"].value_counts())
    print(f"total missing values: {int(df.isnull().sum().sum())}")

    missing_deployable = [c for c in DEPLOYABLE_FEATURES if c not in df.columns]
    missing_reference = [c for c in REFERENCE_ONLY if c not in df.columns]
    if missing_deployable or missing_reference:
        raise ValueError(
            f"Missing deployable columns: {missing_deployable}; "
            f"missing reference columns: {missing_reference}"
        )

    reference_features = DEPLOYABLE_FEATURES + REFERENCE_ONLY
    extra_numeric = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in reference_features
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if extra_numeric:
        print(
            "NOTE: additional numeric columns not in DEPLOYABLE/REFERENCE lists:",
            extra_numeric,
        )

    print("\n=== REFERENCE (full numeric) feature set ===")
    reference_results = run_feature_set("REFERENCE_FULL", reference_features, df)

    print("\n=== DEPLOYABLE feature set ===")
    deployable_results = run_feature_set("DEPLOYABLE", DEPLOYABLE_FEATURES, df)

    best = pick_best_deployable(deployable_results)
    print(f"\n>>> Selected best DEPLOYABLE model: {best['name']}")
    print(f"    Phishing recall: {best['phishing_recall']:.4f}")
    print(f"    Phishing F1:     {best['phishing_f1']:.4f}")
    print(f"    Holdout macro-F1:{best['holdout_macro_f1']:.4f}")

    out_path = Path(__file__).resolve().parent / "best_phishing_model.pkl"
    payload = {
        "model": best["model"],
        "model_name": best["name"],
        "features": DEPLOYABLE_FEATURES,
        "metrics": {
            "phishing_recall": best["phishing_recall"],
            "phishing_f1": best["phishing_f1"],
            "holdout_macro_f1": best["holdout_macro_f1"],
            "cv_macro_f1_mean": best["cv_macro_f1_mean"],
            "cv_macro_f1_std": best["cv_macro_f1_std"],
        },
        "label_map": {0: "Phishing", 1: "Legitimate"},
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved best deployable model to: {out_path}")

    print("\n=== DEPLOYABLE SUMMARY ===")
    for r in sorted(
        deployable_results, key=lambda x: (-x["phishing_recall"], -x["phishing_f1"])
    ):
        print(
            f"{r['name']:20s}  phishing_recall={r['phishing_recall']:.4f}  "
            f"phishing_f1={r['phishing_f1']:.4f}  macro_f1={r['holdout_macro_f1']:.4f}  "
            f"cv_macro_f1={r['cv_macro_f1_mean']:.4f}"
        )

    print("\n=== REFERENCE SUMMARY ===")
    for r in sorted(
        reference_results, key=lambda x: (-x["phishing_recall"], -x["phishing_f1"])
    ):
        print(
            f"{r['name']:20s}  phishing_recall={r['phishing_recall']:.4f}  "
            f"phishing_f1={r['phishing_f1']:.4f}  macro_f1={r['holdout_macro_f1']:.4f}  "
            f"cv_macro_f1={r['cv_macro_f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
