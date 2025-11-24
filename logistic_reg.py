import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_DIR = Path.cwd() / "data"
OUTPUT_DIR = Path.cwd() / "outputs_logreg"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
ZIP_PATH = DATA_DIR / "bank-additional.zip"
CSV_REL_PATH = "bank-additional/bank-additional-full.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def download_and_extract():
    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print(f"Saved {ZIP_PATH}")
    else:
        print("Zip already exists, skipping download.")

    csv_target = DATA_DIR / "bank-additional-full.csv"
    if not csv_target.exists():
        print("Extracting CSV from zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            member = CSV_REL_PATH
            z.extract(member, path=DATA_DIR)
            extracted = DATA_DIR / member
            shutil.move(str(extracted), str(csv_target))
            extracted.parent.rmdir()
        print("Extraction done.")
    else:
        print("CSV already extracted.")
    return csv_target


def load_and_preview(csv_path):
    print("\n" + "=" * 60)
    print("Loading Bank Marketing Dataset...")
    print("=" * 60)
    df = pd.read_csv(csv_path, sep=';')
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df['y'].value_counts())
    print(f"\nClass imbalance ratio: {(df['y']=='no').sum() / (df['y']=='yes').sum():.2f}:1")
    return df


def preprocess(df):
    df = df.copy()
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    expected_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    numeric_cols = [c for c in expected_numeric if c in df.columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ['y']]

    df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)
    X_num = df[numeric_cols].astype(float)

    X = pd.concat([X_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    y = df['y'].values

    print(f"\nFeatures after encoding: {X.shape[1]} total")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Encoded categorical features: {X.shape[1] - len(numeric_cols)}")

    return X, y


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / 'roc_curve.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  └─ ROC curve saved: {path.name}")
    return path, auc


def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\nTraining Logistic Regression...")

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver='lbfgs'
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  ✓ Accuracy: {acc:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ AUC: {auc:.4f}")

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - Logistic Regression")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    cm_path = OUTPUT_DIR / "confusion_matrix_logreg.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    result_file = OUTPUT_DIR / "logreg_results.txt"
    with open(result_file, "w") as f:
        f.write("Logistic Regression Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['no', 'yes']))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm_path": str(cm_path)
    }, clf, y_proba


def main():
    csv_path = download_and_extract()
    df = load_and_preview(csv_path)
    X, y = preprocess(df)

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION EXPERIMENT")
    print("=" * 60)

    result, clf, y_proba = run_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    print("\nGenerating ROC curve...")
    plot_roc_curve(y_test, y_proba)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(result)

    print("\nAll outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
