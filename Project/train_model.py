"""Train, evaluate, package, and document the heart disease classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_disease_cleaned.csv"
MODEL_DIR = ROOT / "model"
IMAGE_DIR = ROOT / "static" / "images"
CASE_STUDY_PATH = ROOT / "case_study.md"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
EXPECTED_COLUMNS = FEATURES + ["target"]
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = data[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    data["target"] = data["target"].astype(int)
    if not set(data["target"].unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")
    return data


def specs() -> dict:
    return {
        "Plain Logistic Regression": (Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000))]), None),
        "Ridge Logistic Regression (L2)": (Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000))]), {"model__C": C_VALUES}),
        "Lasso Logistic Regression (L1)": (Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))]), {"model__C": C_VALUES}),
    }


def evaluate(name, pipeline, x_test, y_test) -> dict:
    prediction = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]
    return {"model": name, "accuracy": float(accuracy_score(y_test, prediction)), "precision": float(precision_score(y_test, prediction, zero_division=0)), "recall": float(recall_score(y_test, prediction, zero_division=0)), "f1_score": float(f1_score(y_test, prediction, zero_division=0)), "auc_roc": float(roc_auc_score(y_test, probability)), "best_c": pipeline.named_steps["model"].get_params().get("C")}


def save_plots(data, x_test, y_test, fitted, best_name, metrics) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.figure(figsize=(12, 9)); sns.heatmap(data.corr(numeric_only=True), cmap="RdBu_r", center=0, square=True, linewidths=.35, cbar_kws={"shrink": .78}); plt.title("Clinical Feature Correlation Heatmap", pad=16, weight="bold"); plt.tight_layout(); plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight"); plt.close()
    counts = data["target"].value_counts().sort_index(); plt.figure(figsize=(7, 4.5)); bars = plt.bar(["No heart disease (0)", "Heart disease (1)"], counts.values, color=["#2f6f8f", "#b65050"], width=.58); plt.title("Class Distribution", pad=14, weight="bold"); plt.ylabel("Patients")
    for bar, value in zip(bars, counts.values): plt.text(bar.get_x() + bar.get_width()/2, value + 3, str(value), ha="center", weight="bold")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "class_distribution.png", dpi=180, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8, 6))
    for name, pipeline in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(x_test)[:, 1]); plt.plot(fpr, tpr, linewidth=2.4, label=f"{name} (AUC {metrics[name]['auc_roc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#8b9aa4", linewidth=1.3, label="No discrimination"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curves: Held-Out Test Set", pad=14, weight="bold"); plt.legend(loc="lower right", fontsize=8); plt.tight_layout(); plt.savefig(IMAGE_DIR / "roc_curves.png", dpi=180, bbox_inches="tight"); plt.close()
    matrix = confusion_matrix(y_test, fitted[best_name].predict(x_test)); plt.figure(figsize=(6.5, 5.2)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["No disease", "Heart disease"], yticklabels=["No disease", "Heart disease"], linewidths=1, linecolor="white"); plt.title(f"Confusion Matrix: {best_name}", pad=14, weight="bold"); plt.xlabel("Predicted class"); plt.ylabel("Actual class"); plt.tight_layout(); plt.savefig(IMAGE_DIR / "best_confusion_matrix.png", dpi=180, bbox_inches="tight"); plt.close()
    rows = []
    for name in ("Ridge Logistic Regression (L2)", "Lasso Logistic Regression (L1)"):
        rows.extend((feature, abs(weight), name.split()[0]) for feature, weight in zip(FEATURES, fitted[name].named_steps["model"].coef_[0]))
    coefficients = pd.DataFrame(rows, columns=["feature", "magnitude", "model"]); order = coefficients.groupby("feature").magnitude.max().sort_values().index
    plt.figure(figsize=(10, 7)); sns.barplot(data=coefficients, x="magnitude", y="feature", hue="model", order=order, palette={"Ridge":"#2f6f8f", "Lasso":"#b65050"}); plt.title("Regularized Model Coefficient Magnitudes", pad=14, weight="bold"); plt.xlabel("Absolute standardized coefficient"); plt.ylabel("Clinical feature"); plt.tight_layout(); plt.savefig(IMAGE_DIR / "coefficient_magnitudes.png", dpi=180, bbox_inches="tight"); plt.close()


def write_case_study(data, metrics, best_name, x_test, y_test, best_pipeline) -> None:
    table = pd.DataFrame(metrics.values())[["model", "accuracy", "precision", "recall", "f1_score", "auc_roc", "best_c"]].copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]: table[col] = table[col].map(lambda value: f"{value:.3f}")
    table.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "Selected C"]
    cases = x_test.copy(); cases["actual"] = y_test; cases["predicted"] = best_pipeline.predict(x_test); cases["probability"] = best_pipeline.predict_proba(x_test)[:, 1]
    case_text = []
    for target in (0, 1):
        row = cases[cases.actual == target].iloc[0]; label = "positive" if row.predicted else "negative"
        case_text.append(f"**Case {target + 1}: observed target {target}.** A {int(row.age)}-year-old patient with resting blood pressure {int(row.trestbps)} mm Hg, cholesterol {int(row.chol)} mg/dL, maximum heart rate {int(row.thalach)} bpm, and exercise angina code {int(row.exang)} was classified as **{label}** at a predicted probability of **{row.probability:.1%}**. This is a model screening output, not an explanation of causality or a diagnosis.")
    CASE_STUDY_PATH.write_text(f'''# Heart Disease Prediction Case Study

## 1. Problem Statement (Case-Based)

A clinic's preventive cardiology team wants a consistent early-screening aid for adults arriving with routine cardiovascular measurements. Before scheduling expensive follow-up tests, clinicians need a way to flag a patient whose age, blood pressure, cholesterol, ECG result, exercise response, and related findings resemble patterns historically associated with heart disease. This coursework model estimates screening risk from routinely recorded data so the team can prioritize clinical review; it does not replace a clinician or establish a diagnosis.

## 2. Dataset & Problem Introduction

This project uses the Cleveland subset of the **UCI Heart Disease** dataset: 303 patient records with 14 clinical attributes, including 13 input features and a binary `target`. The source is the UCI Machine Learning Repository's Heart Disease dataset. The supplied cleaned file uses `target = 1` for heart disease and `target = 0` for no heart disease. In this context, the task is binary classification: use the 13 available routine findings to estimate whether a record belongs to the positive class.

## 3. EDA — Exploration + Implementation + Outputs

The data preparation keeps documented columns, converts them to numeric values, removes incomplete or duplicate records, and verifies that the target remains binary. Scaling is fitted inside each training pipeline so test-set information cannot affect the transformation.

```python
frame = pd.read_csv("data/heart_disease_cleaned.csv")
frame = frame[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
frame = frame.dropna().drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(
    frame[FEATURES], frame["target"], test_size=0.20,
    stratify=frame["target"], random_state=42
)
```

The generated outputs below show the relationships among the clinical variables and the balance of the prediction target.

![Correlation heatmap](/static/images/correlation_heatmap.png)

![Class distribution](/static/images/class_distribution.png)

## 4. Algorithm Implementation

Three logistic classifiers were trained on the same stratified 80/20 split. Plain logistic regression provides a baseline. Ridge (L2) shrinks correlated feature weights smoothly, while Lasso (L1) can shrink some weights to zero. Ridge and Lasso tune `C` (the inverse regularization strength) with stratified 5-fold cross-validation using ROC-AUC.

```python
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)),
])
search = GridSearchCV(ridge, {{"model__C": C_VALUES}}, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=42))
search.fit(X_train, y_train)
```

**Held-out test-set results** (best model selected by AUC-ROC: **{best_name}**):

{table.to_markdown(index=False)}

![ROC comparison](/static/images/roc_curves.png)

![Best-model confusion matrix](/static/images/best_confusion_matrix.png)

![Ridge and Lasso coefficient magnitudes](/static/images/coefficient_magnitudes.png)

### What this means in practice

{case_text[0]}

{case_text[1]}

### Limitations

The Cleveland data is small, historical, and drawn from a single hospital population, so its performance may not generalize to other communities or current clinical practice. It does not include a complete medical history, imaging, clinician judgment, or outcome calibration study. This application is for academic use only and must not be used to diagnose, triage, or treat a person.
''', encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(); x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], data.target, test_size=.20, stratify=data.target, random_state=42)
    cv = StratifiedKFold(n_splits=min(5, int(y_train.value_counts().min())), shuffle=True, random_state=42)
    fitted, metrics = {}, {}
    for name, (pipeline, params) in specs().items():
        estimator = pipeline if params is None else GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=1)
        estimator.fit(x_train, y_train); fitted[name] = estimator if params is None else estimator.best_estimator_; metrics[name] = evaluate(name, fitted[name], x_test, y_test)
    best_name = max(metrics, key=lambda name: metrics[name]["auc_roc"]); best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, MODEL_DIR / "best_pipeline.joblib"); joblib.dump(fitted, MODEL_DIR / "all_models.joblib"); joblib.dump(best_pipeline.named_steps["model"], MODEL_DIR / "best_model.joblib"); joblib.dump(best_pipeline.named_steps["scaler"], MODEL_DIR / "scaler.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"best_model": best_name, "models": metrics}, indent=2), encoding="utf-8"); pd.DataFrame(metrics.values()).to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    metadata = {"features": FEATURES, "defaults": {key: float(data[key].median()) for key in FEATURES}, "ranges": {key: [float(data[key].min()), float(data[key].max())] for key in FEATURES}, "categories": {key: sorted(float(x) for x in data[key].unique()) for key in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]}, "sample_count": int(len(data))}
    (MODEL_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plots(data, x_test, y_test, fitted, best_name, metrics); write_case_study(data, metrics, best_name, x_test, y_test, best_pipeline)
    print("\nModel comparison (held-out test set)"); print(pd.DataFrame(metrics.values())[["model", "accuracy", "precision", "recall", "f1_score", "auc_roc", "best_c"]].to_string(index=False, float_format=lambda value: f"{value:.3f}")); print(f"\nBest model: {best_name}")


if __name__ == "__main__": main()
