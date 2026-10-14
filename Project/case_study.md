# Heart Disease Prediction Case Study

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
search = GridSearchCV(ridge, {"model__C": C_VALUES}, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=42))
search.fit(X_train, y_train)
```

**Held-out test-set results** (best model selected by AUC-ROC: **Ridge Logistic Regression (L2)**):

| Model                          |   Accuracy |   Precision |   Recall |   F1-score |   AUC-ROC |   Selected C |
|:-------------------------------|-----------:|------------:|---------:|-----------:|----------:|-------------:|
| Plain Logistic Regression      |      0.869 |       0.812 |    0.929 |      0.867 |     0.95  |          1   |
| Ridge Logistic Regression (L2) |      0.869 |       0.812 |    0.929 |      0.867 |     0.955 |          0.5 |
| Lasso Logistic Regression (L1) |      0.869 |       0.812 |    0.929 |      0.867 |     0.945 |          1   |

![ROC comparison](/static/images/roc_curves.png)

![Best-model confusion matrix](/static/images/best_confusion_matrix.png)

![Ridge and Lasso coefficient magnitudes](/static/images/coefficient_magnitudes.png)

### What this means in practice

**Case 1: observed target 0.** A 59-year-old patient with resting blood pressure 138 mm Hg, cholesterol 271 mg/dL, maximum heart rate 182 bpm, and exercise angina code 0 was classified as **negative** at a predicted probability of **22.3%**. This is a model screening output, not an explanation of causality or a diagnosis.

**Case 2: observed target 1.** A 35-year-old patient with resting blood pressure 126 mm Hg, cholesterol 282 mg/dL, maximum heart rate 156 bpm, and exercise angina code 1 was classified as **positive** at a predicted probability of **79.5%**. This is a model screening output, not an explanation of causality or a diagnosis.

### Limitations

The Cleveland data is small, historical, and drawn from a single hospital population, so its performance may not generalize to other communities or current clinical practice. It does not include a complete medical history, imaging, clinician judgment, or outcome calibration study. This application is for academic use only and must not be used to diagnose, triage, or treat a person.
