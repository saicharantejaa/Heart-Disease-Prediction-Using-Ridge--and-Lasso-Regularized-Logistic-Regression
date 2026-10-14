# Heart Disease Prediction Using Ridge and Lasso Regularized Logistic Regression

A machine learning project comparing standard Logistic Regression with Ridge ($L_2$) and Lasso ($L_1$) regularization techniques to predict coronary heart disease using the UCI Cleveland Heart Disease dataset.

Includes complete model training pipelines, cross-validation, hyperparameter tuning, metric evaluations, diagnostic visualizations, and an interactive Flask web application.

---

## Repository Structure

```text
├── Project/
│   ├── data/
│   │   └── heart_disease_cleaned.csv    # Preprocessed UCI Cleveland dataset
│   ├── model/                           # Serialized models, scaler, metrics
│   │   ├── all_models.joblib
│   │   ├── best_model.joblib
│   │   ├── best_pipeline.joblib
│   │   ├── feature_metadata.json
│   │   ├── metrics.json
│   │   ├── model_comparison.csv
│   │   └── scaler.joblib
│   ├── notebooks/
│   │   └── eda.py                       # Exploratory data analysis script
│   ├── static/
│   │   ├── images/                      # Generated evaluation & EDA charts
│   │   └── style.css                    # Web interface styling
│   ├── templates/                       # Flask HTML templates
│   │   ├── predict.html                 # Patient risk screening form
│   │   └── report.html                  # Full academic case study & report
│   ├── app.py                           # Flask web application entrypoint
│   ├── train_model.py                   # Model training and evaluation script
│   ├── case_study.md                    # Detailed written case study analysis
│   ├── requirements.txt                 # Python package dependencies
│   └── README.md                        # Project-level guide
├── .gitignore                           # Git ignore rules
└── README.md                            # Main repository documentation
```

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saicharantejaa/Heart-Disease-Prediction-Using-Ridge--and-Lasso-Regularized-Logistic-Regression.git
   cd Heart-Disease-Prediction-Using-Ridge--and-Lasso-Regularized-Logistic-Regression/Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**:
   ```bash
   python train_model.py
   ```

4. **Launch the web application**:
   ```bash
   python app.py
   ```
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Machine Learning Overview

- **Algorithm**: Logistic Regression
- **Regularization Techniques**:
  - **No Regularization** (Baseline)
  - **Ridge ($L_2$)**: Penalizes large coefficients, prevents multicollinearity issues.
  - **Lasso ($L_1$)**: Induces feature sparsity, performing automatic feature selection by zeroing out coefficients.
- **Optimization & Evaluation**:
  - Stratified 5-Fold Cross-Validation
  - Hyperparameter tuning across penalty strengths ($C \in [0.001, 100]$)
  - Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC

---

## Dataset

- **UCI Machine Learning Repository**: Heart Disease Dataset (Cleveland database)
- **Instances**: 303 clinical records (cleaned)
- **Features**: 13 diagnostic attributes + binary target (presence of heart disease)

---

## License & Disclaimer

Academic and educational use only. This project is not a medical device and should not be used for clinical diagnoses or patient care.