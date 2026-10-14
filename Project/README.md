# CardioScreen: Heart Disease Prediction Case Study

An academic machine learning web application built around the UCI Cleveland Heart Disease dataset. It trains, evaluates, and compares standard, Ridge ($L_2$), and Lasso ($L_1$) regularized logistic regression models, generates visual diagnostics, and provides an interactive clinical patient-screening interface.

## Quick Start

1. **Navigate into the Project folder**:
   ```bash
   cd Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train and evaluate the models**:
   ```bash
   python train_model.py
   ```
   *(This generates the trained models in `model/`, evaluation metrics, and EDA/diagnostic plots in `static/images/`.)*

4. **Run the Flask web application**:
   ```bash
   python app.py
   ```

5. **Open in browser**:
   Navigate to `http://127.0.0.1:5000` to view the comprehensive case study report, model comparisons, and interactive screening form at `/predict`.

## Key Features

- **Model Comparison**: Rigorous evaluation of unregularized Logistic Regression vs. Ridge ($L_2$) and Lasso ($L_1$) regularization.
- **Cross-Validation & Hyperparameter Tuning**: Stratified 5-Fold cross-validation with grid search over regularization strengths ($C$).
- **Explainability**: Patient risk prediction includes feature contribution breakdown for clinical transparency.
- **Web Interface**: Clean, responsive UI with interactive prediction tool and downloadable clinical summary reports.

## Dataset

- **Source**: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Features**: 13 clinical attributes (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise angina, ST depression, slope, number of major vessels, thalassemia).

> **Disclaimer**: Academic and educational use only. This application does not constitute professional medical diagnosis, triage, or clinical advice.
