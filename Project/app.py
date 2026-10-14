"""Flask application for the academic heart disease case study."""
from __future__ import annotations

import json
import textwrap
from io import BytesIO
from pathlib import Path

import joblib
import markdown
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
FEATURE_LABELS = {"age":"Age (years)","sex":"Sex","cp":"Chest pain type","trestbps":"Resting blood pressure (mm Hg)","chol":"Serum cholesterol (mg/dL)","fbs":"Fasting blood sugar > 120 mg/dL","restecg":"Resting ECG result","thalach":"Maximum heart rate","exang":"Exercise-induced angina","oldpeak":"ST depression (oldpeak)","slope":"ST slope","ca":"Major vessels (ca)","thal":"Thalassemia code"}
CATEGORY_LABELS = {"sex":{0:"Female (0)",1:"Male (1)"},"cp":{1:"Typical angina (1)",2:"Atypical angina (2)",3:"Non-anginal pain (3)",4:"Asymptomatic (4)"},"fbs":{0:"False (0)",1:"True (1)"},"restecg":{0:"Normal (0)",1:"ST-T abnormality (1)",2:"Left ventricular hypertrophy (2)"},"exang":{0:"No (0)",1:"Yes (1)"},"slope":{1:"Upsloping (1)",2:"Flat (2)",3:"Downsloping (3)"},"ca":{0:"0",1:"1",2:"2",3:"3"},"thal":{3:"Normal (3)",6:"Fixed defect (6)",7:"Reversible defect (7)"}}
app = Flask(__name__)
app.config["SECRET_KEY"] = "academic-heart-case-study-only"


def ensure_artifacts():
    required = [MODEL_DIR / "best_pipeline.joblib", MODEL_DIR / "all_models.joblib", MODEL_DIR / "metrics.json", ROOT / "case_study.md"]
    if not all(path.exists() for path in required):
        from train_model import main
        main()


def load_assets():
    ensure_artifacts()
    return (
        joblib.load(MODEL_DIR / "best_pipeline.joblib"),
        json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8")),
        json.loads((MODEL_DIR / "feature_metadata.json").read_text(encoding="utf-8")),
        joblib.load(MODEL_DIR / "all_models.joblib"),
    )


def display_value(feature: str, value: float) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature].get(int(value), str(int(value)))
    units = {"age": " years", "trestbps": " mm Hg", "chol": " mg/dL", "thalach": " bpm"}
    return f"{value:g}{units.get(feature, '')}"


def build_explanation(contributions: list[dict]) -> str:
    upward = [item for item in contributions if item["value"] > 0][:2]
    downward = [item for item in contributions if item["value"] < 0][:1]
    up_text = " and ".join(f"{item['feature']} ({item['input_value']})" for item in upward)
    down_text = " and ".join(f"{item['feature']} ({item['input_value']})" for item in downward)
    if up_text and down_text:
        return f"The strongest signals increasing the model's risk estimate are {up_text}, while {down_text} is the leading signal reducing it."
    if up_text:
        return f"The strongest signals increasing the model's risk estimate are {up_text}."
    if down_text:
        return f"The strongest signal reducing the model's risk estimate is {down_text}."
    return "The model found no strong directional signal relative to the training population for this record."


def build_pdf(report: dict) -> BytesIO:
    lines = [
        "CardioScreen - Academic Patient Screening Report", "",
        f"Classification: {'Heart disease pattern detected' if report['prediction'] else 'No heart disease pattern detected'}",
        f"Risk category: {report['risk_category']}",
        f"Probability: {report['probability']:.1%} heart disease | {report['no_disease_probability']:.1%} no disease", "",
        "Patient inputs:",
    ]
    lines.extend(f"  {label}: {value}" for label, value in report["inputs"])
    lines.extend(["", "All model predictions:"])
    lines.extend(f"  {model['name']}: {model['probability']:.1%} heart disease ({model['label']})" for model in report["model_predictions"])
    lines.extend(["", f"Selected model: {report['best_model']}", f"Held-out performance: {report['accuracy']:.1%} accuracy, {report['auc_roc']:.3f} AUC-ROC", "", "Coefficient-based directional signals:"])
    lines.extend(f"  {item['feature']} ({item['input_value']}): {item['direction']} risk" for item in report["contributions"])
    lines.extend(["", *textwrap.wrap(report["explanation"], width=92), "", "Academic use only. This is not a medical diagnosis, triage, or treatment tool."])
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69)); figure.patch.set_facecolor("white"); figure.text(.08, .95, "CardioScreen", fontsize=21, weight="bold", color="#164b65"); figure.text(.08, .925, "Heart disease prediction case study", fontsize=10, color="#61717d")
        y = .88
        for line in lines:
            figure.text(.08, y, line, fontsize=9.2, color="#173e54" if line.endswith(":") else "#293f4c")
            y -= .022
            if y < .07:
                pdf.savefig(figure, bbox_inches="tight"); plt.close(figure); figure = plt.figure(figsize=(8.27, 11.69)); y = .94
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
    buffer.seek(0)
    return buffer


@app.get("/")
def report():
    _, metrics, metadata, _ = load_assets()
    writeup = markdown.markdown((ROOT / "case_study.md").read_text(encoding="utf-8"), extensions=["fenced_code", "tables"])
    return render_template("report.html", metrics=list(metrics["models"].values()), best_model=metrics["best_model"], sample_count=metadata["sample_count"], writeup=writeup)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    pipeline, metrics, metadata, all_models = load_assets()
    values, result = metadata["defaults"].copy(), None
    if request.method == "POST":
        try:
            for feature in metadata["features"]:
                value = float(request.form[feature]); minimum, maximum = metadata["ranges"][feature]
                if not minimum <= value <= maximum:
                    raise ValueError(f"{FEATURE_LABELS[feature]} must be between {minimum:g} and {maximum:g} for this dataset.")
                values[feature] = value
            patient = pd.DataFrame([values], columns=metadata["features"])
            probability = float(pipeline.predict_proba(patient)[0, 1])
            prediction = int(pipeline.predict(patient)[0])
            model_metrics = metrics["models"][metrics["best_model"]]
            standardized = pipeline.named_steps["scaler"].transform(patient)[0]
            contributions = [{"feature": FEATURE_LABELS[feature], "value": float(contribution), "direction": "increases" if contribution >= 0 else "reduces", "input_value": display_value(feature, values[feature])} for feature, contribution in zip(metadata["features"], pipeline.named_steps["model"].coef_[0] * standardized)]
            contributions.sort(key=lambda item: abs(item["value"]), reverse=True)
            model_predictions = [{"name": name, "probability": float(model.predict_proba(patient)[0, 1]), "label": "Heart disease" if int(model.predict(patient)[0]) else "No disease"} for name, model in all_models.items()]
            risk_category = "Low risk" if probability < .30 else "Moderate risk" if probability <= .60 else "High risk"
            result = {"prediction": prediction, "probability": probability, "no_disease_probability": 1 - probability, "risk_category": risk_category, "best_model": metrics["best_model"], "accuracy": model_metrics["accuracy"], "auc_roc": model_metrics["auc_roc"], "contributions": contributions, "explanation": build_explanation(contributions), "model_predictions": model_predictions, "all_agree": len({model["label"] for model in model_predictions}) == 1, "summary": [("Age", display_value("age", values["age"])), ("Cholesterol", display_value("chol", values["chol"])), ("Max heart rate", display_value("thalach", values["thalach"])), ("Chest pain", display_value("cp", values["cp"]))], "inputs": [(FEATURE_LABELS[feature], display_value(feature, values[feature])) for feature in metadata["features"]]}
            session["last_report"] = result
        except (KeyError, ValueError) as error:
            flash(str(error), "error")
    fields = [{"name": x, "label": FEATURE_LABELS[x], "value": values[x], "range": metadata["ranges"][x], "choices": metadata["categories"].get(x), "choice_labels": CATEGORY_LABELS.get(x, {})} for x in metadata["features"]]
    return render_template("predict.html", fields=fields, result=result)


@app.get("/download-report")
def download_report():
    report = session.get("last_report")
    if not report:
        flash("Run a patient screening before downloading its report.", "error")
        return redirect(url_for("predict"))
    return send_file(build_pdf(report), as_attachment=True, download_name="cardioscreen_patient_report.pdf", mimetype="application/pdf")


@app.get("/case-study")
def case_study_file():
    return redirect(url_for("report") + "#case-study")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
