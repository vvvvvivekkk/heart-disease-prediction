"""Interactive web interface for the Heart Disease Prediction model.

Run it:
    python app.py

Then open the local URL it prints (http://127.0.0.1:7860) in your browser,
enter a patient's details, and click "Predict".
"""

from __future__ import annotations

import joblib
import pandas as pd
import torch

from src.data import FEATURE_COLUMNS
from src.evaluate import load_model

import gradio as gr

# ---- load the trained model once at startup -------------------------------
DEVICE = torch.device("cpu")
PREPROCESSOR = joblib.load("results/preprocessor.joblib")
MODEL = load_model("results", DEVICE)


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal):
    """Return a label dict (for the gauge) and a plain-English verdict."""
    row = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])[FEATURE_COLUMNS]
    X = PREPROCESSOR.transform(row).astype("float32")
    with torch.no_grad():
        prob = float(torch.sigmoid(MODEL(torch.from_numpy(X))).item())

    labels = {"Heart disease present": prob, "No disease": 1.0 - prob}
    if prob >= 0.5:
        verdict = (f"### 🔴 Likely heart disease\n"
                   f"Estimated probability: **{prob*100:.1f}%**\n\n"
                   f"*A model estimate — not a medical diagnosis. Please consult a doctor.*")
    else:
        verdict = (f"### 🟢 Heart disease unlikely\n"
                   f"Estimated probability: **{prob*100:.1f}%**\n\n"
                   f"*A model estimate — not a medical diagnosis. Please consult a doctor.*")
    return labels, verdict


# dropdown choices as (label shown, value sent) -----------------------------
CP = [("Typical angina (0)", 0), ("Atypical angina (1)", 1),
      ("Non-anginal pain (2)", 2), ("Asymptomatic (3)", 3)]
RESTECG = [("Normal (0)", 0), ("ST-T abnormality (1)", 1), ("LV hypertrophy (2)", 2)]
SLOPE = [("Upsloping (0)", 0), ("Flat (1)", 1), ("Downsloping (2)", 2)]
CA = [(str(i), i) for i in range(5)]
THAL = [("0", 0), ("Normal (1)", 1), ("Fixed defect (2)", 2), ("Reversible defect (3)", 3)]
YESNO = [("No", 0), ("Yes", 1)]
SEX = [("Female", 0), ("Male", 1)]

with gr.Blocks(title="Heart Disease Prediction") as demo:
    gr.Markdown(
        "# ❤️ Heart Disease Prediction\n"
        "Enter a patient's details and click **Predict**. The model (an Artificial "
        "Neural Network trained on the UCI Cleveland dataset) estimates the "
        "probability of heart disease.\n\n"
        "*Educational project — not a medical device.*"
    )
    with gr.Row():
        with gr.Column():
            age = gr.Slider(24, 77, value=57, step=1, label="Age")
            sex = gr.Radio(choices=SEX, value=1, label="Sex")
            cp = gr.Dropdown(choices=CP, value=0, label="Chest pain type")
            trestbps = gr.Slider(90, 200, value=130, step=1, label="Resting blood pressure (mm Hg)")
            chol = gr.Slider(120, 570, value=236, step=1, label="Cholesterol (mg/dl)")
            fbs = gr.Radio(choices=YESNO, value=0, label="Fasting blood sugar > 120 mg/dl?")
            restecg = gr.Dropdown(choices=RESTECG, value=0, label="Resting ECG result")
        with gr.Column():
            thalach = gr.Slider(70, 205, value=174, step=1, label="Max heart rate achieved")
            exang = gr.Radio(choices=YESNO, value=0, label="Exercise-induced angina?")
            oldpeak = gr.Slider(0.0, 6.2, value=0.0, step=0.1, label="ST depression (oldpeak)")
            slope = gr.Dropdown(choices=SLOPE, value=1, label="Slope of peak exercise ST segment")
            ca = gr.Dropdown(choices=CA, value=1, label="Major vessels coloured (0–4)")
            thal = gr.Dropdown(choices=THAL, value=2, label="Thalassemia")

    btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        out_label = gr.Label(num_top_classes=2, label="Prediction")
        out_text = gr.Markdown()

    inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
              exang, oldpeak, slope, ca, thal]
    btn.click(predict, inputs=inputs, outputs=[out_label, out_text])

    gr.Markdown("#### Example patients (click one to load, then press Predict)")
    gr.Examples(
        examples=[
            [44, 0, 2, 108, 141, 0, 1, 175, 0, 0.6, 1, 0, 2],  # real case: has disease
            [51, 1, 0, 140, 298, 0, 1, 122, 1, 4.2, 1, 3, 3],  # real case: no disease
        ],
        inputs=inputs,
        label="Real patients from the dataset",
    )

if __name__ == "__main__":
    demo.launch()

