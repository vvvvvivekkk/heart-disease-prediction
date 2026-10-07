# Dataset

`heart.csv` is the **Cleveland Heart Disease** dataset — the most widely used
subset of the UCI "Heart Disease" database (Detrano et al.).

- **Rows:** 303 patient records (one exact-duplicate row is dropped during
  preprocessing, leaving 302).
- **Target:** `target` — 1 = heart disease present, 0 = absent.
- **Class balance:** 165 positive / 138 negative.

## Columns

| Column | Description |
|--------|-------------|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1/0) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1/0) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment (0–2) |
| ca | Number of major vessels coloured by fluoroscopy (0–4) |
| thal | Thalassemia (0–3) |
| target | Diagnosis of heart disease (1/0) |

## Source & licence

Original data: **UCI Machine Learning Repository — Heart Disease Data Set**
(<https://archive.ics.uci.edu/dataset/45/heart+disease>). The database was
donated for research and is distributed for academic use. This project uses it
for a university course assignment only (non-commercial, educational).

> Please cite the original donors (Hungarian Institute of Cardiology; University
> Hospital, Zurich; University Hospital, Basel; and V.A. Medical Center, Long
> Beach / Cleveland Clinic Foundation — R. Detrano, M.D., Ph.D.) if you reuse
> the data.
