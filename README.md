# OsteoML

Journal-facing research website and model API for the osteoporosis risk stratification model.

## Contents

- `web/`: website, calculator UI, Python model API, and model artifacts.
- `web/model_artifacts/model.pkl`: saved stacking model, tracked with Git LFS.
- `web/model_artifacts/scaler_source.csv`: source matrix used to reconstruct the scaler.
- `web/model_artifacts/shap_background.csv`: SHAP background source.
- `render.yaml`: Render free web-service blueprint.
- `requirements.txt`: Python runtime dependencies.

## Local Run

```bash
pip install -r requirements.txt
python web/model_service.py
```

Open `http://127.0.0.1:8038/`.

## Free Deployment

Use Render's free web service tier and connect this GitHub repository. Render will read `render.yaml`, install dependencies, and run:

```bash
python web/model_service.py
```

The same service hosts the website and the `/api/predict` model endpoint.

## Research Boundary

This website is for research communication and manuscript review. It is not a diagnostic medical device and does not replace DXA testing or clinician assessment.
