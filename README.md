# OsteoML

Journal-facing research website and model API for the osteoporosis risk stratification model.

## Contents

- `web/`: website, calculator UI, Python model API, and model artifacts.
- `web/model/model.pkl`: saved stacking model, tracked with Git LFS.
- `web/model/scaler_source.csv`: source matrix used to reconstruct the scaler.
- `web/model/shap_background.csv`: SHAP background source.
- `Dockerfile`: Hugging Face Spaces Docker deployment.
- `requirements.txt`: Python runtime dependencies.

## Local Run

```bash
pip install -r requirements.txt
python web/model_service.py
```

Open `http://127.0.0.1:8038/`.

## Free Deployment

Use Hugging Face Spaces with the Docker SDK. The Space builds the Dockerfile and exposes the app on port `7860`.

The same service hosts the website and the `/api/predict` model endpoint.

## Research Boundary

This website is for research communication and manuscript review. It is not a diagnostic medical device and does not replace DXA testing or clinician assessment.
