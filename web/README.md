# OsteoML Predictor Website

This folder contains a journal-facing research website for the osteoporosis machine-learning prediction model.

## Files

- `PRD.md`: product requirements document in Chinese.
- `DEVELOPMENT_SPEC.md`: development and deployment specification.
- `index.html`: website entry page.
- `calculator.html`: standalone calculator page.
- `styles.css`: visual design and responsive layout.
- `app.js`: calculator interaction and demonstration risk score.
- `model_service.py`: local API that loads the saved stacking model.
- `assets/`: local visual assets used by the website.
- `output/playwright/`: verification screenshots.

## Local Preview

From the project root:

```bash
python3 -m http.server 8037 --directory web
```

Then open:

```text
http://127.0.0.1:8037/
```

Standalone calculator:

```text
http://127.0.0.1:8037/calculator.html
```

## Local Model API

Run the model API in a second terminal:

```bash
python3 web/model_service.py
```

Health check:

```bash
curl http://127.0.0.1:8038/api/health
```

The API loads:

```text
web/model_artifacts/model.pkl
```

The browser calculator automatically uses this API when it is available. If the API is not running, it falls back to the transparent frontend demonstration score.

When `include_shap: true` is sent to `/api/predict`, the API returns a local SHAP explanation for the predicted class. The standalone calculator requests this after the user clicks **Calculate with model**. A single SHAP calculation can take about 10-15 seconds for the saved stacking model.

## Deployment

This version can be deployed as a static website plus a small model API. For a manuscript production website, the preprocessing pipeline should be frozen as an explicit serialized artifact rather than reconstructed at runtime.

## Research Boundary

When the model API is running, the standalone calculator uses the saved stacking model. The preprocessing step is reconstructed from the bundled scaler source matrix for website review. It is not a diagnostic medical device and does not replace DXA testing or clinician judgement.
