#!/usr/bin/env python3
"""Local model API for the OsteoML website.

This service loads the saved StackingClassifier and reconstructs the feature
preprocessing needed for browser inputs. It is intended for local manuscript
website review; production hosting should freeze and version the preprocessing
pipeline explicitly.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import warnings
import base64
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


WEB_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = WEB_ROOT / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
RAW_DATA_PATH = ARTIFACT_DIR / "scaler_source.csv"
SHAP_BACKGROUND_PATH = ARTIFACT_DIR / "shap_background.csv"

FEATURE_COLS = [
    "BMI",
    "Weight",
    "BRI",
    "Waist",
    "Age",
    "Height",
    "ALP",
    "Race",
    "Creatinine",
    "eGFR",
    "Physical_Activity",
    "Energy",
    "Income",
    "DBP",
    "Calcium",
    "Education",
    "Gender",
    "SBP",
]
CONTINUOUS_COLS = [
    "Age",
    "Height",
    "BMI",
    "Waist",
    "SBP",
    "DBP",
    "Creatinine",
    "ALP",
    "Energy",
    "Calcium",
]
DERIVED_COLS = ["WHtR", "Weight", "eGFR", "BRI", "Body_Fat", "BMR"]
SCALE_COLS = CONTINUOUS_COLS + DERIVED_COLS
CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]
EXPLANATION_BACKGROUND_SIZE = 20


def load_scaler() -> StandardScaler:
    raw = pd.read_csv(RAW_DATA_PATH)
    x = raw[[c for c in raw.columns if c not in ["T", "T_Score", "BMD"]]]
    y = raw["T"]
    x_train, _, _, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(x_train[SCALE_COLS])
    return scaler


def load_explanation_background() -> pd.DataFrame:
    train = pd.read_csv(SHAP_BACKGROUND_PATH)[FEATURE_COLS]
    return train.sample(EXPLANATION_BACKGROUND_SIZE, random_state=42)


MODEL = joblib.load(MODEL_PATH)
SCALER = load_scaler()
BACKGROUND = load_explanation_background()


def model_predict_for_shap(x: Any) -> np.ndarray:
    return MODEL.predict_proba(pd.DataFrame(x, columns=FEATURE_COLS))


EXPLAINER = shap.Explainer(model_predict_for_shap, BACKGROUND)


def as_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def as_float_any(payload: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        if key in payload and payload.get(key) not in ("", None):
            return float(payload[key])
    return default


def map_gender(value: Any) -> float:
    if isinstance(value, str):
        return 2.0 if value.lower() == "female" else 1.0
    return float(value)


def map_race(value: Any) -> float:
    mapping = {
        "mexican_american": 1.0,
        "other_hispanic": 2.0,
        "non_hispanic_white": 3.0,
        "non_hispanic_black": 4.0,
        "asian": 5.0,
        "other": 5.0,
    }
    if isinstance(value, str):
        return mapping.get(value, 5.0)
    return float(value)


def map_activity(value: Any) -> float:
    mapping = {"high": 1.0, "moderate": 2.0, "low": 4.0}
    if isinstance(value, str):
        return mapping.get(value, 2.0)
    return float(value)


def egfr(creatinine_umol_l: float, age: float, gender: float) -> float:
    cr_mgdl = creatinine_umol_l / 88.4
    if gender == 2:
        kappa = 0.7
        alpha = -0.329 if cr_mgdl <= 0.7 else -1.209
        return (
            144
            * min(cr_mgdl / kappa, 1) ** alpha
            * max(cr_mgdl / kappa, 1) ** -1.209
            * 0.993**age
        )
    kappa = 0.9
    alpha = -0.411 if cr_mgdl <= 0.9 else -1.209
    return (
        141
        * min(cr_mgdl / kappa, 1) ** alpha
        * max(cr_mgdl / kappa, 1) ** -1.209
        * 0.993**age
    )


def bri(height_cm: float, waist_cm: float) -> float:
    h_m = height_cm / 100
    w_m = waist_cm / 100
    term = (w_m / (2 * math.pi)) ** 2 / (0.5 * h_m) ** 2
    if term >= 1:
        return 15.0
    return 364.2 - 365.5 * math.sqrt(1 - term)


def build_features(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    age = as_float(payload, "age", 66)
    height = as_float_any(payload, ["height", "height_cm"], 158)
    weight = as_float_any(payload, ["weight", "weight_kg"], 54)
    waist = as_float_any(payload, ["waist", "waist_cm"], 82)
    sbp = as_float(payload, "sbp", 132)
    dbp = as_float(payload, "dbp", 78)
    alp = as_float(payload, "alp", 96)
    energy = as_float(payload, "energy", 1900)
    calcium = as_float(payload, "calcium", 930)
    creatinine = as_float(payload, "creatinine", 72.5)
    gender = map_gender(payload.get("sex", payload.get("gender", "female")))
    race = map_race(payload.get("race", payload.get("race_ethnicity", "asian")))
    activity = map_activity(payload.get("activity", payload.get("physical_activity", 2)))
    education = as_float(payload, "education", 3)
    income = as_float(payload, "income", 2)

    bmi = weight / ((height / 100) ** 2)
    whtr = waist / height
    egfr_value = egfr(creatinine, age, gender)
    bri_value = bri(height, waist)
    body_fat = 1.2 * bmi + 0.23 * age - 10.8 * (1 if gender == 1 else 0) - 5.4
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == 1 else -161)

    raw = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Age": age,
                "Height": height,
                "BMI": bmi,
                "Waist": waist,
                "SBP": sbp,
                "DBP": dbp,
                "Creatinine": creatinine,
                "ALP": alp,
                "Energy": energy,
                "Calcium": calcium,
                "Physical_Activity": activity,
                "Education": education,
                "Income": income,
                "Race": race,
                "WHtR": whtr,
                "Weight": weight,
                "eGFR": egfr_value,
                "BRI": bri_value,
                "Body_Fat": body_fat,
                "BMR": bmr,
            }
        ]
    )
    processed = raw.copy()
    processed[SCALE_COLS] = SCALER.transform(processed[SCALE_COLS])
    ost = math.floor(0.2 * (weight - age))
    derived = {"BMI": bmi, "OST": ost, "eGFR": egfr_value, "BRI": bri_value}
    return processed[FEATURE_COLS], derived


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    features, derived = build_features(payload)
    proba = MODEL.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    probabilities = {name: float(proba[i]) for i, name in enumerate(CLASS_NAMES)}
    result = {
        "mode": "actual_model",
        "model_version": "stacking_20260113",
        "preprocessing_version": "reconstructed_oe_rf_v2_rf",
        "prediction": CLASS_NAMES[pred_idx],
        "probabilities": probabilities,
        "derived": derived,
        "feature_order": FEATURE_COLS,
        "note": (
            "The saved StackingClassifier is loaded. Preprocessing is reconstructed "
            "from local NHANES v2_rf data for website demonstration."
        ),
    }
    if bool(payload.get("include_shap")):
        result["explanation"] = compute_local_shap(features, pred_idx)
    return result


def compute_local_shap(features: pd.DataFrame, class_idx: int) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = EXPLAINER(
            features,
            max_evals=2 * len(FEATURE_COLS) + 1,
            silent=True,
        )
    values = explanation.values[0, :, class_idx]
    base_value = float(explanation.base_values[0, class_idx])
    output_value = float(base_value + values.sum())
    rows = []
    for feature, contribution, value in zip(
        FEATURE_COLS, values, features.iloc[0].to_numpy()
    ):
        rows.append(
            {
                "feature": feature,
                "value": float(value),
                "contribution": float(contribution),
            }
        )
    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "method": "SHAP PermutationExplainer",
        "class_name": CLASS_NAMES[class_idx],
        "base_value": base_value,
        "output_value": output_value,
        "unit": "probability",
        "top_features": rows[:10],
        "plots": make_shap_plots(explanation, features, class_idx),
    }


def shap_class_explanation(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    class_idx: int,
) -> shap.Explanation:
    return shap.Explanation(
        values=explanation.values[0, :, class_idx],
        base_values=explanation.base_values[0, class_idx],
        data=features.iloc[0].to_numpy(),
        feature_names=FEATURE_COLS,
    )


def fig_to_data_url() -> str:
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_native_waterfall(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    class_idx: int,
    title: str,
) -> str:
    class_exp = shap_class_explanation(explanation, features, class_idx)
    plt.figure(figsize=(9.5, 5.6))
    shap.plots.waterfall(class_exp, max_display=10, show=False)
    plt.title(title, fontsize=13, pad=12)
    return fig_to_data_url()


def make_native_bar(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    class_idx: int,
    title: str,
) -> str:
    class_exp = shap_class_explanation(explanation, features, class_idx)
    plt.figure(figsize=(8.8, 5.2))
    shap.plots.bar(class_exp, max_display=10, show=False)
    plt.title(title, fontsize=13, pad=12)
    return fig_to_data_url()


def make_shap_plots(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    predicted_class_idx: int,
) -> list[dict[str, str]]:
    plots = []
    for idx, class_name in enumerate(CLASS_NAMES):
        plots.append(
            {
                "title": f"{class_name} class waterfall",
                "kind": "waterfall",
                "class_name": class_name,
                "src": make_native_waterfall(
                    explanation,
                    features,
                    idx,
                    f"SHAP waterfall for {class_name}",
                ),
            }
        )
    for idx, class_name in enumerate(CLASS_NAMES):
        plots.append(
            {
                "title": f"{class_name} class feature importance",
                "kind": "bar",
                "class_name": class_name,
                "src": make_native_bar(
                    explanation,
                    features,
                    idx,
                    f"SHAP bar plot for {class_name}",
                ),
            }
        )
    return plots


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "mode": "actual_model",
                    "model_path": str(MODEL_PATH),
                    "model_version": "stacking_20260113",
                },
            )
            return
        self.serve_static_file()

    def serve_static_file(self) -> None:
        requested = self.path.split("?", 1)[0].split("#", 1)[0]
        if requested in ("", "/"):
            requested = "/index.html"
        if requested.startswith("/"):
            requested = requested[1:]
        static_root = WEB_ROOT
        target = (static_root / requested).resolve()
        if static_root not in target.parents and target != static_root:
            self.send_json(403, {"error": "Forbidden"})
            return
        if not target.is_file():
            self.send_json(404, {"error": "Not found"})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            self.send_json(200, predict(payload))
        except Exception as exc:  # pragma: no cover - local diagnostic path
            self.send_json(500, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[model-service] {self.address_string()} {fmt % args}")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8038"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"OsteoML model API running at http://{host}:{port}")
    print(f"Loaded model: {MODEL_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
