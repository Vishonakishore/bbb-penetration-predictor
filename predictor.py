"""Trusted local model artifact loading and prediction helpers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from features import featurize_smiles, FEATURE_NAMES

MODEL_DIR = Path(__file__).parent / "models"


def load_artifacts(model_dir: str | Path = MODEL_DIR):
    path = Path(model_dir)
    model_file, metadata_file = path / "random_forest.joblib", path / "metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Model artifacts are missing. Train them first with: python train_model.py")
    model = joblib.load(model_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Saved model feature configuration does not match this application.")
    if metadata.get("target_labels", {}).get("1") != "BBB-permeable" or metadata.get("target_labels", {}).get("0") != "BBB-non-permeable":
        raise ValueError("Saved BBBP target mapping is missing or inconsistent with the verified p_np convention.")
    return model, metadata


def assess_applicability(features: pd.DataFrame, metadata: dict) -> dict:
    """Return an explicitly heuristic warning from training descriptor ranges and distance."""
    domain = metadata.get("applicability_domain")
    if not domain:
        return {"within_heuristic_domain": None, "warning": "Applicability reference unavailable; treat this prediction cautiously."}
    values = features.loc[:, FEATURE_NAMES].to_numpy(dtype=float)[0]
    low, high = domain["minimum"], domain["maximum"]
    outside = [name for i, name in enumerate(FEATURE_NAMES) if values[i] < low[i] or values[i] > high[i]]
    mean, scale = np.asarray(domain["mean"]), np.asarray(domain["scale"])
    distance = float(np.linalg.norm((values - mean) / scale))
    far = distance > float(domain["centroid_distance_p95"])
    reasons = []
    if outside:
        reasons.append("outside observed training ranges: " + ", ".join(outside))
    if far:
        reasons.append("descriptor-space distance exceeds the training-set 95th percentile")
    return {"within_heuristic_domain": not reasons, "outside_range_features": outside,
            "standardized_centroid_distance": distance,
            "warning": ("Potentially outside the model's applicability domain (heuristic): " + "; ".join(reasons)) if reasons else
                       "No heuristic domain warning. This does not establish that the molecule is represented by the training data."}


def predict_one(smiles: str, model, metadata: dict) -> dict:
    mol, descriptors, frame = featurize_smiles(smiles)
    prediction = int(model.predict(frame)[0])
    labels = metadata["target_labels"]
    result = {"prediction": prediction, "label": labels[str(prediction)], "descriptors": descriptors, "mol": mol,
              "applicability": assess_applicability(frame, metadata)}
    if hasattr(model, "predict_proba"):
        result["probabilities"] = {labels[str(int(cls))]: float(prob) for cls, prob in zip(model.classes_, model.predict_proba(frame)[0])}
    return result


def predict_batch(frame: pd.DataFrame, model, metadata: dict) -> pd.DataFrame:
    if "smiles" not in frame.columns:
        raise ValueError("Uploaded CSV must contain a column named 'smiles'.")
    out = frame.copy().reset_index(drop=True)
    labels = metadata["target_labels"]
    out["prediction_status"] = "error"
    out["predicted_class"] = pd.NA
    out["probability_0"] = pd.NA
    out["probability_1"] = pd.NA
    out["applicability_warning"] = pd.NA
    out["error"] = pd.NA
    for idx, smiles in out["smiles"].items():
        try:
            _, _, features = featurize_smiles(smiles)
            domain = assess_applicability(features, metadata)
            cls = int(model.predict(features)[0])
            out.at[idx, "prediction_status"] = "ok"
            out.at[idx, "predicted_class"] = labels[str(cls)]
            if domain["within_heuristic_domain"] is not True:
                out.at[idx, "applicability_warning"] = domain["warning"]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                for c, prob in zip(model.classes_, probs):
                    out.at[idx, f"probability_{int(c)}"] = float(prob)
        except Exception as exc:
            out.at[idx, "error"] = str(exc)
    return out
