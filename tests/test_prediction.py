import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from features import FEATURE_NAMES
from predictor import predict_one, predict_batch, load_artifacts, assess_applicability

@pytest.fixture
def model_and_metadata():
    x = pd.DataFrame(np.zeros((4, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    model = DummyClassifier(strategy="constant", constant=1).fit(x, [0, 1, 0, 1])
    domain={"mean":[0.0]*len(FEATURE_NAMES),"scale":[1.0]*len(FEATURE_NAMES),"minimum":[-1.0]*len(FEATURE_NAMES),"maximum":[1.0]*len(FEATURE_NAMES),"centroid_distance_p95":3.0}
    return model, {"target_labels":{"0":"BBB-non-permeable","1":"BBB-permeable"},"feature_names":FEATURE_NAMES,"applicability_domain":domain}

def test_single_prediction_and_mapping(model_and_metadata):
    model, metadata = model_and_metadata
    result = predict_one("CCO", model, metadata)
    assert result["label"] == "BBB-permeable"
    assert result["probabilities"]["BBB-permeable"] == 1
    assert "within_heuristic_domain" in result["applicability"]

def test_applicability_warning_for_out_of_range_molecule(model_and_metadata):
    model, metadata = model_and_metadata
    frame=pd.DataFrame([[0.0]*len(FEATURE_NAMES)],columns=FEATURE_NAMES)
    frame.loc[0,FEATURE_NAMES[0]]=999
    result=assess_applicability(frame,metadata)
    assert result["within_heuristic_domain"] is False
    assert FEATURE_NAMES[0] in result["outside_range_features"]

def test_batch_keeps_invalid_rows(model_and_metadata):
    model, metadata = model_and_metadata
    result = predict_batch(pd.DataFrame({"smiles":["CCO","not a smiles"]}), model, metadata)
    assert len(result) == 2
    assert result.iloc[0].prediction_status == "ok"
    assert result.iloc[1].prediction_status == "error"
    assert pd.notna(result.iloc[1].error)

def test_batch_requires_smiles(model_and_metadata):
    model, metadata = model_and_metadata
    with pytest.raises(ValueError): predict_batch(pd.DataFrame({"other": [1]}), model, metadata)

def test_missing_artifacts():
    with pytest.raises(FileNotFoundError): load_artifacts("work\\missing-model-artifacts")

def test_saved_model_loads_and_predicts_when_trained():
    from pathlib import Path
    if not (Path("models/random_forest.joblib").exists() and Path("models/metadata.json").exists()):
        pytest.skip("Train models before testing persisted model loading.")
    model, metadata=load_artifacts()
    result=predict_one("CCO",model,metadata)
    assert result["label"] in metadata["target_labels"].values()
    assert 0.0 <= sum(result["probabilities"].values()) <= 1.000001
    assert result["applicability"]["warning"]
