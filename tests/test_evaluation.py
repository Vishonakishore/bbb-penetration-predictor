import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from train_model import calculate_metrics, scaffold_split, target_label_mapping
import pandas as pd
import json
from pathlib import Path
import pytest

def test_metrics_match_sklearn_directly():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])
    result = calculate_metrics(y_true, y_pred, scores)
    assert result["accuracy"] == accuracy_score(y_true, y_pred)
    assert result["precision"] == precision_score(y_true, y_pred, zero_division=0)
    assert result["recall"] == recall_score(y_true, y_pred, zero_division=0)
    assert result["f1"] == f1_score(y_true, y_pred, zero_division=0)
    assert result["roc_auc"] == roc_auc_score(y_true, scores)
    assert result["confusion_matrix"] == confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]

def test_auc_undefined_for_one_class():
    result = calculate_metrics([1, 1], [1, 0], [0.8, 0.2])
    assert result["roc_auc"] is None
    assert result["test_single_class_limitation"] is True

def test_moleculenet_label_mapping():
    assert target_label_mapping([0, 1]) == {"0":"BBB-non-permeable", "1":"BBB-permeable"}

def test_label_mapping_rejects_non_binary():
    import pytest
    with pytest.raises(ValueError): target_label_mapping([0, 2])

def test_scaffold_split_has_no_overlap_and_reports_classes():
    frame = pd.DataFrame({
        "smiles":["CCO", "CCN", "CCC", "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCOCC1", "c1ccc2ccccc2c1", "c1ccoc1", "C1CCNCC1"],
        "p_np":[0,1,0,1,0,1,0,1,0,1]
    })
    train, test, stats = scaffold_split(frame, test_fraction=.25, seed=3)
    assert set(train).isdisjoint(test)
    assert stats["scaffold_overlap_count"] == 0
    assert stats["train_class_counts"] == {str(int(k)):int(v) for k,v in frame.iloc[train].p_np.value_counts().items()}
    assert stats["test_class_counts"] == {str(int(k)):int(v) for k,v in frame.iloc[test].p_np.value_counts().items()}

def test_saved_metrics_match_saved_predictions_when_trained():
    root=Path(__file__).parents[1]
    predictions_path=root/"reports"/"heldout_test_predictions.csv"
    metadata_path=root/"models"/"metadata.json"
    if not predictions_path.exists() or not metadata_path.exists():
        pytest.skip("Train models to generate held-out prediction artifacts.")
    predictions=pd.read_csv(predictions_path)
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_by_model=[]
    for name, rows in predictions.groupby("model"):
        actual=rows["actual"].to_numpy(dtype=int)
        actual_by_model.append(actual)
        score=rows["probability_permeable"].to_numpy(dtype=float)
        direct=calculate_metrics(actual,rows["predicted"].to_numpy(dtype=int),score)
        saved=metadata["heldout_model_metrics"][name]
        for metric in ("accuracy","precision","recall","f1","roc_auc"):
            assert np.isclose(direct[metric],saved[metric])
        assert direct["confusion_matrix"]==saved["confusion_matrix"]
    assert len(actual_by_model)==3
    assert all(np.array_equal(actual_by_model[0],other) for other in actual_by_model[1:])
