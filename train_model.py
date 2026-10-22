"""Train BBBP descriptor models with a scaffold-isolated final test set."""
from __future__ import annotations
import json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, rdkit
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, RocCurveDisplay)
from rdkit import Chem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_loader import load_dataset
from features import featurize_smiles, FEATURE_NAMES, DESCRIPTOR_VERSION

ROOT = Path(__file__).parent
SEED = 42
TARGET_LABELS = {0: "BBB-non-permeable", 1: "BBB-permeable"}


def scaffold_key(smiles: str) -> str:
    """Murcko scaffold key. All acyclic compounds share the empty scaffold group."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in scaffold split: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def target_label_mapping(values) -> dict[str, str]:
    """MoleculeNet BBBP convention: p_np=1 permeable, p_np=0 non-permeable."""
    present = {int(value) for value in values}
    if not present or not present.issubset(TARGET_LABELS):
        raise ValueError(f"BBBP p_np must contain only binary values 0/1; found {sorted(present)}")
    return {str(key): TARGET_LABELS[key] for key in sorted(present)}


def calculate_metrics(y_true, y_pred, y_score=None) -> dict:
    """Metrics with class 1 as positive and an explicit undefined-AUC value."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None and np.unique(y_true).size == 2 else None,
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "test_single_class_limitation": np.unique(y_true).size < 2,
    }

def scaffold_split(frame: pd.DataFrame, test_fraction: float = .2, seed: int = SEED):
    """Randomly hold out scaffold groups; never stratify by labels or split a group."""
    scaffolds: dict[str, list[int]] = {}
    for i, smiles in enumerate(frame.smiles):
        scaffold = scaffold_key(smiles)
        scaffolds.setdefault(scaffold, []).append(i)
    groups = list(scaffolds.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    test_group_count = max(1, round(len(groups) * test_fraction))
    test_groups, train_groups = groups[:test_group_count], groups[test_group_count:]
    test = sorted(i for group in test_groups for i in group)
    train = sorted(i for group in train_groups for i in group)
    if not train or not test:
        raise ValueError("Scaffold split failed: dataset has too few distinct scaffolds.")
    # Label distributions are observed and recorded, never used to assign groups.
    overall_counts = frame.p_np.value_counts().to_dict()
    train_scaffolds = {scaffold_key(frame.iloc[i].smiles) for i in train}
    test_scaffolds = {scaffold_key(frame.iloc[i].smiles) for i in test}
    overlap = train_scaffolds & test_scaffolds
    if overlap:
        raise AssertionError(f"Scaffold leakage detected: {len(overlap)} scaffold(s) occur in both splits.")
    return train, test, {"scaffold_count": len(groups), "train_scaffold_count":len(train_scaffolds), "test_scaffold_count":len(test_scaffolds), "scaffold_overlap_count":len(overlap), "test_scaffold_fraction":len(test_scaffolds)/len(groups), "test_fraction_actual": len(test)/len(frame),
                                        "train_class_counts": {str(k): int(v) for k,v in frame.iloc[train].p_np.value_counts().items()},
                                        "test_class_counts": {str(k): int(v) for k,v in frame.iloc[test].p_np.value_counts().items()},
                                            "overall_class_proportions": {str(k): float(v/len(frame)) for k,v in overall_counts.items()}}

def main():
    data, data_stats = load_dataset()
    descriptor_rows, kept = [], []
    for idx, row in data.iterrows():
        try:
            _, _, x = featurize_smiles(row.smiles)
            descriptor_rows.append(x.iloc[0].to_numpy(dtype=float)); kept.append(idx)
        except Exception:
            continue
    data_stats["descriptor_failures"] = len(data)-len(kept)
    data = data.loc[kept].reset_index(drop=True)
    X = pd.DataFrame(descriptor_rows, columns=FEATURE_NAMES)
    y = data.p_np.astype(int)
    train_idx, test_idx, split_stats = scaffold_split(data)
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
    }
    # CV and model comparisons use training data only. The RF is primary by project specification.
    cv_results = {}
    cv_splits = min(5, int(y_train.value_counts().min()))
    if cv_splits >= 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
        for name, estimator in models.items():
            scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring={"accuracy":"accuracy", "f1":"f1", "roc_auc":"roc_auc"}, error_score=np.nan)
            cv_results[name] = {metric: (None if not np.isfinite(scores[f"test_{metric}"].mean()) else float(np.nanmean(scores[f"test_{metric}"]))) for metric in ("accuracy", "f1", "roc_auc")}
    else:
        cv_results = {name: {"note": "CV unavailable: training data has fewer than two examples in one class."} for name in models}
    fitted = {}
    for name, estimator in models.items():
        fitted[name] = estimator.fit(X_train, y_train)
    primary = fitted["Random Forest"]
    yhat = primary.predict(X_test)
    test_metrics, test_predictions = {}, []
    for name, estimator in fitted.items():
        predicted = estimator.predict(X_test)
        scores = None
        if hasattr(estimator, "predict_proba") and 1 in estimator.classes_:
            scores = estimator.predict_proba(X_test)[:, list(estimator.classes_).index(1)]
        test_metrics[name] = calculate_metrics(y_test, predicted, scores)
        if name == "Random Forest":
            yhat = predicted
            y_score = scores
        for pos, row in enumerate(test_idx):
            record = {"smiles":data.iloc[row].smiles,"actual":int(y_test.iloc[pos]),"model":name,"predicted":int(predicted[pos])}
            if scores is not None: record["probability_permeable"] = float(scores[pos])
            test_predictions.append(record)
    metrics = test_metrics["Random Forest"]
    (ROOT/"models").mkdir(exist_ok=True); (ROOT/"reports").mkdir(exist_ok=True)
    joblib.dump(primary, ROOT/"models"/"random_forest.joblib")
    labels = target_label_mapping(y.unique())
    # Empirical descriptor-space reference for a transparent, heuristic applicability warning.
    domain_mean = X_train.mean().to_numpy(dtype=float)
    domain_scale = X_train.std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    train_z = (X_train.to_numpy(dtype=float)-domain_mean)/domain_scale
    domain = {"mean":domain_mean.tolist(),"scale":domain_scale.tolist(),
              "minimum":X_train.min().to_numpy(dtype=float).tolist(),"maximum":X_train.max().to_numpy(dtype=float).tolist(),
              "centroid_distance_p95":float(np.quantile(np.linalg.norm(train_z,axis=1),.95)),
              "interpretation":"Heuristic descriptor-space warning; not a validated applicability-domain test."}
    metadata = {"feature_names": FEATURE_NAMES, "descriptor_version": DESCRIPTOR_VERSION, "target_labels": labels,
                "target_column":"p_np", "target_convention":"MoleculeNet BBBP p_np: 1 = permeable, 0 = non-permeable",
                "dataset_source": data_stats["source"], "dataset_retrieved_at": data_stats["retrieved_at"],
                "training_config":{"seed":SEED,"test_scaffold_fraction_target":.2,"split":"Seeded random 20% of Bemis-Murcko scaffold groups; all acyclic molecules share the empty scaffold group","cv":"StratifiedKFold on training set only",
                                   "random_forest":{"n_estimators":400,"min_samples_leaf":2,"class_weight":"balanced"},
                                   "logistic_regression":{"max_iter":2000,"class_weight":"balanced","scaled":True},
                                   "svm":{"kernel":"rbf","C":1.0,"class_weight":"balanced","scaled":True}},
                "split_stats":split_stats,"dataset_stats":data_stats,"test_metrics":metrics,"heldout_model_metrics":test_metrics,"cv_results":cv_results,
                "applicability_domain":domain,
                "library_versions":{"python":platform.python_version(),"scikit_learn":sklearn.__version__,"rdkit":rdkit.__version__,"pandas":pd.__version__,"numpy":np.__version__,"joblib":joblib.__version__}}
    (ROOT/"models"/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    cv_comparison = pd.DataFrame([{"model":n, **cv_results[n]} for n in models])
    cv_comparison.to_csv(ROOT/"reports"/"model_comparison_cv.csv",index=False)
    comparison = pd.DataFrame([{"model":n, **{k:v for k,v in test_metrics[n].items() if k in ("accuracy","precision","recall","f1","roc_auc")}} for n in models])
    comparison.to_csv(ROOT/"reports"/"model_comparison_test.csv",index=False)
    pd.DataFrame(test_predictions).to_csv(ROOT/"reports"/"heldout_test_predictions.csv",index=False)
    fig, ax = plt.subplots(figsize=(7,4)); comparison.set_index("model")[["accuracy","precision","recall","f1","roc_auc"]].plot(kind="bar",ax=ax); ax.set_title("Same held-out scaffold test set"); ax.set_ylim(0,1); fig.tight_layout(); fig.savefig(ROOT/"reports"/"model_comparison_test.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); cv_comparison.set_index("model")[["accuracy","f1","roc_auc"]].plot(kind="bar",ax=ax); ax.set_title("Training-only cross-validation"); ax.set_ylim(0,1); fig.tight_layout(); fig.savefig(ROOT/"reports"/"model_comparison_cv.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5,4)); sns.heatmap(metrics["confusion_matrix"],annot=True,fmt="d",cmap="Blues",xticklabels=[labels["0"],labels["1"]],yticklabels=[labels["0"],labels["1"]],ax=ax); ax.set(xlabel="Predicted",ylabel="Actual",title="Random Forest held-out scaffold test"); fig.tight_layout(); fig.savefig(ROOT/"reports"/"confusion_matrix.png",dpi=160); plt.close(fig)
    if metrics["roc_auc"] is not None:
        fig,ax=plt.subplots(figsize=(5,4)); RocCurveDisplay.from_estimator(primary,X_test,y_test,ax=ax); fig.tight_layout(); fig.savefig(ROOT/"reports"/"roc_curve.png",dpi=160); plt.close(fig)
    importances = pd.Series(primary.feature_importances_,index=FEATURE_NAMES).sort_values()
    fig,ax=plt.subplots(figsize=(7,5)); importances.plot.barh(ax=ax,color="#178b8c"); ax.set_title("Random Forest descriptor importance"); fig.tight_layout(); fig.savefig(ROOT/"reports"/"feature_importance.png",dpi=160); plt.close(fig)
    print(json.dumps({"data":data_stats,"split":split_stats,"cv_results":cv_results,"test_metrics":metrics},indent=2))

if __name__ == "__main__": main()
