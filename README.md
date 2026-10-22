# BBB Penetration Predictor

A local Streamlit research and education application that predicts the MoleculeNet BBBP binary label from SMILES with RDKit molecular descriptors and a Random Forest classifier.

## Features

- Validates SMILES and calculates the same ordered descriptors at training and prediction time.
- Downloads and caches the public BBBP CSV, reports cleaning and class statistics, and excludes conflicting duplicate structures.
- Compares Random Forest, scaled Logistic Regression, and scaled RBF SVM using training-only cross-validation and the same held-out test set.
- Uses a reproducible random 20% scaffold-group holdout (including one shared empty scaffold group for acyclic molecules) and reports actual row/class counts and overlap.
- Predicts a single SMILES or every row in an uploaded CSV, retaining invalid rows with error details.
- Saves a local trusted model artifact and metadata; the app does not retrain on reruns.

## Algorithms and features

Random Forest is the required primary model. Logistic Regression is a linear baseline and SVM with an RBF kernel is a nonlinear comparator; the latter two scale descriptors in pipelines. Features: molecular weight, LogP, TPSA, H-bond donors and acceptors, rotatable bonds, aromatic rings, total rings, fraction sp3, heavy atoms, formal charge, molar refractivity, heteroatom count, and aliphatic rings. These properties provide interpretable molecular context, not a deterministic BBB rule.

## Dataset and citation

BBBP is retrieved from [MoleculeNet's BBBP CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv), commonly cited as Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning*, Chemical Science (2018), DOI: 10.1039/C7SC02664A. The code validates `smiles` and `p_np` rather than assuming an arbitrary CSV schema. `p_np=1` means permeable and `0` non-permeable, matching MoleculeNet BBBP convention.

Cleaning drops missing/non-binary labels and invalid SMILES, canonicalizes molecules, removes duplicate structures, and excludes all copies of structures with conflicting labels. The original row counts and exclusion statistics are saved in metadata. No training examples or labels are synthesized.

## Windows PowerShell setup

Python 3.11 or 3.12 is recommended. RDKit wheels are available through supported pip distributions; if installation fails, use a conda environment with `conda install -c conda-forge rdkit` and then install remaining requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

To run tests: `python -m pytest -q`. To retrain after changing code or data: `python train_model.py`. The dataset downloads automatically to `data/BBBP.csv`; if download is unavailable, place the official CSV at that path and rerun.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and Streamlit Community Cloud instructions. Include the generated model, metadata, and report PNGs when publishing; the dataset itself is only needed for retraining.

## Evaluation

Seed 42 controls the random scaffold-group holdout and estimators. Twenty percent of unique Bemis–Murcko groups (including a single shared group for acyclic molecules) are selected for test, so row fraction may differ from 20%. Class distributions and scaffold overlap are recorded; labels do not drive test-group assignment. If a test class is absent, ROC-AUC is reported undefined. Stratified CV uses training data only; all three fixed models are compared on the same test set, while RF remains primary by specification and is not selected based on test scores. Test metrics include accuracy, precision, recall, F1, ROC-AUC when defined, confusion matrix, and classification report.

## Project structure

```text
app.py              Streamlit UI
features.py         Shared RDKit descriptor logic
data_loader.py      Dataset retrieval, validation, cleaning
predictor.py        Artifact loading and single/batch prediction
train_model.py      Scaffold splitting, CV, final evaluation, artifacts/charts
tests/              Pytest coverage
data/               Cached dataset (ignored by Git)
models/             Local model/metadata (ignored by Git)
reports/            Evaluation charts/tables (ignored by Git)
```

## Limitations and disclaimer

BBB penetration is a complex biological property that cannot be reliably decided from a few descriptors. The dataset can contain class imbalance and experimental inconsistencies. A benchmark score is not evidence of clinical validity. Scaffold testing reduces some forms of structural leakage but cannot remove all dataset bias. Model probabilities may not be calibrated, and out-of-distribution compounds may be unreliable. This is a research and educational estimate, not medical advice or a substitute for laboratory experiments or expert review; it makes no claims about clinical effectiveness, drug safety, or actual human brain exposure.
