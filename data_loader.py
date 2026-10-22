"""Download, validate, and reproducibly clean MoleculeNet BBBP."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
from rdkit import Chem

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
DATA_PATH = Path(__file__).parent / "data" / "BBBP.csv"
REQUIRED_COLUMNS = {"smiles", "p_np"}


def validate_schema(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"BBBP dataset schema is missing required column(s): {', '.join(sorted(missing))}. Found: {', '.join(map(str, frame.columns))}")


def load_dataset(path: str | Path = DATA_PATH, download: bool = True) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found at {path}. Run python train_model.py with network access.")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(DATA_URL, timeout=45)
            response.raise_for_status()
            path.write_bytes(response.content)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not download BBBP dataset from {DATA_URL}: {exc}. Download the CSV manually into {path} and retry.") from exc
    raw = pd.read_csv(path)
    validate_schema(raw)
    stats = {"original_rows": len(raw), "missing_values": int(raw.isna().sum().sum()), "source": DATA_URL,
             "retrieved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    work = raw.copy()
    work["smiles"] = work["smiles"].astype("string").str.strip()
    work["p_np"] = pd.to_numeric(work["p_np"], errors="coerce")
    bad_label = work["p_np"].isna() | ~work["p_np"].isin([0, 1])
    work = work.loc[~bad_label & work["smiles"].notna() & work["smiles"].ne("")].copy()
    work["_mol"] = work["smiles"].map(Chem.MolFromSmiles)
    stats["invalid_smiles"] = int(work["_mol"].isna().sum())
    work = work.loc[work["_mol"].notna()].copy()
    stats["valid_molecules_before_dedup"] = len(work)
    work["_canonical"] = work["_mol"].map(Chem.MolToSmiles)
    # Conflicting labels for an identical canonical structure are excluded; otherwise keep first occurrence.
    conflicts = work.groupby("_canonical")["p_np"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    stats["conflicting_duplicate_structures"] = len(conflict_keys)
    work = work.loc[~work["_canonical"].isin(conflict_keys)]
    stats["duplicate_molecules_removed"] = int(work.duplicated("_canonical").sum())
    work = work.drop_duplicates("_canonical", keep="first").drop(columns=["_mol", "_canonical"]).reset_index(drop=True)
    stats["valid_molecules"] = len(work)
    stats["class_distribution"] = {str(int(k)): int(v) for k, v in work["p_np"].value_counts().sort_index().items()}
    stats["excluded_missing_or_invalid_labels"] = int(bad_label.sum())
    return work, stats
