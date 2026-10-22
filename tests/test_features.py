import numpy as np
import pytest
from features import featurize_smiles, FEATURE_NAMES, InvalidSmilesError

def test_valid_smiles_and_descriptors():
    mol, descriptors, frame = featurize_smiles("CCO")
    assert mol is not None
    assert descriptors["molecular_weight"] > 40
    assert list(frame.columns) == FEATURE_NAMES
    assert frame.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(frame.to_numpy()).all()

def test_invalid_smiles():
    with pytest.raises(InvalidSmilesError): featurize_smiles("not a smiles !!!")

def test_empty_smiles():
    with pytest.raises(InvalidSmilesError): featurize_smiles(" ")
