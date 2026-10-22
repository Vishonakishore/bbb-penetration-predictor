import pandas as pd
import pytest
from data_loader import validate_schema

def test_dataset_schema():
    validate_schema(pd.DataFrame({"smiles":["CCO"],"p_np":[1]}))

def test_missing_dataset_column():
    with pytest.raises(ValueError, match="schema"):
        validate_schema(pd.DataFrame({"smiles":["CCO"]}))
