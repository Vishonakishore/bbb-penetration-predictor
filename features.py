"""Deterministic RDKit descriptor generation shared by training and inference."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

DESCRIPTOR_VERSION = "rdkit-descriptors-v1"
DESCRIPTOR_FUNCTIONS = {
    "molecular_weight": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_bond_donors": Lipinski.NumHDonors,
    "h_bond_acceptors": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "aromatic_rings": Lipinski.NumAromaticRings,
    "ring_count": Lipinski.RingCount,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "heavy_atom_count": Lipinski.HeavyAtomCount,
    "formal_charge": Chem.GetFormalCharge,
    "molar_refractivity": Descriptors.MolMR,
    "heteroatom_count": Lipinski.NumHeteroatoms,
    "n_aliphatic_rings": Lipinski.NumAliphaticRings,
}
FEATURE_NAMES = list(DESCRIPTOR_FUNCTIONS)


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed or described safely."""


def featurize_smiles(smiles: str) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Return (RDKit molecule, named descriptors, one-row ordered feature frame)."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidSmilesError("Enter a non-empty SMILES string.")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise InvalidSmilesError("RDKit could not parse this SMILES. Check atom, bond, and ring syntax.")
    values: dict[str, float] = {}
    for name, function in DESCRIPTOR_FUNCTIONS.items():
        try:
            value = float(function(mol))
        except Exception as exc:
            raise InvalidSmilesError(f"Could not calculate descriptor {name}: {exc}") from exc
        if not math.isfinite(value):
            raise InvalidSmilesError(f"Descriptor {name} is not finite for this molecule.")
        values[name] = value
    return mol, values, pd.DataFrame([[values[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
