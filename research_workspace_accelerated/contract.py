"""Accelerator research profile contract.

This profile deliberately reuses the proven candidate API while carrying a PyTorch seed.
It is an experimental starting point, not the byte-identical official NumPy baseline, and
its provenance is recorded as such by the driver.
"""
from research_workspace.contract import (ContractError, Data, ENTRY_POINT, Spec,
                                         check_scores, columns_of, primary, train_subset)

MUTABLE_FILES = ("features.py", "model.py", "train.py", "config.json")
