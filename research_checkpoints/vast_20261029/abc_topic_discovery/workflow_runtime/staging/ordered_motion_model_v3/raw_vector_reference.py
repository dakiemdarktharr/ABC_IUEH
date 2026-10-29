"""Lazy, hash-pinned factory for the contemporaneous 1,683-param raw-vector arm.

This module imports only the Python standard library at load time. Calling
``create_raw_vector_reference`` imports the pinned local ABC_IUEH Torch model;
do that only within the coordinator's scheduled RAM window. No custom weights,
layers, or initialization are defined here.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any


SOURCE_GIT_REVISION = "6e7b8a6aaf6d7013cc87ebdbce847d5d13b07837"
SOURCE_RELATIVE_PATH = Path("ABC_IUEH/src/abc_iueh/arc_model.py")
SOURCE_SHA256 = "ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd"
EXPECTED_ARM = "vector_angle"
EXPECTED_PARAMETER_COUNT = 1683


def pinned_source_path() -> Path:
    workspace = Path(__file__).resolve().parents[3]
    return workspace / SOURCE_RELATIVE_PATH


def create_raw_vector_reference(seed: int) -> Any:
    """Instantiate the existing `ArcModel('vector_angle', seed=seed)` factory.

    The SHA check rejects source drift before any Torch import. The model uses
    its existing deterministic NumPy initialization and trainable parameter
    count; this adapter creates no additional or replacement weights.
    """
    source = pinned_source_path()
    if not source.is_file():
        raise FileNotFoundError(f"pinned raw-vector source is missing: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"raw-vector source SHA-256 mismatch: {digest}")

    package_root = str(source.parents[1])  # ABC_IUEH/src
    existing = sys.modules.get("abc_iueh.arc_model")
    if existing is not None:
        existing_file = Path(existing.__file__).resolve()
        if existing_file != source.resolve():
            raise RuntimeError(f"different abc_iueh.arc_model already loaded: {existing_file}")

    inserted = package_root not in sys.path
    if inserted:
        sys.path.insert(0, package_root)
    try:
        # This is the first Torch-using operation in this module. Invoke only
        # after coordinator scheduling and the required available-RAM check.
        from abc_iueh.arc_model import ArcModel
    finally:
        if inserted:
            sys.path.remove(package_root)

    imported_file = Path(sys.modules["abc_iueh.arc_model"].__file__).resolve()
    if imported_file != source.resolve():
        raise RuntimeError(f"imported raw-vector source differs from pin: {imported_file}")
    model = ArcModel(EXPECTED_ARM, seed=seed)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"raw-vector parameter count mismatch: {actual_count}")
    return model

