"""Correct workspace-root adapter for the pinned ARC vector model.

This is separate from the frozen historical model-v3 reference. It imports no
numerical libraries until the factory is called.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

SOURCE_SHA256 = "ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd"


def pinned_source_path() -> Path:
    workspace = Path(__file__).resolve().parents[5]
    return workspace / "ABC_IUEH" / "src" / "abc_iueh" / "arc_model.py"


def create_raw_vector_reference(seed: int):
    source = pinned_source_path().resolve(strict=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("pinned ARC model source digest mismatch")
    package_root = str(source.parents[1])
    existing = sys.modules.get("abc_iueh.arc_model")
    if existing is not None and Path(existing.__file__).resolve() != source:
        raise RuntimeError("a different ARC model was already loaded")
    inserted = package_root not in sys.path
    if inserted:
        sys.path.insert(0, package_root)
    try:
        from abc_iueh.arc_model import ArcModel
    finally:
        if inserted:
            sys.path.remove(package_root)
    if Path(sys.modules["abc_iueh.arc_model"].__file__).resolve() != source:
        raise RuntimeError("loaded ARC model path differs from pin")
    model = ArcModel("vector_angle", seed=seed)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 1683:
        raise RuntimeError("ARC parameter count differs from pin")
    return model
