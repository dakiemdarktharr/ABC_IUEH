"""Private train-only bundle contract for the Linux Vast pilot."""

from .bundle import (
    BUNDLE_KIND,
    CONTEXT_SAMPLES,
    FOLD_ID,
    PURPOSE,
    ROLE,
    TARGET_SAMPLES,
    PreparedTrainBundle,
    SegmentInput,
    attestation_signing_bytes,
    export_authorized_fit10_bundle,
    open_prepared_train_bundle,
)

__all__ = [
    "BUNDLE_KIND",
    "CONTEXT_SAMPLES",
    "FOLD_ID",
    "PURPOSE",
    "ROLE",
    "TARGET_SAMPLES",
    "PreparedTrainBundle",
    "SegmentInput",
    "attestation_signing_bytes",
    "export_authorized_fit10_bundle",
    "open_prepared_train_bundle",
]
