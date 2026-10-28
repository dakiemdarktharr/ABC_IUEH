"""Post-run resume-artifact hardening: synthetic CPU fixtures only.

The measured pilot completed and its original source was archived before this
module was added. These tests do not read human signals or rerun efficacy fits.
"""
import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import abc_iueh.arc_experiment as experiment
from abc_iueh.geometry import exp_so3


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def synthetic_store_and_folds():
    t = np.arange(424, dtype=np.float64)
    records = []
    for index in range(15):
        q = exp_so3(np.column_stack((.001*t+.01*index,
                                    .0000005*t*t,
                                    .004*np.sin(t/19+index*.02)))).astype(np.float32)
        records.append((f"synthetic_train_{index:02d}", "session", "segment", q))
    store = experiment.FlatTrainWindows(records, device="cpu")
    store.precompute_costs(chunk_size=8)
    return store, experiment.make_inner_folds(store.users)


def saved_reference(store, fold, bindings):
    reference = experiment.fit_fold_references(store, fold, sample_size=8, seed=20261028)
    return {**experiment._serial_reference_payload(reference), "bindings": copy.deepcopy(bindings)}


def test_canonical_saved_fold_manifest_valid_resume_is_accepted():
    users = [f"synthetic_train_{index:02d}" for index in range(15)]
    expected = experiment.canonical_fold_payload(users, seed=20261028)
    reloaded = json.loads(json.dumps(expected))
    experiment.validate_saved_folds(reloaded, users, seed=20261028)
    assert reloaded == expected


@pytest.mark.parametrize("mutation", ["fake_digest", "copied_reordered_folds", "boolean_fold_id",
                                     "duplicate_user", "unknown_key"])
def test_saved_fold_manifest_rejects_wrong_content_even_with_self_consistent_digest(mutation):
    users = [f"synthetic_train_{index:02d}" for index in range(15)]
    payload = experiment.canonical_fold_payload(users, seed=20261028)
    if mutation == "fake_digest":
        payload["fold_digest"] = "0"*64
    elif mutation == "copied_reordered_folds":
        payload["folds"][0], payload["folds"][1] = payload["folds"][1], payload["folds"][0]
        payload["fold_digest"] = experiment._digest_json(payload["folds"])
    elif mutation == "boolean_fold_id":
        # False compares equal to integer0 in ordinary dict equality.
        payload["folds"][0]["fold"] = False
        payload["fold_digest"] = experiment._digest_json(payload["folds"])
    elif mutation == "duplicate_user":
        payload["folds"][0]["validation_users"][0] = payload["folds"][0]["fit_users"][0]
        payload["fold_digest"] = experiment._digest_json(payload["folds"])
    else:
        payload["unchecked_extra_option"] = True
    with pytest.raises(ValueError, match="fold manifest"):
        experiment.validate_saved_folds(payload, users, seed=20261028)


def test_valid_reference_reload_is_exact_and_does_not_modify_saved_payload(synthetic_store_and_folds):
    store, folds = synthetic_store_and_folds
    bindings = {"synthetic_source_sha256": "a"*64, "synthetic_config_sha256": "b"*64}
    payload = saved_reference(store, folds[0], bindings)
    original = copy.deepcopy(payload)
    verified = experiment.validate_reloaded_fold_reference(payload, store, folds[0], bindings,
                                                          sample_size=8, seed=20261028)
    assert payload == original
    assert experiment._serial_reference_payload(verified) == {k:v for k,v in payload.items() if k != "bindings"}


def test_copied_wrong_fold_reference_rejected_despite_identical_global_bindings(synthetic_store_and_folds):
    store, folds = synthetic_store_and_folds
    bindings = {"synthetic_source_sha256": "a"*64}
    copied = saved_reference(store, folds[0], bindings)
    with pytest.raises(ValueError, match="fold reference"):
        experiment.validate_reloaded_fold_reference(copied, store, folds[1], bindings,
                                                    sample_size=8, seed=20261028)


def test_heldout_calibration_origin_rejected_even_after_updating_its_self_hash(synthetic_store_and_folds):
    store, folds = synthetic_store_and_folds
    bindings = {"synthetic_source_sha256": "a"*64}
    payload = saved_reference(store, folds[0], bindings)
    heldout_index = int(store.indexes_for_users(folds[0]["validation_users"])[0])
    payload["calibration_indexes"][0] = heldout_index
    payload["calibration_indexes"] = sorted(payload["calibration_indexes"])
    payload["sample_schedule_sha256"] = hashlib.sha256(
        np.asarray(payload["calibration_indexes"], dtype="<i8").tobytes()).hexdigest()
    with pytest.raises(ValueError, match="fold reference"):
        experiment.validate_reloaded_fold_reference(payload, store, folds[0], bindings,
                                                    sample_size=8, seed=20261028)


@pytest.mark.parametrize("mutation", ["static_weights", "prior_probabilities", "other_fit_only_sample",
                                     "boolean_index", "boolean_fold", "unknown_key"])
def test_reference_content_is_rederived_from_correct_fit_data_not_trusted_by_self_hash(
        synthetic_store_and_folds, mutation):
    store, folds = synthetic_store_and_folds
    fold = folds[0]
    bindings = {"synthetic_source_sha256": "a"*64}
    payload = saved_reference(store, fold, bindings)
    if mutation == "static_weights":
        payload["static_weights"] = [1.0, 0.0, 0.0, 0.0]
        payload["static_fit"]["weights"] = payload["static_weights"].copy()
    elif mutation == "prior_probabilities":
        values = payload["prior_probabilities"]
        maximum = int(np.argmax(values))
        other = (maximum+1) % 21
        delta = min(.01, values[maximum]/2)
        values[maximum] -= delta
        values[other] += delta
        assert np.isclose(sum(values), 1.0) and min(values) >= 0
    elif mutation == "other_fit_only_sample":
        allowed = store.indexes_for_users(fold["fit_users"])
        unused = sorted(set(allowed.tolist())-set(payload["calibration_indexes"]))[0]
        payload["calibration_indexes"][0] = int(unused)
        payload["calibration_indexes"] = sorted(payload["calibration_indexes"])
        payload["sample_schedule_sha256"] = hashlib.sha256(
            np.asarray(payload["calibration_indexes"], dtype="<i8").tobytes()).hexdigest()
    elif mutation == "boolean_index":
        payload["calibration_indexes"][0] = False
    elif mutation == "boolean_fold":
        payload["fold"] = False
    else:
        payload["unchecked_elapsed_seconds"] = 0.0
    with pytest.raises(ValueError, match="fold reference"):
        experiment.validate_reloaded_fold_reference(payload, store, fold, bindings,
                                                    sample_size=8, seed=20261028)


def test_reference_global_binding_mismatch_rejected_before_rederivation(synthetic_store_and_folds, monkeypatch):
    store, folds = synthetic_store_and_folds
    bindings = {"synthetic_source_sha256": "a"*64}
    payload = saved_reference(store, folds[0], bindings)
    def forbidden(*args, **kwargs):
        raise AssertionError("bad global binding reached analytic reference rederivation")
    monkeypatch.setattr(experiment, "fit_fold_references", forbidden)
    with pytest.raises(ValueError, match="fold reference"):
        experiment.validate_reloaded_fold_reference(payload, store, folds[0],
                                                    {"synthetic_source_sha256": "changed"},
                                                    sample_size=8, seed=20261028)
