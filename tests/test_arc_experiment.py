"""Synthetic-only runner boundaries, fold isolation and exact-prefix recovery."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.arc_model import ARMS
import abc_iueh.arc_experiment as experiment
from abc_iueh.geometry import exp_so3,geodesic_deg
from abc_iueh.sources import sha256_file,write_json
from abc_iueh.stage import LockedDevelopmentStage

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def motion(length,shift=0):
    t = np.arange(length,dtype=np.float64)
    return exp_so3(np.column_stack((.001*t+shift,.0000005*t*t,.004*np.sin(t/19)))).astype(np.float32)


def store_fixture():
    records = []
    for index in range(15):
        records.append((f"synthetic_train_{index:02d}","session","segment",motion(424,index*.01)))
    return experiment.FlatTrainWindows(records,device="cpu")


def small_config():
    config = json.loads((ROOT/"configs/arc_pilot.json").read_text())
    config["training"]["steps"] = 3
    config["training"]["batch_size"] = 2
    config["training"]["resume_checkpoint_interval_steps"] = 1
    config["train_diagnostics"]["sampled_fold_fit_origins"] = 8
    config["train_diagnostics"]["checkpoints"] = [0,1,2,3]
    config["static_fit"]["sampled_fold_fit_origins"] = 8
    config["evaluation"]["batch_size"] = 2
    return config


def source_fixture(tmp_path):
    source,split = tmp_path/"source.h5",tmp_path/"split.json"
    users = [f"synthetic_user_{index:02d}" for index in range(35)]
    partitions = {"train":users[:15],"development":users[15:20],"final":users[20:]}
    with h5py.File(source,"w") as file:
        file.attrs["quaternion_order"],file.attrs["sample_rate_hz"] = "wxyz",100
        for index,user in enumerate(users):
            q = motion(424,index*.01) if user in partitions["train"] else np.full((424,4),np.nan,dtype=np.float32)
            file.create_dataset(f"users/{user}/sessions/session/segments/segment/quaternion",data=q)
    write_json(split,{"mode":"locked_confirmatory","unit":"user_before_windowing","training_allowed":True,
                      "holdout_pretraining_allowed":False,"source_hdf5_sha256":sha256_file(source),"partitions":partitions})
    baseline = tmp_path/"baseline"
    baseline.mkdir()
    with h5py.File(baseline/"features.h5","w") as file:
        file.create_dataset("train/u/s/g/origin_index",data=[399])
        file.create_dataset("development/v/s/g/origin_index",data=[399])
        file.create_dataset("development/v/s/g/features/h1_a0",data=np.zeros((1,10)))
    return LockedDevelopmentStage(source,split),baseline


def test_guard_actual_seven_field_stage_paths_and_pre_slice_partition_refusal(tmp_path):
    stage,baseline = source_fixture(tmp_path)
    original = h5py.Dataset.__getitem__
    with experiment.train_only_hdf5_guard(stage,baseline,tmp_path/"output") as reads:
        records = list(stage.iter_segments("train"))
        assert len(records)==15 and all(np.isfinite(row[3]).all() for row in records)
        for partition in ("development","final"):
            before = dict(reads)
            with pytest.raises(PermissionError,match="before any dataset slice"):
                next(stage.iter_segments(partition))
            assert dict(reads)==before
        with h5py.File(stage.hdf5_path,"r") as source:
            for partition in ("development","final"):
                dataset = source[f"users/{stage.partitions[partition][0]}/sessions/session/segments/segment/quaternion"]
                with pytest.raises(PermissionError):
                    dataset[:]
                with pytest.raises(PermissionError):
                    np.asarray(dataset)
                with pytest.raises(PermissionError):
                    dataset.read_direct(np.empty(dataset.shape,dtype=np.float32))
        with h5py.File(baseline/"features.h5","r") as cache:
            assert cache["development/v/s/g/origin_index"][:].tolist()==[399]
            with pytest.raises(PermissionError):
                cache["development/v/s/g/features/h1_a0"][:]
    assert h5py.Dataset.__getitem__ is original
    assert stage.verify_unchanged()["segment_reads"]=={"train":15,"development":0}


def test_flat_windows_exact_origins_targets_boundaries_and_no_full_window_materialization():
    a,b = motion(423),motion(425,.4)
    store = experiment.FlatTrainWindows([("b","s","g",b),("a","s","g",a)])
    assert len(store)==10
    np.testing.assert_array_equal(store.origins_cpu,np.r_[np.arange(399,403),np.arange(399,405)+423])
    context,target,_ = store.batch(np.arange(len(store)))
    np.testing.assert_array_equal(context[0].numpy(),a[:400])
    np.testing.assert_array_equal(target[0].numpy(),a[419])
    np.testing.assert_array_equal(context[4].numpy(),b[:400])
    np.testing.assert_array_equal(target[-1].numpy(),b[-1])
    assert store.q_cpu.shape==(848,4) and not hasattr(store,"all_contexts")
    changed = experiment.FlatTrainWindows([("a","s","g",a),("b","s","g",b.copy())])
    assert changed.row_order_sha256==store.row_order_sha256


def test_folds_user_disjoint_and_shared_schedule_has_no_validation_origins():
    store = store_fixture()
    folds = experiment.make_inner_folds(store.users)
    assert len(folds)==3 and sorted(user for fold in folds for user in fold["validation_users"])==list(store.users)
    assert folds==experiment.make_inner_folds(tuple(reversed(store.users)))
    for fold in folds:
        assert len(fold["fit_users"])==10 and len(fold["validation_users"])==5
        assert not set(fold["fit_users"]).intersection(fold["validation_users"])
        first = experiment.make_schedule(store,fold["fit_users"],fold=fold["fold"],seed=20261026,steps=3,batch_size=2)
        second = experiment.make_schedule(store,fold["fit_users"],fold=fold["fold"],seed=20261026,steps=3,batch_size=2)
        np.testing.assert_array_equal(first,second)
        assert set(store.codes_cpu[first].ravel()).issubset({store.user_code[u] for u in fold["fit_users"]})


def test_fold_reference_fit_unchanged_when_only_validation_rotations_change():
    store = store_fixture()
    store.precompute_costs(chunk_size=4)
    fold = experiment.make_inner_folds(store.users)[0]
    before = experiment.fit_fold_references(store,fold,sample_size=8)
    records = []
    for index,user in enumerate(store.users):
        q = motion(424,index*.01)
        if user in fold["validation_users"]:
            q = motion(424,.7+index*.03)[::-1].copy()
        records.append((user,"session","segment",q))
    changed = experiment.FlatTrainWindows(records)
    changed.precompute_costs(chunk_size=4)
    after = experiment.fit_fold_references(changed,fold,sample_size=8)
    for field in ("calibration_indexes","static_weights","prior_probabilities"):
        np.testing.assert_array_equal(before[field],after[field])
    assert before["static_fit"]==after["static_fit"] and before["sample_schedule_sha256"]==after["sample_schedule_sha256"]
    val = store.indexes_for_users(fold["validation_users"])
    assert not np.array_equal(store.batch(val)[1].numpy(),changed.batch(val)[1].numpy())


def test_synthetic_unit_resume_reproduces_primary_final_states_and_common_initialization(tmp_path,monkeypatch):
    # Wall-time selection is tested separately. A deterministic clock here
    # avoids lazy-autograd/OS scheduling making a three-update fixture lack a
    # positive timed checkpoint, without changing any gradients or updates.
    clock = [100.0]
    def tick():
        clock[0] += .001
        return clock[0]
    monkeypatch.setattr(experiment,"time",SimpleNamespace(perf_counter=tick,time_ns=time.time_ns))
    store = store_fixture()
    store.precompute_costs(chunk_size=8)
    fold = experiment.make_inner_folds(store.users)[0]
    reference = experiment.fit_fold_references(store,fold,sample_size=8)
    config = small_config()
    bindings = {"synthetic":True,"source_sha256":"fixture"}
    full = experiment.fit_unit(store,fold,reference,20261026,tmp_path/"full",bindings,config,synthetic=True)
    paused = experiment.fit_unit(store,fold,reference,20261026,tmp_path/"resume",bindings,config,
                                 budget_deadline=-1.0,synthetic=True)
    assert paused["status"]=="paused_resource_budget" and paused["completed_step"]==1
    old_states = {p.name:sha256_file(p) for p in (tmp_path/"resume").glob("state_step_*.pt")}
    with pytest.raises(ValueError,match="binding mismatch"):
        experiment.fit_unit(store,fold,reference,20261026,tmp_path/"resume",{"synthetic":True,"source_sha256":"changed"},
                            config,synthetic=True)
    assert old_states=={p.name:sha256_file(p) for p in (tmp_path/"resume").glob("state_step_*.pt")}
    resumed = experiment.fit_unit(store,fold,reference,20261026,tmp_path/"resume",bindings,config,synthetic=True)
    assert full["final_model_sha256"]==resumed["final_model_sha256"]
    assert full["initial_model_sha256"]==resumed["initial_model_sha256"]
    assert full["schedule_sha256"]==resumed["schedule_sha256"]
    assert set(full["diagnostics"])=={"0","1","2","3"}
    assert "static_mixture" in full["diagnostics"]["0"]["fixed_reference_mean_geodesic_deg"]
    states = sorted((tmp_path/"resume").glob("state_step_*.pt"))
    assert len(states)==4
    final = torch.load(states[-1],map_location="cpu",weights_only=True)
    assert all(final["actual_tensor_execution"][arm]["parameters_and_gradients_on_execution_device"] for arm in ARMS)
    with pytest.raises(FileExistsError,match="completed unit"):
        experiment.fit_unit(store,fold,reference,20261026,tmp_path/"resume",bindings,config,synthetic=True)
    with h5py.File(tmp_path/"full/errors.h5","r") as file:
        for method in config["evaluation"]["methods"]:
            for user in fold["validation_users"]:
                values = file[f"{method}/20261026/{user}"][:]
                assert values.shape==(5,) and np.isfinite(values).all()
                assert values.sum()==pytest.approx(full["user_sums"][method]["20261026"][user],abs=1e-12)


def test_partial_validation_resume_and_completed_output_readonly_replay(tmp_path):
    store = store_fixture()
    store.precompute_costs(chunk_size=8)
    fold = experiment.make_inner_folds(store.users)[0]
    reference = experiment.fit_fold_references(store,fold,sample_size=8)
    models = experiment._models(20261026,"cpu")
    path = tmp_path/"validation.h5"
    with pytest.raises(experiment.ResourcePause):
        experiment.evaluate_validation(store,models,fold,reference,20261026,path,batch_size=2,
                                       budget_deadline=time.perf_counter()-1)
    assert not path.exists() and path.with_suffix(".h5.part").exists()
    first = experiment.evaluate_validation(store,models,fold,reference,20261026,path,batch_size=2)
    saved_hash = sha256_file(path)
    second = experiment.evaluate_validation(store,models,fold,reference,20261026,path,batch_size=2)
    assert first==second and sha256_file(path)==saved_hash
    models["nll21"].head.bias.data[0] += 1
    with pytest.raises(ValueError,match="another checkpoint"):
        experiment.evaluate_validation(store,models,fold,reference,20261026,path,batch_size=2)


def test_equal_time_selection_uses_only_training_times_not_losses(tmp_path):
    bindings = {"synthetic":True}
    for step in (0,100,200):
        times = {arm:step*(index+1)/100 for index,arm in enumerate(ARMS)}
        torch.save({"step":step,"bindings":bindings,"schedule_sha256":"same","training_seconds":times,
                    "losses":torch.tensor([999999-step])},tmp_path/f"state_step_{step:06d}.pt")
    selected = experiment.select_equal_time_states(tmp_path,{arm:10 for arm in ARMS},bindings=bindings,schedule_hash="same")
    assert selected["time_cap_seconds"]==10 and not selected["criterion_uses_accuracy_or_loss"]
    assert all(record["steps"]==200 for record in selected["arms"].values())
    chosen = experiment.select_equal_time_states(tmp_path,{arm:5 for arm in ARMS},bindings=bindings,schedule_hash="same")
    assert chosen["arms"]["nll21"]["steps"]==200
    assert chosen["arms"]["vector_angle"]["steps"]==100
    assert chosen["arms"]["vector_angle"]["examples"]==12800
    with pytest.raises(ValueError,match="positive saved checkpoint"):
        experiment.select_equal_time_states(tmp_path,{arm:2 for arm in ARMS},bindings=bindings,schedule_hash="same")


def test_full_synthetic_six_units_metrics_and_derived_aggregation_replay(tmp_path,monkeypatch):
    clock = [100.0]
    def tick():
        clock[0] += .001
        return clock[0]
    monkeypatch.setattr(experiment,"time",SimpleNamespace(perf_counter=tick,time_ns=time.time_ns))
    store = store_fixture()
    store.precompute_costs(chunk_size=8)
    folds = experiment.make_inner_folds(store.users)
    config = small_config()
    config["training"]["steps"] = 1
    config["train_diagnostics"]["checkpoints"] = [0,1]
    config["evaluation"]["batch_size"] = 8
    config["uncertainty"]["bootstrap_resamples"] = 20
    bindings = {"synthetic":True}
    for fold in folds:
        reference = experiment.fit_fold_references(store,fold,sample_size=8)
        for seed in config["training"]["seeds"]:
            experiment.fit_unit(store,fold,reference,seed,tmp_path/f"fold_{fold['fold']}"/f"seed_{seed}",
                                bindings,config,synthetic=True)
    first = experiment._combine(tmp_path,store,folds,config,{"synthetic":True},bindings)
    files = {name:sha256_file(tmp_path/name) for name in ("private_paired_errors.h5","private_matched_time_paired_errors.h5",
                                                       "private_user_statistics.json","private_matched_time_user_statistics.json")}
    second = experiment._combine(tmp_path,store,folds,config,{"synthetic":True},bindings)
    assert first==second and first["status"]=="synthetic_helper_only_max3_updates"
    assert files=={name:sha256_file(tmp_path/name) for name in files}
    text = json.dumps(first)
    assert not any(user in text for user in store.users)
    assert first["matched_training_time_secondary"]["selections"][0]["criterion_uses_accuracy_or_loss"] is False


def test_production_config_rejects_synthetic_override_cpu_and_unknown_fields(tmp_path):
    config = json.loads((ROOT/"configs/arc_pilot.json").read_text())
    assert experiment.validate_arc_config(config)==config
    for change in ({"final_access_allowed":True},{"unknown_ignored_option":1}):
        changed = copy.deepcopy(config)
        changed.update(change)
        with pytest.raises(ValueError,match="frozen"):
            experiment.validate_arc_config(changed)
    with pytest.raises(ValueError,match="frozen"):
        experiment.validate_arc_config(small_config())
    wrong_type = copy.deepcopy(config)
    wrong_type["schema_version"] = True
    with pytest.raises(ValueError,match="type"):
        experiment.validate_arc_config(wrong_type)
    path = tmp_path/"config.json"
    write_json(path,config)
    with pytest.raises(ValueError,match="requires CUDA"):
        experiment.run_arc_pilot(None,tmp_path/"none",tmp_path/"out",config,config_path=path,device="cpu")
    assert not (tmp_path/"out").exists()
