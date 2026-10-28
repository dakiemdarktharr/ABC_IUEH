"""Fixed train-only inner-CV arc pilot, with immutable resumable units.

No real fit runs at import. Production requires CUDA and the exact frozen
configuration. Synthetic tests use a separate, explicitly bounded helper path.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import torch

from .arc_model import ARMS, PARAMETER_COUNTS, ArcModel, nearest_probabilities
from .evaluation import fit_static_mixture, predict_endpoint
from .geometry import geodesic_deg, inverse_wxyz, log_so3, multiply_wxyz, normalize_wxyz
from .sources import sha256_file, utc_now, write_json
from .stage import LockedDevelopmentStage, validate_feature_bundle

_FROZEN_CONFIG = json.loads(r'''
{
  "schema_version": 1,
  "study_id": "20261028_arc_pilot_fixed5000",
  "stage": "train_inner_arc_pilot_only",
  "status_before_efficacy": "fixed_before_new_human_data_model_fits_or_inner_validation_errors",
  "source_partition": "train",
  "development_signal_access_allowed": false,
  "final_access_allowed": false,
  "source_hdf5_sha256": "0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62",
  "split_sha256": "9d7f342c15231a3f7f61fd8c9b95b75adc6aff641e7071d95a43a3832db49256",
  "baseline_feature_cache_sha256": "41235cfba9cd00201dedabf06c65c1df56d424cd724d5aeb51f94bb4d6edc691",
  "baseline_config": "configs/baseline_features.json",
  "context_samples": 400,
  "actual_context_samples": 342,
  "horizon_samples": 20,
  "stride": 1,
  "folds": {
    "count": 3,
    "seed": 20261028,
    "assignment": "lexically_sort_outer_train_user_ids_then_numpy_default_rng_seed_permutation_then_split_5_5_5",
    "outer_train_users_required": 15,
    "fit_users_per_fold": 10,
    "validation_users_per_fold": 5,
    "fold_ids": [
      0,
      1,
      2
    ],
    "all_sessions_of_user_together": true,
    "each_user_validates_once": true,
    "all_fitted_statistics_use_fold_fit_users_only": true
  },
  "model": {
    "encoder": "causal_separable_tcn_3x16_k5_d1_4_16_64",
    "tokens": "current_frame_one_step_SO3_log_radians_per_sample_times_100",
    "feature_scale": 100.0,
    "fitted_scaler": false,
    "initialization": "NumPy_default_rng_training_seed_Normal0_std0.02_float32_module_parameter_order_all_biases_zero",
    "heads": {
      "nll21": "Linear16to21",
      "risk21": "Linear16to21",
      "scalar_angle": "Linear16to16_ReLU_Linear16to1",
      "scalar_pinball": "Linear16to16_ReLU_Linear16to1",
      "vector_angle": "Linear16to16_ReLU_Linear16to3"
    },
    "vector_formula": "q_current*Exp_SO3(k*v_body+(k/20)*r_body);v_body=vector(conj(q_current)*[0,v_world]*q_current)",
    "receptive_field_increments": 341,
    "candidate_indices": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20
    ],
    "candidate_formula": "Exp_SO3(j*v_world)*q_current",
    "short_arc_condition": "20*norm(v_world)<pi",
    "long_arc_fallback": "CAV_W1_exposure20",
    "zero_speed_output": "arc_constrained_heads_q_current_vector_control_may_predict_nonzero_residual",
    "near_zero_threshold_rad_per_sample": 0.0,
    "tie_policy": "uniform_exact_minimum_candidate_angle_ties",
    "parameters": {
      "nll21": 1717,
      "scalar_angle": 1649,
      "scalar_pinball": 1649,
      "risk21": 1717,
      "vector_angle": 1683
    },
    "full_stateless_conv_dense_MACs": {
      "nll21": 425904,
      "scalar_angle": 425840,
      "scalar_pinball": 425840,
      "risk21": 425904,
      "vector_angle": 425872
    },
    "MAC_exclusions": [
      "quaternion_geometry",
      "normalization",
      "CDF",
      "comparisons",
      "output_generation",
      "activations",
      "bias",
      "padding"
    ]
  },
  "arms": [
    "nll21",
    "scalar_angle",
    "scalar_pinball",
    "risk21",
    "vector_angle"
  ],
  "training": {
    "steps": 5000,
    "batch_size": 128,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "epsilon": 1e-08,
    "weight_decay": 0.0,
    "seeds": [
      20261026,
      20261027
    ],
    "sampling": "uniform_global_fold_fit_origins_with_replacement",
    "schedule_rng": "numpy.default_rng(numpy.SeedSequence([training_seed,fold_id]))",
    "same_origin_schedule_across_arms_per_fold_seed": true,
    "cloned_encoder_initial_state_across_arms_per_fold_seed": true,
    "early_stopping": false,
    "checkpoint_selection": "primary_fixed_final5000_secondary_predeclared_time_only",
    "resume_checkpoint_interval_steps": 100,
    "hyperparameter_sweep_allowed": false,
    "losses": {
      "nll21": "categorical_cross_entropy_uniform_exact_ties",
      "scalar_angle": "mean_actual_endpoint20_atan2_geodesic_radians",
      "scalar_pinball": "mean_uniform_tie_expected_0.5_absolute_exposure_error_divided_by20",
      "risk21": "mean_squared_error_over21_actual_candidate_geodesic_costs_in_radians",
      "vector_angle": "mean_actual_endpoint20_atan2_geodesic_radians"
    },
    "loss_units_are_different_team_choices": true,
    "surrogate_or_loss_scaling_changes_after_results_allowed": false
  },
  "train_diagnostics": {
    "sampled_fold_fit_origins": 65536,
    "replacement": false,
    "seed": 20261028,
    "effective_seed": "seed+fold_id",
    "checkpoints": [
      0,
      1000,
      2500,
      5000
    ],
    "checkpoint_selection_allowed": false
  },
  "static_fit": {
    "candidate_experts": [
      "ZV",
      "CAV_W1",
      "CAV_W5",
      "CAV_W20"
    ],
    "simplex_step": 0.25,
    "simplex_points": 35,
    "sampled_fold_fit_origins": 65536,
    "replacement": false,
    "seed": 20261028,
    "effective_seed": "seed+fold_id",
    "objective": "actual_emitted_endpoint20_geodesic_degrees",
    "old_all15_static_weights_allowed": false
  },
  "prior": {
    "fit": "uniform_tie_exposure_histogram_on_the_same65536_fold_fit_origins",
    "smoothing": 0.0,
    "validation_labels_allowed": false
  },
  "evaluation": {
    "batch_size": 256,
    "methods": [
      "nll21_median",
      "nll21_mean",
      "nll21_map",
      "scalar_angle",
      "scalar_pinball",
      "risk21",
      "vector_angle",
      "static_mixture",
      "prior_median",
      "prior_mean",
      "prior_map",
      "ZV",
      "CAV_W1",
      "CAV_W5",
      "CAV_W20"
    ],
    "coverage": "all_existing_R80_cache_origins_of_each_inner_validation_user",
    "metric": "actual_endpoint20_mean_geodesic_degrees",
    "two_seed_aggregation": "arithmetic_mean_of_paired_errors_not_prediction_ensemble",
    "primary_method": "nll21_median",
    "primary_comparators": [
      "static_mixture",
      "scalar_angle",
      "risk21"
    ],
    "mandatory_gate_comparators": [
      "static_mixture",
      "scalar_angle",
      "risk21",
      "scalar_pinball"
    ],
    "vector_control": "must_report_competitive_claim_limit_if_vector_angle_is_better",
    "scalar_and_mean_decoder_support": "continuous_same_arc_envelope_not_identical_discrete21_action_set"
  },
  "uncertainty": {
    "cluster": "whole_user_all_sessions_and_overlapping_origins",
    "bootstrap_resamples": 2000,
    "seed": 20261028,
    "estimand": "window_weighted_ratio_of_sum_errors_to_sum_counts",
    "paired_bootstrap_draws_for_all_methods": true,
    "relative_improvement_recomputed_within_each_draw": true,
    "CI_scope": "conditional_on_two_fixed_seeds_and_three_fixed_cross_validation_models_not_independent_retraining"
  },
  "equal_compute_secondary": {
    "enabled": true,
    "predeclared_before_any_human_data_fit": true,
    "primary_fixed5000_unchanged": true,
    "training_extra_fits": 0,
    "checkpoint_interval_steps": 100,
    "target_per_fold_seed": "minimum_final5000_synchronized_training_seconds_across_all_five_arms",
    "selection": "last100step_checkpoint_with_cumulative_synchronized_training_time_at_most_target",
    "selection_uses_losses_or_accuracy": false,
    "rounding_maximum_steps": 100,
    "coverage": "same_full_inner_validation_origins_as_primary",
    "record_separately": [
      "actual_steps",
      "actual_examples",
      "cumulative_training_seconds",
      "target_training_seconds",
      "next_checkpoint_time_gap",
      "quantization_rounding"
    ],
    "baselines": "identical_fold_fit_baseline_errors_from_primary",
    "summary": "separate_error_group_and_metrics_call_with_explicit_secondary_scope_wrapper",
    "if_no_checkpoint_fits_target": "secondary_inconclusive_no_same_compute_claim_no_extra_fit",
    "strong_positive_paper_interpretation": "must_also_survive_equal_compute_secondary_otherwise_qualify"
  },
  "gate": {
    "absolute_improvement_degrees_minimum": 0.05,
    "relative_improvement_fraction_minimum": 0.03,
    "both_seed_deltas_strictly_negative": true,
    "improved_folds_minimum": 2,
    "comparison_logic": "all_four_mandatory_gate_comparators_conjunctively",
    "CI_is_reported_not_a_new_gate": true,
    "interpretation": "exploratory_pilot_screen_not_confirmatory_success_or_acceptance"
  },
  "resources": {
    "device_required": "cuda",
    "cpu_training_fallback": false,
    "precision": "float32_no_autocast_no_TF32",
    "deterministic_algorithms": true,
    "maximum_pilot_GPU_hours": 5.0,
    "maximum_overall_research_GPU_hours": 20.0,
    "projection_prefix_steps": 100,
    "budget_action": "pause_at_serialized_resume_checkpoint_and_report_incomplete_no_step_selection",
    "same_compute_maximum_training_time_ratio": 1.2,
    "time_comparison": "synchronized_training_time_at_equal_updates_and_examples_report_each_arm",
    "if_time_ratio_exceeded": "qualify_same_compute_claim_no_automatic_extra_fit",
    "static_neural_compute_matched_claim_allowed": false,
    "cloud_approved_cost": 0
  },
  "paper_scope": "single_conditional_Alyx_corpus_user_disjoint_inner_pilot_no_foundation_or_high_novelty_certification",
  "official_timing_dispute": "20samples_100Hz_100ms_unresolved",
  "BOXRR_enabled": false,
  "new_development_or_final_stage_scheduled": false,
  "private_artifacts": [
    "user_ids",
    "fold_assignments",
    "error_vectors",
    "per_user_statistics",
    "weights",
    "sampling_schedules"
  ],
  "public_aggregate_only": true
}
''')
_DECODE_METHODS = ("nll21_median", "nll21_mean", "nll21_map")


class ResourcePause(RuntimeError):
    """A saved exact-prefix state needs more authorized resource budget."""


def validate_arc_config(config):
    if _FROZEN_CONFIG is None or config != _FROZEN_CONFIG:
        raise ValueError("configuration differs from the frozen arc pilot execution contract")
    # Python bool compares equal to integer1; retain an exact type/schema check.
    def types(actual, expected):
        if type(actual) is not type(expected):
            raise ValueError("frozen arc config value has an incorrect type")
        if isinstance(expected, dict):
            for key in expected:
                types(actual[key], expected[key])
        elif isinstance(expected, list):
            for a, b in zip(actual, expected):
                types(a, b)
    types(config, _FROZEN_CONFIG)
    return config


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot():
    """Resident memory and available physical memory, without environment dumps."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_phys", "available_phys", "total_page", "available_page",
                                                        "total_virtual", "available_virtual", "available_extended")]
        class ProcessCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "quota_peak_paged", "quota_paged",
                                                    "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("cannot measure available host memory")
        counters = ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise RuntimeError("cannot measure process memory")
        return {"available_host_bytes": int(status.available_phys), "process_rss_bytes": int(counters.working_set),
                "process_peak_rss_bytes": int(counters.peak_working_set)}
    import resource
    page = os.sysconf("SC_PAGE_SIZE")
    available = os.sysconf("SC_AVPHYS_PAGES") * page
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"available_host_bytes": int(available), "process_rss_bytes": int(peak), "process_peak_rss_bytes": int(peak)}


@contextmanager
def train_only_hdf5_guard(stage, baseline_dir, output_dir):
    """Enforce values read by file and user, including read_direct/__array__."""
    source = Path(stage.hdf5_path).resolve()
    cache = (Path(baseline_dir) / "features.h5").resolve()
    output = Path(output_dir).resolve()
    train = set(stage.partitions["train"])
    counters = Counter()
    original_getitem, original_direct, original_array = h5py.Dataset.__getitem__, h5py.Dataset.read_direct, h5py.Dataset.__array__
    original_iter = stage.iter_segments

    def train_reader(partition):
        if partition!="train":
            raise PermissionError("arc reader refuses non-training partitions before any dataset slice")
        yield from original_iter(partition)

    def check(data):
        filename = Path(data.file.filename).resolve()
        name = data.name
        if filename == source:
            fields = name.strip("/").split("/")
            if (len(fields)==7 and fields[0]=="users" and fields[1] in train and fields[2]=="sessions"
                    and fields[4]=="segments" and fields[6]=="quaternion"):
                return "source/train_quaternion"
            raise PermissionError("arc pilot blocks development/final source values")
        if filename == cache:
            if name.startswith(("/train/", "/development/")) and name.endswith("/origin_index"):
                return name.split("/")[1] + "/origin_index_integrity_only"
            raise PermissionError("arc pilot allows cache index integrity values only")
        if filename.is_relative_to(output):
            return "private_output/error_or_checkpoint_values"
        raise PermissionError("HDF5 value read is outside fixed pilot source/cache/output paths")

    def getitem(data, selection, *args, **kwargs):
        key = check(data)
        result = original_getitem(data, selection, *args, **kwargs)
        counters[key + "/calls"] += 1
        counters[key + "/elements"] += int(np.size(result))
        return result

    def direct(data, destination, *args, **kwargs):
        key = check(data)
        result = original_direct(data, destination, *args, **kwargs)
        counters[key + "/read_direct_calls"] += 1
        counters[key + "/read_direct_destination_elements"] += int(np.size(destination))
        return result

    def array(data, *args, **kwargs):
        check(data)
        return original_array(data, *args, **kwargs)

    h5py.Dataset.__getitem__, h5py.Dataset.read_direct, h5py.Dataset.__array__ = getitem, direct, array
    stage.iter_segments = train_reader
    try:
        yield counters
    finally:
        h5py.Dataset.__getitem__, h5py.Dataset.read_direct, h5py.Dataset.__array__ = original_getitem, original_direct, original_array
        stage.iter_segments = original_iter


@dataclass(frozen=True)
class Segment:
    user: str
    session: str
    segment: str
    sample_start: int
    samples: int
    window_start: int
    windows: int


class FlatTrainWindows:
    """Only raw flat rotations and origin indexes; never materialize all windows."""
    def __init__(self, records, *, device="cpu"):
        ordered = sorted(records, key=lambda record: record[:3])
        if not ordered:
            raise ValueError("no training recordings")
        self.users = tuple(sorted({record[0] for record in ordered}))
        self.user_code = {user: i for i, user in enumerate(self.users)}
        arrays, origins, codes, segments = [], [], [], []
        sample_position = window_position = 0
        self.row_order_sha256 = {}
        digests = {user: hashlib.sha256(b"abc_arc_user_origins_v1\0") for user in self.users}
        for user, session, segment, q in ordered:
            q = np.asarray(q)
            if q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
                raise ValueError("training source must contain finite float32 quaternion segments")
            count = max(0, len(q)-419)
            if not count:
                continue
            arrays.append(q)
            local = np.arange(399, len(q)-20, dtype=np.int64)
            origins.append(local+sample_position)
            codes.append(np.full(count, self.user_code[user], dtype=np.uint8))
            segments.append(Segment(user, session, segment, sample_position, len(q), window_position, count))
            digests[user].update(_digest_json([user, session, segment, count]).encode())
            digests[user].update(local.astype("<i8").tobytes())
            sample_position += len(q)
            window_position += count
        if not arrays or set(s.user for s in segments) != set(self.users):
            raise ValueError("every outer training user needs eligible windows")
        self.q_cpu = np.concatenate(arrays)
        self.origins_cpu = np.concatenate(origins)
        self.codes_cpu = np.concatenate(codes)
        self.segments = tuple(segments)
        self.row_order_sha256 = {user: digest.hexdigest() for user,digest in digests.items()}
        self.device = torch.device(device)
        self.q = torch.as_tensor(self.q_cpu, device=self.device)
        self.origins = torch.as_tensor(self.origins_cpu, device=self.device)
        self.offsets = torch.arange(-399,1,device=self.device)
        self.costs = None

    def __len__(self):
        return len(self.origins_cpu)

    def indexes_for_users(self, users):
        if not users or not set(users).issubset(self.users):
            raise ValueError("window user set is outside outer training cohort")
        return np.flatnonzero(np.isin(self.codes_cpu, [self.user_code[user] for user in users]))

    def batch(self, global_indexes):
        indexes = torch.as_tensor(global_indexes, dtype=torch.long, device=self.device)
        origin = self.origins[indexes]
        context = self.q[origin[:,None]+self.offsets[None,:]]
        target = self.q[origin+20]
        return context, target, None if self.costs is None else self.costs[indexes]

    def geometry(self, indexes):
        origin = self.origins_cpu[np.asarray(indexes, dtype=np.int64)]
        current = self.q_cpu[origin]
        unit = normalize_wxyz(current)
        velocities = np.zeros((len(origin),4,3), dtype=np.float64)
        for expert, width in enumerate((1,5,20),start=1):
            velocities[:,expert] = log_so3(multiply_wxyz(unit, inverse_wxyz(self.q_cpu[origin-width])))/width
        return current, velocities, self.q_cpu[origin+20]

    def precompute_costs(self, *, chunk_size=4096):
        model = ArcModel("nll21",seed=20261026).to(self.device).eval()
        self.costs = torch.empty((len(self),21),dtype=torch.float32,device=self.device)
        with torch.inference_mode():
            for start in range(0,len(self),chunk_size):
                stop = min(len(self),start+chunk_size)
                context,target,_ = self.batch(np.arange(start,stop,dtype=np.int64))
                values = model.candidate_costs(context,target)
                if not torch.isfinite(values).all().item() or torch.any((values<0)|(values>torch.pi+1e-6)).item():
                    raise ValueError("invalid actual-candidate angular training labels")
                self.costs[start:stop] = values
        _sync(self.device)
        del model


def make_inner_folds(users, *, seed=20261028):
    ordered = np.asarray(sorted(users))
    if len(ordered) != 15 or len(set(ordered)) != 15:
        raise ValueError("inner pilot requires exactly fifteen unique outer training users")
    shuffled = np.random.default_rng(seed).permutation(ordered)
    folds = []
    for fold in range(3):
        validation = tuple(sorted(shuffled[fold*5:(fold+1)*5].tolist()))
        fitting = tuple(user for user in ordered.tolist() if user not in validation)
        folds.append({"fold":fold,"fit_users":fitting,"validation_users":validation})
    return folds


def make_schedule(store, fit_users, *, fold, seed, steps, batch_size):
    eligible = store.indexes_for_users(fit_users)
    rng = np.random.default_rng(np.random.SeedSequence([seed,fold]))
    schedule = eligible[rng.integers(0,len(eligible),size=(steps,batch_size),dtype=np.int64)]
    if not np.isin(store.codes_cpu[schedule], [store.user_code[user] for user in fit_users]).all():
        raise PermissionError("validation origin entered optimizer schedule")
    return schedule


def fit_fold_references(store, fold, *, sample_size=65536, seed=20261028):
    eligible = store.indexes_for_users(fold["fit_users"])
    if len(eligible)<sample_size:
        raise ValueError("not enough fitting origins for fixed reference calibration")
    indexes = np.sort(np.random.default_rng(seed+fold["fold"]).choice(eligible,size=sample_size,replace=False))
    current,velocities,target = store.geometry(indexes)
    weights,static = fit_static_mixture(current,velocities,target,step=.25)
    if store.costs is None:
        raise ValueError("candidate cost labels must be bound before fitting prior")
    costs = store.costs[torch.as_tensor(indexes,device=store.device)]
    prior = nearest_probabilities(costs).double().mean(0).cpu().numpy()
    prior = (prior/prior.sum()).astype(np.float32)
    return {"fold":fold["fold"],"calibration_indexes":indexes,"static_weights":weights,"static_fit":static,
            "prior_probabilities":prior,"sample_schedule_sha256":hashlib.sha256(indexes.astype("<i8").tobytes()).hexdigest()}


def _models(seed,device):
    models = {arm:ArcModel(arm,seed).to(device) for arm in ARMS}
    if {arm:sum(p.numel() for p in model.parameters()) for arm,model in models.items()} != PARAMETER_COUNTS:
        raise ValueError("arc parameter contract changed")
    if len({model.encoder_sha256() for model in models.values()})!=1:
        raise ValueError("all arms must clone the same encoder initialization")
    return models


def _learned_errors(models,context,target):
    target_numpy = target.detach().cpu().numpy()
    nll = models["nll21"]
    probabilities = torch.softmax(nll.logits(context),dim=-1)
    predictions = {"nll21_"+decode:nll.decode_probabilities(context,probabilities,decode)[:,-1] for decode in ("median","mean","map")}
    predictions.update({arm:model(context)[:,-1] for arm,model in models.items() if arm!="nll21"})
    errors = {}
    for method,prediction in predictions.items():
        array = prediction.detach().cpu().numpy()
        if array.dtype!=np.float32 or array.shape!=(len(target),4) or not np.isfinite(array).all():
            raise ValueError("actual emitted endpoint must be finite float32 [B,4]")
        errors[method] = geodesic_deg(array,target_numpy)
    return errors


def train_diagnostic(store,models,indexes,*,batch_size=256,reference=None):
    sums = {arm:0.0 for arm in ARMS}
    errors = {}
    references = {"CAV_W1":0.0}
    if reference is not None:
        references["static_mixture"] = 0.0
    with torch.inference_mode():
        for model in models.values():
            model.eval()
        for start in range(0,len(indexes),batch_size):
            batch = indexes[start:start+batch_size]
            context,target,costs = store.batch(batch)
            for arm,model in models.items():
                loss = model.training_loss(context,target,reduction="sum",candidate_costs=costs)
                sums[arm] += float(loss.item())
            for method,values in _learned_errors(models,context,target).items():
                errors[method] = errors.get(method,0.0)+float(values.sum(dtype=np.float64))
            current,velocities,truth = store.geometry(batch)
            references["CAV_W1"] += float(geodesic_deg(predict_endpoint(current,velocities,[0,1,0,0]),truth).sum(dtype=np.float64))
            if reference is not None:
                references["static_mixture"] += float(geodesic_deg(predict_endpoint(current,velocities,reference["static_weights"]),truth).sum(dtype=np.float64))
    return {"scope":"fixed_fold_fit_only_in_sample_no_checkpoint_selection","windows":len(indexes),
            "natural_unit_mean_loss":{arm:value/len(indexes) for arm,value in sums.items()},
            "actual_endpoint20_mean_geodesic_deg":{method:value/len(indexes) for method,value in errors.items()},
            "fixed_reference_mean_geodesic_deg":{method:value/len(indexes) for method,value in references.items()},
            "learned_minus_reference_deg":{method:{ref:(value-refvalue)/len(indexes) for ref,refvalue in references.items()}
                                            for method,value in errors.items()}}


def evaluate_validation(store,models,fold,reference,seed,path,*,batch_size=256,budget_deadline=None):
    """One complete fixed-step user-held-out evaluation, streamed to private HDF5."""
    methods = tuple(_DECODE_METHODS)+tuple(arm for arm in ARMS if arm!="nll21")+(
        "static_mixture","prior_median","prior_mean","prior_map","ZV","CAV_W1","CAV_W5","CAV_W20")
    sums = {method:{str(seed):{}} for method in methods}
    counts = {}
    seconds = {arm:0.0 for arm in ARMS}
    temporary = path.with_suffix(path.suffix+".part")
    evaluation_binding = _digest_json({"states":{arm:_state_hash(model) for arm,model in models.items()},
                                      "seed":seed,"fold":fold["fold"],"row_digests":store.row_order_sha256,
                                      "static_weights":np.asarray(reference["static_weights"]).tolist(),
                                      "prior_probabilities":np.asarray(reference["prior_probabilities"]).tolist()})
    from .arc_metrics import row_order_digest
    prior = torch.as_tensor(reference["prior_probabilities"],device=store.device).view(1,21)
    active_path = path if path.exists() else temporary
    mode = "r" if path.exists() else "r+" if temporary.exists() else "w"
    with h5py.File(active_path,mode) as output,torch.inference_mode():
        if mode=="w":
            output.attrs["row_order_sha256"] = row_order_digest(store.row_order_sha256)
            output.attrs["evaluation_binding"] = evaluation_binding
            output.attrs["validation_seconds"] = json.dumps(seconds)
        elif output.attrs.get("evaluation_binding")!=evaluation_binding:
            raise ValueError("partial/completed validation belongs to another checkpoint/reference/row order")
        else:
            seconds = json.loads(output.attrs["validation_seconds"])
        for model in models.values():
            model.eval()
        for user in fold["validation_users"]:
            indexes = store.indexes_for_users((user,))
            counts[user] = len(indexes)
            datasets = {}
            for method in methods:
                key = f"{method}/{seed}/{user}"
                if key in output:
                    data = output[key]
                    if data.shape!=(len(indexes),) or data.dtype!=np.dtype("f8") or data.attrs.get("row_order_sha256")!=store.row_order_sha256[user]:
                        raise ValueError("validation dataset origin binding changed")
                else:
                    data = output.create_dataset(key,shape=(len(indexes),),dtype="f8",chunks=True,compression="lzf")
                    data.attrs["row_order_sha256"] = store.row_order_sha256[user]
                datasets[method] = data
                sums[method][str(seed)][user] = 0.0
            committed = int(output.attrs.get("committed_"+user,0))
            if not 0<=committed<=len(indexes) or mode=="r" and committed!=len(indexes):
                raise ValueError("invalid validation committed-row marker")
            for method,data in datasets.items():
                for start in range(0,committed,65536):
                    sums[method][str(seed)][user] += float(data[start:min(committed,start+65536)].sum(dtype=np.float64))
            for start in range(committed,len(indexes),batch_size):
                if budget_deadline is not None and time.perf_counter()>=budget_deadline:
                    raise ResourcePause("validation paused at committed origin prefix")
                stop = min(len(indexes),start+batch_size)
                batch = indexes[start:stop]
                context,target,_ = store.batch(batch)
                target_numpy = target.cpu().numpy()
                errors = {}
                for arm,model in models.items():
                    _sync(store.device)
                    before = time.perf_counter()
                    if arm=="nll21":
                        probabilities = torch.softmax(model.logits(context),dim=-1)
                        outputs = {"nll21_"+decode:model.decode_probabilities(context,probabilities,decode)[:,-1]
                                   for decode in ("median","mean","map")}
                    else:
                        outputs = {arm:model(context)[:,-1]}
                    for method,prediction in outputs.items():
                        emitted = prediction.cpu().numpy()
                        if emitted.dtype!=np.float32 or emitted.shape!=(len(batch),4):
                            raise ValueError("metric requires actual emitted float32 endpoint20")
                        errors[method] = geodesic_deg(emitted,target_numpy)
                    _sync(store.device)
                    seconds[arm] += time.perf_counter()-before
                current,velocities,baseline_target = store.geometry(batch)
                for method,weights in (("ZV",[1,0,0,0]),("CAV_W1",[0,1,0,0]),("CAV_W5",[0,0,1,0]),
                                       ("CAV_W20",[0,0,0,1]),("static_mixture",reference["static_weights"])):
                    errors[method] = geodesic_deg(predict_endpoint(current,velocities,weights),baseline_target)
                for decode in ("median","mean","map"):
                    prediction = models["nll21"].decode_probabilities(context,prior.expand(len(batch),-1),decode)[:,-1]
                    errors["prior_"+decode] = geodesic_deg(prediction.cpu().numpy(),target_numpy)
                for method,values in errors.items():
                    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>180+1e-6):
                        raise ValueError("invalid geodesic validation vector")
                    datasets[method][start:stop] = values
                    sums[method][str(seed)][user] += float(values.sum(dtype=np.float64))
                output.attrs["committed_"+user] = stop
                output.attrs["validation_seconds"] = json.dumps(seconds)
                output.flush()
    if active_path==temporary:
        temporary.rename(path)
    # Canonical bounded summation gives exactly the same receipt after replay,
    # independent of the forward batch size or a previous interruption point.
    with h5py.File(path,"r") as completed:
        for method in methods:
            for user in fold["validation_users"]:
                data = completed[f"{method}/{seed}/{user}"]
                total_sum = 0.0
                for start in range(0,len(data),65536):
                    values = data[start:min(len(data),start+65536)]
                    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>180+1e-6):
                        raise ValueError("completed validation vector contains invalid angles")
                    total_sum += float(values.sum(dtype=np.float64))
                sums[method][str(seed)][user] = total_sum
    return {"user_sums":sums,"user_counts":counts,"validation_seconds":seconds,"errors_sha256":sha256_file(path)}


def _exclusive_json(path,value):
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path.name}")
    write_json(path,value)


def _save_unit_state(path,models,optimizers,*,step,losses,training_seconds,diagnostics,bindings,schedule_hash,flags,loader_seconds):
    if path.exists():
        raise FileExistsError("immutable resume state already exists")
    payload = {"scope":"arc_pilot_inner_fit_only","step":step,"models":{arm:model.state_dict() for arm,model in models.items()},
               "optimizers":{arm:optimizer.state_dict() for arm,optimizer in optimizers.items()},
               "losses":losses[:,:step].detach().cpu(),"training_seconds":training_seconds,"diagnostics":diagnostics,
               "bindings":bindings,"schedule_sha256":schedule_hash,"actual_tensor_execution":flags,"loader_seconds":loader_seconds,
               "torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state() if next(next(iter(models.values())).parameters()).device.type=="cuda" else None}
    temporary = path.with_suffix(".pt.part")
    if temporary.exists():
        raise FileExistsError("partial resume state requires explicit recovery")
    torch.save(payload,temporary)
    temporary.rename(path)


def select_equal_time_states(directory,final_training_seconds,*,bindings,schedule_hash,batch_size=128,checkpoint_interval=100):
    """Select only by cumulative synchronized training seconds, never losses."""
    cap = min(final_training_seconds.values())
    if set(final_training_seconds)!=set(ARMS) or not all(np.isfinite(value) and value>0 for value in final_training_seconds.values()):
        raise ValueError("final training times must be positive finite for every arm")
    chosen,records,previous_times = {},[],{arm:-1.0 for arm in ARMS}
    for path in sorted(Path(directory).glob("state_step_*.pt")):
        state = torch.load(path,map_location="cpu",weights_only=True)
        if state["bindings"]!=bindings or state["schedule_sha256"]!=schedule_hash:
            raise ValueError("equal-time checkpoint source/config/schedule binding mismatch")
        if state["step"]==0:
            continue
        if state["step"]%checkpoint_interval:
            raise ValueError("equal-time checkpoint is outside fixed interval")
        records.append(state)
        for arm in ARMS:
            seconds = state["training_seconds"][arm]
            if not np.isfinite(seconds) or seconds<=previous_times[arm] or seconds<0:
                raise ValueError("checkpoint training times must be finite and strictly cumulative")
            previous_times[arm] = seconds
            if seconds<=cap:
                previous = chosen.get(arm)
                if previous is None or state["step"]>previous["steps"]:
                    chosen[arm] = {"checkpoint_basename":path.name,"steps":state["step"],"training_seconds":seconds,
                                   "examples":state["step"]*batch_size}
    if set(chosen)!=set(ARMS):
        raise ValueError("equal-time selection requires a feasible positive saved checkpoint for every arm")
    for arm,selection in chosen.items():
        following = next((record for record in records if record["step"]>selection["steps"]),None)
        selection["unused_time_seconds"] = cap-selection["training_seconds"]
        selection["next_checkpoint_training_seconds"] = None if following is None else following["training_seconds"][arm]
        selection["rounding_interval_steps"] = 0 if following is None else following["step"]-selection["steps"]
        if selection["rounding_interval_steps"]>checkpoint_interval:
            raise ValueError("equal-time rounding exceeds one fixed checkpoint interval")
    return {"scope":"secondary_matched_training_time_not_matched_examples","time_cap_seconds":cap,
            "selection":"last_fixed100step_checkpoint_with_cumulative_training_time_at_or_below_cap",
            "criterion_uses_accuracy_or_loss":False,"checkpoint_interval_steps":checkpoint_interval,"arms":chosen}


def fit_unit(store,fold,reference,seed,directory,bindings,config,*,budget_deadline=None,synthetic=False):
    """All five arms, exactly matched fixed batches. Only synthetic overrides <=3 steps."""
    steps = config["training"]["steps"]
    if synthetic:
        if not 1<=steps<=3:
            raise ValueError("synthetic integration helper is limited to three updates")
    elif steps!=5000:
        raise ValueError("production arc fit must execute exactly5000 fixed steps")
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"receipt.json").exists():
        raise FileExistsError("completed unit receipt is immutable")
    schedule = make_schedule(store,fold["fit_users"],fold=fold["fold"],seed=seed,steps=steps,
                             batch_size=config["training"]["batch_size"])
    schedule_hash = hashlib.sha256(schedule.astype("<i8").tobytes()).hexdigest()
    schedule_path = directory/"private_origin_schedule.npy"
    if schedule_path.exists():
        if not np.array_equal(np.load(schedule_path,allow_pickle=False),schedule):
            raise ValueError("saved fitting schedule differs from frozen schedule")
    else:
        np.save(schedule_path,schedule,allow_pickle=False)
    models = _models(seed,store.device)
    initial_hashes = {arm:_state_hash(model) for arm,model in models.items()}
    initial_encoder = models["nll21"].encoder_sha256()
    options = config["training"]
    optimizers = {arm:torch.optim.Adam(model.parameters(),lr=options["learning_rate"],betas=tuple(options["betas"]),
                                      eps=options["epsilon"],weight_decay=options["weight_decay"])
                  for arm,model in models.items()}
    losses = torch.empty((len(ARMS),steps),dtype=torch.float32,device=store.device)
    training_seconds = {arm:0.0 for arm in ARMS}
    diagnostics,flags,loader_seconds = {},{},0.0
    prior_states = sorted(directory.glob("state_step_*.pt"))
    first_step = 0
    if prior_states:
        state = torch.load(prior_states[-1],map_location=store.device,weights_only=True)
        if state["bindings"]!=bindings or state["schedule_sha256"]!=schedule_hash:
            raise ValueError("resume state source/config/schedule binding mismatch")
        first_step = state["step"]
        for arm in ARMS:
            models[arm].load_state_dict(state["models"][arm])
            optimizers[arm].load_state_dict(state["optimizers"][arm])
        losses[:,:first_step] = state["losses"].to(store.device)
        training_seconds,diagnostics,flags = state["training_seconds"],state["diagnostics"],state["actual_tensor_execution"]
        loader_seconds = state["loader_seconds"]
        torch.set_rng_state(state["torch_cpu_rng"].cpu())
        if store.device.type=="cuda" and state["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["torch_cuda_rng"].cpu(),store.device)
    train_indexes = store.indexes_for_users(fold["fit_users"])
    sample_size = config["train_diagnostics"]["sampled_fold_fit_origins"]
    if len(train_indexes)<sample_size:
        raise ValueError("fixed training diagnostic subset exceeds fitting population")
    diagnostic_indexes = np.sort(np.random.default_rng(config["train_diagnostics"]["seed"]+fold["fold"]).choice(
        train_indexes,size=sample_size,replace=False))
    if "0" not in diagnostics:
        diagnostics["0"] = train_diagnostic(store,models,diagnostic_indexes,batch_size=config["evaluation"]["batch_size"],reference=reference)
    if not prior_states:
        _save_unit_state(directory/"state_step_000000.pt",models,optimizers,step=0,losses=losses,
                         training_seconds=training_seconds,diagnostics=diagnostics,bindings=bindings,schedule_hash=schedule_hash,
                         flags=flags,loader_seconds=loader_seconds)
    schedule_device = torch.as_tensor(schedule,device=store.device)
    interval = config["training"]["resume_checkpoint_interval_steps"]
    for step in range(first_step,steps):
        _sync(store.device)
        before = time.perf_counter()
        context,target,costs = store.batch(schedule_device[step])
        _sync(store.device)
        loader_seconds += time.perf_counter()-before
        for index,arm in enumerate(ARMS):
            model,optimizer = models[arm],optimizers[arm]
            model.train()
            _sync(store.device)
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(context,target,candidate_costs=costs)
            loss.backward()
            if step==0:
                flags[arm] = {"context_device":str(context.device),"target_device":str(target.device),
                              "candidate_labels_device":str(costs.device),"loss_device":str(loss.device),
                              "parameters_and_gradients_on_execution_device":all(p.device==store.device and p.grad is not None and
                                                                                 p.grad.device==store.device for p in model.parameters()),
                              "initial_parameter_gradient_l2":float(torch.sqrt(sum((p.grad.double()**2).sum() for p in model.parameters())).item())}
            optimizer.step()
            losses[index,step] = loss.detach()
            _sync(store.device)
            training_seconds[arm] += time.perf_counter()-before
        completed = step+1
        if completed in config["train_diagnostics"]["checkpoints"]:
            diagnostics[str(completed)] = train_diagnostic(store,models,diagnostic_indexes,
                                                           batch_size=config["evaluation"]["batch_size"],reference=reference)
        checkpoint_due = completed%interval==0 or completed==steps
        if checkpoint_due:
            if not torch.isfinite(losses[:,:completed]).all().item():
                raise ValueError("nonfinite arc fit loss")
            checkpoint = directory/f"state_step_{completed:06d}.pt"
            _save_unit_state(checkpoint,models,optimizers,step=completed,losses=losses,training_seconds=training_seconds,
                             diagnostics=diagnostics,bindings=bindings,schedule_hash=schedule_hash,flags=flags,loader_seconds=loader_seconds)
            snapshot = memory_snapshot()
            progress = {"event":"arc_pilot_fit","fold":fold["fold"],"seed":seed,"step":completed,"steps":steps,
                        "per_arm_training_seconds":training_seconds,"memory":snapshot}
            if completed==config["resources"]["projection_prefix_steps"]:
                progress["projected_all6_units_training_seconds_at_prefix_rate"] = sum(training_seconds.values())/completed*steps*6
            print(json.dumps(progress),flush=True)
            if completed<steps and (budget_deadline is not None and time.perf_counter()>=budget_deadline or
                                    not synthetic and snapshot["available_host_bytes"]<1024**3):
                return {"status":"paused_resource_budget","fold":fold["fold"],"seed":seed,"completed_step":completed,
                        "checkpoint":checkpoint.name,"training_seconds":training_seconds,"memory":snapshot}
    metrics = evaluate_validation(store,models,fold,reference,seed,directory/"errors.h5",batch_size=config["evaluation"]["batch_size"],
                                  budget_deadline=budget_deadline)
    equal_time = select_equal_time_states(directory,training_seconds,bindings=bindings,schedule_hash=schedule_hash,
                                          batch_size=options["batch_size"],checkpoint_interval=interval)
    selection_path = directory/"matched_time_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text())!=equal_time:
            raise ValueError("frozen timing-only selection changed on replay")
    else:
        _exclusive_json(selection_path,equal_time)
    final_hashes = {arm:_state_hash(model) for arm,model in models.items()}
    for arm,selection in equal_time["arms"].items():
        saved = torch.load(directory/selection["checkpoint_basename"],map_location=store.device,weights_only=True)
        models[arm].load_state_dict(saved["models"][arm])
        selection["examples"] = selection["steps"]*options["batch_size"]
        selection["selected_state_sha256"] = _state_hash(models[arm])
    timed_metrics = evaluate_validation(store,models,fold,reference,seed,directory/"matched_time_errors.h5",
                                        batch_size=config["evaluation"]["batch_size"],budget_deadline=budget_deadline)
    peaks = {"peak_allocated_bytes":torch.cuda.max_memory_allocated(store.device),
             "peak_reserved_bytes":torch.cuda.max_memory_reserved(store.device)} if store.device.type=="cuda" else {
                 "peak_allocated_bytes":0,"peak_reserved_bytes":0}
    receipt = {"status":"complete","scope":"synthetic_helper_only_max3_updates" if synthetic else "production_inner_fold_fit",
               "fold":fold["fold"],"seed":seed,"steps":steps,"batch_size":options["batch_size"],
               "bindings":bindings,"schedule_sha256":schedule_hash,"initial_encoder_sha256":initial_encoder,
               "initial_model_sha256":initial_hashes,"final_model_sha256":final_hashes,
               "natural_unit_loss_scopes":config["training"]["losses"],"diagnostics":diagnostics,
               "training_seconds":training_seconds,"loader_seconds":loader_seconds,"actual_tensor_execution":flags,
               "memory":memory_snapshot(),**peaks,**metrics,"matched_time":{"selection":equal_time,**timed_metrics},
               "first100_mean_batch_loss":{arm:float(losses[i,:min(100,steps)].mean().item()) for i,arm in enumerate(ARMS)},
               "last100_mean_batch_loss":{arm:float(losses[i,-min(100,steps):].mean().item()) for i,arm in enumerate(ARMS)}}
    _exclusive_json(directory/"receipt.json",receipt)
    return receipt


def _all_sources():
    directory = Path(__file__).parent
    return {name:sha256_file(directory/name) for name in ("arc_experiment.py","arc_model.py","arc_metrics.py","stage.py",
                                                         "geometry.py","evaluation.py","sources.py","evidence.py","selector_experiment.py")}


def _combine_scope(output,store,folds,config,bindings,*,matched_time=False):
    from .arc_metrics import summarize_arc_pilot,row_order_digest
    combined = output/("private_matched_time_paired_errors.h5" if matched_time else "private_paired_errors.h5")
    replay = combined.exists()
    active_path = combined if replay else combined.with_name(combined.name+f".part.{time.time_ns()}")
    sums = {method:{str(seed):{} for seed in config["training"]["seeds"]} for method in config["evaluation"]["methods"]}
    counts,user_folds = {},{}
    resources = {arm:[] for arm in ARMS}
    with h5py.File(active_path,"r" if replay else "w") as destination:
        if replay:
            if destination.attrs.get("row_order_sha256")!=row_order_digest(store.row_order_sha256):
                raise ValueError("derived aggregate origin binding mismatch")
        else:
            destination.attrs["row_order_sha256"] = row_order_digest(store.row_order_sha256)
        for fold in folds:
            user_folds.update({user:fold["fold"] for user in fold["validation_users"]})
            for seed in config["training"]["seeds"]:
                directory = output/f"fold_{fold['fold']}"/f"seed_{seed}"
                receipt = json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
                values = receipt["matched_time"] if matched_time else receipt
                error_path = directory/("matched_time_errors.h5" if matched_time else "errors.h5")
                if receipt["status"]!="complete" or receipt["bindings"]!=bindings or sha256_file(error_path)!=values["errors_sha256"]:
                    raise ValueError("completed unit binding or error hash mismatch")
                with h5py.File(error_path,"r") as source:
                    for method in config["evaluation"]["methods"]:
                        parent = destination[f"{method}/{seed}"] if replay else destination.require_group(f"{method}/{seed}")
                        for user in fold["validation_users"]:
                            source_data = source[f"{method}/{seed}/{user}"]
                            if replay:
                                if user not in parent or parent[user].shape!=source_data.shape:
                                    raise ValueError("derived aggregate is missing immutable unit rows")
                                for start in range(0,len(source_data),65536):
                                    stop = min(len(source_data),start+65536)
                                    if not np.array_equal(parent[user][start:stop],source_data[start:stop]):
                                        raise ValueError("derived aggregate differs from immutable unit errors")
                            else:
                                source.copy(source_data,parent,name=user)
                            sums[method][str(seed)][user] = values["user_sums"][method][str(seed)][user]
                for user,count in values["user_counts"].items():
                    if user in counts and counts[user]!=count:
                        raise ValueError("seed counts differ")
                    counts[user] = count
                for arm in ARMS:
                    training_seconds = values["selection"]["arms"][arm]["training_seconds"] if matched_time else receipt["training_seconds"][arm]
                    resources[arm].append({"fold":fold["fold"],"seed":seed,"training_seconds":training_seconds,
                                           "validation_seconds":values["validation_seconds"][arm],
                                           "peak_allocated_bytes":receipt["peak_allocated_bytes"],"peak_reserved_bytes":receipt["peak_reserved_bytes"]})
        summary = summarize_arc_pilot(destination,sums,counts,user_folds,row_order_sha256=store.row_order_sha256,
                                      resources=resources,resamples=config["uncertainty"]["bootstrap_resamples"],seed=config["uncertainty"]["seed"])
    stats_path = output/("private_matched_time_user_statistics.json" if matched_time else "private_user_statistics.json")
    statistics = {"user_sums":sums,"user_counts":counts,"user_folds":user_folds,"row_order_sha256":store.row_order_sha256}
    if stats_path.exists():
        if json.loads(stats_path.read_text())!=statistics:
            raise ValueError("existing private statistics differ from immutable unit receipts")
    else:
        _exclusive_json(stats_path,statistics)
    if not replay:
        active_path.rename(combined)
    return summary


def _combine(output,store,folds,config,runtime,bindings):
    primary = _combine_scope(output,store,folds,config,bindings)
    secondary = _combine_scope(output,store,folds,config,bindings,matched_time=True)
    selections = []
    for fold in folds:
        for seed in config["training"]["seeds"]:
            unit = json.loads((output/f"fold_{fold['fold']}"/f"seed_{seed}"/"receipt.json").read_text())
            selections.append({"fold":fold["fold"],"seed":seed,**unit["matched_time"]["selection"]})
    steps = config["training"]["steps"]
    return {"status":"complete_exploratory_outer_train_inner_cv_only" if steps==5000 else "synthetic_helper_only_max3_updates",
            "matched_updates_primary":{"scope":f"fixed{steps}_updates_same{steps*config['training']['batch_size']}_examples_per_fit","metrics":primary},
            "matched_training_time_secondary":{"scope":"timing_only_fixed100step_prefix_selection_different_example_counts",
                                               "metrics":secondary,"selections":selections},
            "runtime":runtime,"bindings":bindings,"development_signal_values_read":0,"final_array_values_read":0}


def run_arc_pilot(stage,baseline_dir,output_dir,config,*,config_path,device="cuda"):
    validate_arc_config(config)
    config_path,baseline,output = Path(config_path),Path(baseline_dir),Path(output_dir)
    if json.loads(config_path.read_text(encoding="utf-8"))!=config:
        raise ValueError("saved arc config differs from execution config")
    if torch.device(device).type!="cuda":
        raise ValueError("real arc pilot requires CUDA and cannot fall back to CPU")
    if stage.hdf5_sha256!=config["source_hdf5_sha256"] or stage.split_sha256!=config["split_sha256"]:
        raise ValueError("source/split differ from frozen pilot")
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    if memory_snapshot()["available_host_bytes"]<1024**3:
        return {"status":"paused_resource_budget","reason":"available_host_memory_below_1GiB_before_data_load","memory":memory_snapshot()}
    from .selector_experiment import _cuda_runtime
    actual_device,runtime = _cuda_runtime(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime.update({"cudnn_version":torch.backends.cudnn.version(),"cudnn_tf32_allowed":torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
                    "model_export_performed":False})
    sources = _all_sources()
    config_hash = sha256_file(config_path)
    previous_sessions = list(output.glob("session_*.json"))
    used = sum(json.loads(path.read_text())["cuda_context_wall_seconds"] for path in previous_sessions)
    budget = config["resources"]["maximum_pilot_GPU_hours"]*3600-used
    if budget<=0:
        return {"status":"paused_resource_budget","reason":"fixed_pilot_budget_exhausted","used_seconds":used}
    deadline = started+budget
    with train_only_hdf5_guard(stage,baseline,output) as reads:
        try:
            feature = validate_feature_bundle(stage,baseline,config_path=config["baseline_config"])
            if feature["cache_sha256"]!=config["baseline_feature_cache_sha256"]:
                raise ValueError("feature cache differs from original origin population")
            bindings = {"source_hdf5_sha256":stage.hdf5_sha256,"split_sha256":stage.split_sha256,"feature_cache_sha256":feature["cache_sha256"],
                        "scalers_sha256":feature["scalers_sha256"],"config_sha256":config_hash,"sources_sha256":sources}
            manifest_path = output/"manifest.json"
            manifest = {"schema_version":1,"scope":config["stage"],"config":config,"bindings":bindings,"created_utc":utc_now()}
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous["config"]!=config or previous["bindings"]!=bindings:
                    raise ValueError("existing pilot output belongs to another frozen source/config")
            else:
                _exclusive_json(manifest_path,manifest)
            if (output/"summary.json").exists():
                return json.loads((output/"summary.json").read_text(encoding="utf-8"))
            folds = make_inner_folds(stage.partitions["train"],seed=config["folds"]["seed"])
            store = FlatTrainWindows(stage.iter_segments("train"),device=actual_device)
            if len(store)!=3330312 or set(store.users)!=set(stage.partitions["train"]):
                raise ValueError("all source origins must equal original R80 outer train population")
            _exclusive_json(output/"private_folds.json",{"folds":folds,"fold_digest":_digest_json(folds)}) if not (output/"private_folds.json").exists() else None
            label_started = time.perf_counter()
            store.precompute_costs()
            runtime["label_cache_preparation_seconds"] = time.perf_counter()-label_started
            runtime["outer_train_quaternion_samples"] = len(store.q_cpu)
            runtime["outer_train_windows"] = len(store)
            runtime["fold_digest"] = _digest_json(folds)
            result = None
            for fold in folds:
                fold_dir = output/f"fold_{fold['fold']}"
                fold_dir.mkdir(exist_ok=True)
                reference_path = fold_dir/"private_references.json"
                if reference_path.exists():
                    reference = json.loads(reference_path.read_text())
                    if reference["bindings"]!=bindings:
                        raise ValueError("fold reference binding mismatch")
                    reference["static_weights"] = np.asarray(reference["static_weights"])
                    reference["prior_probabilities"] = np.asarray(reference["prior_probabilities"],dtype=np.float32)
                else:
                    reference = fit_fold_references(store,fold,sample_size=config["static_fit"]["sampled_fold_fit_origins"],seed=config["static_fit"]["seed"])
                    serial = {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in reference.items()}
                    _exclusive_json(reference_path,{**serial,"bindings":bindings})
                for seed in config["training"]["seeds"]:
                    directory = fold_dir/f"seed_{seed}"
                    if (directory/"receipt.json").exists():
                        continue
                    result = fit_unit(store,fold,reference,seed,directory,bindings,config,budget_deadline=deadline)
                    if result["status"]!="complete":
                        return result
            result = _combine(output,store,folds,config,runtime,bindings)
            result["data_audit"] = stage.verify_unchanged()
            result["read_set_counters"] = dict(reads)
            result["sources_unchanged"] = _all_sources()==sources
            result["config_unchanged"] = sha256_file(config_path)==config_hash
            if not result["sources_unchanged"] or not result["config_unchanged"]:
                raise ValueError("code/config changed during pilot")
            _exclusive_json(output/"summary.json",result)
            return result
        except ResourcePause as error:
            return {"status":"paused_resource_budget","phase":str(error),"final_holdout_evaluated":False,
                    "development_signal_values_read":0,"final_array_values_read":0}
        finally:
            _sync(actual_device)
            _exclusive_json(output/f"session_{time.time_ns()}.json",{"created_utc":utc_now(),"cuda_context_wall_seconds":time.perf_counter()-started,
                                                                   "memory":memory_snapshot(),"data_audit":stage.verify_unchanged(),"read_set_counters":dict(reads)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/arc_pilot.json")
    parser.add_argument("--output",default="data/arc_pilot/20261028")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--hdf5",default="data/processed/alyx.h5")
    parser.add_argument("--split",default="data/user_split.json")
    parser.add_argument("--baseline-dir",default="data/baseline_runs/20261028_r80")
    args = parser.parse_args()
    config = validate_arc_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    result = run_arc_pilot(LockedDevelopmentStage(args.hdf5,args.split),args.baseline_dir,args.output,config,
                           config_path=args.config,device=args.device)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
