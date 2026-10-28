"""Explicit team choices; disputed challenge statements remain disputed."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    target_rate_hz: float = 100.0
    native_rate_min_hz: float = 85.0
    native_rate_max_hz: float = 110.0
    trim_seconds: float = 5.0
    max_gap_seconds: float = 0.05
    norm_epsilon: float = 1e-8
    min_segment_samples: int = 1000
    filter_window_seconds: float = 5.0
    min_filter_window_samples: int = 10
    sweep_threshold_deg: float = 17.0
    velocity_threshold_deg_s: float = 15.0
    threshold_operator: str = "gt"
    tail_window_policy: str = "partial_elapsed"
    session_cap_seconds: float = 1200.0
    context_samples: int = 400
    horizon_samples: int = 20
    split_seed: int = 20261026
    minimum_confirmatory_users: int = 30
    development_users: int = 5
    minimum_final_users: int = 12

    def __post_init__(self):
        integer_fields = ("min_segment_samples", "min_filter_window_samples", "context_samples",
                          "horizon_samples", "minimum_confirmatory_users", "development_users", "minimum_final_users")
        for name in integer_fields:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        numeric_fields = ("target_rate_hz", "native_rate_min_hz", "native_rate_max_hz", "trim_seconds",
                          "max_gap_seconds", "norm_epsilon", "filter_window_seconds", "sweep_threshold_deg",
                          "velocity_threshold_deg_s", "session_cap_seconds")
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric data")
        if self.minimum_confirmatory_users <= self.development_users + self.minimum_final_users:
            raise ValueError("minimum cohort must leave train, development and final users")
        if self.threshold_operator not in {"gt", "ge"}:
            raise ValueError("threshold_operator must be gt or ge")
        if self.tail_window_policy not in {"partial_elapsed", "drop"}:
            raise ValueError("tail_window_policy must be partial_elapsed or drop")
        if not 0 < self.native_rate_min_hz <= self.native_rate_max_hz:
            raise ValueError("invalid native rate interval")
        if self.context_samples != 400 or self.horizon_samples != 20:
            raise ValueError("official static interface remains 400 -> 20 samples")
        if self.max_gap_seconds <= 0 or self.target_rate_hz <= 0:
            raise ValueError("rates and gap limit must be positive")
        if self.min_segment_samples < self.context_samples + self.horizon_samples:
            raise ValueError("segment must support a complete context and forecast")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
