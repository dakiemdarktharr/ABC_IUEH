import pytest
from abc_iueh.config import PipelineConfig


@pytest.mark.parametrize("kwargs", [{"development_users": -1}, {"minimum_final_users": 0},
                         {"development_users": 1.5}, {"development_users": True},
                         {"split_seed": -1}, {"target_rate_hz": float("nan")},
                         {"minimum_confirmatory_users": 10}])
def test_invalid_configuration_rejected_before_splitting(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)
