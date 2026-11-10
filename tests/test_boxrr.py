"""Generated XROR frames only. No dataset download, real user or trained model."""
import numpy as np


def test_startup_duplicates_removed_without_hiding_interior_clock_errors():
    from abc_iueh.boxrr import remove_startup_clock_duplicates
    t = np.array([0., 0., 0., .01, .02, .02, .01])
    q = np.arange(28).reshape(7, 4)
    clean_t, clean_q, removed = remove_startup_clock_duplicates(t, q)
    assert removed == 2
    np.testing.assert_array_equal(clean_t, [0., .01, .02, .02, .01])
    np.testing.assert_array_equal(clean_q, q[[0, 3, 4, 5, 6]])
import pytest

from abc_iueh.boxrr import head_rotation
from abc_iueh.config import PipelineConfig
from abc_iueh.preprocess import prepare_session


def recording(rate=100):
    times = np.arange(25 * rate, dtype=float) / rate
    # Distinctive nonzero rotations around a fixed unit axis in source frame.
    angle = 0.7 * np.sin(1.7 * times)
    x, y, z = (np.sin(angle / 2)[:, None] * np.array([1, 2, 3]) / np.sqrt(14)).T
    w = np.cos(angle / 2)
    devices = [{"type": "CONTROLLER", "axes": ["x"]},
               {"type": "HMD", "joint": "HEAD", "axes": ["k", "1", "j", "i"]}]
    frames = np.column_stack((times, times * 0 + 13, z, w, y, x))
    return {"info": {"hardware": {"devices": devices}}, "frames": frames}, np.column_stack((w, x, y, z))


def test_decode_then_filter_and_resample():
    doc, expected = recording()
    t, q = head_rotation(doc)
    np.testing.assert_array_equal(t, doc["frames"][:, 0])
    np.testing.assert_allclose(q, expected)
    result = prepare_session(t, q, PipelineConfig())
    assert result.accepted and len(result.segments) == 1
    segment = result.segments[0]
    assert segment.quaternion.dtype == np.float32
    np.testing.assert_allclose(np.diff(segment.timestamps), .01, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(segment.quaternion, axis=1), 1., atol=1e-6)
    assert segment.quaternion[:400].shape == (400, 4)
    assert segment.quaternion[400:420].shape == (20, 4)


def test_native_rate_checked_before_resampling():
    doc, _ = recording(60)
    assert prepare_session(*head_rotation(doc), PipelineConfig()).reason == "native_rate_out_of_range"


@pytest.mark.parametrize("fault", ["missing_device", "multiple_heads", "duplicate_axis", "width"])
def test_ambiguous_layout_rejected(fault):
    doc, _ = recording()
    devices = doc["info"]["hardware"]["devices"]
    if fault == "missing_device":
        devices.pop()
    elif fault == "multiple_heads":
        devices.append(devices[-1])
    elif fault == "duplicate_axis":
        devices[-1]["axes"] = ["1", "i", "i", "k"]
    else:
        doc["frames"] = doc["frames"][:, :-1]
    with pytest.raises(ValueError):
        head_rotation(doc)
