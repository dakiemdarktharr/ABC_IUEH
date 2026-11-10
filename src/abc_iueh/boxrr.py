"""Read head rotation from an already decoded BOXRR-23 XROR frame table.

Upstream MetaGuard/XROR stores elapsed seconds first, then each device's
declared axes. Its quaternion axes are i, j, k, 1. Keep the source frame.
This module performs no network, archive or source-file access.
"""
import numpy as np


def remove_startup_clock_duplicates(timestamps, quaternion):
    """Keep the first startup row; remove only the adjacent repeated initial clock.

    Real XROR recordings commonly start with several t=0 frames. These are
    inside the later five-second trim. Interior duplicates/backward clocks
    remain untouched so the strict session validator can reject them.
    """
    end = 1
    while end < len(timestamps) and timestamps[end] == timestamps[0]:
        end += 1
    removed = end - 1
    if removed:
        keep = np.r_[0, np.arange(end, len(timestamps))]
        return timestamps[keep], quaternion[keep], removed
    return timestamps, quaternion, 0


def head_rotation(recording):
    """Return elapsed seconds and scalar-first quaternions for one recording."""
    devices = recording["info"]["hardware"]["devices"]
    if not isinstance(devices, list):
        raise ValueError("XROR devices must be an ordered list")
    offset = 1
    columns = None
    for device in devices:
        axes = device["axes"]
        if not isinstance(axes, list) or not all(isinstance(axis, str) for axis in axes):
            raise ValueError("XROR device must declare its axes in order")
        if device.get("type") == "HMD" and device.get("joint", "HEAD") == "HEAD":
            if columns is not None:
                raise ValueError("multiple head devices in XROR recording")
            rotation_axes = ("1", "i", "j", "k")
            if any(axes.count(axis) != 1 for axis in rotation_axes):
                raise ValueError("head device must declare four distinct quaternion axes")
            columns = [offset + axes.index(axis) for axis in rotation_axes]
        offset += len(axes)
    if columns is None:
        raise ValueError("XROR recording has no head-mounted device")
    frames = np.asarray(recording["frames"])
    if frames.ndim != 2 or frames.shape[1] != offset or frames.dtype.kind not in "fiu":
        raise ValueError("decoded numeric frame width must match declared axes plus timestamp")
    return frames[:, 0].astype(np.float64), frames[:, columns].astype(np.float64)
