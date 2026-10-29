"""Static [1,400,4]→[1,20,4] opset-17 export helper.

The helper checks the 2 GiB available-RAM gate before importing Torch. It is
provided for coordinator-scheduled export only and has not been run here.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


RAM_FLOOR_BYTES = 2 * 1024 * 1024 * 1024


def available_ram_bytes() -> int:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages * page_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="destination path for the ONNX graph")
    args = parser.parse_args()
    available = available_ram_bytes()
    if available < RAM_FLOOR_BYTES:
        raise SystemExit(f"need >=2 GiB available RAM before Torch import; found {available} bytes")

    import torch

    from ordered_motion_model import OrderedMotionProduct

    model = OrderedMotionProduct().cpu().eval()
    example = torch.zeros(1, 400, 4, dtype=torch.float32)
    example[..., 0] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (example,), str(args.output), export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["context"], output_names=["prediction"],
        dynamic_axes=None, dynamo=False,
    )


if __name__ == "__main__":
    main()

