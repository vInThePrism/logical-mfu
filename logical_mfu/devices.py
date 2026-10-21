"""Dense peak throughput per device and precision (FLOP/s per GPU).

The MFU denominator is the vendor's **dense** peak for the precision you train in.
Not the 2:4-sparse number (2x larger), and not a measured GEMM ceiling. A measured
ceiling is useful, but report it next to MFU, not inside it, so numbers stay
comparable with papers and other teams.

Values are datasheet numbers for the SXM / standard variants. Check your exact SKU
(PCIe and NVL parts often have lower clocks) and add rows with ``register_device``.
"""
from __future__ import annotations

from typing import Dict

T = 1e12

# device -> precision -> dense FLOP/s
PEAKS: Dict[str, Dict[str, float]] = {
    "V100":     {"fp16": 125 * T},
    "A100":     {"bf16": 312 * T, "fp16": 312 * T, "tf32": 156 * T},
    "A800":     {"bf16": 312 * T, "fp16": 312 * T, "tf32": 156 * T},
    "A10":      {"bf16": 125 * T, "fp16": 125 * T},
    "L40S":     {"bf16": 362.05 * T, "fp16": 362.05 * T, "fp8": 733 * T},
    "H100":     {"bf16": 989.4 * T, "fp16": 989.4 * T, "fp8": 1978.9 * T, "tf32": 494.7 * T},
    "H800":     {"bf16": 989.4 * T, "fp16": 989.4 * T, "fp8": 1978.9 * T, "tf32": 494.7 * T},
    "H200":     {"bf16": 989.4 * T, "fp16": 989.4 * T, "fp8": 1978.9 * T},
    "H20":      {"bf16": 148 * T, "fp16": 148 * T, "fp8": 296 * T},
    "B200":     {"bf16": 2250 * T, "fp16": 2250 * T, "fp8": 4500 * T},
    "MI300X":   {"bf16": 1307.4 * T, "fp16": 1307.4 * T, "fp8": 2614.9 * T},
}

_ALIASES = {
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H800": "H800",
    "NVIDIA A100-SXM4-80GB": "A100",
    "NVIDIA A100-SXM4-40GB": "A100",
    "NVIDIA B200": "B200",
}


def register_device(name: str, **peaks_tflops: float) -> None:
    """``register_device("Ascend910B", fp16=..., bf16=...)`` with values in TFLOP/s."""
    PEAKS[name] = {k: v * T for k, v in peaks_tflops.items()}


def peak_flops(device: str, precision: str = "bf16") -> float:
    key = _ALIASES.get(device, device)
    if key not in PEAKS:
        for k in PEAKS:  # tolerate torch.cuda.get_device_name() strings
            if k.lower() in device.lower():
                key = k
                break
    try:
        return PEAKS[key][precision]
    except KeyError as e:
        raise KeyError(f"no dense {precision} peak for {device!r}; add it with register_device()") from e
