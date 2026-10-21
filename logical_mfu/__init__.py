"""Kernel-invariant MFU: count the work from the model, never from the kernels."""
from .devices import peak_flops, register_device
from .flops import StepFlops, Terms, attention_pairs, mfu, step_flops
from .runtime import BatchStats, batch_stats
from .spec import LMSpec, TrainSpec, VisionSpec

__all__ = [
    "LMSpec", "VisionSpec", "TrainSpec", "step_flops", "StepFlops", "Terms", "attention_pairs",
    "mfu", "peak_flops", "register_device", "BatchStats", "batch_stats",
]
