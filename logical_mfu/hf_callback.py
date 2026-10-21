"""Per-step MFU/HFU logging for a Hugging Face ``Trainer`` (LLaMA-Factory uses one too).

Usage::

    meter = MFUMeter(lm=LMSpec.from_hf_config(cfg), train=TrainSpec(llm="lora", lora_rank=16),
                     device="H800", precision="bf16")
    trainer = MyTrainer(..., callbacks=[MFUCallback(meter)])

    class MyTrainer(Trainer):
        def training_step(self, model, inputs, *a, **kw):
            meter.observe(inputs)          # every micro-batch, before forward
            return super().training_step(model, inputs, *a, **kw)

Experimental, pure data parallelism only: each rank must observe a distinct batch.
TP/PP accounting is unsupported. The callback records metrics in state.log_history;
it does not invoke Trainer.log or external logger integrations. Depending on Trainer
version, data fetch can occur outside this step window. observe() may copy tensor
statistics to the host inside the timed window; use explicit offline inputs for
controlled comparisons until the proposed capture protocol is implemented.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from .devices import peak_flops
from .flops import StepFlops, step_flops
from .runtime import batch_stats
from .spec import LMSpec, TrainSpec, VisionSpec

try:
    from transformers import TrainerCallback
except Exception:  # transformers is optional for the core math
    TrainerCallback = object  # type: ignore


class MFUMeter:
    def __init__(self, lm: LMSpec, train: TrainSpec, device: str, precision: str = "bf16",
                 vision: Optional[VisionSpec] = None, recompute: str = "none",
                 lm_head_on_loss_tokens: bool = False):
        self.lm, self.train, self.vision = lm, train, vision
        self.peak = peak_flops(device, precision)
        self.recompute = recompute
        self.lm_head_on_loss_tokens = lm_head_on_loss_tokens
        self.reset()

    def reset(self) -> None:
        self.acc = StepFlops()
        self.tokens = 0

    def observe(self, batch: Mapping[str, Any]) -> None:
        s = batch_stats(batch)
        self.tokens += s.tokens
        self.acc += step_flops(
            self.lm, self.train, s.seg_lens, s.image_patches, self.vision,
            lm_head_tokens=s.loss_tokens if self.lm_head_on_loss_tokens else None,
            recompute=self.recompute,
        )

    def collect(self, step_time_s: float) -> dict:
        model, hw, tokens = self.acc.model_flops, self.acc.hardware_flops, float(self.tokens)
        world = 1
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
                t = torch.tensor([model, hw, tokens], dtype=torch.float64,
                                 device="cuda" if torch.cuda.is_available() else "cpu")
                dist.all_reduce(t)
                model, hw, tokens = t.tolist()
        except ImportError:
            pass
        denom = step_time_s * world * self.peak
        self.reset()
        return {
            "mfu": model / denom,
            "hfu": hw / denom,
            "tokens_per_s_per_gpu": tokens / step_time_s / world,
            "model_tflops_per_gpu": model / step_time_s / world / 1e12,
        }


class MFUCallback(TrainerCallback):
    """Times each optimizer step with wall clock (after a CUDA sync) and logs MFU/HFU."""

    def __init__(self, meter: MFUMeter, warmup_steps: int = 5):
        self.meter, self.warmup = meter, warmup_steps
        self._t0 = None

    @staticmethod
    def _sync():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def on_step_begin(self, args, state, control, **kw):
        self._sync()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, logs=None, **kw):
        self._sync()
        dt = time.perf_counter() - self._t0
        stats = self.meter.collect(dt)
        if state.global_step > self.warmup and state.is_world_process_zero:
            state.log_history.append({"step": state.global_step, **{f"perf/{k}": v for k, v in stats.items()}})
