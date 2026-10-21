"""CLI: python -m logical_mfu --config config.json --seq 4096x8 --step-time 2.1 --gpus 8 --device H800"""
from __future__ import annotations

import argparse
import json

from . import LMSpec, TrainSpec, mfu, peak_flops, step_flops


def _seqs(s: str):
    out = []
    for part in s.split(","):
        if "x" in part:
            L, n = part.split("x")
            out += [int(L)] * int(n)
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Hugging Face config.json")
    ap.add_argument("--seq", required=True, help="document lengths per step, e.g. 4096x8 or 2048,1024,512")
    ap.add_argument("--step-time", type=float, help="seconds per optimizer step")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--device", default="H100")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--train", default="full", choices=["full", "lora", "frozen"])
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--recompute", default="none", choices=["none", "selective", "full"])
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    cfg = cfg.get("text_config", cfg)
    lm = LMSpec.from_hf_config(cfg)
    f = step_flops(lm, TrainSpec(llm=a.train, lora_rank=a.lora_rank), _seqs(a.seq), recompute=a.recompute)
    print(f"params: {lm.num_params()/1e9:.2f}B total, {lm.num_params(active_only=True)/1e9:.2f}B active")
    print(f.table())
    if a.step_time:
        p = peak_flops(a.device, a.precision)
        print(f"MFU = {mfu(f.model_flops, a.step_time, a.gpus, p):.1%}   "
              f"HFU = {mfu(f.hardware_flops, a.step_time, a.gpus, p):.1%}   "
              f"(peak {p/1e12:.0f} TFLOP/s dense {a.precision})")


if __name__ == "__main__":
    main()
