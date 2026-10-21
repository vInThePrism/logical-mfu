# logical-mfu: kernel-invariant MFU / HFU

Count the work from the model, never from the kernels.

```
MFU = logical model FLOPs / (step time × #GPUs × dense peak for the training precision)
```

The numerator comes from ~15 config fields plus runtime counts per micro-batch (real tokens,
document lengths after packing, image patches). It is split into **forward / input-gradient /
weight-gradient per module and per trainable state**, so dense, MoE, LoRA, frozen towers and
packed sequences are modeled under the documented assumptions, and no backend change (FlashAttention version, grouped GEMM, fused
LoRA, Cut Cross-Entropy, recompute) can move it. Background: [notes/01](mfu-from-first-principles.md).

Pure Python, no dependencies for the math. `torch`/`transformers` only for the Trainer callback.

## CLI

```bash
pip install -e .
python -m logical_mfu --config path/to/config.json --seq 8192x4 --step-time 3.0 --gpus 8 --device H800
```

```
params: 30.53B total, 3.35B active
component                      fwd       dgrad       wgrad       total
llm.qkv                     32.99T      32.99T      32.99T      98.96T
llm.attn_core              105.57T     211.13T       0.00T     316.70T
...
MODEL (MFU)                                                    914.71T
```

## Python

```python
from logical_mfu import LMSpec, TrainSpec, VisionSpec, step_flops, mfu, peak_flops, batch_stats

lm = LMSpec.from_hf_config(cfg)                      # cfg = json.load(open("config.json"))
train = TrainSpec(llm="lora", lora_rank=16)          # or "full" / "frozen"; vision="frozen", merger="full"
s = batch_stats(batch)                               # cu_seqlens / position_ids / attention_mask, image_grid_thw
f = step_flops(lm, train, s.seg_lens, s.image_patches, vision=None, recompute="full")
print(f.table())
print(mfu(f.model_flops, step_time, n_gpus, peak_flops("H800", "bf16")))
```

Hugging Face `Trainer`: see [`hf_callback.py`](../logical_mfu/hf_callback.py) (`MFUMeter` + `MFUCallback`). Each rank counts its
own micro-batches; totals are all-reduced at log time.

## Rules (all of them)

| Term | Runs when |
|---|---|
| forward `2·T·in·out` | always |
| input gradient `2·T·in·out` | something upstream is trainable |
| weight gradient `2·T·in·out` | this weight is trainable |
| attention core: fwd `4·pairs·q_dim`, bwd 2× fwd | bwd only if something upstream is trainable |

Parallel modules (q/k/v, gate/up, routed vs shared experts) are not upstream of each other.

## Tests

```bash
pytest tests    # 13 tests: published param counts, 6N, MoE active params, LoRA 4N, packing,
                # layout invariance, frozen tower, window attention, HFU, MFU ratio = throughput ratio
```

## Conventions (choose once, then freeze)

- `causal_count`: `"exact"` (default, L(L+1)/2), `"half"`, or `"full"` (Megatron's L²).
- LM head on all positions (default) or only loss positions (`lm_head_tokens=`).
- Peaks in `devices.py` are datasheet dense numbers for SXM parts; add yours with `register_device`.

## Not modeled

Norms, biases, activations, RoPE, softmax and the router's top-k (all O(T·d), not O(T·d²));
MLA attention; encoder-decoder models. Add them as new module types if you need them.

## Runtime capture limits

Explicit cu_seqlens and document-id masks take priority over position resets. Three-dimensional multimodal position IDs require explicit segment boundaries; repeated temporal image coordinates cannot identify documents. Two-dimensional resets may be used only when the training attention actually isolates those segments. The experimental callback supports pure data parallelism, may omit data fetch time, and records metrics in state.log_history rather than Trainer logging integrations. See the root README and proposed protocol before comparing complete jobs.
