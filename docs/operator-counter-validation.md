# Operator-counter coverage and validation

Source-reported examples from the original training field notes; original traces are not bundled. These observations motivate checks, not a claim that this package reproduced them.

## Two families of MFU tools, two different ways to be wrong

**Closed-form estimators** (the Megatron-style `6·N·tokens` family, plus an attention
term) compute FLOPs once from the model config. They assume every layer is identical.
That assumption breaks for almost every model people fine-tune today:

| Breaks the "homogeneous layers" assumption | Why |
|---|---|
| MoE | only `top_k` experts are active per token; shared experts add a dense term |
| GQA / MQA | K/V projections are narrower than Q |
| Vision-language | a ViT runs before the LLM; its cost depends on image patches, not text tokens |
| LoRA / frozen modules | frozen weights need no weight-gradient GEMM |
| Packing | attention cost is Σ sᵢ² over *segments*, not (Σ sᵢ)² over the packed row |

A public data point: one published job's MFU went from 24.5 % to 17.8 % just by
billing each layer type separately instead of assuming homogeneity.

**Operator-interception counters** (`torch.utils.flop_counter.FlopCounterMode`,
DeepSpeed's profiler) count FLOPs per dispatched op. They see real shapes, but only
for ops that pass through the torch dispatcher *and* have a registered formula. Fused
kernels — FlashAttention 3/4, cut-cross-entropy, fused LoRA, grouped-GEMM experts —
either bypass the dispatcher (Triton/CuTe kernels launched directly) or fold several
GEMMs into one custom op. The measured FLOPs collapse and MFU *falls*. I measured a
**~370× under-count** on a fused-LoRA run (14.6 TFLOP/step counted vs ~5,400 actual).
`register_flop_formula` cannot fix it: there is no dispatch event to hook.

So the same logical work gives different numbers depending on the execution path —
exactly what a utilisation metric must not do.


## How to validate it

* **Autograd probe**: on a small model (e.g. 8B + LoRA), run a few steps with
  fusion off and `FlopCounterMode` on, gradient checkpointing off; the analytic
  numerator must match within a small tolerance. This checks the dX/dW frontier.
* **Fusion A/B**: toggle a fused kernel; the per-step FLOPs must be bit-identical
  (they were, in my runs: `4,986,491,390,394,368` on both sides), so any MFU change
  comes from time alone.
* **Completeness**: hardware counters (ncu) see everything, so `HFU ≥ MFU` must hold;
  `HFU < MFU` means the analytic formula over-counts.


The complete accounting model is defined in [MFU from first principles](mfu-from-first-principles.md). Hardware-counter comparisons require matching FLOPs and peak conventions; a blanket HFU ≥ MFU assertion is not a substitute for checking those conventions.
