# MFU from first principles: a kernel-invariant numerator

Code: [`tools/logical_mfu`](https://github.com/vInThePrism/logical-mfu)

## The definition, and what it implies

```
MFU = model FLOPs per step / (step time × number of GPUs × dense peak FLOP/s per GPU)
```

Fix the model, the data and the training semantics (what is trainable, how loss is computed).
Then the numerator is fixed. Fix the hardware and the precision, and the denominator is fixed.
**So the only thing that should move MFU is step time.**

That gives a rule for any MFU tool:

> A change in the physical execution path must not change the logical FLOPs numerator.

FA2 → FA3 → FA4, a per-expert loop → grouped GEMM, unfused → fused LoRA, full logits → Cut
Cross-Entropy, recomputation on → off: none of these changes the work the model asks for. If the
numerator moves when you swap one of them, you are measuring the meter, not the optimization.

## Why the usual estimators drift

| Estimator | What goes wrong |
|---|---|
| `6 × N × tokens` with one N for the whole model | Assumes every layer is dense and fully trained. Wrong for MoE (counts all experts), LoRA (counts weight gradients that never run), frozen vision towers (counts backward that never runs). |
| Operator counters (`FlopCounterMode`, profiler FLOPs) | Count what *runs*, so they change with the backend. A custom fused kernel the counter does not know is counted as 0. Recomputed forward passes are counted as useful work. |
| `(Σ L)²` for attention on packed rows | Three 2K documents packed in one 6K row cost 3 × 2K², not 6K². Using the row length over-counts 3×. |

Operator counting is still useful, but as a **cross-check**: a small batch, activation checkpointing
off, on the reference (unfused) path. It should agree with the formula. If it does not, one of them is
wrong, and you find out before the number goes on a dashboard.

## The method: three terms per module

Every module with a weight does up to three matmuls per token:

| Term | FLOPs (linear `in → out`) | Runs when |
|---|---|---|
| forward | `2 · in · out` | always |
| input gradient (dgrad) | `2 · in · out` | something **upstream** of this module is trainable |
| weight gradient (wgrad) | `2 · in · out` | this module's weight is trainable |

Attention's QKᵀ and PV have no weights. Forward is `4 · pairs · q_dim` (two matmuls, 2 FLOPs per
multiply-add). Backward is 2× forward (dQ and dK from dS; dV and dP), and it runs only if something
upstream is trainable.

"6N" is just the case where all three terms run everywhere. The other cases fall out of the same rules:

| Situation | Cost per weight | Why |
|---|---|---|
| Full fine-tuning | 6N | all three terms |
| LoRA: frozen base weight | 4N (2N in the first layer's q/k/v) | no wgrad, but the gradient must still flow down to the adapters below |
| LoRA adapter (rank r) | ~6 · r · (in + out) | small, but real model work |
| Frozen vision tower under the LLM | 2N | nothing below it needs a gradient |
| Trainable merger on a frozen tower | first linear 4N, second 6N | the first linear has nothing trainable upstream; the second has the first |
| MoE | 6 × *active* N | dropless routing gives exactly T × top-k token-expert pairs, however they are executed |

One detail that is easy to get wrong: **parallel modules are not upstream of each other.** A LoRA
adapter on `q_proj` does not make `k_proj` need an input gradient. The tool processes q/k/v (and
gate/up) as parallel groups for this reason.

## Inputs: config (static) + batch (runtime)

- **Static, about 15 config fields:** hidden size, layers, heads, KV heads, head dim, intermediate
  size, vocab, experts, top-k, MoE intermediate size, shared experts, dense-layer pattern, tied
  embeddings; and the vision tower's size, window size and merge size.
- **Runtime, per micro-batch:** real (non-padding) tokens, document lengths after packing
  (from `cu_seqlens`, `position_ids` resets, or a document-id attention mask), and image patch counts
  (from `image_grid_thw`).

Each rank counts its own micro-batches and the totals are summed across ranks at log time. With
data-parallel imbalance (one rank gets the long documents), this is still exact.

## Conventions you must choose once, and then freeze

- Causal attention: exact `L(L+1)/2` pairs, `L²/2`, or the full `L²` (Megatron's convention).
  The tool defaults to exact.
- LM head: all positions (the usual convention), or only positions with a loss.

Either choice is fine. Changing it between two runs is not.

## HFU: the same denominator, a different numerator

HFU counts what the hardware really executes, including recomputation:

```
HFU numerator = model FLOPs + recomputed forward FLOPs
```

With full activation checkpointing a layer runs forward twice: 8N instead of 6N. So HFU ≥ MFU.
MFU tells you how much useful work you get; HFU tells you how busy the hardware is. A large gap means
compute is going into recomputation. Shrinking recompute (selective recompute, CP/SP to cut
activation memory) shows up as the gap closing.

## The denominator

Use the vendor's **dense** peak for the precision you train in: BF16 dense for BF16 training. Not
the 2:4-sparse number, which is 2× larger. H100/H800 SXM: ~989 TFLOP/s BF16 dense.

You can also measure an *achievable* peak with a large GEMM microbenchmark. It is a useful ceiling,
but report it next to MFU, not inside it, so your numbers stay comparable with papers and other teams.

## The sanity check

If the numerator is really fixed, **the MFU ratio equals the throughput ratio**:

| Change | Step time | MFU |
|---|---|---|
| MoE SFT, kernel + overlap work | 14.4 s → 7.2 s (2.00×) | 10.2% → 20.4% (2.00×) |
| ZeRO-3 → Megatron-Core, 32K context | 17.0 s → 10.0 s (1.70×) | 24.0% → 40.8% (1.70×) |
| Triton RMSNorm → CuTe DSL RMSNorm, + Cut Cross-Entropy | throughput +16.5% (1.165×) | 38.3% → 44.6% (1.165×) |

The second row is the strong test: the whole training backend changed, and the numerator did not.

## In one sentence

MFU measures the model, not the kernels: count the work once, from the model and the data, and let
step time be the only thing that moves the number.

## Counter coverage and measurement

See [operator-counter validation](operator-counter-validation.md), the [proposed measurement protocol](design/measurement-protocol.md), and the [training measurement methodology](https://github.com/vInThePrism/training-perf-field-notes/blob/main/docs/measurement-methodology.md).
