# Proposed runtime capture

**Status: Proposed.** This specifies future integration behavior; it is not documentation of a shipped injection module or CLI. The existing HF callback is experimental and uses a narrower timing/capture approach.

## Capture at the model boundary

A root forward pre-hook with keyword support can observe the batch after collation and packing. Retain the required tensor references, not derived counts, while timing is open. Resolve lengths after timing closes; `.item()`, `.tolist()`, tensor-to-Python conversion, and device reductions can perturb the measured execution.

Collect every micro-batch in a gradient-accumulation group into one optimizer-step record. Gate capture by training/timing state so evaluation and calibration forwards do not leak into it. Reference retention also has a memory cost that an implementation must measure.

## Declare packing semantics

Potential metadata include explicit cumulative lengths, segment-ID masks, binary masks, and position IDs. Their meaning depends on the collator and attention backend. In isolated packing, lengths are attention-isolated segments; in packing that allows attention across the row, position resets must not split the attention workload. A 32k row containing 16 isolated equal segments has roughly one sixteenth the quadratic pair count of a single 32k sequence.

Binary masks count valid tokens; segment-ID masks require grouping rather than summing numeric IDs. Multi-axis position IDs in vision-language models need a model-specific contract; do not assume every axis has identical boundaries. Padded, logical nonpadding, and loss-supervised counts should remain distinct diagnostics with documented interpretation.

## Timing integration requirements

Open timing before the first batch request and close after optimizer/scheduler work. HF callback events may occur after accumulation-group prefetch; therefore integration must inspect the supported trainer version and select a fetch boundary explicitly. If that boundary is unsupported, report the limitation. Avoid import-time monkeypatching as an undocumented default.

## Proposed checks

- Captured micro-batch count matches actual accumulation for that step, allowing an explicitly recorded short final group.
- Segment lengths match the declared attention isolation and actual executed batch.
- Nonpadding tokens never exceed physical padded slots.
- Evaluation forwards do not enter training records.
- Statistics resolution does not perform synchronization inside the timing window.
- Per-rank accounting is explicit; rank-zero shapes do not establish global work under TP/PP or imbalanced data parallelism.

The design input recorded one first-batch stall of 0.467 seconds versus 0.0046 seconds steady state. The associated implementation and trace are not bundled; this is motivation for including fetch time, not a benchmark for this package.
