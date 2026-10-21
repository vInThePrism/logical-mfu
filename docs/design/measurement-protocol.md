# Proposed measurement protocol

**Status: Proposed.** This is a design input adapted from an earlier measurement-tool memo. The implementation of that tool is not included. The current optional HF callback does not implement this complete protocol.

## Logical-work contract

Use model geometry, training state, and the actual optimizer-step workload. Capture every gradient-accumulation micro-batch at the model boundary, after collation. Declare whether attention isolates packed segments; position resets alone do not prove isolation. Separate physical padded execution, logical nonpadding work, and diagnostic token counts explicitly. Do not infer a numerator from labels alone: masked labels do not imply absent forward work.

Count a logical reference for fused implementations under an invariant formula version. Treat recomputation as modeled physical overhead, separate from MFU. Report unsupported model structures rather than silently applying a dense-model estimate. Generic tensor/pipeline-parallel accounting requires a distinct per-rank work model.

## Timing contract

Open a host-clock measurement before fetching the first micro-batch in the optimizer-step group. Close it after optimizer and scheduler work and before evaluation, logging, and checkpointing. Synchronize the relevant GPU work before the end timestamp. Capture references within the window; resolve device-dependent statistics after timing closes. A callback that starts at HF `on_step_begin` may omit prefetch time.

Steady-state and end-to-end metrics answer different questions. End-to-end time includes startup, evaluation, checkpoint persistence, final drain, and other declared workload costs. Compare equivalent durable checkpoint payloads and recovery behavior.

## Aggregation

Retain every raw step and mark warm-up or other explicitly excluded windows. Headline MFU is `sum(step_flops) / (sum(step_seconds) * number_of_gpus * dense_peak_flops_per_gpu)`. Do not replace this with a median or unweighted mean of ratios. Quantiles diagnose jitter. Automatic outlier rejection can hide real stalls.

## Reporting and experiment controls

A future report should retain integer FLOPs, seconds, ratio and percentage views, raw step records, formula/version assumptions, unsupported operations, and optional end-to-end totals. Record exact GPU SKU/count/precision, dense peak, model revision, full training or LoRA mode, dataset, seed, batch/accumulation, packing contract, software environment, evaluation/checkpoint workload, and timing exclusions.

Historical design inputs reported a −4.25% difference between median and time-weighted aggregation for one packed workload. No raw data for that comparison is bundled. Hardware peaks and all model-specific formulas must be independently checked for the target execution mode.
