# Proposed integration work

**Status: Proposed; not implemented by the design documents.**

The original migration memo referred to modules that were not included with the source archives. No missing implementation is imported from an existing repository.

1. Reconcile HF timing with a boundary before accumulation-group batch fetch.
2. Resolve device-dependent batch statistics outside timed execution.
3. Require explicit packing and attention-isolation contracts.
4. Export all optimizer-step records with time-weighted aggregation and exclusions.
5. Add model/operation coverage reporting and supported distributed-accounting contracts.
6. Validate equivalent checkpoint payload, final drain, and restoration for end-to-end comparisons.
7. Compare analytic formulas against a small unfused coverage-controlled reference.

See the [proposed protocol](measurement-protocol.md) and [runtime capture](runtime-capture.md).
