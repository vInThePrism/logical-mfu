# Logical MFU

**Explainable FLOPs accounting for LLM training, with a kernel-invariant MFU numerator.**

[Quick start](#quick-start) · [What it counts](#what-it-counts) · [Conventions](#conventions-and-limits) · [Integration status](#integration-status)

A change of attention backend or kernel implementation should not redefine the useful work assigned to the same model and batch. Logical MFU computes that work from model geometry, training state, and sequence lengths, then reports a component-level breakdown.

```text
Model geometry + training state + actual attention segments
                            ↓
           forward / input-gradient / weight-gradient FLOPs
                            ↓
MFU = logical model FLOPs / (step seconds × GPUs × dense peak FLOP/s)
```

The accounting core is pure Python. You do not need a GPU to explore the formulas.

## Quick start

```bash
git clone https://github.com/vInThePrism/logical-mfu.git
cd logical-mfu
python -m pip install -e .
```

Try a small, explicit model specification:

```python
from logical_mfu import LMSpec, TrainSpec, step_flops

model = LMSpec(
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    head_dim=64,
    vocab_size=4096,
    intermediate_size=768,
)
result = step_flops(model, TrainSpec(llm="full"), seg_lens=[128, 96])
print(result.table())
```

The output separates forward, input-gradient, and weight-gradient work for modeled components. Here `[128, 96]` describes two attention-isolated sequences, not merely two spans separated by a position-ID reset.

For a supported Hugging Face model configuration:

```bash
python -m logical_mfu \
  --config path/to/config.json \
  --seq 8192x4 \
  --step-time 3.0 --gpus 8 --device H800
```

Replace the config path and measurements with your own. The lengths must represent the same global optimizer-step workload as the timing and GPU count. The CLI example is not a benchmark result.

## What it counts

| Component | Accounting rule |
|---|---|
| Linear forward | Count the modeled matrix multiplication |
| Input gradient | Count it when the modeled upstream computation needs a gradient |
| Weight gradient | Count it when the weight is trainable |
| Attention core | Derive query/key pair counts from attention segments and the selected causal convention |
| MoE experts | Count active token-expert assignments under the supported dropless model |
| Vision tower and merger | Use explicit geometry, patch counts, and trainable state through the Python API |

LoRA, frozen modules, and packed sequences require different accounting from a blanket `6 × parameters × tokens`. See [the derivation](docs/mfu-from-first-principles.md).

## Conventions and limits

- Keep the causal-pair and LM-head conventions fixed when comparing runs.
- Use a dense hardware peak for the stated device and precision; verify the exact SKU.
- Norms, biases, activations, RoPE, softmax, and top-k are outside this model's FLOPs scope.
- MLA, encoder-decoder architectures, arbitrary computation graphs, and generic TP/PP accounting are not supported.
- Packing metadata must describe actual attention isolation. Position-ID resets alone do not establish that contract. The helper rejects 3-D multimodal position IDs unless explicit segment boundaries or a document-id mask is provided.

The returned `hardware_flops` adds modeled activation recomputation to logical model FLOPs. Its corresponding HFU is an **estimate under that model**, not a hardware-counter measurement of all executed instructions.

## Integration status

The repository includes a CLI, Python API, existing formula tests, and an optional Hugging Face callback. Treat the callback as experimental: its timing boundary and tensor-to-host statistics collection need to be reconciled with a stricter measurement protocol before making end-to-end claims.

The [measurement protocol](docs/design/measurement-protocol.md) and [runtime capture design](docs/design/runtime-capture.md) describe proposed improvements. They are not a list of completed features.

To run the included formula tests:

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

## Related work and contributions

[Training Performance Field Notes](https://github.com/vInThePrism/training-perf-field-notes) explains measurement failures in context. [AI Infra Notebook](https://github.com/vInThePrism/ai-infra-notebook) develops the broader system concepts.

For a new model adapter, document its geometry, trainable-state assumptions, FLOPs convention, and unsupported operations. Counterexamples and small independently checkable calculations are especially useful.

Code is MIT-licensed. Imported explanatory notes retain their source licenses, including CC BY 4.0 where applicable.

## Continue learning and contributing

See [where new work belongs](docs/content-routing.md), [contribution guidance](CONTRIBUTING.md), [release validation](VALIDATION.md), and the [license and source map](LICENSES.md).
