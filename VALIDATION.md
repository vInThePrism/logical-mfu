# Release validation

Validated on 2026-10-21 on macOS arm64 with Python 3.14.0. The isolated validation environment used pytest 9.1.1, NumPy 2.5.4, and PyTorch 2.14.1 where needed. These are the versions exercised for this release, not a compatibility matrix.

Original private datasets, raw cluster traces, and historical run logs are not bundled. Historical measurements in notes were not independently reproduced in this release.

## Checks completed

- Editable package installation succeeded.
- `python -m pytest tests -q`: **13 passed**.
- The root README Python example and CLI help execute.
- Targeted checks confirmed document-id masks take precedence over position IDs, and ambiguous 3-D position IDs raise instead of inventing segments.

## Boundaries

The optional HF callback was not exercised in a live Trainer job. It is experimental, pure-DP only, and may omit data fetch time or add statistics-transfer overhead. Generic TP/PP and hardware-counter HFU are unsupported. See the README and proposed design documents.
