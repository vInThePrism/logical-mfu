# Working in logical-mfu

Purpose: Logical FLOPs accounting.

logical_mfu/ contains the CPU accounting core, CLI, runtime batch helper, and experimental HF callback. docs/design/ is proposed work. Generic TP/PP integration and hardware-counter HFU are unsupported.

## Commands

Run from the repository root unless a command says otherwise.

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

## Editing rules

- Keep changes focused on the request. Do not add test files, test dependencies, or scaffolding unless requested or required by an existing build. Run relevant existing checks.
- Keep README commands and relative links aligned with the actual files.
- Report measured, estimated, source-reported, and unverified results accurately. Include environment and assumptions with performance claims.
- Preserve source copyright and license provenance; consult LICENSES.md before moving or adapting material.
- Never commit credentials, private paths, datasets, raw internal logs, or generated caches.
- Use docs/content-routing.md to keep the four companion repositories distinct.
