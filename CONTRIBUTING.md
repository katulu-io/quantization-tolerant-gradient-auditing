# Contributing

This is a small, single-purpose artifact: a standalone, reproducible demo
accompanying the FLTA 2026 paper "Quantization-Tolerant Gradient Auditing for Active Inference-Attack Detection in Federated Learning". The scope is kept
deliberately narrow - it reproduces the method, it is not the production system.
Contributions that improve clarity, correctness, or reproducibility are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, torch, pytest
pip install matplotlib                  # optional, only for --plot
```

## Run the checks

```bash
pytest -q                               # the smoke test (6 tests)
python inference_audit_demo.py          # the audit table (exits non-zero on unexpected roles)
python inference_audit_demo.py --sweep  # the operating-conditions sweep
```

CI (`.github/workflows/ci.yml`) runs the test, the demo, and the sweep on every
push and pull request; please keep it green.

## Guidelines

- **Keep it standalone.** The detector primitives are inlined on purpose so the
  demo depends only on `numpy` and `torch`, with no proprietary-data
  dependency. Do not add an import on the parent project.
- **Keep it deterministic.** `run_demo` is seeded; tests assert exact roles and
  the operating conditions contrast. If you change the model, scales, or thresholds, update
  the expected values and the README/`docs/` output blocks together.
- **Match the surrounding style.** Plain, typed Python; concise docstrings;
  no new runtime dependencies for the core demo (`matplotlib` stays optional and
  import-guarded).
- **Document behaviour changes.** New flags or behaviours belong in
  [`docs/usage.md`](docs/usage.md); changes to the mechanism belong in
  [`docs/method.md`](docs/method.md); changes to the boundary belong in
  [`docs/limitations.md`](docs/limitations.md).

## Reporting issues

Please include the exact command, the printed table or sweep, your Python and
torch versions, and the OS/architecture (the quantized checksum is designed to
be robust across architectures, so cross-platform reports are useful).
