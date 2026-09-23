# Usage

## Install

The demo depends only on `numpy` and `torch` (CPU is fine). `pytest` is needed
for the test, `matplotlib` only for `--plot`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, torch, pytest
pip install matplotlib                  # optional, only for --plot
```

## Three ways to run

### 1. The audit table (default)

```bash
python inference_audit_demo.py
```

Runs a synthetic federation (an honest majority plus one client of each attacker
type) and prints, per client, the two-dimensional feature and the resolved role:

```
  client            truth checksum-dev  latency-flags        role             note
  --------------------------------------------------------------------------------
  honest A         honest            0              0      honest            clean
  cosine-MI      attacker            0             24   inference         detected
  active-GAN     attacker           24             24   inference         detected
  byzantine      attacker           24              0   byzantine         detected
  offline-dec    attacker            0              0      honest EVADES (boundary)
```

The process exits non-zero if any role is unexpected, so the table doubles as a
self-check.

### 2. The operating conditions sweep

```bash
python inference_audit_demo.py --sweep
```

Reproduces the paper's operating conditions finding on an honest-only baseline: the
latency channel is robust while the honest round dominates the wide-area jitter,
and degrades once the jitter approaches the round. It contrasts a long-round
federation (~600 ms) with a short-round one (~110 ms):

```
    WAN jitter   long round ~600 ms    short round ~110 ms
          0 ms                 0.00                   0.00
         50 ms                 0.00                   0.21
        100 ms                 0.00                   0.37
```

(honest false-positive rate; the short round degrades far sooner.)

### 3. The feature-space figure

```bash
pip install matplotlib
python inference_audit_demo.py --plot          # writes inference_audit_feature_space.png
```

A scatter of the clients in the (latency-anomaly, checksum-deviation) plane,
coloured by resolved role. The figure is git-ignored (regenerate on demand).

## Command-line reference

| flag | default | meaning |
|---|---|---|
| `--seed` | `0` | RNG seed (the demo is deterministic per seed) |
| `--rounds` | `24` | number of audit rounds |
| `--kappa` | `2.5` | latency z-score threshold (higher = fewer latency flags) |
| `--probe-ms` | `450` | extra in-band analytic time of a thorough attacker, ms |
| `--jitter-ms` | `0` | modelled wide-area jitter added to every client, ms |
| `--sweep` | off | run the operating-conditions jitter sweep instead of the table |
| `--plot` | off | also save the feature-space figure (needs matplotlib) |
| `--plot-path` | `inference_audit_feature_space.png` | output path for `--plot` |

Examples:

```bash
# push the cohort into the fragile operating conditions: a short round with heavy jitter
python inference_audit_demo.py --kappa 2.5 --jitter-ms 150

# a cheaper attacker (shorter analytic pass) is harder to catch on latency
python inference_audit_demo.py --probe-ms 80
```

## Interpreting the output

- **checksum-dev** - rounds in which the client's quantized challenge gradient
  exceeded the mismatch tolerance. Non-zero means the client mutated the model.
- **latency-flags** - rounds in which the client's modified z-score exceeded
  `kappa`. Two or more means a persistent latency anomaly.
- **role** - the resolved class: `honest`, `byzantine`, or `inference`.
- **note** - `detected` / `clean` / `EVADES (boundary)` / `MISMATCH`. An
  attacker resolving to `honest` is an evasion, by design (see
  [limitations.md](limitations.md)).

## Test

```bash
pytest -q
```

Six tests assert the roles, the passive-attacker-via-latency claim, the
offline-decoupled evasion boundary, and the long-robust / short-fragile operating conditions
contrast.

## Docker (no network, no data)

```bash
docker build -t inference-audit-demo .
docker run --rm --network none inference-audit-demo            # the demo
docker run --rm --network none inference-audit-demo --sweep    # the sweep
docker run --rm --network none inference-audit-demo pytest -q  # the test
```

## Continuous integration

`.github/workflows/ci.yml` installs CPU PyTorch and runs the test, the demo, and
the sweep on every push and pull request.
