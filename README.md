# Inference-audit demo — Quantization-Tolerant Gradient Auditing

Self-contained, reproducible example accompanying the paper
**"Quantization-Tolerant Gradient Auditing for Active Inference-Attack Detection in Federated Learning"** (FLTA 2026).

It demonstrates the paper's core mechanism on synthetic data: a server-side audit
that, each round, asks every client for a checksum of the gradient on a secret
challenge batch and the time the local computation took, and reads the two
signals to separate honest, model-corrupting (Byzantine), and model-mining
clients — **without** differential-privacy noise.

The paper's honest split is reproduced here. The **checksum** is the reliable,
deployable half: a quantization-tolerant, **block-level** fingerprint (quantize
to the relative grid `q * mean(|grad|)`, hash blocks of 256 coordinates, compare
on the fraction of differing blocks against `tau = 0.05`) that detects active,
model-mutating attacks and survives benign cross-hardware floating-point
non-determinism that defeats an exact hash (run `--determinism`). The **timing**
half detects a passive attacker only in clean conditions and is **not robust**
under realistic wide-area jitter (run `--sweep`). The demo shows both, plainly.

This artifact is an independent implementation. The detector
primitives needed (the quantised-gradient mismatch, the median/MAD latency
z-score, and the two-signal role rule) are inlined in `inference_audit_demo.py`,
so the only dependencies are `numpy` and `torch` and there is no proprietary
data.

## Documentation

- [docs/method.md](docs/method.md) - how the audit works: the two signals
  (quantized-gradient checksum, median/MAD latency z-score), the role rule, the
  math, what is real vs modelled, and a function reference and paper mapping.
- [docs/usage.md](docs/usage.md) - install, the three run modes, full CLI
  reference, output interpretation, Docker, and CI.
- [docs/limitations.md](docs/limitations.md) - the evasion surface (four ways to
  pass the audit) and the operating conditions (when the latency channel is
  reliable).

## Run

```bash
pip install -r requirements.txt
python inference_audit_demo.py                # the per-client audit table
python inference_audit_demo.py --determinism  # the checksum's cross-hardware tolerance (the headline)
python inference_audit_demo.py --determinism --pair cross-isa   # model a different measured machine pair
python inference_audit_demo.py --sweep        # why the timing half is not robust (jitter sweep)
python inference_audit_demo.py --plot         # also save the feature-space figure (needs matplotlib)
```

### The determinism result (the reliable half)

```
  1,072,129 parameters -> 4,189 blocks; the server asks for a challenge-derived subsample of 256
  modelling the measured pair 'cross-both': different ISA and OS (arm64 macOS vs x86_64 Linux), block mismatch 0.0321

  client                           exact hash  block mismatch  subsampled   verdict
  ----------------------------------------------------------------------------------
  honest, same hardware                 match          0.0000      0.0000    accept
  honest, cross-both                  DIFFERS          0.0322      0.0117    accept
  active-GAN (mutates model)          DIFFERS          0.9838      0.9688      flag
  active-ascent (mutates model)       DIFFERS          0.9831      0.9648      flag
```

An **exact** hash flags the honest cross-hardware client (its raw bits differ),
so it would false-positive an honest silo. The **quantization-tolerant**
fingerprint keeps that client's block mismatch below the tolerance `tau = 0.05`
and accepts it, while two distinct active, model-mutating attacks move
essentially every block and are flagged. This is the paper's central result. On real
hardware the paper measures honest drift across four machines and finds it grows
with the distance between the two stacks: 0.0042 same architecture and OS, 0.0208
across operating systems, 0.0250 across instruction sets, 0.0321 when both differ.
`--pair` models any of them and defaults to the 0.0321 worst case, where `tau = 0.05`
has only 1.6x of headroom.

The `subsampled` column answers from 256 blocks instead of all 4,189 and reaches
the same verdicts — the **block subsampling** that keeps the response size and
the server's hashing cost independent of model size.

Useful flags: `--seed`, `--rounds`, `--kappa` (latency z-score threshold),
`--probe-ms` (a thorough attacker's extra in-band time), `--jitter-ms`
(modelled wide-area jitter added to every client).

### The audit table

```
  client            truth checksum-dev  latency-flags        role             note
  --------------------------------------------------------------------------------
  honest A         honest            0              0      honest            clean
  cosine-MI      attacker            0             24   inference         detected
  active-GAN     attacker           24             24   inference         detected
  byzantine      attacker           24              0   byzantine         detected
  offline-dec    attacker            0              0      honest EVADES (boundary)
```

| client | checksum deviation | latency anomaly | resolved role |
|--------|--------------------|-----------------|---------------|
| honest | no | no | honest |
| cosine-MI (passive membership inference) | no | yes | inference |
| active-GAN (active, model-mutating, Hitaj et al.) | yes | yes | inference |
| byzantine (state corruption) | yes | no | byzantine |
| **offline-decoupled** (mines saved models offline) | no | no | honest — **evades** |

The two active, model-mutating attacks (active-GAN here) are caught by the
**checksum**, the reliable half of the audit, and separated from the Byzantine
corruptor by the two-signal rule. The passive cosine-MI attacker is caught by the
**latency** channel in these clean conditions, but that detection is **not
robust** — see the jitter sweep below. The **offline-decoupled** attacker presents
neither signal and **evades**: the audit *raises the cost* of in-band inference,
it does not abolish it. Reporting these boundaries honestly is part of the paper.

### The operating-conditions sweep

`--sweep` shows why the timing half is **not robust**: the cohort latency
channel's honest false-positive rate climbs as wide-area jitter grows, and it
degrades sooner when the honest round is short relative to the jitter. It
contrasts a long-round federation (~600 ms, tabular) with a short-round one
(~110 ms, CIFAR-like):

```
    WAN jitter   long round ~600 ms    short round ~110 ms
          0 ms                 0.00                   0.00
         50 ms                 0.00                   0.21
        100 ms                 0.00                   0.37
```

(honest false-positive rate; the short round degrades far sooner.)

## Test

```bash
pytest -q
```

The smoke test asserts every client resolves to its expected role, that the
passive attacker is caught by latency alone, that the offline-decoupled attacker
evades, and that the operating conditions sweep shows the long-robust / short-fragile contrast.

## Reproduce in Docker (no network, no data)

```bash
docker build -t inference-audit-demo .
docker run --rm --network none inference-audit-demo            # the demo
docker run --rm --network none inference-audit-demo pytest -q  # the smoke test
```

## Notes

- The checksum deviations are **real** (quantised challenge gradients of a small
  model). The latencies are **modelled**, as in the paper's controlled latency
  experiments: the audit only ever observes a per-round time, so a behaviour's
  extra compute is added as a latency delta.
- An honest majority is used so the cohort median/MAD reflect honest behaviour —
  a concrete illustration of the paper's small-cohort MAD discussion (Section
  VI-F): too many simultaneous attackers contaminate the cohort statistic.
- This demo reproduces the method, not the paper's industrial-dataset numbers.
- The passive attacker in this demo is a stand-in. In the paper, the passive
  threat is a real, state-of-the-art attack: LiRA (Carlini et al. 2022, 256
  shadow models) reaches an online AUC of 0.85 and a 15.8% true-positive rate at 0.1% false-positive rate on real CIFAR-10. Being passive, it never touches the
  challenge gradient, so — like the `cosine-MI` client here — it **evades the
  checksum** and falls to the timing channel, which the paper shows is not robust.
  Detecting passive inference stays the domain of differential privacy.
- `--plot` needs matplotlib (`pip install matplotlib`); it is not required for
  the demo or the test.

## License & acknowledgment

Released under the Apache License 2.0 (see `LICENSE` and `NOTICE`), © 2026 Katulu GmbH.

This project is supported by the Federal Ministry for Economic Affairs and Climate Action (BMWK) on the basis of a decision by the German Bundestag.
