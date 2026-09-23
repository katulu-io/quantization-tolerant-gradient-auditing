# Limitations and the evasion surface

The audit raises the cost of passive inference, it does not abolish it. Stating
that boundary honestly is part of the paper, and the demo reproduces one
instance of it directly. Two boundaries matter: the **evasion surface** (attacks
that pass the audit) and the **operating conditions** (under which the
latency channel is reliable).

## The evasion surface: four ways to pass

Both audit signals are *in-band* - they only see the synchronous round. An
adversary who moves the inference out of the round, or separates the audited
computation from the attacked one, is not observed. There are four such bypasses:

| evasion | why it passes |
|---|---|
| **offline-decoupled** | returns an honest update and checksum in-round, then mines the saved models on a separate thread. Presents neither signal. **Demonstrated in the demo** (the `offline-dec` client resolves to `honest`). |
| **timing-padding** | pads its honest response up to the cohort median so no latency anomaly appears. The audit only ever sees a wall-clock time, so a cohort-median padding erases the relative anomaly. |
| **clean-copy** | answers the challenge on an untouched copy of the model (reproducing the reference checksum) while tampering with a separate replica. Defeats the checksum channel. |
| **gradient-orthogonal** | perturbs its update orthogonally to the audited gradient (AgrEvader-style), leaving the challenge gradient intact. Defeats the checksum channel. |

A purely post-hoc GAN inversion on an *exported, frozen* model is the
offline-decoupled case: it never mutates the in-round model, so the checksum is
blind to it. The audit catches only the **active, training-time** GAN attack
that mutates the shared model, not a passive inversion of an exported one.

The contribution is operational: standard attack tooling runs inside the
synchronous loop, so every evasion costs the attacker its off-the-shelf scripts
and forces bespoke infrastructure (a decoupled pipeline, online median
estimation, a clean replica). In a contractually governed federation that
machinery is itself evidence of intent.

## The timing channel is not robust

Unlike the checksum, the latency channel is **not robust**. It detects an
always-on passive attacker only in clean conditions, and its honest
false-positive rate climbs steeply under realistic wide-area jitter; the
false-positive-free per-client baseline is, in turn, blind to the always-on
attacker. The detection is also lost once the honest round is short relative to
the jitter. Run the sweep to see the false-positive rate climb:

```bash
python inference_audit_demo.py --sweep
```

- **Long-round federation (~600 ms, heavy local training):** the honest
  false-positive rate stays at zero through hundreds of milliseconds of jitter.
- **Short-round federation (~110 ms, a few fast steps):** the false-positive
  rate climbs once the jitter approaches the round.

So the latency channel is scoped to federations whose honest computation
dominates the jitter - a precondition to state, not assume. The audit uses the
server-observable wall-clock time throughout. A thread-scoped work clock would
strip the network noise, but it is client-reported and therefore forgeable by
the very adversary being audited; only an attested (trusted-execution)
measurement would be both jitter-robust and trustworthy, which the paper leaves
to future work. Timing-based detection of a passive attacker is, accordingly,
not robust.

The checksum channel, by contrast, is timing-independent: it detects a
model-mutating attack regardless of the latency distribution.

## What the demo does and does not claim

- It reproduces the **mechanism** (the two-signal separation, the evasion
  boundary, the operating conditions dependence) on synthetic data.
- It does **not** reproduce the paper's industrial-dataset numbers. The
  mentioned true-positive / false-positive rates come from the evaluation in
  the paper using a proprietary industrial dataset.
- The detector primitives mirror our reference implementation. The demo
  is a faithful, standalone reproduction, not the production system.
