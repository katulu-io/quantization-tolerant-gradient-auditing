#!/usr/bin/env python
# Copyright 2026 Katulu GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-contained demo: the challenge-response inference audit on synthetic data.

Standalone, reproducible example for the paper "Quantization-Tolerant Gradient 
Auditing for Active Inference-Attack Detection in Federated Learning". Its 
implementation is independent. The detector primitives needed are inlined below,
so the only dependencies are numpy and torch and there is no proprietary data.

A tiny synthetic federation is run for a handful of rounds. Each client exhibits
one behaviour; the server audits it with one two-dimensional feature, the
(checksum-deviation, latency-anomaly) pair:

  - honest            - evaluates the challenge faithfully, normal round time.
  - cosine-MI         - PASSIVE membership inference: does not touch the model
                        (checksum matches) but runs an extra analytic pass that
                        costs measurable time (latency anomaly).
  - active-GAN        - the active, model-mutating GAN attack of Hitaj et al.:
                        repurposes the shared model as a discriminator, so its
                        challenge gradient deviates AND it spends extra time.
  - byzantine         - corrupts the model state for non-inference reasons:
                        checksum deviates, no persistent latency cost.
  - offline-decoupled - an attacker that returns an honest update and checksum
                        in-round and mines the saved models OFFLINE: it presents
                        NEITHER signal, so it EVADES the audit. This is the
                        method's honest boundary (the paper's evasion surface).

The checksum deviations are REAL (quantised challenge gradients of a small
model); the latencies are MODELLED, as in the paper's controlled latency
experiments. ``--sweep`` reproduces the operating-conditions finding: the latency
channel degrades once wide-area jitter approaches the round time.

The fingerprint is the paper's (Section IV-A): quantize to the relative grid
``q * mean(|grad|)``, split into blocks of BLOCK_SIZE coordinates, hash each,
and compare on the fraction of differing BLOCKS against the tolerance TAU --
not on single coordinates. ``--determinism`` also exercises the challenge-derived
block subsampling that bounds the response size, and ``--pair`` selects which of
the paper's four measured machine pairs to model: honest drift grows with the
distance between the two stacks, from 0.0042 (same architecture and OS) to
0.0321 (both differ), against a tolerance of 0.05.

Run:
    pip install -r requirements.txt
    python inference_audit_demo.py            # the four-behaviour table
    python inference_audit_demo.py --sweep    # the jitter operating-conditions
    python inference_audit_demo.py --plot     # also save the feature-space figure
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import torch


# ---------------------------- detector primitives  --------------------------------
# See the paper, Section IV-A (checksum, latency z-score) and the role rule role(c).

def quantize(grad: torch.Tensor, q: float) -> torch.Tensor:
    """Round each coordinate to the relative grid ``q * mean(|grad|)``."""
    step = q * (grad.detach().abs().mean() + 1e-12)
    return torch.round(grad.detach() / step)


def block_count(grad: torch.Tensor, block_size: int = None) -> int:
    """Number of fixed-size blocks the fingerprint splits a gradient into."""
    bs = BLOCK_SIZE if block_size is None else block_size
    return max(1, int(np.ceil(grad.numel() / bs)))


def challenge_blocks(n_blocks: int, key: int, subsample: int = None) -> np.ndarray | None:
    """The challenge-derived subset of block indices the server asks for.

    Because the challenge key is already secret, the server derives from it a
    random subset of ``subsample`` blocks and requires only those, bounding the
    response and the hashing cost independently of model size. The subset is
    unpredictable for the same reason the challenge batch is, so a client can
    neither precompute its answer nor choose which coordinates are safe to
    tamper with. ``None`` means "every block" (used when the model is small
    enough that subsampling would not reduce anything)."""
    s = SUBSAMPLE if subsample is None else subsample
    if s is None or s >= n_blocks:
        return None
    return np.random.default_rng(key).choice(n_blocks, size=s, replace=False)


def fingerprint(grad: torch.Tensor, q: float, block_size: int = None,
                indices: np.ndarray | None = None) -> dict[int, str]:
    """Block-quantized fingerprint: a SHA-256 digest per block of fixed size.

    Quantizes to the relative grid, splits into blocks of ``block_size``
    coordinates and hashes each, so the comparison is over whole blocks rather
    than single coordinates. Fixing the block *size* rather than the block
    *count* keeps the honest mismatch fraction bounded as models grow, where a
    fixed count saturates. Only ``indices`` are hashed when given, which is what
    makes the subsampled response cheap."""
    bs = BLOCK_SIZE if block_size is None else block_size
    qg = quantize(grad, q).to(torch.int64).numpy()
    blocks = np.array_split(qg, block_count(grad, bs))
    idx = range(len(blocks)) if indices is None else indices
    return {int(i): hashlib.sha256(blocks[int(i)].tobytes()).hexdigest()[:16] for i in idx}


def mismatch_fraction(grad_a: torch.Tensor, grad_b: torch.Tensor, q: float,
                      block_size: int = None,
                      indices: np.ndarray | None = None) -> float:
    """Fraction of fingerprint BLOCKS that differ between two gradients.

    This is the paper's comparison (Section IV-A): a block counts as differing
    if any coordinate in it lands on a different quantization level. A client is
    checksum-flagged when this fraction exceeds the tolerance ``TAU``."""
    fa = fingerprint(grad_a, q, block_size, indices)
    fb = fingerprint(grad_b, q, block_size, indices)
    if not fa:
        return 0.0
    return sum(fa[i] != fb[i] for i in fa) / len(fa)


def modified_zscore_per_round(latencies: np.ndarray,
                              mad_floor_fraction: float = 0.1,
                              mad_floor_seconds: float = 1e-3) -> np.ndarray:
    """Per-round median+MAD modified z-score (Iglewicz/Hoaglin). The MAD is
    floored to a fraction of the median and to an absolute floor (the paper's
    tau_min = 1 ms) so tight honest cohorts do not blow up the scores."""
    scores = np.zeros_like(latencies)
    for r, row in enumerate(latencies):
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        med = np.median(row[mask])
        mad = np.median(np.abs(row[mask] - med))
        mad = max(mad, mad_floor_fraction * max(abs(med), 1e-9), mad_floor_seconds)
        if mad < 1e-12:
            continue
        scores[r, mask] = 0.6745 * (row[mask] - med) / mad
    return scores


def combine_time_correctness(deviations: np.ndarray,
                             latency_flags: np.ndarray) -> list[str]:
    """The two-signal role rule: a persistent latency anomaly (>= 2 rounds)
    maps to ``inference``; a checksum deviation without it maps to
    ``byzantine``; neither maps to ``honest``."""
    if deviations.size == 0:
        return []
    n_rounds, n_clients = deviations.shape
    min_late_rounds = 1 if n_rounds <= 1 else 2
    out: list[str] = []
    for c in range(n_clients):
        any_dev = bool(deviations[:, c].any())
        late_count = int(latency_flags[:, c].sum())
        if late_count >= min_late_rounds:
            out.append("inference")
        elif any_dev:
            out.append("byzantine")
        else:
            out.append("honest")
    return out


# --- configuration -----------------------------------------------------------
N_FEATURES = 20
CHALLENGE_SIZE = 32
Q = 1e-2            # quantisation resolution (determinism margin)
BLOCK_SIZE = 256    # coordinates per fingerprint block (paper, Section IV-A)
TAU = 0.05          # checksum tolerance (fraction of differing BLOCKS)
SUBSAMPLE = 256     # challenge-derived blocks requested per round (the paper's S)
ROUND_BASE_S = 0.60     # honest round time (a long-round tabular federation)
HONEST_JITTER_S = 0.02  # honest per-round latency jitter

# Each client: (label, behaviour). The honest majority keeps the cohort
# median/MAD honest, as in the paper's benchmark.
CLIENTS = [("honest A", "honest"), ("honest B", "honest"), ("honest C", "honest"),
           ("cosine-MI", "cosine-MI"), ("active-GAN", "active-GAN"),
           ("byzantine", "byzantine"), ("offline-dec", "offline-decoupled")]
# The detector role each behaviour SHOULD resolve to (offline-decoupled evades,
# so the audit honestly reports it as honest).
EXPECTED = {"honest": "honest", "cosine-MI": "inference", "active-GAN": "inference",
            "byzantine": "byzantine", "offline-decoupled": "honest"}
# Ground-truth attackers (for the table's "truth" column).
ATTACKERS = {"cosine-MI", "active-GAN", "byzantine", "offline-decoupled"}
# Behaviours that mutate the shared model (-> checksum deviation).
MUTATORS = {"active-GAN", "byzantine"}
# Behaviours that spend extra in-band analytic time (-> latency anomaly).
SLOW = {"cosine-MI", "active-GAN"}


def small_model(seed: int, width: int = 64) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, width), torch.nn.ReLU(),
        torch.nn.Linear(width, width), torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    )


def flat_grad(model, x, y, loss_fn) -> torch.Tensor:
    model.zero_grad()
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in grads]).detach()


def mutated_copy(model, kind: str, seed: int, rng: np.random.Generator,
                 width: int = 64) -> torch.nn.Module:
    """Return a copy of the model after the attacker's mutation, if any."""
    clone = small_model(seed, width)
    clone.load_state_dict(model.state_dict())
    if kind == "active-GAN":
        # repurpose the shared model as a GAN discriminator: push its
        # confidence on a target class for random latent inputs (model mutates).
        z = torch.tensor(rng.normal(size=(8, N_FEATURES)), dtype=torch.float32)
        target = torch.ones((8, 1))
        opt = torch.optim.SGD(clone.parameters(), lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(3):
            opt.zero_grad()
            loss_fn(clone(z), target).backward()
            opt.step()
        clone.zero_grad()
    elif kind == "active-ascent":
        # a second, distinct active attack: ascend the loss on a probe batch to
        # sharpen the model around its data and amplify the membership signal
        # (model mutates, so the checksum catches it too).
        z = torch.tensor(rng.normal(size=(8, N_FEATURES)), dtype=torch.float32)
        yb = torch.tensor((rng.random((8, 1)) > 0.5), dtype=torch.float32)
        opt = torch.optim.SGD(clone.parameters(), lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(3):
            opt.zero_grad()
            (-loss_fn(clone(z), yb)).backward()   # ASCEND (opposite of honest training)
            opt.step()
        clone.zero_grad()
    elif kind == "byzantine":
        # corrupt the state for reasons unrelated to reading the model.
        with torch.no_grad():
            for p in clone.parameters():
                p.add_(torch.tensor(rng.normal(scale=0.05, size=tuple(p.shape)),
                                    dtype=torch.float32))
    return clone


def latency(kind: str, rng: np.random.Generator, probe_s: float, jitter_s: float,
            round_base_s: float = ROUND_BASE_S) -> float:
    """Modelled per-round time the server observes (seconds). ``jitter_s`` is
    extra wide-area noise added to every client (the operating-conditions knob);
    ``round_base_s`` is the honest round time (long vs short federations)."""
    base = round_base_s + rng.normal(scale=HONEST_JITTER_S)
    if kind in SLOW:
        base += probe_s            # the extra analytic / optimisation pass costs time
    if jitter_s > 0:
        base += abs(rng.normal(scale=jitter_s))   # one-sided WAN jitter
    return float(max(base, 1e-3))


def run_demo(seed: int = 0, rounds: int = 24, kappa: float = 2.5,
             probe_s: float = 0.45, jitter_s: float = 0.0,
             round_base_s: float = ROUND_BASE_S):
    """Run the synthetic audit and return ``(clients, deviations, latency_flags,
    roles)``; arrays are ``(rounds, clients)``. Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    w = torch.tensor(rng.normal(size=(N_FEATURES, 1)), dtype=torch.float32)
    x_pool = torch.tensor(rng.normal(size=(512, N_FEATURES)), dtype=torch.float32)
    y_pool = (torch.sigmoid(x_pool @ w) > 0.5).float()
    model = small_model(seed)

    n = len(CLIENTS)
    deviations = np.zeros((rounds, n), dtype=int)
    latencies = np.zeros((rounds, n), dtype=float)

    for r in range(rounds):
        idx = rng.choice(len(x_pool), size=CHALLENGE_SIZE, replace=False)
        cx, cy = x_pool[idx], y_pool[idx]
        g_ref = flat_grad(model, cx, cy, loss_fn)   # server reference gradient
        # the same secret that picks the challenge batch picks the blocks the
        # server will ask for, so the client cannot know them in advance
        blocks = challenge_blocks(block_count(g_ref), key=(seed + 1) * 9973 + r)
        for c, (_, kind) in enumerate(CLIENTS):
            if kind in MUTATORS:
                g_client = flat_grad(mutated_copy(model, kind, seed, rng), cx, cy, loss_fn)
            else:                                    # honest / passive / offline: model untouched
                g_client = g_ref
            deviations[r, c] = int(
                mismatch_fraction(g_client, g_ref, Q, indices=blocks) > TAU)
            latencies[r, c] = latency(kind, rng, probe_s, jitter_s, round_base_s)

    z = modified_zscore_per_round(latencies)
    latency_flags = (z > kappa).astype(int)
    roles = combine_time_correctness(deviations, latency_flags)
    return CLIENTS, deviations, latency_flags, roles


def print_table(clients, deviations, latency_flags, roles, kappa) -> bool:
    n = len(clients)
    print(f"\nChallenge-response audit on synthetic data "
          f"({n} clients, {deviations.shape[0]} rounds, kappa={kappa}, "
          f"tau={TAU} of blocks)\n")
    print(f"  {'client':<12} {'truth':>10} {'checksum-dev':>12} {'latency-flags':>14} "
          f"{'role':>11} {'note':>16}")
    print("  " + "-" * 80)
    ok = True
    for c, (label, kind) in enumerate(clients):
        dev = int(deviations[:, c].sum())
        late = int(latency_flags[:, c].sum())
        role = roles[c]
        truth = "attacker" if kind in ATTACKERS else "honest"
        ok = ok and role == EXPECTED[kind]
        if kind in ATTACKERS and role == "honest":
            note = "EVADES (boundary)"
        elif role == EXPECTED[kind]:
            note = "detected" if kind in ATTACKERS else "clean"
        else:
            note = "MISMATCH"
        print(f"  {label:<12} {truth:>10} {dev:>12} {late:>14} {role:>11} {note:>16}")
    print()
    print("  Reading: a checksum deviation -> the model was mutated (active attack);")
    print("  a persistent latency anomaly -> extra in-band computation. The CHECKSUM")
    print("  is the reliable half (it flags the active-GAN attack here, and run")
    print("  --determinism for the cross-hardware tolerance that makes it deployable).")
    print("  The latency half detects the passive cosine-MI attacker in these clean")
    print("  conditions, but that detection is NOT robust (run --sweep). The offline-")
    print("  decoupled attacker presents neither signal and EVADES.")
    print(f"\n  result: {'roles as expected' if ok else 'UNEXPECTED roles'}\n")
    return ok


def _honest_fpr(seeds, jitter_s, round_base_s, kappa, n_clients=6, rounds=24):
    """Honest-only false-positive rate (the paper's honest baseline): an
    all-honest cohort, fraction of clients latency-flagged in >= 2 rounds (the
    persistence threshold the role rule uses). No attackers, so the cohort
    median/MAD is not skewed -- this isolates the jitter effect on honest
    clients."""
    vals = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        lat = np.array([[latency("honest", rng, 0.0, jitter_s, round_base_s)
                         for _ in range(n_clients)] for _ in range(rounds)])
        flags = (modified_zscore_per_round(lat) > kappa).astype(int)
        vals.append(float(np.mean(flags.sum(axis=0) >= 2)))
    return float(np.mean(vals))


def jitter_sweep(seeds=(0, 1, 2, 3), jitters_ms=(0, 25, 50, 100, 200, 400), kappa=2.5):
    """Reproduce the operating-conditions / modality finding: the latency channel is
    robust while the honest round dominates the wide-area jitter, and degrades
    once jitter approaches the round. We contrast a long-round federation
    (~600 ms, tabular) with a short-round one (~110 ms, CIFAR-like): the honest
    false-positive rate climbs much sooner for the short round."""
    long_s, short_s = 0.60, 0.11
    print(f"\nOperating-conditions sweep (honest false-positive rate, kappa={kappa}, "
          f"{len(seeds)} seeds)\n")
    print(f"  {'WAN jitter':>12} {'long round ~600 ms':>20} {'short round ~110 ms':>22}")
    print("  " + "-" * 56)
    for j in jitters_ms:
        fpr_long = _honest_fpr(seeds, j / 1000.0, long_s, kappa)
        fpr_short = _honest_fpr(seeds, j / 1000.0, short_s, kappa)
        print(f"  {j:>9d} ms {fpr_long:>20.2f} {fpr_short:>22.2f}")
    print("\n  The long-round federation tolerates wide-area jitter; the short-round")
    print("  one degrades once jitter approaches the round. The latency channel is")
    print("  therefore scoped to federations whose honest computation dominates the")
    print("  jitter -- a precondition to state, not assume (paper, Sections VI-E/F).\n")


def save_plot(clients, deviations, latency_flags, roles, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("  [--plot] matplotlib not installed; run `pip install matplotlib`.")
        return
    colours = {"honest": "#188038", "inference": "#c5221f", "byzantine": "#1a73e8"}
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for c, (label, kind) in enumerate(clients):
        x = int(latency_flags[:, c].sum())
        y = int(deviations[:, c].sum())
        ax.scatter(x, y, s=90, color=colours.get(roles[c], "#666"), zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(f" {label}", (x, y), fontsize=8, va="center")
    ax.set_xlabel("latency-anomaly rounds")
    ax.set_ylabel("checksum-deviation rounds")
    ax.set_title("Challenge-response audit: two-signal feature space")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=col, label=role)
               for role, col in colours.items()]
    ax.legend(handles=handles, title="resolved role", loc="center right", fontsize=8)
    ax.margins(0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  [--plot] wrote {path}")


def _exact_hash(grad: torch.Tensor) -> str:
    """Exact whole-gradient SHA-256 of the raw float32 bytes (no tolerance)."""
    return hashlib.sha256(grad.detach().numpy().astype(np.float32).tobytes()).hexdigest()


DETERMINISM_WIDTH = 1024   # ~1.07M parameters, so the block statistics are realistic

# The paper measures honest cross-hardware drift on four machines and finds that
# it tracks how far apart the two software/hardware stacks are. Each entry is
# (measured block mismatch, modelled relative FP drift that reproduces it on the
# DETERMINISM_WIDTH model, description of the machine pair). The drift values are
# calibrated to the measurements -- they are a model of them, not an independent
# estimate of FP32 error.
MEASURED_PAIRS = {
    "same-stack": (0.0042, 2.92e-7, "two machines, same ISA and OS"),
    "cross-os":   (0.0208, 1.34e-6, "same ISA, different OS (arm64 macOS vs Linux)"),
    "cross-isa":  (0.0250, 1.65e-6, "same OS, different ISA (x86_64 vs aarch64 Linux)"),
    "cross-both": (0.0321, 1.98e-6, "different ISA and OS (arm64 macOS vs x86_64 Linux)"),
}
# The worst honest pair the paper measured: the number tau actually has to clear.
DEFAULT_PAIR = "cross-both"


def determinism_demo(seed: int = 0, pair: str = DEFAULT_PAIR) -> bool:
    """Demonstrate the paper's central checksum result (Section IV-A): an exact
    hash false-positives on benign cross-platform floating-point drift, while the
    quantization-tolerant fingerprint absorbs that drift yet still catches an
    active model mutation. The drift here is modelled; the paper measures it for
    real across four machines, and ``pair`` selects which of those measured
    machine pairs to model (see ``MEASURED_PAIRS``).

    This runs on a wider model than the federation table above, because the block
    fingerprint is only meaningful when the gradient holds a realistic number of
    blocks: the federation's toy model yields 22, where a single drifting
    coordinate already moves 1/22 = 0.045 of the blocks and would sit at the
    tolerance by itself. At ``DETERMINISM_WIDTH`` there are a few thousand, as in
    a deployment."""
    measured, drift_scale, pair_desc = MEASURED_PAIRS[pair]
    rng = np.random.default_rng(seed)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = small_model(seed, DETERMINISM_WIDTH)
    w = torch.tensor(rng.normal(size=(N_FEATURES, 1)), dtype=torch.float32)
    x = torch.tensor(rng.normal(size=(CHALLENGE_SIZE, N_FEATURES)), dtype=torch.float32)
    y = (torch.sigmoid(x @ w) > 0.5).float()
    g_ref = flat_grad(model, x, y, loss_fn)                      # server reference

    # An honest client on different hardware: same computation, tiny relative FP
    # drift, scaled to reproduce the selected measured pair.
    scale = g_ref.abs().mean().item()
    g_drift = g_ref + torch.tensor(
        rng.normal(scale=drift_scale * scale, size=tuple(g_ref.shape)),
        dtype=torch.float32)
    # two distinct active attackers: both mutate the model
    g_gan = flat_grad(mutated_copy(model, "active-GAN", seed, rng, DETERMINISM_WIDTH),
                      x, y, loss_fn)
    g_asc = flat_grad(mutated_copy(model, "active-ascent", seed, rng, DETERMINISM_WIDTH),
                      x, y, loss_fn)

    n_blocks = block_count(g_ref)
    sub = challenge_blocks(n_blocks, key=seed)
    rows = [("honest, same hardware", g_ref),
            (f"honest, {pair}", g_drift),
            ("active-GAN (mutates model)", g_gan), ("active-ascent (mutates model)", g_asc)]
    print(f"\nDeterminism of the challenge-gradient checksum "
          f"(q={Q}, block={BLOCK_SIZE} coords, tolerance tau={TAU})")
    print(f"  {g_ref.numel():,} parameters -> {n_blocks:,} blocks; "
          f"the server asks for a challenge-derived subsample of {len(sub):,}")
    print(f"  modelling the measured pair '{pair}': {pair_desc}, "
          f"block mismatch {measured}\n")
    print(f"  {'client':<30} {'exact hash':>12} {'block mismatch':>15} "
          f"{'subsampled':>11} {'verdict':>9}")
    print("  " + "-" * 82)
    ok = True
    for label, g in rows:
        exact = "match" if _exact_hash(g) == _exact_hash(g_ref) else "DIFFERS"
        frac = mismatch_fraction(g, g_ref, Q)
        frac_sub = mismatch_fraction(g, g_ref, Q, indices=sub)
        flagged = frac > TAU
        verdict = "flag" if flagged else "accept"
        honest = label.startswith("honest")
        # honest accepted, attacks flagged -- on the full fingerprint and on the
        # subsample alike, which is the point of subsampling being safe
        ok = ok and (flagged != honest) and ((frac_sub > TAU) != honest)
        print(f"  {label:<30} {exact:>12} {frac:>15.4f} {frac_sub:>11.4f} {verdict:>9}")
    print()
    honest_frac = mismatch_fraction(g_drift, g_ref, Q)
    attack_min = min(mismatch_fraction(g, g_ref, Q) for g in (g_gan, g_asc))
    print("  Reading: the exact hash DIFFERS for the honest cross-hardware client,")
    print("  so a bitwise checksum would falsely flag it. The tolerant fingerprint")
    print("  keeps its block mismatch below tau and ACCEPTS it, while both active")
    print("  model-mutating attacks move essentially every block and are FLAGGED.")
    print(f"  The subsampled column shows the same verdicts from a {len(sub)}-block")
    print(f"  response rather than all {n_blocks:,}, which keeps the audit's cost")
    print("  independent of model size.")
    print()
    print(f"  Headroom: tau={TAU} clears this honest pair by {TAU / honest_frac:.1f}x "
          f"and sits {attack_min / TAU:.1f}x below the attack floor.")
    if pair == DEFAULT_PAIR:
        print("  This is the WORST honest pair the paper measured. Run --pair with")
        print("  same-stack, cross-os or cross-isa to see how the honest side falls")
        print("  as the two stacks get closer; the margin is widest when they match.")
    print(f"\n  result: {'determinism as expected' if ok else 'UNEXPECTED'}\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Challenge-response inference-audit demo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--kappa", type=float, default=2.5, help="latency z-score threshold")
    ap.add_argument("--probe-ms", type=float, default=450.0,
                    help="extra in-band analytic time of a thorough attacker (ms)")
    ap.add_argument("--jitter-ms", type=float, default=0.0,
                    help="modelled wide-area jitter added to every client (ms)")
    ap.add_argument("--sweep", action="store_true",
                    help="run the operating-conditions jitter sweep instead of the table")
    ap.add_argument("--determinism", action="store_true",
                    help="demonstrate the checksum's cross-hardware tolerance (exact hash vs tolerant fingerprint)")
    ap.add_argument("--pair", choices=sorted(MEASURED_PAIRS), default=DEFAULT_PAIR,
                    help="which measured machine pair --determinism models "
                         f"(default: {DEFAULT_PAIR}, the worst honest pair in the paper)")
    ap.add_argument("--plot", action="store_true",
                    help="also save the two-signal feature-space figure (needs matplotlib)")
    ap.add_argument("--plot-path", default="inference_audit_feature_space.png")
    args = ap.parse_args()

    if args.determinism:
        sys.exit(0 if determinism_demo(seed=args.seed, pair=args.pair) else 1)
    if args.sweep:
        jitter_sweep(kappa=args.kappa)
        return

    clients, deviations, latency_flags, roles = run_demo(
        seed=args.seed, rounds=args.rounds, kappa=args.kappa,
        probe_s=args.probe_ms / 1000.0, jitter_s=args.jitter_ms / 1000.0)
    ok = print_table(clients, deviations, latency_flags, roles, args.kappa)
    if args.plot:
        save_plot(clients, deviations, latency_flags, roles, args.plot_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
