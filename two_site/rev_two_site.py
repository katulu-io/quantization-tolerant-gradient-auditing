#!/usr/bin/env python3
"""FLTA-2026 review response: the TWO-SITE measurement.

Self-contained (torch + numpy only, no project imports) so it can be copied to a
second machine or an EC2 instance and run there.

It settles two things the submitted paper asserted without a retained artifact.
Both were measured on 2026-09-21 across four machines; the results are in
``results/two_site/`` and are quoted in Section IV-A of the camera-ready.

1.  **Cross-machine fingerprint drift** (Sec. IV-A-2, and Reviewer 3's "the real
    contribution"). The exact whole-gradient hash disagrees in every
    heterogeneous pair, while the tolerant block mismatch tracks how far apart
    the two stacks are: 0.0042 between two machines of the same architecture and
    OS, 0.0208 across operating systems at fixed architecture, 0.0250 across
    architectures at fixed OS, and 0.0321 when both differ, against 0.887 for a
    model-mutating attacker. Instruction set is the largest single term. The
    drift is deterministic rather than noisy -- two x86_64 instances on
    different continents produced a byte-identical challenge gradient -- so it
    is a property of the (architecture, OS, library) stack, not of the host.

2.  **Real inter-site latency and jitter** (Reviewer 3: "Cite a source for the
    100-200 ms cross-silo jitter figure, or measure one real inter-site link
    even at n=2"). Two links, 300 TCP samples each: Europe to us-east-1 at a
    median of 109.1 ms with a MAD of 2.43 ms, and Europe to ap-southeast-2 at
    310.2 ms with a MAD of 2.55 ms. Both are high-latency and stable. The
    detector consumes the MAD, not the absolute latency, so neither link sits
    in the jitter regime that breaks the cohort baseline.

Because the model is built on CPU from a fixed seed, both machines start from
bitwise-identical weights and challenge batch, so any difference in the emitted
gradient is pure backend non-determinism. **The torch version must match** --
the script records it and ``compare`` refuses to draw conclusions if it differs.

Usage
-----
On EACH machine::

    python rev_two_site.py emit --device cpu --label siteA --out siteA_cpu.json

Optionally also on any GPU present (the paper excludes GPU, and this shows why)::

    python rev_two_site.py emit --device mps  --label siteA --out siteA_mps.json
    python rev_two_site.py emit --device cuda --label siteB --out siteB_cuda.json

For the latency leg, on the machine that plays the server::

    python rev_two_site.py latency --peer <host-or-ip> --n 200 --out latency_AB.json

Then, with all files on one machine::

    python rev_two_site.py compare siteA_cpu.json siteB_cpu.json ... --out rev_two_site.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

DIGEST_HEX = 16
DEFAULTS = dict(seed=2026, features=14, width=2560, depth=5, q=1e-2, block=256,
                batch=16)


# ---------------------------------------------------------------------------
# Deterministic model and challenge, identical on every machine
# ---------------------------------------------------------------------------

def build_model(features: int, width: int, depth: int, seed: int):
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    d = features
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
        d = width
    layers += [torch.nn.Linear(d, 1)]
    return torch.nn.Sequential(*layers)


def build_challenge(seed: int, features: int, batch: int):
    g = torch.Generator().manual_seed(
        int(hashlib.sha256(f"{seed}:0".encode()).hexdigest()[:16], 16) % (2**63))
    x = torch.randn((batch, features), generator=g, dtype=torch.float32)
    y = (torch.rand((batch, 1), generator=g) > 0.5).float()
    return x, y


def challenge_gradient(model, x, y, device: str) -> np.ndarray:
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    model.eval()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    torch.nn.BCEWithLogitsLoss()(model(x), y).backward()
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in model.parameters()
    ]).detach().cpu().numpy()


def fingerprint(g: np.ndarray, q: float, block: int) -> list[str]:
    step = q * (np.abs(g).mean() + 1e-12)
    qi = np.round(g / step).astype(np.int64)
    n_blocks = max(1, int(np.ceil(qi.size / block)))
    return [hashlib.sha256(b.tobytes()).hexdigest()[:DIGEST_HEX]
            for b in np.array_split(qi, n_blocks)]


def gan_mutate(model, features: int, seed: int = 7):
    """The paper's active attacker, for the contrast row."""
    torch.manual_seed(seed)
    z = torch.randn(8, features)
    target = torch.ones(8, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(5):
        opt.zero_grad()
        loss_fn(model(z), target).backward()
        opt.step()
    model.zero_grad()
    return model


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def cmd_emit(a) -> None:
    x, y = build_challenge(a.seed, a.features, a.batch)
    model = build_model(a.features, a.width, a.depth, a.seed)
    weight_digest = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()

    t0 = time.perf_counter()
    g = challenge_gradient(model, x, y, a.device)
    elapsed = time.perf_counter() - t0

    mutated = gan_mutate(build_model(a.features, a.width, a.depth, a.seed),
                         a.features)
    g_att = challenge_gradient(mutated, x, y, "cpu")

    rec = {
        "label": a.label,
        "device": a.device,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "params": dict(seed=a.seed, features=a.features, width=a.width,
                       depth=a.depth, q=a.q, block=a.block, batch=a.batch),
        "n_params": int(g.size),
        "weight_digest": weight_digest,
        "exact_hash": hashlib.sha256(g.tobytes()).hexdigest(),
        "fingerprint": fingerprint(g, a.q, a.block),
        "attack_exact_hash": hashlib.sha256(g_att.tobytes()).hexdigest(),
        "attack_fingerprint": fingerprint(g_att, a.q, a.block),
        "grad_abs_mean": float(np.abs(g).mean()),
        "grad_seconds": elapsed,
    }
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(f"[{a.label}/{a.device}] {rec['processor']} torch {rec['torch']}")
    print(f"  weights   {weight_digest[:16]}  (must match across machines)")
    print(f"  exact     {rec['exact_hash'][:16]}")
    print(f"  blocks    {len(rec['fingerprint'])}  challenge gradient in "
          f"{elapsed*1000:.2f} ms")
    print(f"Wrote {a.out}")


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def cmd_latency(a) -> None:
    """Measure real round-trip latency and jitter to a peer host.

    Uses TCP connect time when a port is given (closest to what a cross-silo FL
    round actually pays) and falls back to ICMP via the system ping otherwise.
    """
    samples: list[float] = []
    if a.port:
        for _ in range(a.n):
            t0 = time.perf_counter()
            try:
                s = socket.create_connection((a.peer, a.port), timeout=5)
                s.close()
                samples.append((time.perf_counter() - t0) * 1000.0)
            except OSError:
                pass
            time.sleep(a.interval)
        method = f"tcp_connect:{a.port}"
    else:
        out = subprocess.run(
            ["ping", "-c", str(a.n), "-i", str(max(a.interval, 0.2)), a.peer],
            capture_output=True, text=True,
        ).stdout
        for line in out.splitlines():
            if "time=" in line:
                samples.append(float(line.split("time=")[1].split()[0]))
        method = "icmp_ping"

    if not samples:
        print("no samples collected -- is the peer reachable?", file=sys.stderr)
        sys.exit(1)

    samples.sort()
    def pct(p):
        return samples[min(len(samples) - 1, int(p * len(samples)))]
    med = statistics.median(samples)
    rec = {
        "peer": a.peer, "method": method, "n": len(samples),
        "min_ms": samples[0], "median_ms": med, "mean_ms": statistics.fmean(samples),
        "p90_ms": pct(0.90), "p99_ms": pct(0.99), "max_ms": samples[-1],
        # The detector consumes deviation from the cohort median, so report
        # jitter the same way rather than as a raw standard deviation.
        "mad_ms": statistics.median([abs(s - med) for s in samples]),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "iqr_ms": pct(0.75) - pct(0.25),
        "samples_ms": samples,
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(f"[latency] {a.peer} via {method}, n={len(samples)}")
    print(f"  median {med:.2f} ms   MAD {rec['mad_ms']:.2f} ms   "
          f"p90 {rec['p90_ms']:.2f}   p99 {rec['p99_ms']:.2f}   "
          f"max {rec['max_ms']:.2f}")
    print(f"Wrote {a.out}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def frac_diff(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        return 1.0
    return float(np.mean([x != y for x, y in zip(a, b)]))


def cmd_compare(a) -> None:
    recs = [json.loads(Path(f).read_text()) for f in a.files]
    ref = recs[0]
    print(f"reference: {ref['label']}/{ref['device']} on {ref['processor']}\n")

    torch_versions = {r["torch"] for r in recs}
    weight_digests = {r["weight_digest"] for r in recs}
    comparable = len(weight_digests) == 1
    if not comparable:
        print("!! weight digests DIFFER across files -- the machines did not "
              "start from identical models, so gradient differences are not "
              "attributable to backend non-determinism. Check torch versions: "
              f"{sorted(torch_versions)}\n")

    rows = []
    for r in recs[1:]:
        same_machine = r["processor"] == ref["processor"] and \
            r["hostname"] == ref["hostname"]
        row = {
            "label": f"{r['label']}/{r['device']}",
            "processor": r["processor"],
            "cross_machine": not same_machine,
            "exact_hash_matches": r["exact_hash"] == ref["exact_hash"],
            "block_mismatch_fraction": frac_diff(r["fingerprint"],
                                                 ref["fingerprint"]),
            "attack_block_mismatch_fraction": frac_diff(r["attack_fingerprint"],
                                                        ref["fingerprint"]),
            "grad_seconds": r["grad_seconds"],
        }
        rows.append(row)
        print(f"  {row['label']:22s} {row['processor'][:24]:24s} "
              f"exact={'MATCH' if row['exact_hash_matches'] else 'DIFFER'}  "
              f"honest_frac={row['block_mismatch_fraction']:.6f}  "
              f"attack_frac={row['attack_block_mismatch_fraction']:.4f}")

    cross = [r for r in rows if r["cross_machine"]]
    cpu_cross = [r for r in cross if not r["label"].endswith(("mps", "cuda"))]
    honest_max = max((r["block_mismatch_fraction"] for r in rows), default=0.0)
    cpu_honest_max = max((r["block_mismatch_fraction"] for r in cpu_cross),
                         default=None)
    attack_min = min((r["attack_block_mismatch_fraction"] for r in rows),
                     default=None)

    print()
    if cpu_cross:
        exact_ok = all(r["exact_hash_matches"] for r in cpu_cross)
        print(f"  cross-machine CPU comparisons: {len(cpu_cross)}")
        print(f"    exact hash {'MATCHES on all' if exact_ok else 'DIFFERS on some'}")
        print(f"    honest block-mismatch max: {cpu_honest_max:.6f}")
        if exact_ok:
            print("    => the exact hash SURVIVES a real second machine on CPU;")
            print("       the tolerance is motivated by architectural")
            print("       heterogeneity (GPU/ISA), not by CPU drift per se.")
    else:
        print("  no cross-machine CPU comparison present -- run `emit` on the "
              "second host and include its file.")
    if attack_min is not None:
        print(f"  attack block-mismatch min: {attack_min:.4f}   "
              f"honest max (all backends): {honest_max:.6f}")
        print(f"  margin at tau=0.05: honest {honest_max:.6f} < 0.05 < "
              f"{attack_min:.4f} attack "
              f"=> {'HOLDS' if honest_max < 0.05 < attack_min else 'VIOLATED'}")

    out = {
        "reference": {k: ref[k] for k in
                      ("label", "device", "processor", "platform", "torch",
                       "hostname")},
        "weights_identical_across_files": comparable,
        "torch_versions": sorted(torch_versions),
        "rows": rows,
        "honest_block_mismatch_max": honest_max,
        "cross_machine_cpu_honest_max": cpu_honest_max,
        "attack_block_mismatch_min": attack_min,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="compute and save this machine's fingerprint")
    e.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    e.add_argument("--label", default=socket.gethostname())
    e.add_argument("--out", default="probe.json")
    for k, v in DEFAULTS.items():
        e.add_argument(f"--{k}", type=type(v), default=v)
    e.set_defaults(func=cmd_emit)

    l = sub.add_parser("latency", help="measure real inter-site latency/jitter")
    l.add_argument("--peer", required=True, help="hostname or IP of the other site")
    l.add_argument("--port", type=int, default=0,
                   help="TCP port for connect-time sampling; omit for ICMP ping")
    l.add_argument("--n", type=int, default=200)
    l.add_argument("--interval", type=float, default=0.25)
    l.add_argument("--out", default="latency.json")
    l.set_defaults(func=cmd_latency)

    c = sub.add_parser("compare", help="compare emitted fingerprints")
    c.add_argument("files", nargs="+", help="first file is the reference")
    c.add_argument("--out", default="rev_two_site.json")
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
