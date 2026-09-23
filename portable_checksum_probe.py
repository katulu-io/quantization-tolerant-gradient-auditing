#!/usr/bin/env python
"""Portable challenge-gradient fingerprint probe for CROSS-MACHINE determinism tests.

Self-contained: depends only on ``torch`` and ``numpy`` (no project imports), so it
can be copied to any machine. It builds a deterministic model and challenge batch
from a shared seed, computes the challenge gradient on a chosen backend, and emits
both an EXACT whole-gradient SHA-256 and the block-quantized tolerant fingerprint.
Run it on two machines (and on CPU + GPU on each), then ``--compare`` the outputs to
measure how much honest cross-hardware floating-point drift the exact hash sees
versus the quantized fingerprint.

Because the model is built on CPU from a fixed seed, every machine starts from the
BITWISE-IDENTICAL weights and challenge batch (requires the same torch version); any
difference in the emitted gradient is therefore pure backend non-determinism.

Usage:
    # on each machine / backend, emit a fingerprint file:
    python portable_checksum_probe.py emit --device cpu --out probeA_cpu.json
    python portable_checksum_probe.py emit --device mps --out probeA_mps.json   # Apple GPU
    python portable_checksum_probe.py emit --device cuda --out probeB_cuda.json  # NVIDIA

    # then compare any set of emitted files against the first (the reference):
    python portable_checksum_probe.py compare probeA_cpu.json probeA_mps.json probeB_cuda.json

Flags (must match across machines for a valid comparison):
    --seed 2026  --features 16  --width 2560 --depth 5  --q 0.01 --block-size 256
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
import numpy as np
import torch


def build_model(features: int, width: int, depth: int, seed: int) -> torch.nn.Module:
    """Deterministic feed-forward model, built on CPU from ``seed`` (identical everywhere)."""
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    d = features
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
        d = width
    layers += [torch.nn.Linear(d, 1)]
    return torch.nn.Sequential(*layers)


def challenge_batch(features: int, seed: int, batch: int = 16):
    """Deterministic challenge batch from ``seed`` (identical everywhere)."""
    key = int(hashlib.sha256(f"{seed}:challenge".encode()).hexdigest()[:16], 16)
    g = torch.Generator().manual_seed(key)
    x = torch.randn((batch, features), generator=g, dtype=torch.float32)
    y = (torch.rand((batch, 1), generator=g) > 0.5).to(torch.float32)
    return x, y


def challenge_gradient(model, x, y, device: str) -> torch.Tensor:
    """Flattened challenge gradient computed on ``device`` (returned as CPU float32)."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_(); p.grad.zero_()
    loss_fn(model(x.to(device)), y.to(device)).backward()
    parts = [(p.grad if p.grad is not None else torch.zeros_like(p))
             .detach().to(torch.float32).reshape(-1).cpu() for p in model.parameters()]
    grad = torch.cat(parts) if parts else torch.zeros(1)
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def fingerprint(grad: torch.Tensor, q: float, block_size: int) -> list[str]:
    """Block-quantized fingerprint: list of per-block SHA-256 digests (blocks of fixed size)."""
    step = q * (grad.abs().mean().item() + 1e-12)
    qg = torch.round(grad / step).to(torch.int64).numpy()
    n_blocks = max(1, int(np.ceil(qg.size / block_size)))
    return [hashlib.sha256(b.tobytes()).hexdigest()[:16] for b in np.array_split(qg, n_blocks)]


def exact_digest(grad: torch.Tensor) -> str:
    return hashlib.sha256(grad.numpy().astype(np.float32).tobytes()).hexdigest()


def cmd_emit(a):
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        sys.exit("MPS not available on this machine")
    if dev == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available on this machine")
    model = build_model(a.features, a.width, a.depth, a.seed)
    x, y = challenge_batch(a.features, a.seed)
    grad = challenge_gradient(model, x, y, dev)
    rec = {
        "config": {"seed": a.seed, "features": a.features, "width": a.width,
                    "depth": a.depth, "q": a.q, "block_size": a.block_size,
                    "nparams": int(grad.numel())},
        "host": {"node": platform.node(), "machine": platform.machine(),
                  "system": platform.system(), "device": dev,
                  "torch": torch.__version__},
        "mean_abs_grad": float(grad.abs().mean()),
        "exact_sha256": exact_digest(grad),
        "fingerprint": fingerprint(grad, a.q, a.block_size),
    }
    Pathlike = a.out or f"probe_{platform.node()}_{dev}.json"
    with open(Pathlike, "w") as f:
        json.dump(rec, f)
    print(f"wrote {Pathlike}  ({rec['host']['system']}/{rec['host']['machine']}/{dev}, "
          f"torch {rec['host']['torch']}, {rec['config']['nparams']} params, "
          f"{len(rec['fingerprint'])} blocks)")


def _frac_mismatch(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        return 1.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def cmd_compare(a):
    recs = [json.load(open(p)) for p in a.files]
    ref = recs[0]
    cfgs = {json.dumps(r["config"], sort_keys=True) for r in recs}
    if len(cfgs) != 1:
        print("WARNING: configs differ across files; comparison is only valid for identical configs.")
    print(f"reference: {a.files[0]}  ({ref['host']['system']}/{ref['host']['machine']}/"
          f"{ref['host']['device']}, torch {ref['host']['torch']})")
    print(f"{'file':<32} {'backend':<22} {'exact==ref':<11} {'block-mismatch frac':<20}")
    for path, r in zip(a.files, recs):
        backend = f"{r['host']['machine']}/{r['host']['device']}"
        exact = "yes" if r["exact_sha256"] == ref["exact_sha256"] else "NO"
        frac = _frac_mismatch(r["fingerprint"], ref["fingerprint"])
        print(f"{path:<32} {backend:<22} {exact:<11} {frac:<20.6f}")
    print("\nReading: exact==ref 'NO' means an exact hash would FALSE-POSITIVE this honest client; "
          "a block-mismatch fraction below the tolerance (default tau=0.05) means the tolerant "
          "fingerprint correctly ACCEPTS it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="compute and write a fingerprint file")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    e.add_argument("--seed", type=int, default=2026)
    e.add_argument("--features", type=int, default=16)
    e.add_argument("--width", type=int, default=2560)
    e.add_argument("--depth", type=int, default=5)
    e.add_argument("--q", type=float, default=0.01)
    e.add_argument("--block-size", type=int, default=256)
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_emit)
    c = sub.add_parser("compare", help="compare emitted files against the first (reference)")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
