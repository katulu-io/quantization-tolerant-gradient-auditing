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
"""Smoke test for the self-contained inference-audit demo.

Asserts that the two-signal (checksum-deviation, latency-anomaly) feature
resolves every client to its expected role, that the passive cosine
membership-inference attacker is caught by latency alone, that the
offline-decoupled attacker EVADES (the method's honest boundary), and that the
operating-conditions sweep reproduces the long-round-robust / short-round-fragile
contrast.

Run:  pytest -q
"""
import pytest
import torch

import inference_audit_demo as demo


def test_fingerprint_compares_blocks_not_coordinates():
    """The fingerprint is block-level (paper, Section IV-A): one differing
    coordinate costs a whole block, so the mismatch is 1/n_blocks, not
    1/n_coordinates. This is the regression guard against the comparison
    silently reverting to per-coordinate."""
    g = torch.arange(1, 4097, dtype=torch.float32) / 4096.0
    n_blocks = demo.block_count(g)
    assert n_blocks == 16                      # 4096 coords / 256 per block

    # Swap two coordinates inside block 0 that sit on different quantization
    # levels. This leaves mean(|g|) -- and therefore the quantization step and
    # every other block -- untouched, so the only change is within one block.
    moved = g.clone()
    moved[0], moved[255] = g[255].clone(), g[0].clone()
    assert demo.quantize(moved, demo.Q)[0] != demo.quantize(g, demo.Q)[0]

    frac = demo.mismatch_fraction(moved, g, demo.Q)
    assert frac == pytest.approx(1 / n_blocks)         # one block, not one coordinate
    assert frac > 10 * (2 / g.numel())                 # and far above the coordinate rate


def test_block_size_is_fixed_so_block_count_grows_with_model():
    """Fixing block *size* rather than block *count* is what keeps the honest
    mismatch fraction bounded as models grow (paper, Section IV-A)."""
    assert demo.block_count(torch.zeros(4096 * 8)) == 8 * demo.block_count(torch.zeros(4096))


def test_challenge_subsample_agrees_with_full_fingerprint():
    """The server asks for a challenge-derived subset of blocks. Different
    challenge keys select different subsets, and the subsample reaches the same
    verdicts as the full fingerprint."""
    g = torch.arange(1, 256 * 600 + 1, dtype=torch.float32) / 1000.0
    n_blocks = demo.block_count(g)
    a = demo.challenge_blocks(n_blocks, key=1)
    b = demo.challenge_blocks(n_blocks, key=2)
    assert len(a) == demo.SUBSAMPLE and len(b) == demo.SUBSAMPLE
    assert set(a.tolist()) != set(b.tolist())          # unpredictable without the key
    assert demo.challenge_blocks(8, key=1) is None     # no subsampling when tiny

    mutated = g + 10 * demo.Q * g.abs().mean()         # moves essentially every block
    assert demo.mismatch_fraction(mutated, g, demo.Q) > demo.TAU
    assert demo.mismatch_fraction(mutated, g, demo.Q, indices=a) > demo.TAU
    assert demo.mismatch_fraction(g, g, demo.Q, indices=a) == 0.0


def test_demo_roles_match_expected():
    clients, deviations, latency_flags, roles = demo.run_demo()
    assert len(clients) == len(roles)
    for c, (label, kind) in enumerate(clients):
        assert roles[c] == demo.EXPECTED[kind], (
            f"{label}: got {roles[c]!r}, expected {demo.EXPECTED[kind]!r}"
        )


def test_passive_attacker_caught_by_latency_only():
    """The cosine-MI attacker has zero checksum deviations but is still flagged
    as inference via the latency channel alone — the paper's core claim."""
    clients, deviations, latency_flags, roles = demo.run_demo()
    idx = [k for _, k in clients].index("cosine-MI")
    assert deviations[:, idx].sum() == 0          # never touches the model
    assert latency_flags[:, idx].sum() >= 2       # persistent latency anomaly
    assert roles[idx] == "inference"


def test_honest_clients_have_no_signal():
    clients, deviations, latency_flags, roles = demo.run_demo()
    honest = [c for c, (_, kind) in enumerate(clients) if kind == "honest"]
    assert honest
    for c in honest:
        assert deviations[:, c].sum() == 0
        assert latency_flags[:, c].sum() == 0
        assert roles[c] == "honest"


def test_byzantine_is_checksum_only():
    clients, deviations, latency_flags, roles = demo.run_demo()
    idx = [k for _, k in clients].index("byzantine")
    assert deviations[:, idx].sum() > 0
    assert latency_flags[:, idx].sum() == 0
    assert roles[idx] == "byzantine"


def test_offline_decoupled_evades():
    """The offline-decoupled attacker presents neither signal and is reported as
    honest — the method's documented boundary (it raises the cost of passive
    inference, it does not abolish it)."""
    clients, deviations, latency_flags, roles = demo.run_demo()
    idx = [k for _, k in clients].index("offline-decoupled")
    assert deviations[:, idx].sum() == 0
    assert latency_flags[:, idx].sum() == 0
    assert roles[idx] == "honest"          # evades, by construction


def test_operating_conditions_long_robust_short_fragile():
    """The honest false-positive rate stays at zero for a long-round federation
    under 100 ms of jitter, but climbs for a short-round one — the operating conditions point
    (paper, Sections VI-E/F)."""
    seeds = (0, 1, 2, 3)
    fpr_long = demo._honest_fpr(seeds, jitter_s=0.100, round_base_s=0.60, kappa=2.5)
    fpr_short = demo._honest_fpr(seeds, jitter_s=0.100, round_base_s=0.11, kappa=2.5)
    assert fpr_long <= 0.05
    assert fpr_short >= 0.20
    assert fpr_short > fpr_long


def test_checksum_determinism_tolerance():
    """The tolerant checksum accepts an honest cross-hardware client (whose exact
    hash differs) and flags active model-mutating attacks — the paper's central
    determinism result (Section IV-A)."""
    assert demo.determinism_demo(seed=0) is True


def _honest_drift_mismatch(pair, seed=0):
    """Reproduce the modelled honest client for one measured pair, using only
    the module's own primitives."""
    import numpy as np
    _, drift_scale, _ = demo.MEASURED_PAIRS[pair]
    rng = np.random.default_rng(seed)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = demo.small_model(seed, demo.DETERMINISM_WIDTH)
    w = torch.tensor(rng.normal(size=(demo.N_FEATURES, 1)), dtype=torch.float32)
    x = torch.tensor(rng.normal(size=(demo.CHALLENGE_SIZE, demo.N_FEATURES)),
                     dtype=torch.float32)
    y = (torch.sigmoid(x @ w) > 0.5).float()
    g_ref = demo.flat_grad(model, x, y, loss_fn)
    g = g_ref + torch.tensor(
        rng.normal(scale=drift_scale * g_ref.abs().mean().item(), size=tuple(g_ref.shape)),
        dtype=torch.float32)
    return demo.mismatch_fraction(g, g_ref, demo.Q)


def test_measured_pairs_reproduce_the_papers_numbers():
    """Each preset's modelled drift must still reproduce the cross-hardware
    mismatch the paper measured for that machine pair. This pins the calibration:
    changing Q, BLOCK_SIZE or DETERMINISM_WIDTH silently invalidates it."""
    for pair, (measured, _, _) in demo.MEASURED_PAIRS.items():
        got = _honest_drift_mismatch(pair)
        assert got == pytest.approx(measured, rel=0.15), (
            f"{pair}: modelled {got:.4f}, paper measured {measured:.4f}"
        )


def test_honest_drift_orders_by_stack_distance_and_stays_under_tau():
    """Honest drift grows as the two stacks diverge, and every measured pair
    stays inside the tolerance -- the paper's headroom claim."""
    order = ["same-stack", "cross-os", "cross-isa", "cross-both"]
    vals = [demo.MEASURED_PAIRS[p][0] for p in order]
    assert vals == sorted(vals), "measured pairs are not ordered by stack distance"
    assert max(vals) < demo.TAU, "a measured honest pair exceeds the tolerance"
    assert demo.DEFAULT_PAIR == "cross-both", "the default must be the worst honest pair"


def test_every_pair_still_separates_honest_from_attack():
    for pair in demo.MEASURED_PAIRS:
        assert demo.determinism_demo(seed=0, pair=pair) is True, pair
