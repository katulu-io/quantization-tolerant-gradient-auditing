# Two-site measurement

The cross-hardware numbers in Section IV-A of the paper come from this
measurement, not from the demo's model of it. Four machines at matched software
(torch 2.11.0, numpy 2.4.4, identical model weight digest), each emitting the
fingerprint of the same challenge gradient:

| pair | what differs | honest block mismatch |
|---|---|---|
| two Apple-silicon laptops | machine only | 0.0042 |
| arm64 macOS vs aarch64 Linux | OS and libraries | 0.0208 |
| x86_64 Linux vs aarch64 Linux | **instruction set only** | 0.0250 |
| arm64 macOS vs x86_64 Linux | both | 0.0321 |

Against a model-mutating attacker at 0.887, so `tau = 0.05` separates every
measured pair -- by 1.6x on the honest side at worst.

The exact hash disagrees in every heterogeneous pair, which is the whole reason
the fingerprint needs a tolerance. The drift is nonetheless deterministic rather
than noisy: two x86_64 instances on different continents produced a
byte-identical challenge gradient.

## Files

- `rev_two_site.py` -- the runner (`emit`, `compare`, `latency` subcommands)
- `requirements.txt` -- pinned versions; these MUST match across sites or the
  comparison is meaningless
- `rev_two_site_ec2.yaml` -- CloudFormation for an x86_64 or arm64 probe host
- `rev_two_site_full.json`, `rev_isa_only.json` -- the comparison outputs
- `latency_AB.json`, `latency_AC_use1.json` -- two real inter-site links

The raw per-machine fingerprints are ~4.9 MB each and are not included; the
comparison outputs carry every number quoted above. Regenerate them with
`rev_two_site.py emit` on two machines and compare.

## The latency measurement

Two intercontinental links, 300 TCP samples each:

| link | median | MAD | p99 |
|---|---|---|---|
| Europe to us-east-1 | 109.1 ms | 2.43 ms | 127.9 ms |
| Europe to ap-southeast-2 | 310.2 ms | 2.55 ms | 330.5 ms |

Both are high-latency and stable. The detector's modified z-score consumes the
MAD, not the absolute latency -- a constant offset is absorbed by the per-round
median -- so neither link sits in the jitter regime that breaks the cohort
baseline.
