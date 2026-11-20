---
id: EXC-2026-11-20-python-ultra
rule: HLT-005-PYTHON-PRODUCT-TRUTH, HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP (cell python-ai)
paths: [python/ultra/]
owner: training (lead)
granted: 2026-11-20
granted_by: owner ruling O1 (docs/ultra/RULINGS.md)
expires: 2027-06-30
proof_lane: just pytest
manifest: agent/boundaries.toml [python] allowed_truth_paths, allowed_non_product_paths
---

# Python training and data exception at `python/ultra/`

**What.** Advanced-ML training (PyTorch transducer trainer, losses,
quantisation-aware training), dataset download and bank building, and the glue
that drives the trainer. The authoritative grant is `python/ultra/EXCEPTION.md`;
this record is the standard-format copy for the audit.

**Why not `python/ai-service/`.** The standard's canonical cell is an AI *service*.
JSR has no Python service: nothing in Python runs at inference time, serves a
request, or ships. The code is an offline trainer, so the owner kept it at
`python/ultra/`, next to the waves and tests it drives.

**Boundaries.** Python never owns product truth, the streaming runtime, scoring,
receipts, proof lanes or release gates; those are Rust (`crates/`). Python
produces model bundles that only the Rust runtime consumes, and reported numbers
come from Rust scorers with `ultra.receipt/v1` receipts.

**Proof lane.** `just pytest` (part of `just fast` and `just required`).

**Expiry and exit.** Expires 2027-06-30 or at the R5 release, whichever is first.
Renewal needs a new owner ruling. Exit: trained models export to a bundle consumed
only by the Rust runtime; Python can then shrink to export tooling or be removed.
