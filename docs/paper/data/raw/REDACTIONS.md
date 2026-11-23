# Receipt redactions (public repository)

neverhuman/jsr is public. The raw receipts below were copied verbatim from a benchmark host, and their
`host` file recorded the machine's hostname. On a cloud VM that hostname is the machine's public IP
address written with dashes. On 2026-11-23 each `host` file below was rewritten to the host's role
label. No other byte of any receipt changed.

- **Rule:** receipt `host` field (hostname) → role label. `xbabe6` is the key the paper's `HOSTS`
  table already uses for this machine, printed as X1.
- **Mapping:** kept in an untracked local file on the integration host, never committed.
- **Chain:** `MANIFEST.txt` now pins the redacted sha256, rebuilt with `tools.paper_data`'s
  `write_manifest`. The original files are unchanged at the origin each MANIFEST line names.
  This record commits to each original with HMAC-SHA256 under a key held untracked on the integration
  host. A plain sha256 of a ~15-byte host name could be brute-forced back to the address. A key holder
  with the originals can check that only `host` changed.
- **Writers:** `docs/paper/run-cold.sh`, `docs/paper/opponents/open-run.sh` and
  `docs/paper/opponents/run-opponent.sh` now record a role label (`JSR_HOST_ROLE` or
  `~/.config/jsr/host-role`) and refuse an address-shaped hostname.
  `python/ultra/tests/test_no_host_addresses.py` (CI, pytest-light) fails on any host address in a
  tracked file, on a receipt `host` file without a role label, and on a script that writes `hostname`
  into a receipt.

| file (under `docs/paper/data/raw/`) | original: HMAC-SHA256 (keyed) | redacted sha256 | rule |
|---|---|---|---|
| `opponents/xeon8358/jsr-d2e-int8-shared8-t8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-shared8-t8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-shared8-t8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-t1/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-t1/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-t1/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-w8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-w8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-d2e-int8-w8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-shared8-t8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-shared8-t8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-shared8-t8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-t1/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-t1/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-t1/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-w8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-w8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/jsr-h22ctl-int8-w8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-t1/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-t1/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-t1/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-w8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-w8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/moonshine-tiny-w8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t1-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t1-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t1-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t8-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t8-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-offline-t8-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-stream160-t1-panel10/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-stream160-t8-panel10/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-stream160-t8-panel10/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/nemotron-labs-unified-stream160-t8-panel10/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t1-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t1-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t1-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t8-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t8-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-t8-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-w8-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-w8-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-offline-w8-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-stream160-t1-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-stream160-t8-panel73/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-stream160-t8-panel73/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/parakeet-unified-stream160-t8-panel73/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-t1/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-t1/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-t1/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-w8/round-1/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-w8/round-2/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
| `opponents/xeon8358/zipformer-int8-w8/round-3/host` | `28953b18bc8fa711f6d1e2cd9fd36895b59aa1280a9013d8db1f2b558c2af67c` | `2f7ccdb680b9d66b2e12c8914ceebf6bd0a07b8a9a5a73129a3592084fe7b3cd` | host → role label |
