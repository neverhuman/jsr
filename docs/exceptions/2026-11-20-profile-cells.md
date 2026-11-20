---
id: EXC-2026-11-20-profile-cells
rule: HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP (cells web, api, domain, application, adapters, workers, db)
paths: [apps/, db/, crates/domain/, crates/application/, crates/adapters/, crates/workers/]
owner: lead
granted: 2026-11-20
expires: 2027-06-30
proof_lane: just score
manifest: agent/boundaries.toml [rust] domain_paths
---

# Scaffold profile cells JSR does not have

The scaffold was generated with profile `rust-api`, which assumes a web app, an
HTTP API, a database and layered domain/application/adapter/worker crates. JSR is
a streaming CPU speech recogniser: a Rust library and command-line tool, with no
web front-end, no HTTP server and no database. The template directories were
deleted rather than kept empty.

Mapping of the standard's cells to JSR:

| Standard cell | JSR |
|---|---|
| `crates/domain` (pure core) | `crates/ultra-core`: numeric, event and receipt contracts; declared as the domain path in `agent/boundaries.toml` |
| `crates/adapters` | `crates/ultra-adapters`: process-isolated external recognisers |
| `crates/workers`, `crates/application` | `crates/ultra-stream` (scheduler) and `crates/ultra-cli` |
| `apps/web`, `apps/api`, `db/` | none; not applicable |
| `contracts/` | `contracts/events/` (canonical) |

Revisit if JSR gains a service surface (HTTP or gRPC) or persistent storage; then
the matching cells apply and this record is closed.
