# DECISIONS

## 2026-11-20: ULTRA v1.0 is the controlling spec
- **Alternatives:** CADENCE-ULTRA v3.
- **Evidence:** ULTRA's ladder (R0–R5) gives one mechanism per rung, and it has the more exact numeric contract.
- **Consequence:** we borrow v3's AVX2 T-LUT kernels and its isqrt-PCEN arm.
- **Revisit:** if the owner prefers v3.

## 2026-11-20: AVX2 first
- **Evidence:** H0 is Zen 2 and has no AVX-512.
- **Consequence:** the AVX-512 kernels are written but stay UNTESTED until a machine with AVX-512 is available.
- **Revisit:** when such a machine is available.

## 2026-11-20: Cargo target on the data drive
- **Evidence:** root NVMe is at 95%, and the disk-steward guard refuses builds above 85%.
- **Consequence:** `.cargo/config.toml` points `target-dir` at `/mnt/xbabe3-external/jsr/cargo-target/main`.
