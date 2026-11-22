//! Wave H24 microbenchmark: 4-bit codebook-scan (`pqscan`) + `FWHT16` + `ReLU²` sparse scan
//! vs `I8Matrix` (`K0`), pinned to a single core.
//!
//! Run on H0 Zen 2 AVX2:
//!   cargo run --release -p ultra-kernels --example pqscan_bench
//! Run on xbabe4 M3 Ultra NEON:
//!   cargo run --release -p ultra-kernels --example pqscan_bench

use std::time::{Duration, Instant};
use ultra_kernels::pqscan::{fwht16_inplace, CodebookMatrix, PQSCAN_K};
use ultra_kernels::{Backend, Frames, Gemv, I8Matrix};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

fn bench_min_ns(min_dur: Duration, mut f: impl FnMut()) -> f64 {
    f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut iters = 0u64;
        let t0 = Instant::now();
        while t0.elapsed() < min_dur {
            f();
            iters += 1;
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

fn make_cb(rows: usize, din: usize, s: usize, seed: &mut u64) -> CodebookMatrix {
    let c = din / s;
    let mut cb = vec![0i8; c * PQSCAN_K * s];
    for v in &mut cb {
        *v = (((lcg(seed) >> 33) % 255) as i32 - 127) as i8;
    }
    let mut codes = vec![0u8; rows * c];
    for v in &mut codes {
        *v = ((lcg(seed) >> 33) & 15) as u8;
    }
    CodebookMatrix::from_parts(rows, din, s, 1, cb, &codes).expect("valid synthetic codebook")
}

fn main() {
    let backend = Backend::detect();
    let min_dur = Duration::from_millis(40);
    let mut seed = 0x2026_1122u64;
    println!(
        "Wave H24 pqscan microbenchmark (arch={}, backend={})",
        std::env::consts::ARCH,
        backend.name()
    );
    println!(
        "{:<12} {:>4} {:>9} | {:>16} | {:>16} | {:>16} | {:>14}",
        "Shape",
        "Rows",
        "K0 ps/MAC",
        "s=2 (tot/tbl%)",
        "s=4 (tot/tbl%)",
        "s=8 (tot/tbl%)",
        "s=4 80%sp"
    );
    println!("{}", "-".repeat(104));

    let shapes = [
        (1536, 384),
        (384, 1536),
        (1152, 384),
        (384, 384),
        (3072, 768),
        (768, 3072),
    ];

    for (rows, din) in shapes {
        // Build K0 baseline (I8Matrix with A7 frames).
        let mut w_i8 = vec![0i8; rows * din];
        for v in &mut w_i8 {
            *v = (((lcg(&mut seed) >> 33) % 255) as i32 - 127) as i8;
        }
        let k0 = I8Matrix::new(rows, din, w_i8).expect("valid i8 matrix");
        let cb2 = make_cb(rows, din, 2, &mut seed);
        let cb4 = make_cb(rows, din, 4, &mut seed);
        let cb8 = make_cb(rows, din, 8, &mut seed);

        for f_n in [1usize, 2, 4, 32] {
            let macs = (rows * din * f_n) as f64;
            let mut x_u8 = vec![0u8; f_n * din];
            let mut x_i8 = vec![0i8; f_n * din];
            for (u, i) in x_u8.iter_mut().zip(&mut x_i8) {
                let r = ((lcg(&mut seed) >> 33) % 127) as u8;
                *u = r;
                *i = (r as i8) - 63;
            }
            let frames = Frames::new(&x_u8, din).expect("valid frames");
            let mut out = vec![0i32; rows * f_n];

            let k0_ns = bench_min_ns(min_dur, || {
                k0.view().gemv(backend, &frames, &mut out).expect("k0 gemv");
            });
            let k0_ps = k0_ns * 1000.0 / macs;
            let _ = k0_ps;

            let mut measure_cb = |cb: &CodebookMatrix, x: &[i8], sparse: bool| -> (f64, f64, f64) {
                // Bit-exactness verify vs scalar first!
                let mut r_ref = vec![0i32; rows * f_n];
                let mut r_fast = vec![0i32; rows * f_n];
                cb.gemv(Backend::Scalar, x, sparse, &mut r_ref)
                    .expect("scalar");
                #[cfg(target_arch = "aarch64")]
                {
                    let tb = cb.build_tables(x, sparse).expect("tables");
                    cb.scan_neon(&tb, sparse, &mut r_fast).expect("neon");
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    cb.gemv(backend, x, sparse, &mut r_fast).expect("simd");
                }
                assert_eq!(r_ref, r_fast, "bit-exact mismatch!");

                let mut tb = ultra_kernels::pqscan::CodebookTables::default();
                cb.build_tables_into(backend, x, sparse, &mut tb)
                    .expect("tables");
                let tbl_ns = bench_min_ns(min_dur, || {
                    cb.build_tables_into(backend, x, sparse, &mut tb)
                        .expect("tables");
                    std::hint::black_box(&tb.entries);
                });
                let scan_ns = bench_min_ns(min_dur, || {
                    #[cfg(target_arch = "aarch64")]
                    cb.scan_neon(&tb, sparse, &mut out).expect("neon");
                    #[cfg(not(target_arch = "aarch64"))]
                    cb.scan_with_tables(backend, &tb, sparse, &mut out)
                        .expect("simd");
                });
                let tot = tbl_ns + scan_ns;
                (k0_ns / tot, 100.0 * tbl_ns / tot, scan_ns * 1000.0 / macs)
            };

            let (sp2, tb2, ps2) = measure_cb(&cb2, &x_i8, false);
            let (sp4, tb4, ps4) = measure_cb(&cb4, &x_i8, false);
            let (sp8, tb8, ps8) = measure_cb(&cb8, &x_i8, false);

            // 80% sparse pairs (ReLU² FFN-out regime).
            let mut x_sp = x_i8.clone();
            let pair_w = 8; // 2 * s=4
            for f in 0..f_n {
                for p in 0..(din / pair_w) {
                    if p % 5 != 0 {
                        x_sp[f * din + p * pair_w..f * din + (p + 1) * pair_w].fill(0);
                    }
                }
            }
            let (sp4_sparse, _tb4_sp, ps4_sp) = measure_cb(&cb4, &x_sp, true);
            let _ = (ps2, ps8);

            println!(
                "{:<12} {:>4} {:>9.2} | {:>6.2}x ({:>4.0}%)   | {:>6.2}x ({:>4.0}%,{:>4.2}ps) | {:>6.2}x ({:>4.0}%)   | {:>5.2}x ({:>4.2}ps)",
                format!("{din}->{rows}"),
                f_n,
                k0_ps,
                sp2,
                tb2,
                sp4,
                tb4,
                ps4,
                sp8,
                tb8,
                sp4_sparse,
                ps4_sp,
            );
        }
    }

    // Measure FWHT16 overhead for 4 frames x 384 dims and 4 frames x 1536 dims.
    for len in [4 * 384usize, 4 * 1536] {
        let mut buf = vec![1.0f32; len];
        let ns = bench_min_ns(min_dur, || {
            fwht16_inplace(&mut buf).expect("fwht16");
        });
        println!(
            "FWHT16 len={len}: {ns:.1} ns ({:.2} ns/16-block)",
            ns / (len as f64 / 16.0)
        );
    }
}
