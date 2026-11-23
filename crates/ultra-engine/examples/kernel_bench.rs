//! Micro-benchmark of the GEMM kernels (exploratory): hot (weights resident in L1/L2) versus
//! streaming (working set larger than L3), plus `--compare` for ISA & `small_m` A/B receipts.
//!
//! `cargo run --release -p ultra-engine --example kernel_bench [-- --compare [--json]]`

use std::time::{Duration, Instant};

use ultra_engine::kernels::small_m::gemm_i8_small_m;
use ultra_engine::kernels::{gemm_f32, gemm_i8, gemm_i8_a7, quantize_row, Isa, PackedF32};
use ultra_engine::quant::PackedI8;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn bench_gmac(min_dur: Duration, macs_per_call: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut best_ns = f64::INFINITY;
    for _ in 0..3 {
        let mut iters = 0u64;
        let t0 = Instant::now();
        while t0.elapsed() < min_dur {
            f();
            iters += 1;
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best_ns {
            best_ns = ns;
        }
    }
    (macs_per_call as f64 / best_ns, best_ns)
}

fn run_compare(as_json: bool) {
    let mut seed = 42u64;
    let isas: Vec<Isa> = [Isa::Avx512Vnni, Isa::Avx2Fma, Isa::Neon, Isa::Scalar]
        .into_iter()
        .filter(|i| i.is_available())
        .collect();
    let mut rows_json = Vec::new();
    for &(n, k, copies, label) in &[
        (1536usize, 384usize, 1usize, "hot_1536x384"),
        (384, 1536, 1, "hot_384x1536"),
        (1536, 384, 64, "stream_64x1536x384"),
    ] {
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let pq = PackedI8::quantize(&w, n, k, None);
        let qs: Vec<PackedI8> = (0..copies).map(|_| pq.clone()).collect();
        for &rows in &[1usize, 2, 4, 8, 16] {
            let x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut seed)).collect();
            let mut a = vec![0i8; rows * pq.kp()];
            let mut sa = vec![0.0f32; rows];
            let mut ca = vec![0.0f32; rows];
            for r in 0..rows {
                (sa[r], ca[r]) = quantize_row(
                    Isa::detect(),
                    &x[r * k..(r + 1) * k],
                    &mut a[r * pq.kp()..r * pq.kp() + k],
                );
            }
            let mut out = vec![0.0f32; rows * n];
            let macs = rows * n * k * copies;
            let mut isa_gmac = Vec::new();
            for &isa in &isas {
                if isa == Isa::Scalar && (copies > 1 || rows > 4) {
                    continue;
                }
                let dur = if isa == Isa::Scalar {
                    Duration::from_millis(25)
                } else {
                    Duration::from_millis(40)
                };
                let (g, ns) = bench_gmac(dur, macs, || {
                    for q in &qs {
                        gemm_i8(isa, q, &a, &sa, &ca, rows, &mut out);
                    }
                });
                let sm = if rows <= 4 && isa != Isa::Scalar {
                    let (gs, nss) = bench_gmac(dur, macs, || {
                        for q in &qs {
                            gemm_i8_small_m(isa, q, &a, &sa, &ca, rows, &mut out);
                        }
                    });
                    Some((gs, nss))
                } else {
                    None
                };
                isa_gmac.push((isa.name(), g, ns, sm));
            }
            if !as_json {
                print!("{label:20} M={rows:2}:");
                for &(name, g, _, sm) in &isa_gmac {
                    if let Some((gs, _)) = sm {
                        print!("  {name}={g:6.1} (small_m={gs:6.1}, {:.2}x)", gs / g);
                    } else {
                        print!("  {name}={g:6.1}");
                    }
                }
                println!(" GMAC/s");
            } else {
                let mut entries = Vec::new();
                for &(name, g, ns, sm) in &isa_gmac {
                    let sm_s = match sm {
                        Some((gs, nss)) => format!(
                            ",\"small_m_gmac_s\":{gs:.3},\"small_m_ns\":{nss:.1},\"small_m_speedup\":{:.4}",
                            gs / g
                        ),
                        None => String::new(),
                    };
                    entries.push(format!(
                        "\"{name}\":{{\"gmac_s\":{g:.3},\"ns\":{ns:.1}{sm_s}}}"
                    ));
                }
                rows_json.push(format!(
                    "{{\"label\":\"{label}\",\"n\":{n},\"k\":{k},\"copies\":{copies},\"m\":{rows},\"isas\":{{{}}}}}",
                    entries.join(",")
                ));
            }
        }
    }
    if as_json {
        println!(
            "{{\"arch\":\"{}\",\"detected_isa\":\"{}\",\"results\":[{}]}}",
            std::env::consts::ARCH,
            Isa::detect().name(),
            rows_json.join(",")
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--compare") {
        run_compare(args.iter().any(|a| a == "--json"));
        return;
    }
    let isa = Isa::detect();
    let mut seed = 1u64;
    for &(n, k, copies, label) in &[
        (1536usize, 384usize, 1usize, "hot 1536x384"),
        (384, 1536, 1, "hot 384x1536"),
        (1536, 384, 96, "stream 96 x 1536x384 (54 MiB i8)"),
    ] {
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let pf = PackedF32::pack(&w, n, k, None);
        let pq = PackedI8::quantize(&w, n, k, None);
        let fs: Vec<PackedF32> = (0..copies.min(24)).map(|_| pf.clone()).collect();
        let qs: Vec<PackedI8> = (0..copies).map(|_| pq.clone()).collect();
        for &rows in &[4usize, 16, 64] {
            let x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut seed)).collect();
            let mut a = vec![0i8; rows * pq.kp()];
            let mut sa = vec![0.0f32; rows];
            let mut ca = vec![0.0f32; rows];
            for r in 0..rows {
                (sa[r], ca[r]) = quantize_row(
                    isa,
                    &x[r * k..(r + 1) * k],
                    &mut a[r * pq.kp()..r * pq.kp() + k],
                );
            }
            let mut out = vec![0.0f32; rows * n];
            let reps = (2_000_000_000 / (rows * n * k * copies)).max(3);
            let t = Instant::now();
            for _ in 0..reps {
                for q in &qs {
                    gemm_i8(isa, q, &a, &sa, &ca, rows, &mut out);
                }
            }
            let gi = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            let a7: Vec<i8> = a.iter().map(|&v| (v / 2).clamp(-63, 63)).collect();
            let t = Instant::now();
            for _ in 0..reps {
                for q in &qs {
                    gemm_i8_a7(isa, q, &a7, &sa, &ca, rows, &mut out);
                }
            }
            let g7 = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            let reps_f = (reps / 4).max(2);
            let t = Instant::now();
            for _ in 0..reps_f {
                for f in &fs {
                    gemm_f32(isa, f, &x, rows, &mut out);
                }
            }
            let gf = (reps_f * fs.len() * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!(
                "{label:36} rows {rows:3}: int8 {gi:6.1}  int8-a7 {g7:6.1}  fp32 {gf:6.1} GMAC/s"
            );
        }
    }
}
