//! H51 real-shape staged 2-bit / 4-centroid group-4 `pqscan` (`vpshufb` AVX2 LUT lookup +
//! scalar reference + exact LUT construction time) comparing H24 `centred_trunc`, `@h51-pq`
//! `nearest_no_bias`, and `fwht16_nearest` directly against `gemm_i8` and `small_m`.

use std::time::{Duration, Instant};

use ultra_engine::kernels::small_m::gemm_i8_small_m;
use ultra_engine::kernels::{gemm_i8, quantize_row, Isa};
use ultra_engine::quant::PackedI8;
use ultra_kernels::pqscan::fwht16_inplace;

const BLOCK_N: usize = 32;

fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    *s
}

struct StagedPq2 {
    n: usize,
    k: usize,
    stages: usize,
    cb: Vec<i8>,
    codes: Vec<u8>,
}

struct LutBuf(Vec<u8>, Vec<u8>, Vec<i32>, Vec<i32>);

impl StagedPq2 {
    fn synthetic(n: usize, k: usize, stages: usize, rng: &mut u64) -> Self {
        assert!(n.is_multiple_of(BLOCK_N) && k.is_multiple_of(16));
        let cb = (0..stages * (k / 4) * 16)
            .map(|_| (((next_u64(rng) >> 33) % 255) as i32 - 127) as i8)
            .collect();
        let codes = (0..stages * (n / BLOCK_N) * (k / 16) * BLOCK_N)
            .map(|_| (next_u64(rng) >> 33) as u8)
            .collect();
        Self {
            n,
            k,
            stages,
            cb,
            codes,
        }
    }

    fn build_lut(&self, a: &[i8], m: usize, nearest: bool, buf: &mut LutBuf) {
        let dot4 =
            |x: &[i8], c: &[i8]| -> i32 { (0..4).map(|i| i32::from(x[i]) * i32::from(c[i])).sum() };
        let (groups, pairs, mut mins) = (self.k / 4, self.k / 8, [0i32; 256]);
        for s in 0..self.stages {
            let cb_s = &self.cb[s * groups * 16..(s + 1) * groups * 16];
            for r in 0..m {
                let ar = &a[r * self.k..(r + 1) * self.k];
                let (mut max_span, mut bias_sum) = (0i32, 0i32);
                for (p, min_slot) in mins[..pairs].iter_mut().enumerate() {
                    let (a0, a1) = (&ar[p * 8..p * 8 + 4], &ar[p * 8 + 4..p * 8 + 8]);
                    let (c0, c1) = (&cb_s[p * 32..p * 32 + 16], &cb_s[p * 32 + 16..p * 32 + 32]);
                    let d0: [i32; 4] = std::array::from_fn(|c| dot4(a0, &c0[c * 4..c * 4 + 4]));
                    let d1: [i32; 4] = std::array::from_fn(|c| dot4(a1, &c1[c * 4..c * 4 + 4]));
                    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
                    let dst = &mut buf.3[p * 16..(p + 1) * 16];
                    for (i1, &v1) in d1.iter().enumerate() {
                        for (i0, &v0) in d0.iter().enumerate() {
                            let v = v0 + v1;
                            dst[i0 | (i1 << 2)] = v;
                            (lo, hi) = (lo.min(v), hi.max(v));
                        }
                    }
                    (*min_slot, bias_sum, max_span) = (lo, bias_sum + lo, max_span.max(hi - lo));
                }
                let sh = (0..16u8)
                    .find(|&k| {
                        let add = if nearest && k > 0 { 1 << (k - 1) } else { 0 };
                        ((max_span + add) >> k) <= 127
                    })
                    .unwrap_or(16);
                let half = if sh > 0 { 1i32 << (sh - 1) } else { 0 };
                let idx = s * m + r;
                let (add, bias_add) = if nearest {
                    (half, 0)
                } else {
                    (0, (pairs as i32) * half)
                };
                (buf.1[idx], buf.2[idx]) = (sh, bias_sum + bias_add);
                let dst = &mut buf.0[idx * pairs * 16..(idx + 1) * pairs * 16];
                for (p, &lo) in mins[..pairs].iter().enumerate() {
                    let (src, out_p) = (&buf.3[p * 16..p * 16 + 16], &mut dst[p * 16..p * 16 + 16]);
                    for (d, &v) in out_p.iter_mut().zip(src) {
                        *d = ((v - lo + add) >> sh).min(127) as u8;
                    }
                }
            }
        }
    }

    fn scan_scalar(&self, buf: &LutBuf, m: usize, out: &mut [i32]) {
        let (bytes, pairs, nb) = (self.k / 16, self.k / 8, self.n / BLOCK_N);
        out[..m * self.n].fill(0);
        for s in 0..self.stages {
            let cs = &self.codes[s * nb * bytes * BLOCK_N..];
            for b in 0..nb {
                let blk = &cs[b * bytes * BLOCK_N..(b + 1) * bytes * BLOCK_N];
                for r in 0..m {
                    let (idx, sh, bias) = (s * m + r, buf.1[s * m + r], buf.2[s * m + r]);
                    let lut = &buf.0[idx * pairs * 16..(idx + 1) * pairs * 16];
                    for j in 0..BLOCK_N {
                        let acc = (0..bytes).fold(0u16, |a, kb| {
                            let c = blk[kb * BLOCK_N + j];
                            let v = lut[kb * 32 + (c & 0xf) as usize]
                                + lut[kb * 32 + 16 + (c >> 4) as usize];
                            a.wrapping_add(u16::from(v))
                        });
                        out[r * self.n + b * BLOCK_N + j] += (i32::from(acc) << sh) + bias;
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn scan_avx2(&self, buf: &LutBuf, m: usize, out: &mut [i32]) {
        use std::arch::x86_64::*;
        let (bytes, pairs, nb) = (self.k / 16, self.k / 8, self.n / BLOCK_N);
        out[..m * self.n].fill(0);
        let mask = _mm256_set1_epi8(0x0f);
        for s in 0..self.stages {
            // SAFETY: s < stages, b < nb, kb < bytes stay inside self.codes and buf.0.
            let cs = unsafe { self.codes.as_ptr().add(s * nb * bytes * BLOCK_N) };
            for b in 0..nb {
                let blk = unsafe { cs.add(b * bytes * BLOCK_N) };
                let [mut even, mut odd] = [[_mm256_setzero_si256(); 4]; 2];
                for kb in 0..bytes {
                    let cb = unsafe { _mm256_loadu_si256(blk.add(kb * BLOCK_N).cast()) };
                    let lo = _mm256_and_si256(cb, mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(cb), mask);
                    for r in 0..m {
                        let lp = unsafe { buf.0.as_ptr().add((s * m + r) * pairs * 16 + kb * 32) };
                        let t0 = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(lp.cast()) });
                        let t1 = _mm256_broadcastsi128_si256(unsafe {
                            _mm_loadu_si128(lp.add(16).cast())
                        });
                        let s8 = _mm256_add_epi8(
                            _mm256_shuffle_epi8(t0, lo),
                            _mm256_shuffle_epi8(t1, hi),
                        );
                        odd[r] = _mm256_add_epi16(odd[r], _mm256_srli_epi16::<8>(s8));
                        even[r] = _mm256_add_epi16(even[r], s8);
                    }
                }
                for r in 0..m {
                    let idx = s * m + r;
                    let ev = _mm256_sub_epi16(even[r], _mm256_slli_epi16::<8>(odd[r]));
                    let (sh, bias, mut e16, mut o16) =
                        (buf.1[idx], buf.2[idx], [0u16; 16], [0u16; 16]);
                    unsafe {
                        _mm256_storeu_si256(e16.as_mut_ptr().cast(), ev);
                        _mm256_storeu_si256(o16.as_mut_ptr().cast(), odd[r]);
                    }
                    let row = &mut out[r * self.n + b * BLOCK_N..r * self.n + (b + 1) * BLOCK_N];
                    for i in 0..16 {
                        row[2 * i] += (i32::from(e16[i]) << sh) + bias;
                        row[2 * i + 1] += (i32::from(o16[i]) << sh) + bias;
                    }
                }
            }
        }
    }

    fn scan_best(&self, buf: &LutBuf, m: usize, out: &mut [i32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && m <= 4 {
            // SAFETY: AVX2 checked above and m <= 4.
            return unsafe { self.scan_avx2(buf, m, out) };
        }
        self.scan_scalar(buf, m, out);
    }

    fn run_var(&self, a: &[i8], m: usize, nearest: bool, dur: Duration) -> (f64, f64, u8) {
        let (p, sm) = (self.k / 8, self.stages * m);
        let mut lut = LutBuf(
            vec![0; sm * p * 16],
            vec![0; sm],
            vec![0; sm],
            vec![0; p * 16],
        );
        let (mut r_o, mut s_o) = (vec![0i32; m * self.n], vec![0i32; m * self.n]);
        self.build_lut(a, m, nearest, &mut lut);
        self.scan_scalar(&lut, m, &mut r_o);
        self.scan_best(&lut, m, &mut s_o);
        assert_eq!(r_o, s_o, "0-ULP mismatch ({m},{},{})", self.k, self.n);
        let lut_ns = bench_ns(dur, || {
            self.build_lut(a, m, nearest, &mut lut);
            std::hint::black_box(&lut.0);
        });
        let scan_ns = bench_ns(dur, || {
            self.scan_best(&lut, m, &mut s_o);
            std::hint::black_box(&s_o);
        });
        (lut_ns, scan_ns, lut.1[0])
    }
}

fn bench_ns(min_dur: Duration, mut f: impl FnMut()) -> f64 {
    f();
    (0..5).fold(f64::INFINITY, |best, _| {
        let (mut iters, t0) = (0u64, Instant::now());
        while t0.elapsed() < min_dur {
            f();
            iters += 1;
        }
        best.min(t0.elapsed().as_nanos() as f64 / iters as f64)
    })
}

fn main() {
    let as_json = std::env::args().any(|a| a == "--json");
    let (isa, dur, mut rng) = (Isa::detect(), Duration::from_millis(30), 0x51_2026_1123u64);
    let (isa_name, mut rows) = (isa.name(), Vec::new());
    let rf = |s: &mut u64| ((next_u64(s) >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    if !as_json {
        println!("H51 staged 2-bit/4-centroid group-4 pqscan (centred_trunc vs nearest_no_bias vs fwht16_nearest, isa={isa_name})\n(M,K,N)        Stg    i8_ns | trunc_tot  near_tot (lutΔ) | fwht_ns fwht_tot  vs_i8");
    }
    for (m, k, n) in [
        (4, 384, 1536),
        (4, 1536, 384),
        (1, 384, 1536),
        (1, 1536, 384),
    ] {
        let w: Vec<f32> = (0..n * k).map(|_| rf(&mut rng)).collect();
        let pq_i8 = PackedI8::quantize(&w, n, k, None);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let x: Vec<f32> = (0..m * k).map(|_| silu(rf(&mut rng) * 6.0)).collect();
        let mut x_rot = x.clone();
        let fwht_ns = bench_ns(dur, || {
            x_rot.copy_from_slice(&x);
            fwht16_inplace(&mut x_rot).expect("fwht16");
            std::hint::black_box(&x_rot);
        });
        let (mut a, mut a_rot, mut sa, mut ca) = (
            vec![0i8; m * pq_i8.kp()],
            vec![0i8; m * k],
            vec![0.0; m],
            vec![0.0; m],
        );
        for r in 0..m {
            (sa[r], ca[r]) = quantize_row(
                isa,
                &x[r * k..r * k + k],
                &mut a[r * pq_i8.kp()..r * pq_i8.kp() + k],
            );
            let _ = quantize_row(isa, &x_rot[r * k..r * k + k], &mut a_rot[r * k..r * k + k]);
        }
        let mut out_f = vec![0.0f32; m * n];
        let i8_ns = bench_ns(dur, || gemm_i8(isa, &pq_i8, &a, &sa, &ca, m, &mut out_f));
        let sm_ns = bench_ns(dur, || {
            gemm_i8_small_m(isa, &pq_i8, &a, &sa, &ca, m, &mut out_f)
        });
        for stages in [1, 2, 4] {
            let mat = StagedPq2::synthetic(n, k, stages, &mut rng);
            let (t_lut, t_scan, t_sh) = mat.run_var(&a, m, false, dur);
            let (n_lut, n_scan, n_sh) = mat.run_var(&a, m, true, dur);
            let (f_lut, f_scan, f_sh) = mat.run_var(&a_rot, m, true, dur);
            let (t_tot, n_tot, f_tot) = (t_lut + t_scan, n_lut + n_scan, fwht_ns + f_lut + f_scan);
            let (n_d, f_sp) = (100.0 * (n_lut - t_lut) / t_lut, i8_ns / f_tot);
            if !as_json {
                let tag = format!("({m},{k},{n})");
                println!("{tag:<14} {stages:>3} {i8_ns:>8.0} | {t_tot:>9.0} {n_tot:>9.0} ({n_d:+5.1}%) | {fwht_ns:>7.0} {f_tot:>8.0} {f_sp:>5.2}x");
            } else {
                let (bpw, sms, t_sp, n_sp) = (
                    stages as f64 * 0.5,
                    i8_ns / sm_ns,
                    i8_ns / t_tot,
                    i8_ns / n_tot,
                );
                let (t_lp, n_lp, f_lp, f_ov) = (
                    100.0 * t_lut / t_tot,
                    100.0 * n_lut / n_tot,
                    100.0 * (fwht_ns + f_lut) / f_tot,
                    100.0 * (f_tot - t_tot) / t_tot,
                );
                rows.push(format!("{{\"m\":{m},\"k\":{k},\"n\":{n},\"stages\":{stages},\"bpw\":{bpw:.1},\"i8_ns\":{i8_ns:.1},\"small_m_ns\":{sm_ns:.1},\"small_m_speedup\":{sms:.4},\"centred_trunc\":{{\"lut_ns\":{t_lut:.1},\"scan_ns\":{t_scan:.1},\"total_ns\":{t_tot:.1},\"lut_pct\":{t_lp:.2},\"speedup_vs_i8\":{t_sp:.4},\"shift\":{t_sh},\"parity_exact\":true}},\"nearest_no_bias\":{{\"lut_ns\":{n_lut:.1},\"scan_ns\":{n_scan:.1},\"total_ns\":{n_tot:.1},\"lut_pct\":{n_lp:.2},\"lut_delta_pct\":{n_d:.2},\"speedup_vs_i8\":{n_sp:.4},\"shift\":{n_sh},\"parity_exact\":true}},\"fwht16_nearest\":{{\"fwht16_ns\":{fwht_ns:.1},\"lut_ns\":{f_lut:.1},\"scan_ns\":{f_scan:.1},\"total_ns\":{f_tot:.1},\"prep_plus_lut_pct\":{f_lp:.2},\"total_delta_pct\":{f_ov:.2},\"speedup_vs_i8\":{f_sp:.4},\"shift\":{f_sh},\"parity_exact\":true}}}}"));
            }
        }
    }
    if as_json {
        let (arch, joined) = (std::env::consts::ARCH, rows.join(","));
        println!("{{\"arch\":\"{arch}\",\"isa\":\"{isa_name}\",\"results\":[{joined}]}}");
    }
}
