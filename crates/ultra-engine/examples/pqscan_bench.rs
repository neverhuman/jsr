//! H51 real-shape staged 2-bit / 4-centroid group-4 `pqscan` (`vpshufb` AVX2 LUT lookup +
//! scalar reference + exact LUT construction time) benchmarked directly against `gemm_i8` and
//! `small_m::gemm_i8_small_m` on ULTRA `S` encoder GEMM shapes.

use std::time::{Duration, Instant};

use ultra_engine::kernels::small_m::gemm_i8_small_m;
use ultra_engine::kernels::{gemm_i8, quantize_row, Isa};
use ultra_engine::quant::PackedI8;

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

    fn alloc_lut(&self, m: usize) -> LutBuf {
        let (p, sm) = (self.k / 8, self.stages * m);
        LutBuf(
            vec![0; sm * p * 16],
            vec![0; sm],
            vec![0; sm],
            vec![0; p * 16],
        )
    }

    fn build_lut(&self, a: &[i8], m: usize, buf: &mut LutBuf) {
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
                let sh = (0..16u8).find(|&k| (max_span >> k) <= 127).unwrap_or(16);
                let idx = s * m + r;
                (buf.1[idx], buf.2[idx]) = (sh, bias_sum);
                let dst = &mut buf.0[idx * pairs * 16..(idx + 1) * pairs * 16];
                for (p, &lo) in mins[..pairs].iter().enumerate() {
                    let (src, out_p) = (&buf.3[p * 16..p * 16 + 16], &mut dst[p * 16..p * 16 + 16]);
                    for (d, &v) in out_p.iter_mut().zip(src) {
                        *d = ((v - lo) >> sh) as u8;
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
            // SAFETY: s < stages, b < nb, kb < bytes stay inside self.codes and buf.luts.
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
        println!("H51 staged 2-bit/4-centroid group-4 pqscan vs gemm_i8 (isa={isa_name})\n(M,K,N)        Stg    i8_ns    sm_ns |   lut_ns  scan_ns   tot_ns  lut%  vs_i8");
    }
    let shapes = [
        (4, 384, 1536),
        (4, 1536, 384),
        (1, 384, 1536),
        (1, 1536, 384),
    ];
    for (m, k, n) in shapes {
        let w: Vec<f32> = (0..n * k).map(|_| rf(&mut rng)).collect();
        let pq_i8 = PackedI8::quantize(&w, n, k, None);
        let x: Vec<f32> = (0..m * k).map(|_| rf(&mut rng)).collect();
        let (mut a, mut sa, mut ca) = (vec![0i8; m * pq_i8.kp()], vec![0.0; m], vec![0.0; m]);
        for r in 0..m {
            let dst = &mut a[r * pq_i8.kp()..r * pq_i8.kp() + k];
            (sa[r], ca[r]) = quantize_row(isa, &x[r * k..r * k + k], dst);
        }
        let mut out_f = vec![0.0f32; m * n];
        let i8_ns = bench_ns(dur, || gemm_i8(isa, &pq_i8, &a, &sa, &ca, m, &mut out_f));
        let sm_ns = bench_ns(dur, || {
            gemm_i8_small_m(isa, &pq_i8, &a, &sa, &ca, m, &mut out_f)
        });
        for stages in [1, 2, 4] {
            let mat = StagedPq2::synthetic(n, k, stages, &mut rng);
            let mut lut = mat.alloc_lut(m);
            let (mut r_o, mut s_o) = (vec![0i32; m * n], vec![0i32; m * n]);
            mat.build_lut(&a, m, &mut lut);
            mat.scan_scalar(&lut, m, &mut r_o);
            mat.scan_best(&lut, m, &mut s_o);
            assert_eq!(r_o, s_o, "0-ULP mismatch ({m},{k},{n}) s={stages}");
            let lut_ns = bench_ns(dur, || {
                mat.build_lut(&a, m, &mut lut);
                std::hint::black_box(&lut.0);
            });
            let scan_ns = bench_ns(dur, || {
                mat.scan_best(&lut, m, &mut s_o);
                std::hint::black_box(&s_o);
            });
            let (tot, lp, sp) = (
                lut_ns + scan_ns,
                100.0 * lut_ns / (lut_ns + scan_ns),
                i8_ns / (lut_ns + scan_ns),
            );
            if !as_json {
                let tag = format!("({m},{k},{n})");
                println!("{tag:<14} {stages:>3} {i8_ns:>8.0} {sm_ns:>8.0} | {lut_ns:>8.0} {scan_ns:>8.0} {tot:>8.0} {lp:>4.1}% {sp:>5.2}x");
            } else {
                let (bpw, sms, ssm, scs) = (
                    stages as f64 * 0.5,
                    i8_ns / sm_ns,
                    sm_ns / tot,
                    i8_ns / scan_ns,
                );
                rows.push(format!("{{\"m\":{m},\"k\":{k},\"n\":{n},\"stages\":{stages},\"bpw\":{bpw:.1},\"i8_ns\":{i8_ns:.1},\"small_m_ns\":{sm_ns:.1},\"small_m_speedup\":{sms:.4},\"lut_ns\":{lut_ns:.1},\"scan_ns\":{scan_ns:.1},\"total_ns\":{tot:.1},\"lut_pct\":{lp:.2},\"speedup_vs_i8\":{sp:.4},\"speedup_vs_small_m\":{ssm:.4},\"scan_only_speedup_vs_i8\":{scs:.4},\"parity_exact\":true}}"));
            }
        }
    }
    if as_json {
        let (arch, joined) = (std::env::consts::ARCH, rows.join(","));
        println!("{{\"arch\":\"{arch}\",\"isa\":\"{isa_name}\",\"results\":[{joined}]}}");
    }
}
