//! Per-kernel output digests on fixed pseudo-random inputs, for cross-ISA bit-identity
//! checks (run on each host and compare): `cargo run --release --example kernel_digest`.

use ultra_core::receipt::sha256_hex;
use ultra_engine::kernels::{self, Isa, PackedF32};
use ultra_engine::quant::PackedI8;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
}

fn h(v: &[f32]) -> String {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    sha256_hex(&b)[..16].to_string()
}

fn main() {
    let isa = Isa::detect();
    let mut s = 7u64;
    let (rows, n, k, d) = (7usize, 100usize, 384usize, 384usize);
    let x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut s)).collect();
    let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.05).collect();
    let b: Vec<f32> = (0..n).map(|_| lcg(&mut s)).collect();
    let pf = PackedF32::pack(&w, n, k, Some(&b));
    let mut o = vec![0.0; rows * n];
    kernels::gemm_f32(isa, &pf, &x, rows, &mut o);
    println!("{} gemm_f32 {}", isa.name(), h(&o));
    let g: Vec<f32> = (0..d).map(|_| lcg(&mut s)).collect();
    let bb: Vec<f32> = (0..d).map(|_| lcg(&mut s)).collect();
    let mut ln = vec![0.0; rows * d];
    kernels::layer_norm(isa, &x, d, &g, &bb, &mut ln);
    println!("{} layer_norm {}", isa.name(), h(&ln));
    let mut si = x.clone();
    kernels::silu_inplace(isa, &mut si);
    println!("{} silu {}", isa.name(), h(&si));
    let mut gl = vec![0.0; rows * d / 2];
    kernels::glu(isa, &x, d / 2, &mut gl);
    println!("{} glu {}", isa.name(), h(&gl));
    let mut y = x.clone();
    kernels::axpy(isa, 0.5, &ln, &mut y);
    println!("{} axpy {}", isa.name(), h(&y));
    let pq = PackedI8::quantize(&w, n, k, Some(&b));
    let mut oq = vec![0.0; rows * n];
    let mut a = vec![0i8; rows * pq.kp()];
    let (mut sa, mut ca) = (vec![0.0; rows], vec![0.0; rows]);
    for r in 0..rows {
        (sa[r], ca[r]) = kernels::quantize_row(
            isa,
            &x[r * k..(r + 1) * k],
            &mut a[r * pq.kp()..r * pq.kp() + k],
        );
    }
    kernels::gemm_i8(isa, &pq, &a, &sa, &ca, rows, &mut oq);
    println!("{} quant+gemm_i8 {}", isa.name(), h(&oq));
    let (dh, ring, qn) = (64usize, 136usize, 4usize);
    let q: Vec<f32> = (0..qn * 3 * dh).map(|_| lcg(&mut s)).collect();
    let kt: Vec<f32> = (0..dh * ring).map(|_| lcg(&mut s)).collect();
    let vr: Vec<f32> = (0..ring * dh).map(|_| lcg(&mut s)).collect();
    let (start, nk) = (100usize, 132usize);
    let first = nk.min(ring - start);
    let mut sc = vec![0.0; qn * nk];
    let mut out = vec![0.0; qn * dh];
    kernels::attend_head(
        isa,
        &q,
        3 * dh,
        qn,
        dh,
        &kt,
        ring,
        [(start, first), (0, nk - first)],
        [
            &vr[start * dh..(start + first) * dh],
            &vr[..(nk - first) * dh],
        ],
        0.125,
        &mut sc,
        &mut out,
        dh,
    );
    println!("{} attention {}", isa.name(), h(&out));
}
