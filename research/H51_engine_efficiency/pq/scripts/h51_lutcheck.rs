//! h51-pq scratch (not for the repo): run the real pqscan kernel (scalar and AVX2) on a test case
//! written by the PyTorch emulation and print the i32 accumulators, to prove the emulation exact.
//! input file: u32 rows, din, s, m, frames; i8 codebooks [m][c][16][s]; u8 codes [m][rows][c]; i8 x [frames][din]
use std::io::Read;
use ultra_kernels::pqscan::CodebookMatrix;
use ultra_kernels::Backend;

fn main() {
    let path = std::env::args().nth(1).expect("input file");
    let mut b = Vec::new();
    std::fs::File::open(&path).unwrap().read_to_end(&mut b).unwrap();
    let u = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]) as usize;
    let (rows, din, s, m, frames) = (u(0), u(1), u(2), u(3), u(4));
    let c = din / s;
    let mut off = 20;
    let ncb = m * c * 16 * s;
    let cb: Vec<i8> = b[off..off + ncb].iter().map(|&v| v as i8).collect();
    off += ncb;
    let ncode = m * rows * c;
    let codes: Vec<u8> = b[off..off + ncode].to_vec();
    off += ncode;
    let x: Vec<i8> = b[off..off + frames * din].iter().map(|&v| v as i8).collect();
    let mat = CodebookMatrix::from_parts(rows, din, s, m, cb, &codes).unwrap();
    let mut o_s = vec![0i32; rows * frames];
    let mut o_v = vec![0i32; rows * frames];
    mat.gemv(Backend::Scalar, &x, false, &mut o_s).unwrap();
    mat.gemv(Backend::detect(), &x, false, &mut o_v).unwrap();
    assert_eq!(o_s, o_v, "scalar != simd");
    let mut out = Vec::with_capacity(rows * frames * 4);
    for v in &o_s {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(format!("{path}.acc"), out).unwrap();
    println!("ok rows={rows} din={din} s={s} m={m} frames={frames} backend={:?}", Backend::detect());
}
