//! Causal stem: two 3x3 stride-2 convolutions (layer 2 as an im2col GEMM) and the
//! projection to `d`, for the feature slice `4 t0 - 6 ..= 4 (t0 + n) - 4` of each job.

use super::{grow, linear, ChunkJob, EncWorkspace, Precision};
use crate::kernels::{self, Isa};
use crate::model::Model;
use crate::prof::Phase;

/// Time `$e` into phase `$p` of `$prof` when profiling is on.
macro_rules! timed {
    ($prof:expr, $p:expr, $e:expr) => {{
        let t = $prof.start();
        let r = $e;
        $prof.stop($p, t);
        r
    }};
}

pub(super) fn stem(
    m: &Model,
    isa: Isa,
    prec: Precision,
    jobs: &[ChunkJob<'_>],
    ws: &mut EncWorkspace,
    rows: usize,
) {
    let a = &m.arch;
    let (ch, f1, f2, nm) = (a.stem_ch, a.f1, a.f2, a.n_mels);
    let t_stem = ws.q.prof.start();
    let col = grow(&mut ws.col, rows * f2 * 9 * ch);
    let mut crow = 0;
    for j in jobs {
        let n1 = 2 * j.n + 1;
        let c1 = grow(&mut ws.c1, n1 * f1 * ch);
        // Layer 1: out[j1][f][c], channel-last.
        for r in 0..n1 {
            let zero = 2 * j.state.t + r < 2;
            for f in 0..f1 {
                let o = &mut c1[(r * f1 + f) * ch..(r * f1 + f + 1) * ch];
                if zero {
                    o.fill(0.0);
                    continue;
                }
                o.copy_from_slice(&m.c1_b);
                for kh in 0..3 {
                    for kw in 0..3 {
                        let xv = j.feats[(2 * r + kh) * nm + 2 * f + kw];
                        let t = kh * 3 + kw;
                        for (ov, &wv) in o.iter_mut().zip(&m.c1_w[t * ch..(t + 1) * ch]) {
                            *ov += wv * xv;
                        }
                    }
                }
                kernels::silu_inplace(isa, o);
            }
        }
        // im2col for layer 2: row (i, f), K = (kh, kw, c).
        for i in 0..j.n {
            for f in 0..f2 {
                let dst = &mut col[(crow + i * f2 + f) * 9 * ch..][..9 * ch];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let src = ((2 * i + kh) * f1 + 2 * f + kw) * ch;
                        dst[(kh * 3 + kw) * ch..][..ch].copy_from_slice(&c1[src..src + ch]);
                    }
                }
            }
        }
        crow += j.n * f2;
    }
    ws.q.prof.stop(Phase::Stem1, t_stem);
    let c2 = grow(&mut ws.c2, rows * f2 * ch);
    linear(isa, prec, &m.c2, col, rows * f2, c2, &mut ws.q);
    timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, c2));
    let x = grow(&mut ws.x, rows * a.d);
    linear(isa, prec, &m.stem_out, c2, rows, x, &mut ws.q);
}
