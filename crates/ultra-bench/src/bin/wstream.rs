//! `wstream`: the spec's Appendix A weight-stream benchmark on the ultra-kernels backends.
//!
//! For each weight count, each frame count F and each kernel, time one GEMV pass of
//! `rows x 1024` weights against F A7 frames: best of 3 repetitions, each looping passes
//! until `min_seconds` have elapsed. All kernels must return the same checksum for a
//! (size, F) point, and when the largest size is 1536M the checksums must equal those
//! printed in Appendix A; otherwise the run aborts with exit code 2.
//!
//! Usage: `wstream [min_seconds] [--sizes 1M,64M,1536M] [--frames 1,2,3,4,6,8,12,16]
//!         [--kernels int8,t1,tlut2] [--threads 1] [--backend avx2|avx512vnni|scalar]`
//! CSV goes to stdout; progress and metadata to stderr.

use std::process::ExitCode;
use std::time::Instant;
use ultra_bench::gen::generate;
use ultra_bench::{appendix_a_checksum, parse_size, size_name, DIN};
use ultra_kernels::{
    gemv_par, Backend, Frames, Gemv, I8Matrix, KernelError, T1Matrix, TLut2Matrix,
};

#[derive(Debug)]
struct Args {
    min_seconds: f64,
    sizes: Vec<usize>,
    frames: Vec<usize>,
    kernels: Vec<String>,
    threads: Vec<usize>,
    backend: Backend,
}

fn list<T>(v: &str, f: impl Fn(&str) -> Option<T>) -> Result<Vec<T>, String> {
    v.split(',')
        .map(|s| f(s.trim()).ok_or_else(|| format!("bad list item {s:?}")))
        .collect()
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        min_seconds: 1.0,
        sizes: list("1M,64M,1536M", parse_size)?,
        frames: vec![1, 2, 3, 4, 6, 8, 12, 16],
        kernels: vec!["int8".into(), "t1".into(), "tlut2".into()],
        threads: vec![1],
        backend: Backend::Avx2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--sizes" => a.sizes = list(&val()?, parse_size)?,
            "--frames" => a.frames = list(&val()?, |s| s.parse().ok().filter(|&f| f >= 1))?,
            "--threads" => a.threads = list(&val()?, |s| s.parse().ok().filter(|&t| t >= 1))?,
            "--kernels" => {
                a.kernels = list(&val()?, |s| {
                    ["int8", "t1", "tlut2"].contains(&s).then(|| s.to_string())
                })?
            }
            "--backend" => {
                let v = val()?;
                a.backend = Backend::ALL
                    .into_iter()
                    .find(|b| b.name() == v)
                    .ok_or_else(|| format!("unknown backend {v}"))?;
            }
            s => a.min_seconds = s.parse().map_err(|_| format!("unknown argument {s}"))?,
        }
    }
    if a.frames.iter().any(|&f| f > ultra_bench::MAX_FRAMES) {
        return Err(format!("frames must be <= {}", ultra_bench::MAX_FRAMES));
    }
    Ok(a)
}

/// One timed point: best seconds per pass and the checksum of the last pass.
fn time_point<M: Gemv>(
    m: M,
    a: &Args,
    x: &Frames<'_>,
    threads: usize,
    out: &mut [i32],
) -> Result<(f64, i64), KernelError> {
    let pass = |out: &mut [i32]| -> Result<i64, KernelError> {
        gemv_par(m, a.backend, x, out, threads)?;
        Ok(out.iter().map(|&v| i64::from(v)).sum())
    };
    let mut cs = pass(out)?; // warm-up, not timed
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let mut passes = 0u32;
        let elapsed = loop {
            cs = pass(out)?;
            passes += 1;
            let e = t0.elapsed().as_secs_f64();
            if e >= a.min_seconds {
                break e;
            }
        };
        best = best.min(elapsed / f64::from(passes));
    }
    Ok((best, cs))
}

struct Mats {
    i8m: Option<I8Matrix>,
    t1: T1Matrix,
    tl: Option<TLut2Matrix>,
}

fn build(a: &Args) -> Result<(Mats, Vec<u8>), KernelError> {
    let maxw = a.sizes.iter().copied().max().unwrap_or(DIN);
    let want = |k: &str| a.kernels.iter().any(|s| s == k);
    let t0 = Instant::now();
    let d = generate(maxw, want("int8"));
    let rows = maxw / DIN;
    let i8m = d.trits.map(|t| I8Matrix::new(rows, DIN, t)).transpose()?;
    let t1 = T1Matrix::from_packed(rows, DIN, d.packed)?;
    let tl = want("tlut2").then(|| TLut2Matrix::from_t1(t1.view()));
    eprintln!(
        "# generated {} weights in {:.1}s",
        size_name(maxw),
        t0.elapsed().as_secs_f64()
    );
    Ok((Mats { i8m, t1, tl }, d.acts))
}

fn run(a: &Args) -> Result<bool, KernelError> {
    let (mats, acts) = build(a)?;
    let maxw = a.sizes.iter().copied().max().unwrap_or(DIN);
    let check_appendix = maxw == 1536 << 20;
    let be = a.backend.name();
    println!("kernel,weights,frames,threads,weight_bytes,ms_per_pass,GMAC_per_s,weight_GB_per_s,checksum");
    for &n in &a.sizes {
        let rows = n / DIN;
        for &threads in &a.threads {
            for &f in &a.frames {
                let x = Frames::new(&acts[..f * DIN], DIN)?;
                let mut out = vec![0i32; rows * f];
                let mut sums = Vec::new();
                for k in &a.kernels {
                    let (bytes, (sec, cs)) = match k.as_str() {
                        "int8" => match &mats.i8m {
                            Some(m) => (
                                n,
                                time_point(
                                    m.view().slice_rows(0, rows)?,
                                    a,
                                    &x,
                                    threads,
                                    &mut out,
                                )?,
                            ),
                            None => continue,
                        },
                        "t1" => (
                            n / 4,
                            time_point(
                                mats.t1.view().slice_rows(0, rows)?,
                                a,
                                &x,
                                threads,
                                &mut out,
                            )?,
                        ),
                        _ => match &mats.tl {
                            Some(m) => (
                                n / 4,
                                time_point(
                                    m.view().slice_rows(0, rows)?,
                                    a,
                                    &x,
                                    threads,
                                    &mut out,
                                )?,
                            ),
                            None => continue,
                        },
                    };
                    let macs = n as f64 * f as f64;
                    println!(
                        "{k}_{be},{},{f},{threads},{bytes},{:.3},{:.1},{:.2},{cs}",
                        size_name(n),
                        sec * 1e3,
                        macs / sec / 1e9,
                        bytes as f64 / sec / 1e9
                    );
                    sums.push((k.clone(), cs));
                }
                if sums.windows(2).any(|w| w[0].1 != w[1].1) {
                    eprintln!("CHECKSUM MISMATCH at {} F={f}: {sums:?}", size_name(n));
                    return Ok(false);
                }
                if let (true, Some(want), Some((_, got))) = (
                    check_appendix,
                    appendix_a_checksum(&size_name(n), f),
                    sums.first(),
                ) {
                    if want != *got {
                        eprintln!(
                            "APPENDIX A MISMATCH at {} F={f}: {got} != {want}",
                            size_name(n)
                        );
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

fn main() -> ExitCode {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("wstream: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!("# {a:?}; available: {:?}", Backend::available());
    match run(&a) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("wstream: {e}");
            ExitCode::from(1)
        }
    }
}
