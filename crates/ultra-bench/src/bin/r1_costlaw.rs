//! `r1_costlaw`: Real-layer `160 ms` Conformer linear-stack speed benchmark comparing
//! `I8Matrix` (`1 byte/weight`, ~52 MiB, overflows 16 MiB Zen 2 CCX L3) against
//! `TLut2Matrix` (`2 bits/weight`, ~13.08 MiB, L3-resident inside a single Zen 2 CCX!)
//! across `R1` Dense `S` (`25 Hz`) and Multirate `S_MR` (`25 Hz` + `12.5 Hz` middle 10 blocks).
//!
//! Usage:
//!   `r1_costlaw [--profile s|s-mr|both] [--streams 1,4,8] [--threads 1,2,4] [--min-seconds 0.5]`

use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;
use ultra_kernels::{gemv_par, Backend, Frames, Gemv, I8Matrix, KernelError, TLut2Matrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rate {
    Hz25,
    Hz12_5,
}

struct LayerPair {
    rate: Rate,
    i8_mat: I8Matrix,
    tlut2_mat: TLut2Matrix,
}

#[derive(Debug)]
struct Args {
    profiles: Vec<String>,
    streams: Vec<usize>,
    threads: Vec<usize>,
    min_seconds: f64,
}

#[derive(Debug, Serialize)]
struct Row {
    measurement_scope: &'static str,
    profile: String,
    weights_total: usize,
    int8_mib: f64,
    tlut2_mib: f64,
    streams: usize,
    threads: usize,
    int8_chunk_ms: f64,
    tlut2_chunk_ms: f64,
    int8_rtfx: f64,
    tlut2_rtfx: f64,
    tlut2_speedup_x: f64,
}

fn parse_csv_usize(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .ok()
                .filter(|&v| v >= 1)
                .ok_or_else(|| format!("invalid positive integer {p:?}"))
        })
        .collect()
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        profiles: vec!["s".into(), "s-mr".into()],
        streams: vec![1, 4, 8],
        threads: vec![1, 2, 4],
        min_seconds: 0.4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{k} requires a value"));
        match k.as_str() {
            "--profile" => {
                let v = val()?;
                args.profiles = match v.as_str() {
                    "s" => vec!["s".into()],
                    "s-mr" => vec!["s-mr".into()],
                    "both" => vec!["s".into(), "s-mr".into()],
                    other => {
                        return Err(format!("unknown --profile {other} (expected s|s-mr|both)"))
                    }
                };
            }
            "--streams" => args.streams = parse_csv_usize(&val()?)?,
            "--threads" => args.threads = parse_csv_usize(&val()?)?,
            "--min-seconds" => {
                args.min_seconds = val()?
                    .parse::<f64>()
                    .ok()
                    .filter(|&x| x > 0.0 && x.is_finite())
                    .ok_or_else(|| "bad --min-seconds".to_owned())?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn make_layer(seed: u64, rows: usize, din: usize, rate: Rate) -> Result<LayerPair, KernelError> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut trits = Vec::with_capacity(rows * din);
    for _ in 0..(rows * din) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D) % 4;
        // ~50% zeros, 25% +1, 25% -1 (realistic ternary sparsity).
        let t = match r {
            0 => -1i8,
            1 => 1i8,
            _ => 0i8,
        };
        trits.push(t);
    }
    let tlut2_mat = TLut2Matrix::from_trits(rows, din, &trits)?;
    let i8_mat = I8Matrix::new(rows, din, trits)?;
    Ok(LayerPair {
        rate,
        i8_mat,
        tlut2_mat,
    })
}

fn append_conformer_block(
    layers: &mut Vec<LayerPair>,
    seed: &mut u64,
    total_w: &mut usize,
    rate: Rate,
) -> Result<(), KernelError> {
    let block_shapes: [(usize, usize); 8] = [
        (1536, 384),
        (384, 1536),
        (1152, 384),
        (384, 384),
        (768, 384),
        (384, 384),
        (1536, 384),
        (384, 1536),
    ];
    for &(r, d) in &block_shapes {
        layers.push(make_layer(*seed, r, d, rate)?);
        *seed += 1;
        *total_w += r * d;
    }
    Ok(())
}

fn build_conformer_stack(multirate: bool) -> Result<(Vec<LayerPair>, usize), KernelError> {
    let mut layers = Vec::with_capacity(132);
    let mut seed = 1u64;
    let mut total_w = 0usize;

    if !multirate {
        for _ in 0..16 {
            append_conformer_block(&mut layers, &mut seed, &mut total_w, Rate::Hz25)?;
        }
    } else {
        for _ in 0..4 {
            append_conformer_block(&mut layers, &mut seed, &mut total_w, Rate::Hz25)?;
        }
        layers.push(make_layer(seed, 384, 768, Rate::Hz12_5)?);
        seed += 1;
        total_w += 384 * 768;
        for _ in 0..10 {
            append_conformer_block(&mut layers, &mut seed, &mut total_w, Rate::Hz12_5)?;
        }
        layers.push(make_layer(seed, 768, 384, Rate::Hz12_5)?);
        seed += 1;
        total_w += 768 * 384;
        for _ in 0..2 {
            append_conformer_block(&mut layers, &mut seed, &mut total_w, Rate::Hz25)?;
        }
    }
    Ok((layers, total_w))
}

fn make_a7_frames(din: usize, f_count: usize) -> Vec<u8> {
    (0..(din * f_count))
        .map(|i| ((i * 37 + 11) % 128) as u8)
        .collect()
}

fn run_stack_once(
    layers: &[LayerPair],
    use_tlut2: bool,
    backend: Backend,
    streams: usize,
    threads: usize,
    acts: &[(usize, Vec<u8>, Vec<u8>)],
    out: &mut [i32],
) -> Result<i64, KernelError> {
    let f25 = 4 * streams;
    let f12 = 2 * streams;
    let mut checksum = 0i64;
    for lp in layers {
        let din = lp.i8_mat.view().din();
        let rows = lp.i8_mat.view().rows();
        let (_, a25, a12) = acts
            .iter()
            .find(|(d, _, _)| *d == din)
            .ok_or(KernelError::Shape("missing activation width"))?;
        let (act_buf, f_cnt) = match lp.rate {
            Rate::Hz25 => (a25.as_slice(), f25),
            Rate::Hz12_5 => (a12.as_slice(), f12),
        };
        let frames = Frames::new(&act_buf[..din * f_cnt], din)?;
        let out_slice = &mut out[..rows * f_cnt];
        if use_tlut2 {
            gemv_par(lp.tlut2_mat.view(), backend, &frames, out_slice, threads)?;
        } else {
            gemv_par(lp.i8_mat.view(), backend, &frames, out_slice, threads)?;
        }
        checksum = checksum
            .wrapping_add(i64::from(out_slice[0]) + i64::from(out_slice[out_slice.len() - 1]));
    }
    Ok(checksum)
}

fn time_ccx_workers(
    layers: &[LayerPair],
    use_tlut2: bool,
    backend: Backend,
    streams_per_worker: usize,
    workers: usize,
    min_s: f64,
    acts: &[(usize, Vec<u8>, Vec<u8>)],
) -> Result<(f64, i64), KernelError> {
    let mut warm_out = vec![0i32; 1536 * 4 * streams_per_worker];
    let cs0 = run_stack_once(
        layers,
        use_tlut2,
        backend,
        streams_per_worker,
        1,
        acts,
        &mut warm_out,
    )?;

    let mut best_chunks_per_s = 0.0f64;
    for _ in 0..3 {
        let barrier = std::sync::Barrier::new(workers);
        let bar_ref = &barrier;
        let res: Result<Vec<(u32, f64)>, KernelError> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                handles.push(scope.spawn(move || -> Result<(u32, f64), KernelError> {
                    let mut out = vec![0i32; 1536 * 4 * streams_per_worker];
                    bar_ref.wait();
                    let t0 = Instant::now();
                    let mut passes = 0u32;
                    loop {
                        let cs = run_stack_once(
                            layers,
                            use_tlut2,
                            backend,
                            streams_per_worker,
                            1,
                            acts,
                            &mut out,
                        )?;
                        if cs != cs0 {
                            return Err(KernelError::Shape("non-deterministic checksum"));
                        }
                        passes += 1;
                        let el = t0.elapsed().as_secs_f64();
                        if el >= min_s {
                            return Ok((passes, el));
                        }
                    }
                }));
            }
            let mut out_v = Vec::with_capacity(workers);
            for h in handles {
                out_v.push(h.join().map_err(|_| KernelError::WorkerPanicked)??);
            }
            Ok(out_v)
        });
        let stats = res?;
        let max_el = stats.iter().map(|(_, e)| *e).fold(1e-9f64, f64::max);
        let total_passes: u64 = stats.iter().map(|(p, _)| u64::from(*p)).sum();
        let chunks_per_s = (total_passes as f64) / max_el;
        best_chunks_per_s = best_chunks_per_s.max(chunks_per_s);
    }
    Ok((1.0 / best_chunks_per_s, cs0))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("r1_costlaw: {e}");
            return ExitCode::from(2);
        }
    };
    let backend = Backend::detect();
    let max_s = args.streams.iter().copied().max().unwrap_or(1);
    let acts: Vec<(usize, Vec<u8>, Vec<u8>)> = [384usize, 768, 1536]
        .into_iter()
        .map(|d| {
            (
                d,
                make_a7_frames(d, 4 * max_s),
                make_a7_frames(d, 2 * max_s),
            )
        })
        .collect();
    let mut rows = Vec::new();

    for prof in &args.profiles {
        let mr = prof == "s-mr";
        let (layers, total_w) = match build_conformer_stack(mr) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("r1_costlaw build failed: {e}");
                return ExitCode::from(1);
            }
        };
        let int8_mib = (total_w as f64) / (1024.0 * 1024.0);
        let tlut2_mib = int8_mib * 0.25;
        for &th in &args.threads {
            for &st in &args.streams {
                let (t_i8, cs_i8) = match time_ccx_workers(
                    &layers,
                    false,
                    backend,
                    st,
                    th,
                    args.min_seconds,
                    &acts,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("r1_costlaw int8 failed: {e}");
                        return ExitCode::from(1);
                    }
                };
                let (t_tl, cs_tl) =
                    match time_ccx_workers(&layers, true, backend, st, th, args.min_seconds, &acts)
                    {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("r1_costlaw tlut2 failed: {e}");
                            return ExitCode::from(1);
                        }
                    };
                if cs_i8 != cs_tl {
                    eprintln!("r1_costlaw: checksum mismatch int8={cs_i8} != tlut2={cs_tl}");
                    return ExitCode::from(2);
                }
                let audio_chunk_s = 0.160 * (st as f64);
                rows.push(Row {
                    measurement_scope: "synthetic_linear_stack_kernel_proxy",
                    profile: prof.clone(),
                    weights_total: total_w,
                    int8_mib: (int8_mib * 100.0).round() / 100.0,
                    tlut2_mib: (tlut2_mib * 100.0).round() / 100.0,
                    streams: st,
                    threads: th,
                    int8_chunk_ms: (t_i8 * 1000.0 * 1000.0).round() / 1000.0,
                    tlut2_chunk_ms: (t_tl * 1000.0 * 1000.0).round() / 1000.0,
                    int8_rtfx: ((audio_chunk_s / t_i8) * 100.0).round() / 100.0,
                    tlut2_rtfx: ((audio_chunk_s / t_tl) * 100.0).round() / 100.0,
                    tlut2_speedup_x: ((t_i8 / t_tl) * 1000.0).round() / 1000.0,
                });
            }
        }
    }
    match serde_json::to_string_pretty(&rows) {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("r1_costlaw json error: {e}");
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
