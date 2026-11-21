//! Engine-agnostic fixed-capture-deadline + audio-grid simulation (`simulate_grid`).
//!
//! Separated from native `SherpaLib` bindings so deterministic fake-clock / fake-recogniser
//! M1–M4 qualification probes exercise the exact same scheduling loop as `run_sp_a_zipformer`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::api::StreamError;
use crate::contrast::{percentile_sorted, self_cpu_seconds};

/// Default LIVE finalization / oldest-unprocessed lateness bound in milliseconds (SP-A A1).
pub const DEFAULT_LATENESS_BOUND_MS: f64 = 800.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeMode {
    Single,
    Batch,
}

impl DecodeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Batch => "batch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatenessQuantiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub last: f64,
    pub first_third_mean: f64,
    pub last_third_mean: f64,
}

/// SP-A Amendment A1 sustainability predicate: drained, finite non-negative quantiles,
/// `p99 <= bound_ms`, and no-growth (`last_third_mean - first_third_mean <= 0.25 * bound_ms`).
pub fn sustains_lateness(lat: &LatenessQuantiles, drained: bool, bound_ms: f64) -> bool {
    let vals = [
        lat.p50,
        lat.p95,
        lat.p99,
        lat.max,
        lat.last,
        lat.first_third_mean,
        lat.last_third_mean,
    ];
    if !drained || !bound_ms.is_finite() || bound_ms <= 0.0 {
        return false;
    }
    if vals.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return false;
    }
    lat.p99 <= bound_ms && (lat.last_third_mean - lat.first_third_mean) <= 0.25 * bound_ms
}

/// Injectable wall/CPU clock for deterministic grid-scheduler probes.
pub trait SimClock {
    fn now_s(&self) -> f64;
    fn sleep_s(&mut self, dt_s: f64);
    fn cpu_s(&self) -> Result<f64, StreamError>;
}

pub(crate) struct RealClock {
    t0: Instant,
    c0: f64,
}

impl RealClock {
    pub(crate) fn new() -> Result<Self, StreamError> {
        Ok(Self {
            t0: Instant::now(),
            c0: self_cpu_seconds()?,
        })
    }
}

impl SimClock for RealClock {
    fn now_s(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    fn sleep_s(&mut self, dt_s: f64) {
        if dt_s > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(dt_s));
        }
    }
    fn cpu_s(&self) -> Result<f64, StreamError> {
        Ok((self_cpu_seconds()? - self.c0).max(0.0))
    }
}

/// Abstract stream-bank operations so `simulate_grid` runs identically over `SherpaZipformerBackend`
/// and deterministic fake recognisers in unit tests.
pub trait SimSlotBank {
    fn num_utterances(&self) -> usize;
    fn utt_samples(&self, utt_pos: usize) -> usize;
    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError>;
    fn has_slot(&self, slot: usize) -> bool;
    fn slot_utt(&self, slot: usize) -> usize;
    fn slot_cursor(&self, slot: usize) -> usize;
    fn slot_finished_input(&self, slot: usize) -> bool;
    fn accept_packet(&mut self, slot: usize, max_samples: usize) -> usize;
    fn finish_slot_input(&mut self, slot: usize);
    fn slot_is_ready(&self, slot: usize) -> bool;
    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64);
    fn take_completed_slot(&mut self, slot: usize) -> String;
    fn on_periodic_tick(&mut self) {}
}

#[derive(Debug, Clone)]
pub struct SimParams {
    pub streams: usize,
    pub decode: DecodeMode,
    pub max_batch: usize,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub paced: bool,
    pub duration_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SustainedSample {
    pub elapsed_s: f64,
    pub backlog_ms: f64,
    pub active_streams: usize,
    pub accepted_audio_s: f64,
    pub processed_audio_s: f64,
}

#[derive(Debug, Clone)]
pub struct SimOutcome {
    pub completed: Vec<(usize, String)>,
    pub accepted_samples: u64,
    pub processed_samples: u64,
    pub audio_accepted_s: f64,
    pub audio_processed_s: f64,
    pub audio_unfinished_s: f64,
    pub audio_completed_s: f64,
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    pub drain_cpu_seconds: Option<f64>,
    pub drain_wall_seconds: Option<f64>,
    pub accepted_packets: u64,
    pub offered_to_decode_packets: u64,
    pub effective_grid_ms: u32,
    pub max_ready_batch: usize,
    pub wakeups: u64,
    pub drained: bool,
    pub tick_compute_ms: Vec<f64>,
    pub lateness_ms: Option<LatenessQuantiles>,
    pub series: Vec<SustainedSample>,
}

fn summarize_lateness(lateness_s: &[f64]) -> Option<LatenessQuantiles> {
    if lateness_s.is_empty() {
        return None;
    }
    let round2 = |v_s: f64| ((v_s * 1000.0) * 100.0).round() / 100.0;
    let last = round2(lateness_s[lateness_s.len() - 1]);
    let third = (lateness_s.len() / 3).max(1);
    let first_mean = lateness_s[..third].iter().sum::<f64>() / (third as f64);
    let last_mean = lateness_s[lateness_s.len() - third..].iter().sum::<f64>() / (third as f64);
    let mut sorted = lateness_s.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(LatenessQuantiles {
        p50: round2(percentile_sorted(&sorted, 50.0)),
        p95: round2(percentile_sorted(&sorted, 95.0)),
        p99: round2(percentile_sorted(&sorted, 99.0)),
        max: round2(percentile_sorted(&sorted, 100.0)),
        last,
        first_third_mean: round2(first_mean),
        last_third_mean: round2(last_mean),
    })
}

fn accept_one_packet<B: SimSlotBank>(
    bank: &mut B,
    streams: usize,
    packet_samples: usize,
    accepted_samples: &mut u64,
    k_acc: &mut u64,
) {
    for slot in 0..streams {
        if bank.has_slot(slot) && !bank.slot_finished_input(slot) {
            let n = bank.accept_packet(slot, packet_samples);
            *accepted_samples += n as u64;
        }
    }
    *k_acc += 1;
}

/// Execute the fixed-capture-deadline + grid-decode loop over any [`SimSlotBank`] and [`SimClock`].
pub fn simulate_grid<B: SimSlotBank, C: SimClock>(
    bank: &mut B,
    params: &SimParams,
    clock: &mut C,
) -> Result<SimOutcome, StreamError> {
    simulate_grid_with_backlog(bank, params, clock, None)
}

/// Same as [`simulate_grid`], optionally reporting per-tick worker lateness (in microseconds) to a
/// shared fleet observer atomic for 10-second sustained-run backlog series (`S5`).
pub(crate) fn simulate_grid_with_backlog<B: SimSlotBank, C: SimClock>(
    bank: &mut B,
    params: &SimParams,
    clock: &mut C,
    backlog_us_sink: Option<&Arc<AtomicU64>>,
) -> Result<SimOutcome, StreamError> {
    let n_utts = bank.num_utterances();
    if params.streams == 0 || params.packet_ms == 0 || params.max_batch == 0 || n_utts == 0 {
        return Err(StreamError::Protocol(
            "streams, packet_ms, max_batch and utterances must all be > 0".into(),
        ));
    }
    let packet_samples = (u64::from(params.packet_ms) * 16_000 / 1000).max(1) as usize;
    let pkt_s = f64::from(params.packet_ms) / 1000.0;
    let grid_every = if params.grid_ms == 0 {
        1u64
    } else {
        ((f64::from(params.grid_ms) / f64::from(params.packet_ms)).round() as u64).max(1)
    };
    let effective_grid_ms = (grid_every as u32) * params.packet_ms;
    let allow_wrap = params.duration_s.is_some();
    let mut queue_cursor = 0usize;

    let refill =
        |bank: &mut B, queue_cursor: &mut usize, cutoff: bool| -> Result<(), StreamError> {
            if cutoff {
                return Ok(());
            }
            for slot in 0..params.streams {
                if !bank.has_slot(slot) {
                    let pos = if *queue_cursor < n_utts {
                        let p = *queue_cursor;
                        *queue_cursor += 1;
                        Some(p)
                    } else if allow_wrap {
                        let p = *queue_cursor % n_utts;
                        *queue_cursor += 1;
                        Some(p)
                    } else {
                        None
                    };
                    if let Some(utt_pos) = pos {
                        bank.open_slot(slot, utt_pos)?;
                    }
                }
            }
            Ok(())
        };

    let any_active = |bank: &B| (0..params.streams).any(|s| bank.has_slot(s));
    refill(bank, &mut queue_cursor, false)?;

    let cpu0 = clock.cpu_s()?;
    let t0 = clock.now_s();
    let deadline_s = |k: u64| t0 + ((k + 1) as f64) * pkt_s;
    let t_cut = match params.duration_s {
        Some(d) => t0 + d.max(0.0),
        None => f64::INFINITY,
    };

    let mut accepted_samples = 0u64;
    let mut processed_samples = 0u64;
    let mut k_acc = 0u64;
    let mut k_dec = 0u64;
    let mut cutoff = false;
    let mut drain_start: Option<(f64, f64)> = None;
    let mut max_ready_batch = 0usize;
    let mut wakeups = 0u64;
    let mut completed: Vec<(usize, String)> = Vec::new();
    let mut audio_completed_samples = 0u64;
    let mut steady_lat_s: Vec<f64> = Vec::new();
    let mut tick_compute_ms: Vec<f64> = Vec::new();
    let mut series: Vec<SustainedSample> = Vec::new();
    let mut next_series_t = t0 + 10.0;

    while any_active(bank) {
        let mut now = clock.now_s();
        if !cutoff {
            if params.paced {
                let target = deadline_s(k_acc).min(t_cut);
                if now < target {
                    clock.sleep_s(target - now);
                    now = clock.now_s();
                }
                let limit_t = now.min(t_cut) + 1e-12;
                while deadline_s(k_acc) <= limit_t && any_active(bank) {
                    accept_one_packet(
                        bank,
                        params.streams,
                        packet_samples,
                        &mut accepted_samples,
                        &mut k_acc,
                    );
                    refill(bank, &mut queue_cursor, false)?;
                }
            } else if deadline_s(k_acc) <= t_cut + 1e-12 || params.duration_s.is_none() {
                accept_one_packet(
                    bank,
                    params.streams,
                    packet_samples,
                    &mut accepted_samples,
                    &mut k_acc,
                );
            }
            if now + 1e-12 >= t_cut || (!params.paced && deadline_s(k_acc) > t_cut + 1e-12) {
                cutoff = true;
                drain_start = Some((clock.cpu_s()?, now));
                for slot in 0..params.streams {
                    if bank.has_slot(slot) && !bank.slot_finished_input(slot) {
                        bank.finish_slot_input(slot);
                    }
                }
            }
        }

        let all_inputs_done =
            (0..params.streams).all(|s| !bank.has_slot(s) || bank.slot_finished_input(s));
        let crossed = (k_acc / grid_every) > (k_dec / grid_every);
        if cutoff || crossed || all_inputs_done {
            let oldest = deadline_s(k_dec);
            let step_t0 = clock.now_s();
            let mut total_ready_decoded = 0usize;
            let mut max_step_lat = 0.0f64;
            // Drain EVERY ready chunk on this grid tick across all backends (R0/ORT loop over
            // ready chunks; ultra-engine fuses up to `max_chunks` chunks per pass; FREEZE §3 B2).
            loop {
                let (ready_n, w) = bank.decode_ready_step(params.decode, params.max_batch);
                if ready_n == 0 {
                    break;
                }
                let step_done_t = clock.now_s();
                max_ready_batch = max_ready_batch.max(ready_n);
                total_ready_decoded += ready_n;
                wakeups += w;
                if params.paced {
                    // FREEZE §3 unified A1: per-chunk step completion against capture deadline,
                    // including drain ticks after cutoff.
                    let lat = (step_done_t - oldest).max(0.0);
                    max_step_lat = max_step_lat.max(lat);
                    for _ in 0..ready_n {
                        steady_lat_s.push(lat);
                    }
                    if let Some(sink) = backlog_us_sink {
                        sink.fetch_max((lat * 1_000_000.0).round() as u64, Ordering::Relaxed);
                    }
                }
            }
            if total_ready_decoded > 0 {
                let done_t = clock.now_s();
                tick_compute_ms.push((done_t - step_t0).max(0.0) * 1000.0);
                if params.paced && done_t >= next_series_t {
                    let active = (0..params.streams).filter(|&s| bank.has_slot(s)).count();
                    series.push(SustainedSample {
                        elapsed_s: ((done_t - t0) * 1000.0).round() / 1000.0,
                        backlog_ms: (max_step_lat * 1000.0 * 100.0).round() / 100.0,
                        active_streams: active,
                        accepted_audio_s: (accepted_samples as f64) / 16_000.0,
                        processed_audio_s: (processed_samples as f64) / 16_000.0,
                    });
                    while next_series_t <= done_t {
                        next_series_t += 10.0;
                    }
                }
            }
            k_dec = k_acc;
            for slot in 0..params.streams {
                if bank.has_slot(slot)
                    && bank.slot_finished_input(slot)
                    && !bank.slot_is_ready(slot)
                {
                    let utt_pos = bank.slot_utt(slot);
                    let cur = bank.slot_cursor(slot);
                    let full_len = bank.utt_samples(utt_pos);
                    let hyp = bank.take_completed_slot(slot);
                    processed_samples += cur as u64;
                    if cur >= full_len {
                        audio_completed_samples += full_len as u64;
                        completed.push((utt_pos, hyp));
                    }
                }
            }
            refill(bank, &mut queue_cursor, cutoff)?;
            bank.on_periodic_tick();
        }
    }

    let cpu_end = clock.cpu_s()?;
    let wall_end = clock.now_s();
    let drained = !any_active(bank);
    let processed_samples = processed_samples.min(accepted_samples);
    let unfinished_samples = accepted_samples.saturating_sub(processed_samples);
    let (drain_cpu_seconds, drain_wall_seconds) = match drain_start {
        Some((dc0, dt0)) => (
            Some(((cpu_end - dc0).max(0.0) * 1000.0).round() / 1000.0),
            Some(((wall_end - dt0).max(0.0) * 1000.0).round() / 1000.0),
        ),
        None => (None, None),
    };

    Ok(SimOutcome {
        completed,
        accepted_samples,
        processed_samples,
        audio_accepted_s: (accepted_samples as f64) / 16_000.0,
        audio_processed_s: (processed_samples as f64) / 16_000.0,
        audio_unfinished_s: (unfinished_samples as f64) / 16_000.0,
        audio_completed_s: (audio_completed_samples as f64) / 16_000.0,
        cpu_seconds: (cpu_end - cpu0).max(0.0),
        wall_seconds: (wall_end - t0).max(0.0),
        drain_cpu_seconds,
        drain_wall_seconds,
        accepted_packets: k_acc,
        offered_to_decode_packets: k_dec,
        effective_grid_ms,
        max_ready_batch,
        wakeups,
        drained,
        tick_compute_ms,
        lateness_ms: summarize_lateness(&steady_lat_s),
        series,
    })
}
