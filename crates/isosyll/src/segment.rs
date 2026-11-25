//! Pause-based cutting of long audio for final-mode decoding.

/// Hop between candidate cut points (20 ms at 16 kHz).
const HOP: usize = 320;
/// RMS window around a candidate cut (200 ms).
const WIN: usize = 3200;
/// Segments target 20..=30 s.
const MIN_SEG: usize = 20 * 16_000;
const MAX_SEG: usize = 30 * 16_000;

/// Return the segment end offsets (the last equals `pcm.len()`) such that every segment is at most
/// `max_samples` long. Audio no longer than `max_samples` is a single segment. Otherwise each cut
/// is the minimum-RMS point (200 ms window, 20 ms hop) within `[start+20 s, start+30 s]`, clamped
/// to `start + max_samples`.
pub fn plan_cuts(pcm: &[i16], max_samples: usize) -> Vec<usize> {
    let n = pcm.len();
    let mut cuts = Vec::new();
    if n == 0 {
        return cuts;
    }
    // Prefix sums of squares for O(1) window energy.
    let mut pre = Vec::with_capacity(n + 1);
    pre.push(0.0f64);
    let mut acc = 0.0f64;
    for &s in pcm {
        acc += f64::from(s) * f64::from(s);
        pre.push(acc);
    }
    let energy = |p: usize| {
        let (a, b) = (p.saturating_sub(WIN / 2), (p + WIN / 2).min(n));
        (pre[b] - pre[a]) / (b - a).max(1) as f64
    };
    let mut start = 0usize;
    while n - start > max_samples {
        let hi = (start + MAX_SEG).min(start + max_samples).min(n - 1);
        let lo = (start + MIN_SEG).min(hi);
        let (mut best, mut best_e) = (lo, f64::INFINITY);
        let mut p = lo;
        while p <= hi {
            let e = energy(p);
            if e < best_e {
                (best, best_e) = (p, e);
            }
            p += HOP;
        }
        cuts.push(best);
        start = best;
    }
    cuts.push(n);
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(len: usize) -> Vec<i16> {
        (0..len)
            .map(|i| if i % 2 == 0 { 8000 } else { -8000 })
            .collect()
    }

    #[test]
    fn short_audio_is_one_segment() {
        let pcm = tone(16_000 * 10);
        assert_eq!(plan_cuts(&pcm, 640_000), vec![pcm.len()]);
        assert!(plan_cuts(&[], 640_000).is_empty());
        let exact = tone(640_000);
        assert_eq!(plan_cuts(&exact, 640_000), vec![640_000]);
    }

    #[test]
    fn cut_lands_in_silent_gap() {
        let mut pcm = tone(16_000 * 100);
        // Silent gaps inside the 20..30 s search windows of the first two segments.
        let gap1 = 25 * 16_000..25 * 16_000 + 8_000; // 25.0 .. 25.5 s
        pcm[gap1.clone()].fill(0);
        let cuts = plan_cuts(&pcm, 640_000);
        assert!(gap1.contains(&cuts[0]), "cut {} not in gap", cuts[0]);
        let start2 = cuts[0];
        let gap2 = start2 + 22 * 16_000..start2 + 22 * 16_000 + 8_000;
        pcm[gap2.clone()].fill(0);
        let cuts = plan_cuts(&pcm, 640_000);
        assert!(gap2.contains(&cuts[1]), "cut {} not in gap", cuts[1]);
        assert_eq!(*cuts.last().unwrap(), pcm.len());
    }

    #[test]
    fn segments_are_bounded_and_cover_input() {
        let pcm = tone(16_000 * 200 + 123);
        let cuts = plan_cuts(&pcm, 640_000);
        let mut prev = 0;
        for &c in &cuts {
            assert!(c > prev && c - prev <= 30 * 16_000);
            prev = c;
        }
        assert_eq!(prev, pcm.len());
        // A small limit is honoured.
        let small = plan_cuts(&pcm, 16_000 * 25);
        let mut p = 0;
        for &c in &small {
            assert!(c - p <= 16_000 * 25);
            p = c;
        }
    }
}
