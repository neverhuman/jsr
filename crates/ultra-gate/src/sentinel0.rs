//! Sentinel-0: two-speed pure-integer wake gate on 160-sample (10 ms) 16 kHz mono PCM16 hops.
//!
//! Zero floating point, zero `unsafe`, zero external dependencies. Every decision is an exact
//! integer function of PCM16 inputs so x86-64 and aarch64 produce bit-identical traces.
#![forbid(unsafe_code)]

/// Samples per gate hop (`10 ms` at `16 kHz`).
pub const HOP: usize = 160;

/// Hops per Martin minimum-statistics sub-window (`250 ms`).
const V: u32 = 25;

/// Number of complete sub-windows in the noise-floor tracker (`U * V = 125` hops = `1.25 s`).
const U: usize = 5;

/// Slow integration length in hops (`80 ms`).
pub const KS: usize = 8;

/// Sentinel ceiling used during warm-up and empty sub-windows.
pub const WARMUP_FLOOR: i32 = 1 << 30;

/// Thresholds in Q8 `log2` energy units (`256` units = `3.0103 dB`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateConfig {
    /// Fast-path (`20 ms`) threshold above effective floor (default `640` ≈ `7.5 dB`).
    pub theta_fast: i32,
    /// Slow-path (`80 ms`) threshold above effective floor (default `256` ≈ `3.0 dB`).
    pub theta_slow: i32,
    /// Fast-path absolute energy floor (default `3943` ≈ `-72 dBFS` white noise).
    pub nabs_fast: i32,
    /// Slow-path absolute energy floor (default `4455` ≈ `-72 dBFS` white noise).
    pub nabs_slow: i32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            theta_fast: 640,
            theta_slow: 256,
            nabs_fast: 3943,
            nabs_slow: 4455,
        }
    }
}

/// `round(256 * log2(1 + m / 256))` for `m = 0..=255` (verified against exact integer arithmetic).
pub const LOG2_LUT: [u8; 256] = [
    0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
    34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62, 63,
    65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 155, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 169,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 178, 179, 180, 181, 182, 183, 184, 185, 185, 186,
    187, 188, 189, 190, 191, 192, 192, 193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203,
    203, 204, 205, 206, 207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218,
    219, 220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231, 232, 233,
    234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245, 246, 247, 247,
    248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
];

/// Q8 `log2` of `v >= 1`: `(e << 8) + LOG2_LUT[the 8 bits below the leading one]`.
/// Returns `0` for `v == 0`.
#[inline]
pub fn ilog2_q8(v: u64) -> i32 {
    if v == 0 {
        return 0;
    }
    let e = 63 - v.leading_zeros();
    let m = (((u128::from(v) << 8) >> e) & 0xFF) as usize;
    ((e as i32) << 8) + i32::from(LOG2_LUT[m])
}

#[derive(Debug, Clone)]
struct MinStat {
    sub: [i32; U],
    cur: i32,
    k: u32,
    i: usize,
}

impl MinStat {
    fn new() -> Self {
        Self {
            sub: [WARMUP_FLOOR; U],
            cur: WARMUP_FLOOR,
            k: 0,
            i: 0,
        }
    }

    fn floor(&self) -> i32 {
        self.sub
            .iter()
            .copied()
            .min()
            .unwrap_or(WARMUP_FLOOR)
            .min(self.cur)
    }

    fn push(&mut self, l: i32) {
        self.cur = self.cur.min(l);
        self.k += 1;
        if self.k == V {
            self.sub[self.i] = self.cur;
            self.i = (self.i + 1) % U;
            self.cur = WARMUP_FLOOR;
            self.k = 0;
        }
    }
}

/// One hop's wake decision and diagnostic levels/floors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HopOut {
    pub hot: bool,
    /// Bit 0 = fast path (`20 ms`), bit 1 = slow path (`80 ms`).
    pub path: u8,
    /// Q8 `log2` levels: `[fast_a, fast_b, slow_a, slow_b]`.
    pub level: [i32; 4],
    /// Effective floors (`WARMUP_FLOOR` during the first `KS - 1` warm-up hops).
    pub floor: [i32; 4],
}

/// Stateful pure-integer Sentinel-0 wake gate.
#[derive(Debug, Clone)]
pub struct Sentinel0 {
    cfg: GateConfig,
    prev_x: i32,
    prev_b: i32,
    ea: [u64; KS],
    eb: [u64; KS],
    n: u64,
    fl: [MinStat; 4],
}

impl Sentinel0 {
    /// Construct a new gate with the supplied integer thresholds.
    pub fn new(cfg: GateConfig) -> Self {
        Self {
            cfg,
            prev_x: 0,
            prev_b: 0,
            ea: [0; KS],
            eb: [0; KS],
            n: 0,
            fl: [
                MinStat::new(),
                MinStat::new(),
                MinStat::new(),
                MinStat::new(),
            ],
        }
    }

    /// Active gate configuration.
    pub fn config(&self) -> GateConfig {
        self.cfg
    }

    /// Total 160-sample hops consumed so far.
    pub fn hops(&self) -> u64 {
        self.n
    }

    /// Consume exactly one 10 ms hop (`160` samples of `16 kHz` mono PCM16).
    pub fn push_hop(&mut self, x: &[i16; HOP]) -> HopOut {
        // Band A: 1st difference at 16 kHz (suppresses DC/rumble, emphasises fricatives/onsets).
        let mut ea: u64 = 0;
        let mut p = self.prev_x;
        for &s in x.iter() {
            let d = i64::from(i32::from(s) - p);
            ea += (d * d) as u64;
            p = i32::from(s);
        }
        self.prev_x = p;

        // Band B: 4-sample box sum (4 kHz voiced band) followed by 1st difference.
        let mut eb: u64 = 0;
        let mut q = self.prev_b;
        for c in x.chunks_exact(4) {
            let b = i32::from(c[0]) + i32::from(c[1]) + i32::from(c[2]) + i32::from(c[3]);
            let d = i64::from(b - q);
            eb += (d * d) as u64;
            q = b;
        }
        self.prev_b = q;

        let i = (self.n % KS as u64) as usize;
        self.ea[i] = ea;
        self.eb[i] = eb;
        let j = (i + KS - 1) % KS;
        let (pa, pb) = if self.n >= 1 {
            (self.ea[j], self.eb[j])
        } else {
            (0, 0)
        };
        let (fa, fb) = (ea + pa, eb + pb);
        let sa: u64 = self.ea.iter().sum();
        let sb: u64 = self.eb.iter().sum();
        let level = [
            ilog2_q8(fa + 1),
            ilog2_q8(fb + 1),
            ilog2_q8(sa + 1),
            ilog2_q8(sb + 1),
        ];

        if self.n < KS as u64 - 1 {
            self.n += 1;
            return HopOut {
                hot: false,
                path: 0,
                level,
                floor: [WARMUP_FLOOR; 4],
            };
        }

        let c = &self.cfg;
        let floor = [
            self.fl[0].floor().max(c.nabs_fast),
            self.fl[1].floor().max(c.nabs_fast),
            self.fl[2].floor().max(c.nabs_slow),
            self.fl[3].floor().max(c.nabs_slow),
        ];
        let fast = level[0] - floor[0] >= c.theta_fast || level[1] - floor[1] >= c.theta_fast;
        let slow = level[2] - floor[2] >= c.theta_slow || level[3] - floor[3] >= c.theta_slow;

        for (f, &l) in self.fl.iter_mut().zip(level.iter()) {
            f.push(l);
        }
        self.n += 1;

        HopOut {
            hot: fast || slow,
            path: u8::from(fast) | (u8::from(slow) << 1),
            level,
            floor,
        }
    }
}
