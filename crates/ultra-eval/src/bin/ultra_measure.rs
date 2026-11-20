//! `ultra-measure [--interval-ms N] -- <cmd> <args...>`
//!
//! Runs the command, samples its whole process tree every 50 ms (default), waits for
//! every descendant, and prints one JSON object on stdout:
//! `{cpu_seconds_user, cpu_seconds_sys, wall_seconds, peak_rss_bytes, peak_pss_bytes,
//! max_threads, exit_code, samples, sampled_cpu_seconds, clk_tck}`.
//! The command's own stdout goes to stderr. Exit status: the command's, or 2 on a
//! usage error, 3 if measurement failed.

use std::time::Duration;

fn usage() -> ! {
    eprintln!("usage: ultra-measure [--interval-ms N] -- <cmd> [args...]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut interval_ms = 50u64;
    let mut i = 0;
    while i < args.len() && args[i] != "--" {
        match args[i].as_str() {
            "--interval-ms" => {
                i += 1;
                interval_ms = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .filter(|v| *v > 0)
                    .unwrap_or_else(|| usage());
            }
            _ => usage(),
        }
        i += 1;
    }
    let cmd = &args[(i + 1).min(args.len())..];
    let Some((program, rest)) = cmd.split_first() else {
        usage();
    };
    match ultra_eval::procfs::measure_command(program, rest, Duration::from_millis(interval_ms)) {
        Ok(m) => {
            match serde_json::to_string(&m) {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("ultra-measure: {e}");
                    std::process::exit(3);
                }
            }
            std::process::exit(m.exit_code.unwrap_or(1));
        }
        Err(e) => {
            eprintln!("ultra-measure: {e}");
            std::process::exit(3);
        }
    }
}
