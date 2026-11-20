//! `ultra` command-line entry point (spec v1.0-B §35).

use ultra_cli::{run_cli_args, EXIT_OK};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run_cli_args(&args) {
        Ok(()) => std::process::exit(EXIT_OK),
        Err(e) => {
            eprintln!("ultra: {e}");
            std::process::exit(e.exit_code());
        }
    }
}
