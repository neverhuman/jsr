//! Unit tests for `args.rs`.

use super::*;

fn p(s: &str) -> Result<Command, String> {
    parse(s.split_whitespace().map(String::from))
}

#[test]
fn transcribe_defaults() {
    let Ok(Command::Transcribe(a)) = p("transcribe a.wav") else {
        panic!("parse");
    };
    assert_eq!(a.mode, Mode::Final);
    assert_eq!((a.json, a.rms_norm, a.threads), (false, false, 1));
    assert_eq!(a.files, vec![PathBuf::from("a.wav")]);
    assert!(a.model.is_none());
}

#[test]
fn transcribe_all_flags_and_many_files() {
    let Ok(Command::Transcribe(a)) =
        p("transcribe a.wav b.wav --mode live --json --model /m --threads 2")
    else {
        panic!("parse");
    };
    assert_eq!(a.mode, Mode::Live);
    assert!(a.json && !a.rms_norm);
    assert!(p("transcribe a.wav --mode live --rms-norm").is_err());
    assert!(p("transcribe a.wav --rms-norm").is_ok());
    assert!(p("stream a.wav --speed 1e-300").is_err());
    assert!(p("stream a.wav --speed 1e300").is_err());
    assert!(p("stream a.wav --speed 0").is_ok());
    assert_eq!(a.threads, 2);
    assert_eq!(a.model, Some(PathBuf::from("/m")));
    assert_eq!(a.files.len(), 2);
    let Ok(Command::Transcribe(b)) = p("transcribe --mode=final --model=/x a.wav") else {
        panic!("parse");
    };
    assert_eq!(b.mode, Mode::Final);
    assert_eq!(b.model, Some(PathBuf::from("/x")));
}

#[test]
fn transcribe_errors() {
    assert!(p("transcribe").is_err());
    assert!(p("transcribe a.wav --mode fast").is_err());
    assert!(p("transcribe a.wav --threads 0").is_err());
    assert!(p("transcribe a.wav --threads x").is_err());
    assert!(p("transcribe a.wav --bogus").is_err());
    assert!(p("transcribe a.wav --model").is_err());
}

#[test]
fn double_dash_allows_dashed_filenames() {
    let Ok(Command::Transcribe(a)) = p("transcribe -- -odd.wav") else {
        panic!("parse");
    };
    assert_eq!(a.files, vec![PathBuf::from("-odd.wav")]);
}

#[test]
fn stream_args() {
    let Ok(Command::Stream(a)) = p("stream x.wav --speed 0 --model /m") else {
        panic!("parse");
    };
    assert_eq!(a.speed, 0.0);
    assert_eq!(a.model, Some(PathBuf::from("/m")));
    let Ok(Command::Stream(d)) = p("stream x.wav") else {
        panic!("parse");
    };
    assert_eq!(d.speed, 1.0);
    assert!(p("stream").is_err());
    assert!(p("stream a.wav b.wav").is_err());
    assert!(p("stream a.wav --speed -1").is_err());
    assert!(p("stream a.wav --speed nan").is_err());
}

#[test]
fn model_verify_and_meta() {
    assert_eq!(
        p("model verify --model /m"),
        Ok(Command::ModelVerify {
            model: Some(PathBuf::from("/m"))
        })
    );
    assert_eq!(p("model verify"), Ok(Command::ModelVerify { model: None }));
    assert!(p("model nope").is_err());
    assert_eq!(p("--version"), Ok(Command::Version));
    assert_eq!(p("-V"), Ok(Command::Version));
    assert_eq!(p("-h"), Ok(Command::Help(HelpTopic::Main)));
    assert_eq!(p(""), Ok(Command::Help(HelpTopic::Main)));
    assert_eq!(
        p("transcribe --help"),
        Ok(Command::Help(HelpTopic::Transcribe))
    );
    assert_eq!(p("stream -h"), Ok(Command::Help(HelpTopic::Stream)));
    assert!(p("frobnicate").is_err());
}
