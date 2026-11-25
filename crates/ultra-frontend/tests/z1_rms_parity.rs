//! Z1 witnesses: the Rust gain matches the Python reference sample for sample.
//! Fixture: `tests/fixtures/z1_rms/` (regenerate with `gen_z1_fixture.py`).

use ultra_frontend::{rms_gain, rms_normalize};

fn fixture() -> (Vec<i16>, Vec<[String; 6]>) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/z1_rms");
    let bytes = std::fs::read(dir.join("z1.bin")).expect("z1.bin");
    let pcm = bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let rows = std::fs::read_to_string(dir.join("z1.tsv"))
        .expect("z1.tsv")
        .lines()
        .map(|l| {
            let f: Vec<String> = l.split('\t').map(String::from).collect();
            [
                f[0].clone(),
                f[1].clone(),
                f[2].clone(),
                f[3].clone(),
                f[4].clone(),
                f[5].clone(),
            ]
        })
        .collect();
    (pcm, rows)
}

#[test]
fn rust_matches_python_reference_sample_for_sample() {
    let (pcm, rows) = fixture();
    assert!(rows.len() >= 6);
    let mut saw_clipping = false;
    for [name, n, gain, clipped, in_off, out_off] in rows {
        let n: usize = n.parse().unwrap();
        let (i, o): (usize, usize) = (in_off.parse().unwrap(), out_off.parse().unwrap());
        let input = &pcm[i..i + n];
        let expected = &pcm[o..o + n];
        let want_gain: f64 = gain.parse().unwrap();
        assert_eq!(
            rms_gain(input).to_bits(),
            want_gain.to_bits(),
            "{name} gain"
        );
        let (got, st) = rms_normalize(input);
        assert_eq!(got.as_slice(), expected, "{name} samples");
        assert_eq!(st.gain.to_bits(), want_gain.to_bits(), "{name} stats gain");
        assert_eq!(
            st.clipped,
            clipped.parse::<usize>().unwrap(),
            "{name} clipped"
        );
        saw_clipping |= st.clipped > 0;
    }
    assert!(saw_clipping, "fixture must exercise int16 clipping");
}
