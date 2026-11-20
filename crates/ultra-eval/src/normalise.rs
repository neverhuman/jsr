//! Pinned, table-driven English text normaliser for benchmark WER (v1.0-B §30.1).
//!
//! Steps, in this fixed order:
//! 1. Unicode lower-casing (`char::to_lowercase`).
//! 2. Apostrophe variants in [`APOSTROPHE_VARIANTS`] become `'`.
//! 3. Every character that is not alphanumeric becomes a space, except an apostrophe
//!    with an alphanumeric character on both sides (an intra-word apostrophe). Hyphens
//!    therefore split words: `well-known` -> `well known`.
//! 4. Whitespace split; each token is expanded by [`WHOLE_WORD_EXPANSIONS`] (exact
//!    match first) or else by the first matching rule of [`SUFFIX_EXPANSIONS`].
//! 5. Words are joined by single spaces (whitespace collapsed, no leading/trailing).
//!
//! The same function is applied to references and hypotheses, so contractions are
//! expanded consistently on both sides. Deliberately NOT done: number spelling
//! unification (`5` vs `five` stay different words), British/American spelling
//! unification, filler removal, and possessive/`'s` disambiguation (only the listed
//! whole words with `'s` are expanded; `john's` stays `john's`). `'d` always expands to
//! `would` (the `had` reading is not recovered). Any rule change is a new evaluation
//! identity: bump [`NORMALISER_ID`]; the pinned hash test fails until you do.

use std::sync::OnceLock;

/// Identity of this normaliser. Reports carry it with [`normaliser_sha256`].
pub const NORMALISER_ID: &str = "ultra-en-wer-norm/v1";

/// Characters treated as an apostrophe.
pub const APOSTROPHE_VARIANTS: &[char] = &['\u{2019}', '\u{2018}', '\u{02BC}', '`', '\u{FF07}'];

/// Whole-token contraction expansions (checked before suffix rules).
pub const WHOLE_WORD_EXPANSIONS: &[(&str, &str)] = &[
    ("can't", "can not"),
    ("cannot", "can not"),
    ("won't", "will not"),
    ("shan't", "shall not"),
    ("let's", "let us"),
    ("it's", "it is"),
    ("that's", "that is"),
    ("what's", "what is"),
    ("there's", "there is"),
    ("here's", "here is"),
    ("he's", "he is"),
    ("she's", "she is"),
    ("who's", "who is"),
    ("where's", "where is"),
    ("how's", "how is"),
    ("y'all", "you all"),
];

/// Suffix contraction rules `(suffix, replacement)`; the stem before the suffix is
/// kept. First match wins; a rule applies only when the stem is non-empty.
pub const SUFFIX_EXPANSIONS: &[(&str, &str)] = &[
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'m", " am"),
    ("'d", " would"),
];

/// Canonical serialisation of every rule above; its SHA-256 is the normaliser hash.
pub fn rule_table() -> String {
    let mut s = String::new();
    s.push_str(NORMALISER_ID);
    s.push_str("\nsteps:lowercase;apostrophe-variants;non-alnum-to-space-keep-intra-word-apostrophe;whole-word;suffix;collapse-ws\n");
    s.push_str("apostrophes:");
    for c in APOSTROPHE_VARIANTS {
        s.push_str(&format!("U+{:04X},", *c as u32));
    }
    s.push_str("\nwhole:");
    for (a, b) in WHOLE_WORD_EXPANSIONS {
        s.push_str(&format!("{a}={b};"));
    }
    s.push_str("\nsuffix:");
    for (a, b) in SUFFIX_EXPANSIONS {
        s.push_str(&format!("{a}={b};"));
    }
    s.push('\n');
    s
}

/// Lower-case hex SHA-256 of [`rule_table`].
pub fn normaliser_sha256() -> &'static str {
    static H: OnceLock<String> = OnceLock::new();
    H.get_or_init(|| ultra_core::receipt::sha256_hex(rule_table().as_bytes()))
}

fn expand(token: &str, out: &mut Vec<String>) {
    if let Some((_, rep)) = WHOLE_WORD_EXPANSIONS.iter().find(|(w, _)| *w == token) {
        out.extend(rep.split(' ').map(str::to_owned));
        return;
    }
    for (suf, rep) in SUFFIX_EXPANSIONS {
        if let Some(stem) = token.strip_suffix(suf) {
            if !stem.is_empty() {
                out.push(stem.to_owned());
                out.extend(rep.split_whitespace().map(str::to_owned));
                return;
            }
        }
    }
    out.push(token.to_owned());
}

/// Normalise text into words.
pub fn normalise_words(text: &str) -> Vec<String> {
    let chars: Vec<char> = text
        .chars()
        .flat_map(char::to_lowercase)
        .map(|c| {
            if APOSTROPHE_VARIANTS.contains(&c) {
                '\''
            } else {
                c
            }
        })
        .collect();
    let mut cleaned = String::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        let keep = if c.is_alphanumeric() {
            true
        } else if c == '\'' {
            let prev = i > 0 && chars[i - 1].is_alphanumeric();
            let next = chars.get(i + 1).is_some_and(|n| n.is_alphanumeric());
            prev && next
        } else {
            false
        };
        cleaned.push(if keep { c } else { ' ' });
    }
    let mut out = Vec::new();
    for tok in cleaned.split_whitespace() {
        expand(tok, &mut out);
    }
    out
}

/// Normalise text into a single-space-joined string.
pub fn normalise(text: &str) -> String {
    normalise_words(text).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_pinned() {
        // Changing any rule must bump NORMALISER_ID and this pinned value together.
        assert_eq!(NORMALISER_ID, "ultra-en-wer-norm/v1");
        assert_eq!(normaliser_sha256(), PINNED);
    }
    const PINNED: &str = "988c59acad873bf0a0c50bdcf434dde0b5125cf51c2cef887bfc63f4f89113e8";

    #[test]
    fn fixtures() {
        let cases = [
            ("Hello, World!", "hello world"),
            ("  It's  a well-known fact.  ", "it is a well known fact"),
            (
                "I can't, won't; they're here",
                "i can not will not they are here",
            ),
            ("We've I'm you'll he'd", "we have i am you will he would"),
            ("John's dog", "john's dog"),
            ("the students' books", "the students books"),
            ("'quoted' words", "quoted words"),
            ("Don\u{2019}t", "do not"),
            ("rock 'n' roll", "rock n roll"),
            ("5 five", "5 five"),
            ("", ""),
            ("?!...", ""),
            ("ÉCOLE Straße", "école straße"),
        ];
        for (inp, want) in cases {
            assert_eq!(normalise(inp), want, "input {inp:?}");
        }
    }

    #[test]
    fn idempotent_on_fixtures() {
        for s in ["It's can't we'd", "A-B c.d", "x'y'z"] {
            let once = normalise(s);
            assert_eq!(normalise(&once), once);
        }
    }

    proptest::proptest! {
        #[test]
        fn idempotent(s in ".{0,64}") {
            let once = normalise(&s);
            proptest::prop_assert_eq!(normalise(&once), once);
        }

        #[test]
        fn output_has_no_double_spaces(s in ".{0,64}") {
            let n = normalise(&s);
            proptest::prop_assert!(!n.contains("  "));
            proptest::prop_assert_eq!(n.trim(), n.as_str());
        }
    }
}
