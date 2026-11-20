# LibriSpeech reference word timings (C1, ruling O14)

Reference word timings for CQ and stable-word delay on LibriSpeech dev/test. They are timing only and independent of every system under test.

## Source
- **What:** Montreal-Forced-Aligner LibriSpeech alignments by Loren Lugosch, made with MFA and its pretrained LibriSpeech acoustic model (Lugosch et al., Interspeech 2019; McAuliffe et al., Interspeech 2017).
- **Where:** Zenodo record 2619474, DOI `10.5281/zenodo.2619474`, version 1.0 (2019-03-31), file `librispeech_alignments.zip`.
  - Size: 623,023,332 bytes.
  - Published md5: `2bab567d0ace651a4ba254e813629f46`.
  - URL: `https://zenodo.org/api/records/2619474/files/librispeech_alignments.zip/content`
- **Licence:** CC BY 4.0, per the Zenodo record metadata. The data is derived from LibriSpeech, also CC BY 4.0. Attribution: Lugosch et al. 2019 and the MFA paper.
- **Why this release:** it carries an explicit licence. The CorentinJ `librispeech-alignments` GitHub release is also widely used, but its repo has no licence file and its downloads sit on Google Drive. The HF dataset `gilkeyio/librispeech-alignments` is a re-packaging of this same Zenodo release that bundles the audio.

## Fetch
- **Scope:** only dev-clean, dev-other, test-clean and test-other.
- **Method:**
  - Read the archive's central directory and the contiguous byte spans of those four splits, using HTTP Range requests on one connection.
  - 48.4 MB transferred in total.
  - Each member is CRC-32-checked by `zipfile`.
  - The whole-archive md5 was **not** verified, because the archive was never fully downloaded (`archive_md5_verified: false` in `SOURCE.json`).
- **Members:** 11,126 = 11,125 TextGrids + `test-other/unaligned.txt`.
- **Command** (from `python/ultra`, under `ionice -c3 nice -n 15 taskset -c 48-63`):
  `python -m ultra.word_timing fetch && python -m ultra.word_timing build && python -m ultra.word_timing spotcheck`

## Files (data drive `/mnt/xbabe3-external/jsr/corpora/librispeech/`)
| file | sha256 |
|---|---|
| `mfa_alignments_zenodo2619474/MEMBERS.sha256` (sha256 of every member) | `a179c1bfe187a02b86260c55c99cd9ebccae84e6fabee7597062be040c1aba40` |
| `mfa_alignments_zenodo2619474/SOURCE.json` | `2864ced0448abc464a8e4cf9328756b601d5093a54fc2931fb859f75bf897873` |
| `word_timing/dev-clean.jsonl` | `af8eb43d2c1aeeb46aab64b73bbb8a6bcbb808f586488ddb9f24067c8a29dd1a` |
| `word_timing/dev-other.jsonl` | `c96a564dc858b013ac75dfd6f011c72e30f74edc64e16790811189f49b68557f` |
| `word_timing/test-clean.jsonl` | `a15f16fec05df8f638ba9f0da14bf30d5e601128106fbc825c0d1dcd269f4680` |
| `word_timing/test-other.jsonl` | `7e9a51c31578ec3c12b246f72da5d39eb3491cde7721dd55d37f45a6c98812ea` |
| `word_timing/MANIFEST.json` (schema, source, per-split coverage + jsonl sha256) | `30e16442edb36f2873f2151c5be9c172a623985a7b7cd341a06b89a7291ad7d2` |
| `word_timing/SPOTCHECK_automated.json` | `3820c1ac8c83211ff6467cffd52f1683cbe5ed1d5e1b50034640dcf309308351` |

## Format: `ultra.word-timing/v1`
- **Lines:** one JSON line per utterance, sorted by id: `{"id","words":[{"w","start_s","end_s"}]}`.
- **Words:** the upper-case LibriSpeech reference words from `ultra.libri.load_split` transcripts. Silences (`""`, `sil`, `sp`) are dropped. Times are in seconds, rounded to ms.
- **Mapping MFA tokens to reference words:**
  - An MFA `<unk>` token takes the reference word at its position.
  - The MFA dictionary splits some apostrophe words into pieces (`WICKER'S` → `wicker` + `s`; `D'AVRIGNY` → `d` + `<unk>`; `BUSH'` → `bush`). These pieces are merged back into one word that runs from the first piece's start to the last piece's end.
  - If the tokens cannot be mapped onto the whole reference, the utterance is left out. This never happened.
- **Loader:** `python/ultra/ultra/word_timing.py` (`load(split) -> {id: [Word]}`); tests in `python/ultra/tests/test_word_timing.py`.

## Coverage against our LibriSpeech transcripts (target ≥ 99 % of reference words)
| split | utts | matched | % utts | ref words | timed | % words | MFA `<unk>` | merged pieces | last word after audio end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev-clean | 2703 | 2703 | 100.000 | 54402 | 54402 | 100.000 | 177 | 27 | 0 |
| dev-other | 2864 | 2864 | 100.000 | 50948 | 50948 | 100.000 | 276 | 41 | 0 |
| test-clean | 2620 | 2620 | 100.000 | 52576 | 52576 | 100.000 | 213 | 22 | 0 |
| test-other | 2939 | 2938 | 99.966 | 52343 | 52324 | 99.964 | 266 | 44 | 0 |

- **The one gap:** `8131-117029-0017` (test-other), which MFA could not align ("beam too narrow", listed in `unaligned.txt`). It has no timing line. Latency scoring must treat it as "no reference timing", not as a miss.
- **Audio duration:** it is ≥ the last word end for every timed utterance (0 overruns).
- **First build:** before the apostrophe merge, coverage was 97.8–99.0 %. Every unmatched utterance was an apostrophe split.

## Spot check (automated substitute; **not** human listening)
This is an automated check. **Nobody listened to any audio.** The 20-utterance listening spot check required by O14 / BENCH_PLAN C1 **remains open for the owner.**

**Method:**
- 20 test-clean utterances drawn at random (seed 20261120) from the timed set.
- For each, compare the MFA last-word end with an energy-based speech offset (frame RMS, 25 ms window, 10 ms hop).
- Threshold = max(p99 frame dB − 35 dB, p10 frame dB + 12 dB); the p10 term is the noise-floor term.
- The same comparison was done for the first-word start against the energy onset.

**Results:**
| detector | median \|MFA end − RMS offset\| | max | median \|MFA start − RMS onset\| |
|---|---:|---:|---:|
| noise-floor aware (primary) | 55 ms | 425 ms | 40 ms |
| p99 − 35 dB only (first attempt, kept for transparency) | 60 ms | 515 ms | 65 ms |

- **Noise floor:** the p99-only detector failed on 6 of 20 files, where the noise floor sits within 35 dB of the peak. It returned onset 0 and offset = end of file. Adding the noise-floor term fixed that.
- **Outliers:** 5 of 20 rows have |Δend| > 100 ms.
  - Two have the energy offset *after* the MFA end: 2094-142345-0058 (−425 ms) and 5142-33396-0042 (−325 ms). These fit trailing breaths or room tails that MFA labels as silence.
  - Three have the MFA end *after* the energy offset: 2094-142345-0006 (+135 ms), 2830-3980-0075 (+215 ms) and 237-134493-0001 (+265 ms). These fit a soft final word that fades below the energy threshold.
  - The energy detector cannot settle either case. A listener should.
- **Bias:** MFA ends sit a few tens of ms after the RMS offset in most rows. This is expected, since energy decays below threshold before the phone boundary.
- **Data:** per-row data is in `SPOTCHECK_automated.json`.
- **Sampled ids** (for the owner's listening check): 672-122797-0031 8455-210777-0039 7127-75947-0008 5639-40744-0019 4446-2275-0027 2830-3980-0075 2094-142345-0058 61-70968-0029 1580-141084-0046 61-70968-0057 2094-142345-0037 2094-142345-0006 4507-16021-0043 2961-960-0017 6829-68769-0052 5142-33396-0042 4992-23283-0010 237-134493-0001 8455-210777-0031 7729-102255-0007
