# Out-of-domain evaluation-only sets (C0, ruling O15)

**Status: PUBLIC-TEST, sealed.**
- Nobody has opened the reference texts. No WER has been computed.
- They get one read, at the qualifying window, after the `bench-freeze-v1` tag exists.
- **Row label:** "out-of-domain for R0/R1; unknown overlap for Nemotron/Moonshine/Whisper".
- LibriSpeech test-clean/other stays the primary board.

## Source
- **Repo:** Open ASR Leaderboard / ESB test-only parquet, Hugging Face dataset `hf-audio/open-asr-leaderboard`. Not gated.
- **Revision:** `de79e57adf9983993bcc72586d7313460825f5eb` (2026-10-07).
- **What was fetched:** only the `test-*.parquet` shards of the `ami` and `voxpopuli` configs, with `huggingface_hub.snapshot_download`, `max_workers=4`, under `ionice -c3 nice -n 15 taskset -c 48-63`.
- **Integrity:** every shard was checked against the HF LFS SHA-256 (19/19 OK).
- **Segmentation:** the leaderboard's own segments, ids and order (sorted by length, longest first), unchanged. This is the leaderboard convention.

| set | config | what | utts | hours | licence |
|---|---|---|---:|---:|---|
| voxpopuli | `voxpopuli` | VoxPopuli EN test | 1,842 | 4.930 | CC0 1.0 (transcribed data); also see the Europarl legal notice https://www.europarl.europa.eu/legal-notice/en/ |
| ami | `ami` | AMI Meeting Corpus, **IHM**, test (ESB "eval" split) | 12,643 | 8.682 | CC BY 4.0 |
| tedlium | — | TED-LIUM 3 test | — | — | **BLOCKED_RIGHTS** (below) |

- **Counts:** utterance counts match the dataset card (`num_examples`).
- **Hours:** decoded sample count / 16 kHz.
- **Native rate:** all audio was already 16 kHz mono (0 resampled).
- **AMI is IHM:** the ESB loading script (`hf-audio/esb-datasets-test-only` @ `a997ad4c`, `datasets-test-only.py`) builds the AMI config from `speech-seq2seq/ami` `audio/ihm/{split}` and sets the licence "CC BY 4.0". VoxPopuli has the licence "CC0, also see https://www.europarl.europa.eu/legal-notice/en/".
- **Not used:** the leaderboard's newer `ami_cleaned` and `voxpopuli_cleaned_aa` configs. The classic `ami` / `voxpopuli` test sets are the standard published rows.

## BLOCKED_RIGHTS: TED-LIUM 3
- **Upstream removal:** TED-LIUM was removed from `hf-audio/open-asr-leaderboard` by PR #9 ("Remove Tedlium dataset", merged 2026-05-19, commit `3edf1c4c`).
- **Reason given:** asked whether TED-LIUM was gone from the benchmark, the maintainer (bezzam, HF) said: "correct unfortunately. The original authors (LIUM) also had to take down the dataset due to copyright/licensing".
- **Not fetched:** the shards still exist at older revisions (for example the parent `20a009a3`). They were **not** fetched from there or from any other mirror. No substitute was used.
- **Owner decision needed:** drop TED-LIUM from ULTRA-Bench v1 (two OOD sets), or name a source with clear rights.

## Files (data drive `/mnt/xbabe3-external/jsr/`)
- **Parquet:** `corpora/ood/open-asr-leaderboard/{ami,voxpopuli}/test-*.parquet` (as downloaded).
- **Audio caches:** `cache/ood_pcm/<set>.audio.npz`. These hold `pcm` (int16, 16 kHz, concatenated), `offsets` and `ids`, and **no text**.
- **Sealed texts:** `cache/ood_pcm/sealed/<set>.texts.npz`, holding `ids` and `texts`. **SEALED.** They are written once by `prepare` and never printed.
- **Manifests:** `cache/ood_pcm/<set>.manifest.json`, with source, revision, licence, the sha256 of every parquet shard and cache file, counts and hours.
- **Permissions:** the drive's filesystem ignores Unix permissions, so the sealed files cannot be made read-only. Sealing is enforced by the loader and by process.

| file | sha256 |
|---|---|
| `voxpopuli.manifest.json` | `cf38aa7ed15e100ae8ab3b7033acd061a653dacc48082e8ddffc4eb761ac2ba9` |
| `voxpopuli.audio.npz` | `3b77370155e20894a2b528e35eb7eaca50d8f1b36e1087a189466ab5b53c24bc` |
| `sealed/voxpopuli.texts.npz` | `15082caeae9d53fa3cbc41368f19117895dd83027f8ba6e66cdc3c49af63a237` |
| `ami.manifest.json` | `28642ac37ad052ac532d699d7d4f880a09a4f13c4b18374c971cda231dbc3531` |
| `ami.audio.npz` | `2ee4593bde17b18880fb45e317c8fa7c1f1e453b86117006633ca0379f853d86` |
| `sealed/ami.texts.npz` | `5c0acd780f281ce2ac993539efab9b0216d96b79c124f185c4ddcc8eb0dd6059` |

**Parquet shard sha256** (same as HF LFS):

| shard | sha256 |
|---|---|
| `voxpopuli/test-00000-of-00004.parquet` | `42da0a15189dd26cd1fd0233a8530e661ca339b90326bc8e225f84c9a8f20c1f` |
| `voxpopuli/test-00001-of-00004.parquet` | `7f1ec665e28cb8b6a4c4a7c2df3559059dc5ae3abcf6989ff0aa0a788c31158b` |
| `voxpopuli/test-00002-of-00004.parquet` | `e6b52256fa2cc4a60f78bff2da4a1ed6dab56e9ff2313b211de8351b0d19ab5c` |
| `voxpopuli/test-00003-of-00004.parquet` | `f95ba08bbb10e30748a15b5595b091be57718f6feaf435d8440e7025be13986b` |
| `ami/test-00000-of-00015.parquet` | `a335a47c309e98fea075ba671357fc94dc8013071bee92d202d6a371f0067dc4` |
| `ami/test-00001-of-00015.parquet` | `eda5746a49acade52323240b12caaa5b16dee14f95be2eeff0ddc7b8963ce297` |
| `ami/test-00002-of-00015.parquet` | `09af25af3decec04f8e087d1cba93b57de92858d64d5705a5f3938ca010ad9bf` |
| `ami/test-00003-of-00015.parquet` | `23312b743b73796ee37b0c9a4aaca2f74f0956dd9300f01bd80a85e02d375766` |
| `ami/test-00004-of-00015.parquet` | `a4d775b9f7e1390edb9eab701df6a29b1293a79d75d6ac8985bcb7a773e0e87a` |
| `ami/test-00005-of-00015.parquet` | `e7d402768e1749e918353db37becb37d2cb5e59a18675346a31eac15c449cd08` |
| `ami/test-00006-of-00015.parquet` | `f6423fc64ddf1931501a5184b701f9e6647326ac94e52a27aa97962811695428` |
| `ami/test-00007-of-00015.parquet` | `a741d770e21087f5aac89db722bcbbbfbf4b932dea6ffa50f2084c97429bf025` |
| `ami/test-00008-of-00015.parquet` | `70f3d40b164339503072ef8008d8dd3b4bb24f83ac519cb6bea85c1b39c5ca4f` |
| `ami/test-00009-of-00015.parquet` | `65e07d823dd83180872b84c2e949ebc610c461018dca27a008d078eaa401407e` |
| `ami/test-00010-of-00015.parquet` | `52787744937328048aff1c1f4d05d31d349c2719dad437792ac74f63197ebdbe` |
| `ami/test-00011-of-00015.parquet` | `964a963e6306d22a765af7aeb9db912553babc61501ea14c5330905fbcec8030` |
| `ami/test-00012-of-00015.parquet` | `210843b76965a8b946cfa1f5f5f5c3634bf56f844486044b958254a0586f36a4` |
| `ami/test-00013-of-00015.parquet` | `b7e1fd3a2e95723a731ff3e5f27a90070074655a45888cfe7e6cb48d3fee06ce` |
| `ami/test-00014-of-00015.parquet` | `9f858a5148c7aa9cb48d55141af03506fafd3500567420633ad4093fb77d2d7c` |

## Sealing (`python/ultra/ultra/ood.py`)
- **`load_audio(name)`:** returns audio and ids only. It is allowed at any time, for timing, cost and density runs.
- **`load_split(name)`:** returns audio plus reference texts. It raises `SealedReadRefused`, a `PermissionError` that carries a repair hint, unless all of these hold:
  1. `ULTRA_SEALED_READ=<look-id>` is set;
  2. `git tag -l bench-freeze-v1` is non-empty **and** the tagged commit is an ancestor of HEAD. A tag planted on an unrelated commit is refused;
  3. no earlier read of the same set under a different look-id is logged in `docs/ultra/LOOKS.jsonl`. O15 allows one read; a retry under the same look-id is allowed and logged again.
- **Logging:** each permitted read appends `{"ts","kind":"ood-sealed-read","look_id","set","user","head","freeze_tag_commit","sealed_sha256"}` to `docs/ultra/LOOKS.jsonl` *before* the texts are opened.
- **Tests:** `python/ultra/tests/test_ood.py` covers:
  - refusal without the tag;
  - refusal without a look-id;
  - audio-only allowed without the tag;
  - a planted fake tag on an orphan commit refused;
  - a permitted read logged, and a second look-id refused;
  - TED-LIUM blocked.
- **Limits:** this guards against accidental reads, not a determined bypass. Anyone can `np.load` the sealed file directly. Process: before the freeze, only manifests and hashes are inspected.
- **Rebuild:** `python -m ultra.ood prepare voxpopuli|ami` (from `python/ultra`). It never prints or returns text.

## Latency on these sets
There are no reference word timings for the OOD sets, so latency (CQ, stable-word, FC) is reported on LibriSpeech only and stated as such (BENCH_PLAN C1).
