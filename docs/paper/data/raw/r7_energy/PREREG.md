# R7 matched energy campaign: pre-registration

Written: 2026-11-26T03:46:49Z, before any R7 measurement run (only the 1-clip, 5 s-idle dry runs of every arm exist, never reported).
Owner of the campaign: agent E1, lane docs/energy-r7.

## Question
Idle-subtracted core energy in joules per audio-second for 14 arms, measured one way on one host (xbabe0), so the hero figure can colour every system by a matched number.

## Method (R6 unchanged)
- Tool: energy_bench.py run (the R6 tool), one repetition per run, 1 worker, one pinned physical core CPU 16 (SMT sibling idle), observer/sampler CPU 31, AMD MSR core energy (0xC001029A), package counter recorded.
- 60 s idle window before and after each run (idle power = mean of both); 5 s settle.
- metric (core_j_per_audio_s) = (core energy of the run - idle core power x run wall time) / audio seconds, read from the receipt field joules_per_audio_s. Also recorded: cpu_s_per_audio_s and speed (= 1 / cpu_s_per_audio_s, x real time).
- 1 thread per recogniser (OMP/MKL/OMP_THREAD_LIMIT = 1; Moonshine adapter sets one ORT thread itself), same binaries, models and flags as the existing timing rows (see run-energy.sh cases r7-*, nemotron, nemotron35, moonshine, vosk, whispercpp, onnxasr, sherpa-offline).
- Panel: 73-clip timing subset ("panel73", one utterance per speaker, every tenth staged file) for all 14 arms. In block 1 only, Isosyll live +LM and Zipformer INT8 additionally run the full 730-utterance DEV panel as ANCHORS.
- 3 counterbalanced blocks over the 14 arms (block 1 forward order, block 2 reversed, block 3 rotated by seven). Reported value = median of the 3 block values.
- Host: QUIET.lock held around each arm; before each arm the host must show no pinned-core neighbours busy; if CI jobs or other heavy processes appear, pause between arms and wait. TIMING.lock is not taken.
- A core-view-unclean repetition (sibling busy, idle drift, worker not on CPU, non-zero exit) is rerun once in place; both runs are kept in the receipt dir and the rerun is the one used. Nothing else is rerun, nothing is tuned.

## Arms (14)
Isosyll live +LM (RC2, 160 ms, LM c4 v2; ultra-cold-wav 3bd85e9e), Isosyll final (commit pass, M5 LM, beam 16; ultra-engine-commit 3bd85e9e), Zipformer INT8 (sherpa streaming), Moonshine Tiny, Moonshine Small, Nemotron 560 ms, Nemotron 160 ms, Nemotron 3.5, Parakeet TDT v3 sherpa-onnx (offline), Parakeet v3 onnx-asr (offline), Vosk Small, Vosk Large, Whisper.cpp stream Tiny, Whisper.cpp stream Base.

## Validity rule (pre-registered)
The full-panel anchors of block 1 must reproduce R6 within +/-5 percent: Isosyll live +LM 0.328 J/audio-s (0.3117 to 0.3444), Zipformer INT8 0.573 J/audio-s (0.5444 to 0.6017). If either misses, R7 is VOID: stop, do not tune, report the numbers as they are. The anchors are not used to rescale any arm.

## What this does not claim
CPU-reported core registers, not wall power; one core of an otherwise idle 64-core part (boost on); numbers describe that condition only.
