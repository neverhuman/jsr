"""Figure 1 outside the PDF: the README block (data/money_readme.md) and the freshness stamp of money.png.

`just paper-refresh` exports docs/paper/money.png from figures/money_standalone.tex, then calls
`python -m tools.paper_data --publish`, which copies the generated block into README.md between its
markers and records the hash of the figure's inputs. A test compares both, so neither can go stale.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

from .common import DATA, PAPER_TITLE, PaperDataError, fmt
from .evidence import HEADLINE, derived
from .words import signed as _sg

START, END = "<!-- money-plot:start -->", "<!-- money-plot:end -->"
V4_START, V4_END = "<!-- v4-findings:start -->", "<!-- v4-findings:end -->"
FIGURE_INPUTS = ("docs/paper/data/cold_dev.csv", "docs/paper/data/money_layout.tex", "docs/paper/data/money_frontier.csv",
                 "docs/paper/figures/money.tex", "docs/paper/figures/money_focused.tex", "docs/paper/figures/style.tex", "docs/paper/figures/money_standalone.tex")
STAMP = "docs/paper/money.png.inputs"

def _story(vs: dict, rec_rows: dict, extras: dict) -> list[str]:
    """Three claims, each worded from its own numbers: the verdict follows the paired bounds, never a target."""
    pairs = {q["mode"]: q for q in vs.get("pairs", [])}
    z = vs.get("zip")
    out = []
    live, final = pairs.get("live"), pairs.get("final")
    if live and z:
        gap = ("with a WER difference the paired test cannot resolve" if live["verdict"] == "inconclusive" else
               f"at {fmt(abs(live['delta']), 2)} pp {'lower' if live['delta'] < 0 else 'higher'} pooled WER "
               f"({fmt(live['wer'], 2)}% vs. {fmt(z['wer'], 2)}%)")
        energy = f", {fmt(live['energy_pct'], 0)}% of its active core energy" if live["energy_pct"] is not None else ""
        out.append(f"- **Always-on at a fraction of the cost.** The live stream runs {fmt(live['speed_x'], 1)}× as fast "
                   f"as Zipformer INT8 on one core{energy} and {fmt(live['rss_pct'], 0)}% of its memory, with a "
                   f"{live['chunk']} ms chunk against Zipformer's {z['chunk']} ms, {gap}.")
    cap = (extras or {}).get("capacity", [])
    if cap:
        c = cap[-1]
        out[-1:] = [out[-1] + f" {c['streams']} real-time live streams fit on one core at "
                    f"{fmt(c['busy_pct'], 0)}% busy (p95 stable-word lag {fmt(c['lag_p95'], 0)} ms)."] if out else []
    if final and z:
        how = {"lower WER": "fewer errors than", "inconclusive": "not significantly different from",
               "higher WER": "more errors than"}[final["verdict"]]
        out.append(f"- **Final text from the same weights.** At utterance ends, a commit pass with a 5-gram LM reaches "
                   f"{fmt(final['wer'], 2)}% pooled WER, {how} Zipformer's {fmt(z['wer'], 2)}% (paired one-sided 95% bounds "
                   f"{_sg(final['lo'], 2)} to {_sg(final['hi'], 2)} pp), at {fmt(final['speed_x'], 2)}× its "
                   "single-core speed.")
    if "RelIntEightDelta" in rec_rows:
        out.append("- **Bit-exact and portable.** Exact int32 accumulation and a portable log-mel front end give "
                   "identical live transcripts on the tested AVX2, AVX-512 VNNI and ARM NEON kernels and batch sizes (final "
                   "text matches across the tested x86 and ARM CPUs), within "
                   f"{fmt(abs(float(rec_rows['RelIntEightDelta']['raw'])), 3)} pp of FP32 "
                   "PyTorch on full LibriSpeech DEV.")
    return out


def _pp(rr: dict, tag: str, which: str) -> tuple[float, float, float]:
    return tuple(float(rr[f"Cand{tag}{which}{k}"]["raw"]) for k in ("Delta", "Lo", "Hi"))


def _look(rr: dict, stem: str) -> tuple[float, float, float]:
    """A v4.1.0 paired difference (look41.py): delta and its one-sided 95% bounds, in pp."""
    return tuple(float(rr[f"{stem}{k}"]["raw"]) for k in ("Delta", "Lo", "Hi"))


def _bounds(t: tuple[float, float, float], verdict: bool = True, unit: bool = False) -> str:
    d, lo, hi = t
    text = f"{_sg(d, 2)} [{_sg(lo, 2)}, {_sg(hi, 2)}]" + (" pp" if unit else "")
    if not verdict:
        return text
    return text + (", lower WER" if hi < 0 else ", higher WER" if lo > 0 else ", inconclusive")


def _hero_block(hero: dict, facts: dict, subtitle: str, prior: dict, rr: dict, extras: dict) -> str:
    """Clean README block: all 12 timed public CPU recognizers vs. Isosyll v4.2.0 final and the unchanged v4.1.0 live point."""
    from .money import HERO_SHOW_FINAL

    d = {k: derived(r) for k, r in hero.items()}
    z = d["Zip"]
    live = d["LiveLm"]
    pair = lambda k: hero[k]["paired_vs_anchor"]
    v = lambda k: str(rr[k]["value"]).replace("$-$", "\u2212").replace("{,}", ",")
    gap_c = _look(rr, "LookLiveCZip")
    fin = d["Final"]
    fin_c, fin_prev_c = _look(rr, "FinCZip"), _look(rr, "FinCPrev")
    isa_runs = str(rr["CandIsaRuns"]["value"])
    speed_x = fmt(pair("LiveLm")["speed"], 1)
    energy_pct = fmt(100.0 * pair("LiveLm")["energy"], 0)
    r7 = lambda o: "—" if o["energy_r7"] is None else fmt(o["energy_r7"], 3)  # one method for the whole column
    lines = [
        START,
        "<!-- Generated by `just paper-refresh` from docs/paper/data (evidence rows and receipts). Do not edit by hand. -->",
        f"**Isosyll v4.2.0 on one CPU core: final {fmt(fin['wer'], 2)}% panel WER · {v('FinFullClean')}% full-DEV dev-clean · "
        f"{fmt(fin['speed'], 1)}× real time · live {fmt(live['speed'], 1)}× real time at {fmt(live['watts'], 3)} J/audio-s**",
        "",
        f"![{PAPER_TITLE}. Single-core throughput vs. pooled WER for all 12 timed public CPU speech recognizers and Isosyll final and live on the "
        "730-utterance LibriSpeech DEV panel, names colored by energy per audio-second](docs/paper/money.png)",
        "",
        f"**Figure 1.** {facts['figure']} [Paper](docs/paper/paper.pdf) · [Release record](docs/ultra/releases/v4.2.0.md).",
        "",
        (
            f"- **Lowest Isosyll WER yet (v4.2.0 final).** The utterance-end final pass (the `isosyll transcribe` default) reaches "
            f"**{fmt(fin['wer'], 2)}%** panel WER (**{fmt(fin['wer_clean'], 2)}%** dev-clean) and **{v('FinFullWer')}%** full DEV "
            f"(**{v('FinFullClean')}%** dev-clean) at **{fmt(fin['speed'], 1)}×** real time and **{fmt(fin['watts'], 3)} J/audio-s**. "
            f"On the unselected DEV C-half it is {_bounds(fin_c, True, True)} vs. Zipformer INT8 and {_bounds(fin_prev_c, True, True)} vs. "
            f"v4.1.0 final; it costs **{v('FinCostCpuPct')}%** CPU vs. v4.1.0 final (its beam-48 confidence gate "
            f"**{v('FinGateCpuPct')}%**, at its {v('FinGateBudgetPct')}% budget, locked by owner choice; paired R9 full-panel blocks)."
        ),
        (
            f"- **Fastest, with the lowest measured energy.** On one quiet Zen 2 core, **Isosyll live** (160 ms chunks, +LM) runs at "
            f"**{fmt(live['speed'], 1)}×** real time and **{fmt(live['watts'], 3)} J/audio-s** "
            f"(**{fmt(live['wer'], 2)}%** panel, **{fmt(live['wer_clean'], 2)}%** dev-clean, **{v('LookLiveFullWer')}%** full DEV WER, "
            f"**{fmt(live['rss_mb'], 0)} MB** RAM): **{speed_x}×** the paired throughput of Zipformer INT8 at **{energy_pct}%** of its "
            f"energy per audio-second (paired R8 full-panel blocks)."
        ),
        (
            "- **Two small artifacts.** A 60 MB INT8 model per mode (live: v4.1.0 weights, unchanged; final: a new weight average), "
            "same architecture and engine; streaming keeps the v4.1.0 commit pass."
        ),
        (
            f"- **Bit-exact pure-Rust engine.** No external runtime or BLAS (AVX2, AVX-512 VNNI, ARM NEON, scalar), "
            f"with identical transcripts across **{isa_runs}** cross-ISA witness runs."
        ),
        "",
        "| System | Class | Speed (× RT) | J/audio-s | Panel WER (%) | Peak RSS (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    ours_rows = [("LiveLm", "**Isosyll live (160 ms, +LM)**", "Ours · streaming")]
    if HERO_SHOW_FINAL:
        ours_rows.insert(0, ("Final", "**Isosyll v4.2.0 final**", "Ours · utterance-end pass"))
    for key, label, cls in ours_rows:
        o = d[key]
        lines.append(f"| {label} | {cls} | **{fmt(o['speed'], 1)}×** | **{r7(o)}** | **{fmt(o['wer'], 2)}** | {fmt(o['rss_mb'], 0)} |")
    comp_rows = [
        ("Zip", "Zipformer INT8 (320 ms)", "Streaming"),
        ("MoonTiny", "Moonshine Tiny Streaming", "Streaming"),
        ("VoskSmall", "Vosk Small", "Streaming"),
        ("VoskLarge", "Vosk Large §", "Streaming"),
        ("NemoFiveSixty", "Nemotron 0.6B (560 ms) §", "Streaming"),
        ("MoonSmall", "Moonshine Small ‡", "Streaming"),
        ("NemoOneSixty", "Nemotron 0.6B (160 ms) ‡", "Streaming"),
        ("WhisperTiny", "Whisper.cpp Tiny §", "Streaming"),
        ("NemoThreeFive", "Nemotron 3.5 (80 ms) ‡", "Streaming"),
        ("WhisperBase", "Whisper.cpp Base §", "Streaming"),
        ("ParakeetOnnx", "Parakeet TDT v3 (onnx-asr)", "Offline whole-clip"),
        ("ParakeetSherpa", "Parakeet TDT v3 (sherpa-onnx)", "Offline whole-clip"),
    ]
    for key, label, cls in comp_rows:
        o = d[key]
        lines.append(f"| {label} | {cls} | {fmt(o['speed'], 1)}× | {r7(o)} | {fmt(o['wer'], 2)} | {fmt(o['rss_mb'], 0)} |")
    lines += [
        "",
        "Measured on one pinned Zen 2 core (730-utterance LibriSpeech DEV selection panel; § = 73-clip timing subset, ‡ = shared-host H0 interim). "
        f"Unselected DEV C-half, v4.1 live vs. Zipformer INT8: {_bounds(gap_c, True, True)}. "
        f"J/audio-s: idle-subtracted core J per audio-second (CPU-domain registers only, not wall power): comparators from "
        f"the R7 matched campaign ({v('RsevenClips')}-clip subset, median of {v('RsevenCleanMin')}–{v('RsevenCleanMax')} core-clean blocks), "
        f"Isosyll final from the R9 and live from the R8 full-panel blocks under the same protocol. "
        f"The {fmt(live['watts'], 3)} J/audio-s and {energy_pct}% above are the paired R8 full-panel run. "
        "Exploratory DEV-only results. The v4.2.0 final pass meets its registered edit gates; live (unchanged since v4.1.0) was "
        "released by owner decision after missing registered edit gates; both were selected on the "
        "DEV selection half and checked once on the unselected half "
        "(protocol and limits in the [paper](docs/paper/paper.pdf) and [release record](docs/ultra/releases/v4.2.0.md)).",
        END,
    ]
    return "\n".join(lines) + "\n"


def v4_block(rr: dict) -> str:
    """Three-line README summary of the V4 negative results, every number read from the curated `Vf*` records."""
    v = lambda k: str(rr[k]["value"]).replace("$-$", "\u2212").replace("{,}", ",")
    lines = [V4_START, "<!-- Generated by `just paper-refresh` from docs/paper/data (curated Vf records). Do not edit by hand. -->",
             "## What we tried in V4 that did not help", "",
             f"- The live gap to Zipformer is acoustic discrimination of common interior words ({v('VfDiagInterior')} of {v('VfDiagPositions')} "
             f"repair positions), not decoding: a perfect ranker over the recorded fixed beam-4 lists reaches {v('VfOracleDepthFourWer')}% WER on "
             f"the selection half, so reranking those lists cannot reach {v('VfOracleTargetWer')}%.",
             f"- Lookahead convolution, layer drop and a {v('VfSpecCapPct')}% time-mask cap (SPEC15) sat inside the {v('VfFloorPp')} pp screen noise floor; "
             f"stronger labels (PL3), ReLU^2, a TDT head, early exit and full-length layer drop (O3-LD) were not adopted.",
             "- The next lever is fresh audio and capacity paid for by the power savings. Details: the paper, "
             "\"What Did Not Move Live Accuracy, and Why\".", V4_END]
    return "\n".join(lines) + "\n"


def release_old() -> str:
    return "v3.0.0"


def readme_block(rows: list[dict], facts: dict, watts: str, extras: dict | None = None, *, subtitle: str = "",
                 focused: bool = False, nemotron_note: str = "", versus: dict | None = None,
                 rec_rows: dict | None = None, release: str = "v3.0.0", hero: dict | None = None, prior: dict | None = None) -> str:
    """Markdown for the top of README.md: summary line, Figure 1, the three claims and the head-to-head table.
    The full 15-comparator table, protocol and limits live in the paper."""
    if hero:
        return _hero_block(hero, facts, subtitle, prior or {}, rec_rows or {}, extras or {})
    drawn = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and r["plot"]]
    lines = [START, "<!-- Generated by `just paper-refresh` from docs/paper/data (evidence rows and receipts). "
             "Do not edit by hand. -->"]
    if drawn and subtitle:
        summary = subtitle.replace("$\\times$", "×").replace("$\\cdot$", "·").replace("\\%", "%").replace("\\,", " ")
        lines += [f"**{summary}**", ""]
    lines += [f"![{PAPER_TITLE}. Single-core throughput against pooled WER on the 730-utterance LibriSpeech DEV panel]"
              "(docs/paper/money.png)", ""]
    if not drawn:
        lines += ["No measured rows yet.", END]
        return "\n".join(lines) + "\n"
    n_comp = sum(1 for r in drawn if r["role"] != "ours") + len((extras or {}).get("complete", []))
    lines += [f"**Figure 1.** {facts['figure']} Same 730 LibriSpeech DEV utterances, one pinned Zen 2 core, median "
              f"of three cold runs including model load. [Read the paper](docs/paper/paper.pdf) for all {n_comp} "
              f"open recognisers we measured, the protocol and the limits · [{release} release record]"
              f"(docs/ultra/releases/{release}.md).", ""]
    lines += [*_story(versus or {}, rec_rows or {}, extras or {}), ""]
    if versus and versus.get("markdown"):
        lines += [versus["markdown"].rstrip("\n"), ""]
    rr = rec_rows or {}
    if "CandFinalFullEdits" in rr:
        v = lambda k: str(rr[k]["value"]).replace("$-$", "\u2212").replace("{,}", ",")
        lines += [f"**Next candidate, measured and not released.** RC2, trained on {v('CandTrainKh')} h, lowers live WER against "
                  f"v3.0.0 by {v('CandLiveLmOldDelta').lstrip('\u2212')} points with the live LM, but trails Zipformer by "
                  f"{v('CandLiveLmCDelta').lstrip('+')} points on unselected DEV and made {v('CandFinalFullEdits')} full-DEV final "
                  f"edits against a registered gate of {v('CandGateFinalFull')}, so the signed release below stays the install target. "
                  "[RC2 qualification record](docs/ultra/releases/v3.1.0-rc2.md).", ""]
    lines += ["All accuracy is LibriSpeech DEV (test sets sealed); results are exploratory. Larger offline and "
              "600M-parameter streaming models (Parakeet, Nemotron) are more accurate at several times the CPU; see "
              "the paper. Refresh with `just paper-refresh`.", END]
    return "\n".join(lines) + "\n"


def figure_digest(root: Path) -> str:
    """sha256 over the files Figure 1 is drawn from: if it changes, money.png must be exported again."""
    h = hashlib.sha256()
    for rel in FIGURE_INPUTS:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return h.hexdigest()


def spliced(readme: str, block: str, start: str = START, end: str = END) -> str:
    if readme.count(start) != 1 or readme.count(end) != 1:
        raise PaperDataError(f"README.md needs exactly one {start} ... {end} block")
    return readme[:readme.index(start)] + block.rstrip("\n") + readme[readme.index(end) + len(end):]


def publish_figure(root: Path) -> None:
    """Copy the generated hero block into README.md and stamp money.png with the digest of its inputs."""
    block = (root / DATA / "money_readme.md").read_text(encoding="utf-8")
    readme = root / "README.md"
    text = spliced(readme.read_text(encoding="utf-8"), block)
    readme.write_text(text, encoding="utf-8")
    if not (root / "docs/paper/money.png").is_file():
        raise PaperDataError("docs/paper/money.png is missing; `just paper-refresh` exports it before publishing")
    (root / STAMP).write_text(figure_digest(root) + "\n", encoding="utf-8")
