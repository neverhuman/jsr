#!/usr/bin/env python3
"""Helpers for ops/paper/build_arxiv.sh: stage the paper sources, write metadata, scan for leaks.

Subcommands (all stdlib only):
  stage    --fls F --work DIR --out DIR     copy exactly the files pdflatex read, strip comments
  metadata --stage DIR --pdf PDF --aux AUX  print ARXIV_METADATA.md fields as JSON-free markdown
  scan     PATH...                          privacy scan; exit 1 on any hit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SHIP_EXT = {".tex", ".sty", ".cls", ".clo", ".def", ".cfg", ".bbl", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".csv"}
SKIP_NAMES = {"paper.pdf"}
VERBATIM = ("verbatim", "verbatim*", "lstlisting", "minted", "comment", "filecontents", "filecontents*")


# ----------------------------------------------------------------------------- comments
def strip_comments(text: str) -> str:
    """Remove TeX comments. Whole-line comments vanish; a trailing comment keeps a bare `%`
    so the swallowed newline (and next-line indentation) behaves exactly as before."""
    out: list[str] = []
    verb_end: str | None = None
    for line in text.split("\n"):
        if verb_end is not None:
            out.append(line)
            if verb_end in line:
                verb_end = None
            continue
        i, n, cut = 0, len(line), None
        while i < n:
            c = line[i]
            if c == "\\":
                m = re.match(r"\\verb\*?(.)", line[i:])
                if m:  # \verb|...| : skip to the closing delimiter
                    j = line.find(m.group(1), i + len(m.group(0)))
                    i = n if j < 0 else j + 1
                    continue
                i += 2  # any escaped char, including \%
                continue
            if c == "%":
                cut = i
                break
            i += 1
        m = re.search(r"\\begin\{(" + "|".join(re.escape(v) for v in VERBATIM) + r")\}", line[: cut if cut is not None else n])
        if m:
            verb_end = "\\end{" + m.group(1) + "}"
            out.append(line)
            if verb_end in line[m.end():]:
                verb_end = None
            continue
        if cut is None:
            out.append(line)
        elif line[:cut].strip() == "":
            continue  # whole-line comment
        else:
            out.append(line[:cut] + "%")
    return "\n".join(out)


def _group(s: str, i: int) -> tuple[str, int]:
    """Return (content, index after) of the brace group starting at s[i] == '{'."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    raise ValueError("unbalanced braces in \\IfFileExists")


def resolve_iffileexists(text: str, shipped: set[str]) -> str:
    """\\IfFileExists{f}{A}{B} -> A if f is shipped (with or without .tex), else B."""
    key = r"\IfFileExists"
    while (k := text.find(key)) >= 0:
        i = k + len(key)
        parts, j = [], i
        for _ in range(3):
            while j < len(text) and text[j] in " \t\n%":
                j += 1
            g, j = _group(text, j)
            parts.append(g)
        name = parts[0].strip()
        here = name in shipped or name + ".tex" in shipped
        text = text[:k] + (parts[1] if here else parts[2]) + text[j:]
    return text


# ----------------------------------------------------------------------------- figure wrapping
FIG_RE = re.compile(r"(\\begin\{figure\*?\}(?:\[[^\]]*\])?)(.*?)(?=\\caption|\\end\{figure\*?\})", re.S)


def wrap_figures(text: str, counter: list[int], names: list[str]) -> str:
    """Wrap each float body that is drawn by TeX (no \\includegraphics) in \\beginpgfgraphicnamed so it
    can be compiled to its own PDF; a leading \\centering stays outside the graphic."""
    def sub(m: re.Match) -> str:
        head, body = m.group(1), m.group(2)
        if "\\includegraphics" in body or not re.search(r"\\(input|tikz|begin\{tikzpicture\})", body):
            return m.group(0)
        counter[0] += 1
        name = f"paperfig{counter[0]}"
        names.append(name)
        lead = ""
        c = re.match(r"(\s*\\centering\s*)", body)
        if c:
            lead, body = c.group(1), body[c.end():]
        return f"{head}{lead}\\beginpgfgraphicnamed{{{name}}}{body.rstrip().rstrip(chr(37)).rstrip()}%\n\\endpgfgraphicnamed\n"
    return FIG_RE.sub(sub, text)


def cmd_wrapfigs(a: argparse.Namespace) -> int:
    counter, names = [0], []
    read = set(read_fls(Path(a.fls), Path(a.dir)))  # figures in unread files are never typeset
    for f in sorted(Path(a.dir).rglob("*.tex")):
        if str(f.relative_to(a.dir)) not in read:
            continue
        t = f.read_text()
        t2 = wrap_figures(t, counter, names)
        if t2 != t:
            f.write_text(t2)
    Path(a.names).write_text("\n".join(names) + ("\n" if names else ""))
    print(f"wrapfigs: {len(names)} figure(s): {' '.join(names)}")
    return 0


# ----------------------------------------------------------------------------- stage
def read_fls(fls: Path, work: Path) -> list[str]:
    root = work.resolve()
    rels: list[str] = []
    for line in fls.read_text(errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        p = Path(line[6:].strip())
        if not p.is_absolute():
            p = root / p
        p = Path(os.path.normpath(p))
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue  # system TeX Live files
        if p.is_file() and p.suffix.lower() in SHIP_EXT and rel.name not in SKIP_NAMES:
            if str(rel) not in rels:
                rels.append(str(rel))
    return sorted(rels)


def cmd_stage(a: argparse.Namespace) -> int:
    work, out = Path(a.work), Path(a.out)
    rels = read_fls(Path(a.fls), work)
    if "paper.tex" not in rels or "paper.bbl" not in rels:
        print("stage: paper.tex or paper.bbl not read by the build", file=sys.stderr)
        return 1
    shipped = set(rels)
    for rel in rels:
        src, dst = work / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".tex" or src.suffix in {".sty", ".cls", ".def", ".cfg", ".clo"}:
            t = resolve_iffileexists(src.read_text(), shipped)
            t = strip_comments(t)
            if rel == "paper.tex":
                t = "\\pdfoutput=1\n" + t
            dst.write_text(t.rstrip("\n") + "\n")
        else:
            dst.write_bytes(src.read_bytes())
    (out / "00README.json").write_text(json.dumps(
        {"process": {"compiler": "pdflatex"},
         "sources": [{"filename": "paper.tex", "usage": "toplevel"}],
         "stamp": False}, indent=2) + "\n")
    print(f"stage: {len(rels)} files -> {out}")
    for r in rels:
        print("  ", r)
    return 0


# ----------------------------------------------------------------------------- metadata
def macros(*files: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    for f in files:
        if not f.exists():
            continue
        t = f.read_text()
        for m in re.finditer(r"\\(?:newcommand|providecommand|renewcommand)\*?\{\\(\w+)\}(?:\[\d\])?\{", t):
            try:
                body, _ = _group(t, m.end() - 1)
            except ValueError:
                continue
            d.setdefault(m.group(1), body)
    return d


def plain(s: str, mac: dict[str, str]) -> str:
    s = strip_comments(s)
    for _ in range(8):
        s2 = re.sub(r"\\(\w+)\s*(\{\})?", lambda m: mac[m.group(1)] if m.group(1) in mac else m.group(0), s)
        if s2 == s:
            break
        s = s2
    rep = [(r"\$\\times\$|\\x\b|\\times", "x"), (r"\\,|\;|\\ |~|\\xspace", " "), (r"\\%", "%"), (r"\\pp\b", " pp "),
           (r"\\textbf\{([^}]*)\}|\\emph\{([^}]*)\}|\\texttt\{([^}]*)\}", lambda m: next(g for g in m.groups() if g is not None)),
           (r"\$([^$]*)\$", r"\1"), (r"\\&", "&"), (r"\{\}", ""), (r"\\-", "")]
    for pat, r in rep:
        s = re.sub(pat, r, s)
    s = re.sub(r"\{\}|\\(?=\s)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def cmd_metadata(a: argparse.Namespace) -> int:
    paper = Path(a.paper)
    mac = macros(paper / "data/metrics.tex", paper / "paper.tex", paper / "sections/fallbacks.tex")
    mac.update({"ultra": "Isosyll", "cpus": "CPU-s/s", "pp": " pp "})
    src = (paper / "paper.tex").read_text()
    title = plain(mac.get("PaperTitle", "") + (": " + mac["PaperDescriptor"] if "PaperDescriptor" in mac else ""), mac)
    # Title as typeset is "PaperTitle" with descriptor as a subtitle; arXiv wants one line.
    m = re.search(r"pdfauthor=\{(.*?)\}\s*,\s*\n", src, re.S)
    authors = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if x.strip()] if m else []
    abstract = plain((paper / "sections/abstract.tex").read_text(), mac)
    pages = re.search(r"^Pages:\s+(\d+)", subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True, check=True).stdout, re.M).group(1)
    aux = Path(a.aux).read_text(errors="replace")
    figs = len(re.findall(r"\\contentsline \{figure\}", aux))
    leftover = re.findall(r"[\\{}$]", abstract)
    print(json.dumps({"title": title, "authors": authors, "abstract": abstract, "pages": int(pages), "figures": figs,
                      "abstract_len": len(abstract), "leftover_markup": leftover}))
    return 0


# ----------------------------------------------------------------------------- scan
PATTERNS = {
    "ipv4": re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.]*\d)"),
    "hostname": re.compile(r"\bxbabe\w*|\bx[89]\b|\bip-\d+-\d+|\b[\w-]+\.(?:internal|local|lan)\b", re.I),
    "path": re.compile(r"(?<![\w:/.])/(?:home|mnt|Users|tmp|root|media)/"),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "account": re.compile(r"neverhuman/jsr|github\.com/neverhuman|neverhumanbot", re.I),
    "token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abp]-[\w-]{10,}|-----BEGIN [A-Z ]*PRIVATE|(?i:bearer\s+[a-z0-9._-]{20,})"),
}
EMAIL_OK = re.compile(r"@users\.noreply\.github\.com$|^noreply@github\.com$")


def scan_text(label: str, text: str) -> list[str]:
    hits = []
    for kind, rx in PATTERNS.items():
        for m in rx.finditer(text):
            if kind == "email" and EMAIL_OK.search(m.group(0)):
                continue
            ctx = text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " ")
            hits.append(f"{label}: {kind}: ...{ctx}...")
    return hits


def cmd_scan(a: argparse.Namespace) -> int:
    hits: list[str] = []
    for p in map(Path, a.paths):
        files = [f for f in sorted(p.rglob("*")) if f.is_file()] if p.is_dir() else [p]
        for f in files:
            hits += scan_text(f"name {f.name}", f.name)
            if f.suffix.lower() == ".pdf":
                txt = subprocess.run(["pdftotext", "-q", str(f), "-"], capture_output=True, text=True).stdout
                info = subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True).stdout
                hits += scan_text(f"{f} (text)", txt) + scan_text(f"{f} (info)", info)
            elif f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gz", ".zip"}:
                continue
            else:
                hits += scan_text(str(f), f.read_text(errors="replace"))
    for h in hits:
        print("PRIVACY HIT", h, file=sys.stderr)
    print(f"scan: {len(hits)} hit(s)")
    return 1 if hits else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--fls", required=True); s.add_argument("--work", required=True); s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_stage)
    w = sub.add_parser("wrapfigs"); w.add_argument("--dir", required=True); w.add_argument("--names", required=True); w.add_argument("--fls", required=True)
    w.set_defaults(f=cmd_wrapfigs)
    m = sub.add_parser("metadata"); m.add_argument("--paper", required=True); m.add_argument("--pdf", required=True); m.add_argument("--aux", required=True)
    m.set_defaults(f=cmd_metadata)
    c = sub.add_parser("scan"); c.add_argument("paths", nargs="+"); c.set_defaults(f=cmd_scan)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
