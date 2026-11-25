"""plbt/v1: book text -> label words (H61, Libri-Light book-text labels).

LibriSpeech transcripts are upper-case words of A-Z and apostrophes, with numbers, abbreviations and
symbols spoken out ("MISTER", "EIGHTEEN SIXTY"). Book text is not. `book_words` turns book text into
tokens of four kinds, and nothing here guesses how a number was read:

  LEX     a word usable as a label as it stands, including single letters (LibriSpeech writes
          initials as letters: T V, THOMAS L HAMER) and letter groups such as U.S. -> U S
  ABBR    an abbreviation with known spoken forms (Mr. -> MISTER; St. -> SAINT or STREET). With one
          form it behaves like a word. With several, the form a recogniser heard decides
  NUM     digits, digits with a unit or currency sign, roman numerals of two or more letters: filled
          from the 1..`cap` heard words aligned to it, which must all be number / unit words
  OTHER   anything else. A row whose aligned span contains one is dropped (`nonlexical`)

Rules, in order:
  1. NFKD, combining marks dropped; apostrophe variants -> "'"; "&c" -> "etc."; "&" -> "and"; upper-case.
  2. Split at white space, hyphens, dashes and "/" (LibriSpeech writes TO DAY, MOTHER IN LAW).
  3. Strip every edge character outside A-Z and 0-9, remembering whether an apostrophe was among them
     on each side (an elision like 'TIS or a quotation mark: tools/plbt/project.py decides).
  4. Classify by the tables below.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CONVENTIONS = "plbt/v1"
LEX, ABBR, NUM, OTHER = "lex", "abbr", "num", "other"
APOSTROPHES = "’‘ʼ`´′‛"
# Spoken forms of abbreviations as LibriSpeech writes them. The first form is used when nothing was heard.
ABBREV: dict[str, tuple[tuple[str, ...], ...]] = {k: tuple(tuple(f.split()) for f in v.split("|")) for k, v in {
    "MR": "MISTER", "MRS": "MISSUS", "MESSRS": "MESSIEURS|MESSRS", "DR": "DOCTOR", "ST": "SAINT|STREET",
    "JR": "JUNIOR", "SR": "SENIOR", "REV": "REVEREND", "CAPT": "CAPTAIN", "LIEUT": "LIEUTENANT",
    "LT": "LIEUTENANT", "COL": "COLONEL", "GEN": "GENERAL", "MAJ": "MAJOR", "SERGT": "SERGEANT", "SGT": "SERGEANT",
    "PROF": "PROFESSOR", "GOV": "GOVERNOR", "ESQ": "ESQUIRE", "MME": "MADAME", "MLLE": "MADEMOISELLE",
    "MT": "MOUNT", "VS": "VERSUS", "ETC": "ET CETERA|AND SO FORTH", "VIZ": "VIZ|NAMELY", "CO": "COMPANY|CO",
    "BROS": "BROTHERS", "VOL": "VOLUME", "CHAP": "CHAPTER", "HON": "HONORABLE|HON|HONOURABLE",
}.items()}
ROMAN_WORDS = frozenset({"MIX", "DIX", "MC", "DI", "LI", "MI"})  # spelled like roman numerals, but words (MC CLOUD)
_UNITS = ("ZERO OH O NAUGHT ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN "
          "FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY "
          "HUNDRED THOUSAND MILLION BILLION FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH "
          "ELEVENTH TWELFTH THIRTEENTH FOURTEENTH FIFTEENTH SIXTEENTH SEVENTEENTH EIGHTEENTH NINETEENTH "
          "TWENTIETH THIRTIETH FORTIETH FIFTIETH SIXTIETH SEVENTIETH EIGHTIETH NINETIETH HUNDREDTH THOUSANDTH "
          "HALF HALVES QUARTER QUARTERS AND A THE OF PER CENT CENTS POINT POUND POUNDS DOLLAR DOLLARS SHILLING "
          "SHILLINGS PENCE PENNY PENNIES FRANC FRANCS DEGREE DEGREES NUMBER O'CLOCK")
NUMBER_KEYS = frozenset(w.replace("'", "") for w in _UNITS.split())
_SPLIT = re.compile(r"[\s\-/‐-―−]+")
_EDGE_L = re.compile(r"^[^A-Z0-9$£]+")
_EDGE_R = re.compile(r"[^A-Z0-9%]+$")
_LEX = re.compile(r"^[A-Z]+(?:'[A-Z]+)*$")
_LETTERS = re.compile(r"^[A-Z](?:\.[A-Z])+$")
_ROMAN = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_NUM = re.compile(r"^[$£]?\d[\d,.]*(?:ST|ND|RD|TH|D|S|%)?$")


@dataclass(frozen=True)
class Word:
    core: str  # upper-case, no edge punctuation; never empty
    kind: str = LEX
    lead: bool = False  # an apostrophe stood before the core in the book
    trail: bool = False  # ... and after it
    cap: int = 0  # NUM: most heard words it may stand for

    @property
    def lexical(self) -> bool:
        return self.kind == LEX

    @property
    def key(self) -> str:
        return self.core.replace("'", "")


def key(word: str) -> str:
    """Matching key of a label / hypothesis word: apostrophes do not count."""
    return word.replace("'", "")


def classify(core: str) -> tuple[str, int]:
    if _LEX.match(core):
        if core in ABBREV:
            return ABBR, 0
        if len(core) > 1 and _ROMAN.match(core) and core not in ROMAN_WORDS:
            return NUM, 3  # XIV -> FOURTEEN / THE FOURTEENTH
        return LEX, 0
    if _NUM.match(core):
        digits = sum(ch.isdigit() for ch in core)
        return NUM, min(8, 2 * digits + 2)  # 1,867 -> ONE THOUSAND EIGHT HUNDRED AND SIXTY SEVEN
    return OTHER, 0


def book_words(text: str) -> list[Word]:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    for ch in APOSTROPHES:
        t = t.replace(ch, "'")
    t = t.upper().replace("&C", " ETC. ").replace("&", " AND ")
    out: list[Word] = []
    for raw in _SPLIT.split(t):
        left = _EDGE_L.match(raw)
        tok = raw[left.end():] if left else raw
        right = _EDGE_R.search(tok)
        core = tok[:right.start()] if right else tok
        if not core:
            continue
        lead = bool(left and "'" in left.group())
        trail = bool(right and "'" in right.group())
        if _LETTERS.match(core):  # U.S -> U S
            out.extend(Word(ch) for ch in core.split("."))
            continue
        kind, cap = classify(core)
        out.append(Word(core, kind, lead, trail, cap))
    return out


def label_words(text: str) -> list[str]:
    """A LibriSpeech-style transcript or hypothesis -> its words (upper-case, split at white space)."""
    return text.upper().split()
