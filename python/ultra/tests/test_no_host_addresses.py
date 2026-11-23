"""neverhuman/jsr is a public repository: tracked files carry no host addresses.

Name hosts by role (x5, xbabe6, H0) and read addresses at run time from the environment (MASTER_ADDR, ...)
or an untracked local fleet hosts file. Allowed literals: loopback, the any-address, and the RFC 5737
documentation networks for test fixtures. IP-shaped host names (a-b-c-d) count too: a cloud VM's hostname
is its public address, which is how one leaked into benchmark receipts. Receipt writers therefore record
a role label (JSR_HOST_ROLE or ~/.config/jsr/host-role), never `hostname`.
"""
from __future__ import annotations

import ipaddress
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ALLOWED = {"127.0.0.1", "0.0.0.0"}
DOC_NETS = tuple(ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
DOTTED = re.compile(rf"(?<![\d.]){OCTET}(?:\.{OCTET}){{3}}(?!\d|\.\d)")
# A dashed address may follow a `word-` prefix (cloud defaults such as `ip-a-b-c-d`), but not a word
# character, a dot or `digit-` directly, so names like `avg3-45-50-55` and dashed number runs do not match.
DASHED = re.compile(rf"(?<![\w.])(?<!\d-){OCTET}(?:-{OCTET}){{3}}(?![\w-]|\.\d)")
ROLE = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")
WRITES_HOSTNAME = re.compile(r"^[ \t]*hostname\b[^\n|;&]*>[ \t]*\S*host\b", re.M)


def _allowed(dotted: str) -> bool:
    if dotted in ALLOWED:
        return True
    address = ipaddress.ip_address(dotted)
    return any(address in net for net in DOC_NETS)


def address_literals(text: str) -> list[str]:
    found = [m.group(0) for m in DOTTED.finditer(text) if not _allowed(m.group(0))]
    found += [m.group(0) for m in DASHED.finditer(text) if not _allowed(m.group(0).replace("-", "."))]
    return found


def _tracked_text_files() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True).stdout
    names = [n for n in out.decode().split("\0") if n]
    assert names, "no tracked files: run this test from a git checkout"
    files = []
    for name in names:
        path = REPO / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" not in data[:8192]:
            files.append((name, data.decode("utf-8", errors="replace")))
    return files


def test_tracked_files_carry_no_host_addresses():
    hits = []
    for name, text in _tracked_text_files():
        for number, line in enumerate(text.splitlines(), 1):
            hits += [f"{name}:{number}: {literal}" for literal in address_literals(line)]
    assert not hits, (
        "host addresses in tracked files of a public repository; name hosts by role and read addresses "
        "from the environment or an untracked hosts file:\n" + "\n".join(hits))


def test_receipt_host_files_hold_role_labels():
    bad = [f"{name}: {text.strip()!r}" for name, text in _tracked_text_files()
           if name.startswith("docs/paper/data/raw/") and name.endswith("/host")
           and (not ROLE.fullmatch(text.strip()) or address_literals(text))]
    assert not bad, "receipt host files must hold a role label:\n" + "\n".join(bad)


def test_receipt_writers_record_role_labels_not_hostname():
    offenders = [name for name, text in _tracked_text_files()
                 if name.endswith(".sh") and WRITES_HOSTNAME.search(text)]
    assert not offenders, f"scripts write `hostname` into a receipt; use a role label: {offenders}"


def _addr(*octets: int, sep: str = ".") -> str:
    """Build an address at run time, so this file itself carries no literal for the scan to find."""
    return sep.join(str(o) for o in octets)


def test_detector_flags_addresses_and_allows_loopback_any_and_documentation_networks():
    private, lan, dashed = _addr(10, 1, 2, 3), _addr(192, 168, 1, 20), _addr(8, 8, 4, 4, sep="-")
    assert address_literals(f"MASTER_ADDR={private} MASTER_PORT=29710") == [private]
    assert address_literals(f"end of a sentence {lan}.") == [lan]
    assert address_literals(f"cloud host {dashed} in a receipt") == [dashed]
    assert address_literals(f"aws default ip-{dashed}.ec2.internal") == [dashed]
    assert address_literals("chunk list 1-2-4-8-16 and 4-8-16-32-64, checkpoint d2-avg3-45-50-55") == []
    assert address_literals("bind 127.0.0.1:8765, listen 0.0.0.0") == []
    assert address_literals("fixtures 192.0.2.10 198.51.100.7 203.0.113.9 203-0-113-9") == []
    assert address_literals("kernel 6.8.0-149, versions 1.2.3.4.5, dates 2026-11-23, ranges 33.2-33.3") == []
    assert WRITES_HOSTNAME.search('    hostname > "$run/host"\n')
    assert not WRITES_HOSTNAME.search('    host_role > "$run/host"\n')
    assert ROLE.fullmatch("xbabe6") and ROLE.fullmatch("Mac-Studio.local") and not ROLE.fullmatch(dashed)
