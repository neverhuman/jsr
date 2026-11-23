"""H52 large tier A coverage (runs on x6, read-only unless --apply-x6): which from-source units are already fully
covered by x6 per-shard labels, and which raw shards are fully covered by published from-source units.
--apply-x6 claims those shards for 'covered-by-from-source' (mkdir NNNN.claim), so the per-shard driver skips them."""
import glob, json, os, sys, time
L = os.path.expanduser("~/jsr-data/librilight/shards/raw/librilight.large")
units = json.load(open(os.path.expanduser("~/jsr-data/h52-labeler/large-units-asc.json")))["units"]
umembers = {f"A{i:05d}": [m[1] for m in u["members"]] for i, u in enumerate(units)}
fs = set()
for f in glob.glob(L + "/r0src/A*.labeled"):
    try:
        fs.update(json.load(open(f)).get("members", []))
    except (OSError, ValueError):
        pass
done = set(open(L + "/members_done.txt").read().split()) if os.path.exists(L + "/members_done.txt") else set()
shard_members = {}
for r in glob.glob(L + "/[0-9]*.ready"):
    k = os.path.basename(r)[:-6]
    try:
        shard_members[k] = set(open(f"{L}/{k}.members").read().split())
    except OSError:
        pass
member_shards = {}
for k, ms in shard_members.items():
    for m in ms:
        member_shards.setdefault(m, set()).add(k)
labelled = {k for k in shard_members if os.path.exists(f"{L}/{k}.labeled")}
x6_full = {m for m in done if member_shards.get(m, set()) <= labelled}
cover_units = sorted(u for u, ms in umembers.items() if all(m in x6_full for m in ms))
skip = sorted(k for k, ms in shard_members.items()
              if k not in labelled and not os.path.exists(f"{L}/{k}.claim") and ms and ms <= fs)
applied = 0
if "--apply-x6" in sys.argv:
    for k in skip:
        try:
            os.mkdir(f"{L}/{k}.claim")
        except FileExistsError:
            continue
        with open(f"{L}/{k}.claim/owner.json", "w") as fh:
            json.dump({"owner": "h52-librilight", "worker": "covered-by-from-source",
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh)
        applied += 1
unl = [k for k in shard_members if k not in labelled and not os.path.exists(f"{L}/{k}.claim")]
inflight = 0
for k in shard_members:
    if k not in labelled and os.path.exists(f"{L}/{k}.claim/owner.json"):
        try:
            inflight += json.load(open(f"{L}/{k}.claim/owner.json")).get("worker") == "x6L"
        except (OSError, ValueError):
            inflight += 1
print(json.dumps({"cover_units": cover_units, "skip_shards": len(skip), "applied": applied, "fs_members": len(fs),
                  "x6_full_members": len(x6_full), "shards": len(shard_members), "labelled": len(labelled),
                  "open_shards": len(unl), "inflight_x6": inflight, "pass2_done": os.path.exists(L + "/PASS2_DONE.json")}))
