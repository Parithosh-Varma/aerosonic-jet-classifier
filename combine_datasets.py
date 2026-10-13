"""Combine ALL datasets into one unified JET/NO-JET/BOOM training manifest.
Sources (auto-detected, missing ones skipped with notice):
  esc50 (airplane=JET, heli/engine/etc=NO-JET) | yt (all *_5s JET, sonic_booms BOOM)
  aerosonicdb (subclass3=JET, subclass0=NO-JET, needs audio/ unzipped)
  desed (backgrounds=NO-JET, needs desed_train/) | drone_neg (needs drone_audio/)
  vggsound (needs vggsound_audio/ + vggsound.csv) | erau (needs erau/ embeddings)
Output: combined_manifest.csv + counts. Train scripts can read it directly.
Usage: python3 combine_datasets.py
"""
import csv
from pathlib import Path

DATA = Path(__file__).parent
rows = []  # (path, label, source)

def add(path, label, source):
    rows.append((str(path), label, source))

# 1) ESC-50
mp = DATA/"esc50_tmp"/"meta"/"esc50.csv"
if mp.exists():
    import csv as _c
    HARD = {"helicopter","engine","train","chainsaw","wind","thunderstorm","siren","car_horn","fireworks","rain"}
    for r in _c.DictReader(open(mp)):
        p = DATA/"esc50_tmp"/"audio"/r["filename"]
        if not p.exists(): continue
        if r["category"] == "airplane": add(p, "JET", "esc50")
        else: add(p, "NOJET", "esc50")

# 2) YT JET dirs (exclude held-out b1_*/f16 test files from train pool? keep, flag source)
for d in ["yt_flybys_5s", "yt_flybys2_5s", "yt_flybys3_5s", "abbotsford_5s", "f16_5s", "auto_5s"]:
    for p in sorted((DATA/d).glob("*.wav")) if (DATA/d).exists() else []:
        if "b1_" in p.name or d == "f16_5s":
            add(p, "JET-heldout", "yt")
        else:
            add(p, "JET", "yt")

# 3) Booms
if (DATA/"sonic_booms").exists():
    for p in sorted((DATA/"sonic_booms").glob("*.wav")):
        add(p, "BOOM", "yt-boom")

# 4) AeroSonicDB (Kaggle archive/ layout OR audio/ + sample_meta.csv at root)
_mapped = False
for am, ad in [(DATA/"archive"/"sample_meta.csv", DATA/"archive"/"audio"),
               (DATA/"sample_meta.csv", DATA/"audio")]:
    if ad.exists() and am.exists():
        import csv as _c
        meta = {r["filename"]: r["subclass"] for r in _c.DictReader(open(am))}
        n = 0
        for p in ad.rglob("*.wav"):
            s = meta.get(p.name)
            if s == "3": add(p, "JET", "aerosonic"); n += 1
            elif s == "0": add(p, "NOJET", "aerosonic"); n += 1
        print(f"aerosonic [{ad}]: {n} mapped")
        _mapped = True
        break
if not _mapped:
    print("aerosonic audio NOT yet (awaiting Kaggle file from user) — will auto-merge on arrival")

# 5) DESED backgrounds
dd = DATA/"desed_train"
if dd.exists():
    n = sum(1 for _ in dd.rglob("*.wav"))
    for p in dd.rglob("*.wav"): add(p, "NOJET", "desed")
    print(f"desed: {n}")
else:
    print("desed NOT present (downloading in background)")

# 6) Drone negatives
dr = DATA/"drone_audio"
if dr.exists():
    n = 0
    for p in dr.rglob("*.wav"):
        add(p, "NOJET", "drone"); n += 1
    print(f"drone: {n}")
else:
    print("drone_audio NOT present (downloading in background)")

# 7) VGGSound (needs audio shards)
vg = DATA/"vggsound_audio"
if vg.exists() and (DATA/"vggsound.csv").exists():
    print("vggsound: wiring skipped (shard mapping heavy) — targeted pull later")
else:
    print("vggsound audio NOT present — csv only, targeted yt-dlp later")

from collections import Counter
c = Counter([label for _, label, _ in rows])
s = Counter([src for _, _, src in rows])
print("TOTAL files:", len(rows), dict(c))
print("by source:", dict(s))
with open(DATA/"combined_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "label", "source"]); w.writerows(rows)
print("saved combined_manifest.csv")
