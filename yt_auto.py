"""Auto-harvest YouTube flyby clips: search -> download -> slice -> model-filter -> dataset.
Tracks seen IDs in yt_auto_seen.txt (never re-pulls). Test-quarantine supported.
Usage:
  python3 yt_auto.py --max-new 2 --max-min 12          # quick run
  python3 yt_auto.py --max-new 5 --max-min 25 --queries "fighter jet flyby" "sonic boom"
  python3 yt_auto.py --max-new 3 --quarantine test_links  # keep as TEST-only, never train
Needs: yt-dlp (pip3 install yt-dlp), ffmpeg.
"""
import argparse, sys, subprocess, os
from pathlib import Path
import torch, torchaudio
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR

DATA = Path(__file__).parent
QDIR = DATA / "quick_jets"
SEEN = DATA / "yt_auto_seen.txt"
MODEL = DATA / "v2_jet.pt"
DEFAULT_QUERIES = ["fighter jet flyby compilation", "jet low pass airshow",
                   "F-16 high speed pass", "sonic boom compilation", "B-1 bomber flyby"]

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip()

def search_ids(query, n=6):
    out = sh([sys.executable, "-m", "yt_dlp", "--extractor-args", "youtube:player_client=android",
              "--flat-playlist", "--print", "%(id)s | %(title)s | %(duration)s", f"ytsearch{n}:{query}"])
    ids = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not parts[0]: continue
        try: dur = float(parts[2])
        except ValueError: continue
        ids.append((parts[0], parts[1][:60], dur))
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=2)
    ap.add_argument("--max-min", type=float, default=12)
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--quarantine", default=None, help="dir for TEST-only clips (default: train pool auto_5s)")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    seen = set(SEEN.read_text().split()) if SEEN.exists() else set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = JetCNN(); m.load_state_dict(torch.load(MODEL, map_location=device)); m.eval()
    print(f"filter model: v2_jet.pt thr={args.thr} quarantine={args.quarantine or 'train-pool'}")

    outdir = DATA/(args.quarantine or "auto_5s"); outdir.mkdir(exist_ok=True)
    got, added = 0, 0
    with open(SEEN, "a") as flog:
        for q in args.queries:
            if got >= args.max_new: break
            for vid, title, dur in search_ids(q):
                if got >= args.max_new: break
                if vid in seen: continue
                if dur > args.max_min*60 or dur < 20: continue
                print(f"[{got+1}/{args.max_new}] {vid} {dur/60:.1f}min {title}")
                target = QDIR/f"auto_{vid}.wav"
                r = subprocess.run([sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav",
                                    "--audio-quality", "0", "--extractor-args", "youtube:player_client=android",
                                    "-o", str(target.with_suffix(".%(ext)s")),
                                    f"https://www.youtube.com/watch?v={vid}"],
                                   capture_output=True, text=True)
                if not target.exists():
                    print("  download FAILED, skip"); flog.write(vid+"\n"); seen.add(vid); continue
                w, sr = torchaudio.load(str(target))
                if w.shape[0] > 1: w = w.mean(0, keepdim=True)
                if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
                w = w.squeeze(0)
                kept = 0; total = 0
                for i in range(30*SR if len(w) > 60*SR else 0, len(w)-5*SR+1, 5*SR):
                    total += 1
                    clip = w[i:i+5*SR]
                    with torch.no_grad():
                        p = torch.sigmoid(m(wav_to_logmel(clip).unsqueeze(0).unsqueeze(0).to(device))).item()
                    if p >= args.thr:
                        torchaudio.save(str(outdir/f"auto_{vid}_{kept:03d}.wav"), clip.unsqueeze(0), SR)
                        kept += 1
                print(f"  kept {kept}/{total}")
                added += kept; got += 1
                flog.write(vid+"\n"); seen.add(vid)
    print(f"DONE: {got} videos, +{added} JET clips -> {outdir} | rerun combine_datasets.py to merge")

if __name__ == "__main__":
    main()
