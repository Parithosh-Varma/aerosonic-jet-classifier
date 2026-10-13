#!/usr/bin/env python3
"""Sound-activated recorder: mic -> pending_review/ (for dashboard review).

NOT MIC-TESTED YET. Per user instruction, live mic test happens only after
the OG (base) model is established. Until then use --self-test (no mic).

How it works:
  loop: record CLIP_SECONDS wav via ffmpeg (macOS avfoundation) to temp,
  measure RMS/dBFS, keep only clips louder than threshold_dbfs,
  save kept clips as pending_review/evt_YYYYMMDD_HHMMSS.wav + log to events.csv.

Requires: ffmpeg installed (present on this machine). No pip deps (stdlib only).

Usage:
  python3 recorder.py                      # live record loop (TEST ONLY AFTER OG MODEL)
  python3 recorder.py --once               # record a single clip then exit
  python3 recorder.py --self-test          # no mic: synth tone -> pending_review/ (safe now)
  python3 recorder.py --threshold -30 --clip 10 --sr 16000
"""
import argparse
import audioop
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(BASE, "config.json")) as f:
        return json.load(f)


def ensure_dirs(cfg):
    for d in (cfg["pending_dir"], cfg["jet_dir"], cfg["nojet_dir"]):
        os.makedirs(os.path.join(BASE, d), exist_ok=True)


def record_clip_ffmpeg(out_path, seconds, sample_rate, device):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "avfoundation", "-i", device,
        "-t", str(seconds),
        "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "pcm_s16le", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")
    return out_path


def wav_rms_dbfs(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
    if not raw:
        return 0.0, -math.inf
    rms = audioop.rms(raw, width)
    if rms <= 0:
        return 0.0, -math.inf
    peak = float(2 ** (width * 8 - 1))
    return rms, 20.0 * math.log10(rms / peak)


def log_event(cfg, filename, kept, db, threshold):
    p = os.path.join(BASE, cfg["events_file"])
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "filename", "kept", "dbfs", "threshold_dbfs"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), filename, int(kept), f"{db:.1f}", threshold])


def synth_test_tone(path, seconds=3, sample_rate=16000, freq=440.0, loud_dbfs=-15.0):
    """Generate a loud sine wav for pipeline self-test. No mic needed."""
    amp = int(32767 * (10 ** (loud_dbfs / 20.0)))
    n = int(seconds * sample_rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n):
            s = int(amp * math.sin(2 * math.pi * freq * i / sample_rate))
            w.writeframes(struct.pack("<h", s))
    return path


def keep_clip(cfg, tmp_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"evt_{ts}.wav"
    dst = os.path.join(BASE, cfg["pending_dir"], name)
    # avoid collision within same second
    k = 1
    while os.path.exists(dst):
        dst = os.path.join(BASE, cfg["pending_dir"], f"evt_{ts}_{k}.wav")
        k += 1
    shutil.move(tmp_path, dst)
    return os.path.basename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="record one clip then exit")
    ap.add_argument("--self-test", action="store_true", help="no-mic synth tone into pending_review/")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--sr", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.threshold is not None:
        cfg["threshold_dbfs"] = args.threshold
    if args.clip is not None:
        cfg["clip_seconds"] = args.clip
    if args.sr is not None:
        cfg["sample_rate"] = args.sr
    ensure_dirs(cfg)

    if args.self_test:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BASE, cfg["pending_dir"], f"selftest_{ts}.wav")
        synth_test_tone(dst, seconds=3, sample_rate=cfg["sample_rate"])
        _, db = wav_rms_dbfs(dst)
        log_event(cfg, os.path.basename(dst), True, db, cfg["threshold_dbfs"])
        print(f"self-test wrote {dst} ({db:.1f} dBFS) -- review it in the dashboard")
        return

    print(f"recording: {cfg['clip_seconds']}s clips @ {cfg['sample_rate']}Hz, "
          f"keep if > {cfg['threshold_dbfs']} dBFS -> {cfg['pending_dir']}/")
    print("Ctrl-C to stop.")
    try:
        while True:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                record_clip_ffmpeg(tmp, cfg["clip_seconds"], cfg["sample_rate"], cfg["audio_device"])
                _, db = wav_rms_dbfs(tmp)
                kept = db >= cfg["threshold_dbfs"]
                if kept:
                    name = keep_clip(cfg, tmp)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] KEPT {name} ({db:.1f} dBFS)")
                    log_event(cfg, name, True, db, cfg["threshold_dbfs"])
                else:
                    os.remove(tmp)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] skip ({db:.1f} dBFS < {cfg['threshold_dbfs']})")
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"clip error: {e}", file=sys.stderr)
                time.sleep(2)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
