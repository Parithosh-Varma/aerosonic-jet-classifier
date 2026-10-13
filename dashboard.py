#!/usr/bin/env python3
"""Jet-noise review dashboard: pending_review/ -> training_dataset/jet|no_jet/.

Run:  streamlit run dashboard.py
Flow: listen to each clip, if it is really a jet engine noise press
      "Yes - jet" (or type yes in the box) -> file moves to training_dataset/jet/.
      "No" moves it to training_dataset/no_jet/. Every decision is appended
      to review_log.csv.
"""
import csv
import json
import os
import shutil
import wave
from datetime import datetime

import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(BASE, "config.json")) as f:
        return json.load(f)


def ensure_dirs(cfg):
    for d in (cfg["pending_dir"], cfg["jet_dir"], cfg["nojet_dir"]):
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
    return cfg


def list_wavs(rel):
    d = os.path.join(BASE, rel)
    if not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.lower().endswith(".wav")]
    fs.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)))
    return fs


def wav_info(path):
    with wave.open(path, "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        return {"seconds": n / sr if sr else 0, "sr": sr, "ch": w.getnchannels()}
    return {}


def log_decision(cfg, filename, decision, src, dst):
    p = os.path.join(BASE, cfg["log_file"])
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "filename", "decision", "src", "dst"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), filename, decision, src, dst])


def move_clip(cfg, filename, decision):
    """decision: 'jet' -> jet_dir, 'nojet' -> nojet_dir. Returns dst path."""
    src = os.path.join(BASE, cfg["pending_dir"], filename)
    dest_dir = cfg["jet_dir"] if decision == "jet" else cfg["nojet_dir"]
    dst = os.path.join(BASE, dest_dir, filename)
    k = 1
    base, ext = os.path.splitext(filename)
    while os.path.exists(dst):
        dst = os.path.join(BASE, dest_dir, f"{base}_{k}{ext}")
        k += 1
    shutil.move(src, dst)
    log_decision(cfg, filename, decision, src, dst)
    return dst


def waveform(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        width, sr = w.getsampwidth(), w.getframerate()
    dtype = {1: "int8", 2: "int16", 4: "int32"}[width]
    y = __import__("numpy").frombuffer(raw, dtype=dtype).astype(float)
    if y.size == 0:
        return None, None
    peak = float(abs(y).max()) or 1.0
    y = y / peak
    # downsample to ~2000 points for fast plot
    step = max(1, len(y) // 2000)
    return y[::step], sr


cfg = ensure_dirs(load_config())
st.set_page_config(page_title="Jet Noise Review", layout="wide")
st.title("Jet Engine Noise — Manual Review")

pend = list_wavs(cfg["pending_dir"])
jets = list_wavs(cfg["jet_dir"])
noj = list_wavs(cfg["nojet_dir"])
c1, c2, c3 = st.columns(3)
c1.metric("Pending review", len(pend))
c2.metric("Accepted (jet)", len(jets))
c3.metric("Rejected (no-jet)", len(noj))

tab_review, tab_jet, tab_nojet = st.tabs(["Review", f"Jet ({len(jets)})", f"No-jet ({len(noj)})"])

with tab_review:
    if not pend:
        st.info(f"No clips waiting. Recorder drops loud clips into `{cfg['pending_dir']}/`. "
                "Run `python3 recorder.py --self-test` for a safe no-mic test file.")
    else:
        sel = st.selectbox("Clip to review", pend, index=0)
        fpath = os.path.join(BASE, cfg["pending_dir"], sel)
        info = wav_info(fpath)
        st.caption(f"{sel} — {info.get('seconds', 0):.1f}s @ {info.get('sr', '?')}Hz, "
                   f"modified {datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')}")
        with open(fpath, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        try:
            y, sr = waveform(fpath)
            if y is not None:
                import matplotlib.pyplot as plt
                import numpy as np
                dur = info.get("seconds", 0) or len(y) / (sr or 1)
                t = np.linspace(0, dur, len(y))
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(t, y, linewidth=0.5)
                ax.set_title("Waveform (normalized)")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("amplitude")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Waveform unavailable: {e}")

        st.subheader("Is this actually a jet engine noise?")
        b1, b2, b3 = st.columns(3)
        if b1.button("Yes — jet engine", type="primary", use_container_width=True):
            dst = move_clip(cfg, sel, "jet")
            st.success(f"Moved to `{dst}`")
            st.rerun()
        if b2.button("No — not jet", use_container_width=True):
            dst = move_clip(cfg, sel, "nojet")
            st.success(f"Moved to `{dst}`")
            st.rerun()
        if b3.button("Delete", use_container_width=True):
            os.remove(fpath)
            log_decision(cfg, sel, "delete", fpath, "-")
            st.success("Deleted.")
            st.rerun()

        # Literal "type yes" path the user asked for:
        typed = st.text_input('Or type "yes" / "no" and press Enter', value="")
        if typed.strip().lower() == "yes":
            dst = move_clip(cfg, sel, "jet")
            st.success(f"'yes' accepted — moved to `{dst}`")
            st.rerun()
        elif typed.strip().lower() == "no":
            dst = move_clip(cfg, sel, "nojet")
            st.success(f"'no' accepted — moved to `{dst}`")
            st.rerun()
        elif typed.strip():
            st.caption('Type exactly "yes" or "no". Buttons above do the same thing.')


def _clip_list_tab(files, rel, header):
    with header:
        if not files:
            st.info("Empty.")
            return
        pick = st.selectbox("Clip", files, key=rel)
        with open(os.path.join(BASE, rel, pick), "rb") as f:
            st.audio(f.read(), format="audio/wav")
        st.caption(pick)


_clip_list_tab(jets, cfg["jet_dir"], tab_jet)
_clip_list_tab(noj, cfg["nojet_dir"], tab_nojet)

st.divider()
st.caption(f"Decisions log: `{cfg['log_file']}` — training data lives in `{cfg['jet_dir']}/` (jet) and `{cfg['nojet_dir']}/` (no-jet).")
