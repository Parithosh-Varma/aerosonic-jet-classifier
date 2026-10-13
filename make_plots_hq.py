"""Regenerate publication figures from real training logs and benchmarks.

Reads history_*.csv (written by train_*.py) and benchmarks.csv (written by
bench.py) — no metric is hardcoded here. Sample-detection panel runs the
saved esc50_jet.pt on a real training clip.

Usage: python3 make_plots_hq.py
Outputs (repo root, tracked): acc_vs_time.png, acc_v1.png, acc_v2.png,
bench.png, sample_jet_0261.png
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import viz
from viz import SERIES_COLORS, METRIC_COLORS

DATA = Path(__file__).parent
viz.apply_theme()


def read_hist(name):
    ep, loss, tr, te = [], [], [], []
    ap = None
    with open(DATA / name) as f:
        reader = csv.DictReader(f)
        has_ap = "test_AP" in reader.fieldnames
        if has_ap:
            ap = []
        for r in reader:
            ep.append(int(r["epoch"]))
            loss.append(float(r["loss"]))
            tr.append(float(r["train_acc"]))
            te.append(float(r["test_acc"]))
            if has_ap:
                ap.append(float(r["test_AP"]))
    return ep, loss, tr, te, ap


def acc_plot(csvname, out, title, sub):
    ep, loss, tr, te, ap = read_hist(csvname)
    best = max(te)
    be = ep[te.index(best)]
    figure, ax = viz.fig("wide")
    ax.fill_between(ep, tr, te, color=SERIES_COLORS["train"], alpha=0.08,
                    zorder=1)
    ax.plot(ep, tr, "o-", color=SERIES_COLORS["train"], lw=2.4, ms=6,
            mfc="white", mew=1.8, label="train acc", zorder=3)
    ax.plot(ep, te, "s-", color=SERIES_COLORS["test"], lw=2.4, ms=6,
            mfc="white", mew=1.8, label="test acc (held out)", zorder=3)
    if ap is not None:
        ax.plot(ep, ap, "^-", color=SERIES_COLORS["test_ap"], lw=2.2, ms=6,
                mfc="white", mew=1.8, label="test AP", zorder=3)
    ax.axhline(best, color=viz.AMBER, ls=(0, (5, 4)), lw=1.6, alpha=0.9)
    ax.plot(be, best, "*", color=viz.AMBER, ms=15, mec="white", mew=1.1,
            zorder=4)
    ax.annotate(f"best {best:.3f} @ ep{be}",
                xy=(be, best), xytext=(12, 18), textcoords="offset points",
                fontsize=11, color=viz.AMBER, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=viz.AMBER, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=viz.AMBER, lw=1.4))
    ax.set_xlabel("epoch", fontweight="bold")
    ax.set_ylabel("accuracy", fontweight="bold")
    lo = min(min(min(tr), min(te)) - 0.08, 0.5)
    ax.set_ylim(max(lo, 0.0), 1.06)
    viz.integer_x(ax)
    ax.set_xlim(min(ep) - 0.4, max(ep) + 0.4)
    ax.legend(framealpha=0.95, loc="lower right")
    viz.title_block(ax, title, sub)
    viz.save(figure, DATA / out, pdf=False)
    print("saved", out, f"best={best:.3f}@{be}")


acc_plot("history_esc50.csv", "acc_vs_time.png",
         "ESC-50 baseline — accuracy vs epoch",
         "JET vs NO-JET · no augmentation, no best-checkpoint")
acc_plot("history_v1.csv", "acc_v1.png",
         "v1 — accuracy vs epoch",
         "ESC-50 + YT military jets · augmented · best checkpoint kept")
acc_plot("history_v2.csv", "acc_v2.png",
         "v2 — accuracy vs epoch",
         "ESC-50 + ~864 YT jets · augmented · current best")
if (DATA / "history_v3.csv").exists():
    acc_plot("history_v3.csv", "acc_v3.png",
             "v3 unified — accuracy vs epoch",
             "aerosonic + YT + ESC · best checkpoint on test AP")
else:
    print("skip acc_v3.png (history_v3.csv missing — run python3 train_v3.py)")

# ---- benchmark bars, driven by benchmarks.csv (never hardcoded) ----
with open(DATA / "benchmarks.csv") as f:
    rows = list(csv.DictReader(f))
short = {"jet_cnn_synth.pt": "synth", "esc50_jet.pt": "esc50",
         "v1_jet.pt": "v1", "v2_jet.pt": "v2"}
labels, esc, ba, b1 = [], [], [], []
for r in rows:
    labels.append(short.get(r["model"], r["model"]))
    esc.append(float(r["esc_acc"]))
    ba.append(float(r["BlueAngels F/A-18 (crowd)"]))
    b1.append(float(r["B-1 jet pass (5-10s)"]))
print("bench source: benchmarks.csv ->", labels)

x = np.arange(len(labels))
w = 0.22
figure, ax = viz.fig("wide")
b0 = ax.bar(x - w, esc, w, label="ESC-fold5 acc",
            color=METRIC_COLORS["esc_acc"], edgecolor="white", linewidth=1.2,
            zorder=3)
b1_ = ax.bar(x, ba, w, label="BlueAngels prob",
             color=METRIC_COLORS["blueangels"], edgecolor="white",
             linewidth=1.2, zorder=3)
b2 = ax.bar(x + w, b1, w, label="B-1 pass prob", color=METRIC_COLORS["b1"],
            edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight="bold", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("score", fontweight="bold")
import matplotlib.pyplot as plt
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
for bars in (b0, b1_, b2):
    viz.clean_bar_labels(ax, bars)
ax.legend(framealpha=0.95, loc="upper left", ncols=3)
viz.title_block(
    ax, "Jet model benchmarks",
    "ESC-50 accuracy vs real-world detection probability · "
    "boom excluded (needs transient head)")
viz.save(figure, DATA / "bench", pdf=False)
print("saved bench.png")

# ---- sample detection: waveform (seconds) + log-mel (Hz-labeled) ----
import torch
import torchaudio
from train_jet import JetCNN, wav_to_logmel, SR, N_MELS

clip = "yt_flybys_5s/jet_0261.wav"
w, sr = torchaudio.load(str(DATA / clip))
w = w.squeeze(0)
m = JetCNN()
m.load_state_dict(torch.load(DATA / "esc50_jet.pt", map_location="cpu"))
m.eval()
with torch.no_grad():
    p = torch.sigmoid(m(wav_to_logmel(w).unsqueeze(0).unsqueeze(0))).item()
mel = wav_to_logmel(w)
dur = len(w) / SR

figure, ax = plt.subplots(2, 1, figsize=viz.SIZES["tall"],
                             gridspec_kw={"hspace": 0.65})
t = np.arange(len(w)) / SR
ax[0].fill_between(t, w.numpy(), color=viz.TEAL, alpha=0.2, linewidth=0)
ax[0].plot(t, w.numpy(), color=viz.GRAPHITE, lw=0.5)
ax[0].set_xlim(0, dur)
ax[0].set_xlabel("time (s)", fontweight="bold")
ax[0].set_ylabel("amplitude", fontweight="bold")
viz.title_block(ax[0], "Waveform",
                f"{clip} · {dur:.1f} s @16 kHz · "
                f"P(JET)={p:.3f} · rms={w.pow(2).mean().sqrt():.3f}", pad=30)

n_frames = mel.shape[1]
extent = [0, n_frames * 256 / SR, 0, N_MELS - 1]
im = ax[1].imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma",
                  extent=extent)
pos, tick_labels = viz.hz_tick_map(N_MELS, SR / 2)
ax[1].set_yticks(pos, tick_labels)
ax[1].set_xlabel("time (s)", fontweight="bold")
ax[1].set_ylabel("frequency (Hz, mel-spaced)", fontweight="bold")
cbar = figure.colorbar(im, ax=ax[1], shrink=0.92, pad=0.015)
cbar.set_label("log energy", fontweight="bold")
viz.title_block(ax[1], "Log-mel spectrogram",
                "64 bands × ~5 s · broadband roar, bright low-frequency "
                "band = flyby signature", pad=30)
figure.tight_layout()
viz.save(figure, DATA / "sample_jet_0261", pdf=False)
print("saved sample_jet_0261.png")
