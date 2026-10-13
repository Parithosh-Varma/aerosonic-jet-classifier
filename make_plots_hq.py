"""Regenerate all plots at publication quality (dpi=220, bigger fonts, styled)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent
DPI = 220
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11})

def read_hist(name):
    ep, loss, tr, te = [], [], [], []
    with open(DATA/name) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
            tr.append(float(r["train_acc"])); te.append(float(r["test_acc"]))
    return ep, loss, tr, te

def acc_plot(csvname, out, title):
    ep, loss, tr, te = read_hist(csvname)
    best = max(te); be = ep[te.index(best)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ep, tr, "o-", lw=2.2, ms=6, label="train acc")
    ax.plot(ep, te, "s-", lw=2.2, ms=6, label="test acc")
    ax.axhline(best, color="green", ls="--", lw=1.5, alpha=0.7, label=f"best {best:.3f} ep{be}")
    ax.set_xlabel("epoch (time)"); ax.set_ylabel("accuracy")
    ax.set_title(title); ax.set_ylim(0, 1.03); ax.legend(framealpha=0.95)
    fig.tight_layout(); fig.savefig(DATA/out, dpi=DPI); plt.close(fig)
    print("saved", out, f"best={best:.3f}@{be}")

acc_plot("history_esc50.csv", "acc_vs_time.png", "ESC-50 JET vs NO-JET — accuracy vs time")
acc_plot("history_v1.csv", "acc_v1.png", "v1 JET (ESC-50 + YT military jets, augmented) — accuracy vs time")
acc_plot("history_v2.csv", "acc_v2.png", "v2 JET (ESC-50 + ~864 YT jets, augmented) — accuracy vs time")

# benchmark bars
import numpy as np
labels = ["synth", "esc50", "v1", "v2"]
esc = [0.909, 0.648, 0.886, 0.932]
ba = [0.00, 0.96, 0.39, 0.07]
b1 = [0.00, 1.00, 0.98, 0.98]
x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(10.5, 5.5))
b0 = ax.bar(x-w, esc, w, label="ESC-fold5 acc", edgecolor="black")
b1_ = ax.bar(x, ba, w, label="BlueAngels prob", edgecolor="black")
b2 = ax.bar(x+w, b1, w, label="B-1 pass prob", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.06); ax.set_ylabel("score")
ax.set_title("Jet model benchmarks — ESC-50 vs real-world (boom excluded: needs transient head)")
ax.legend(framealpha=0.95)
for bars in (b0, b1_, b2):
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
fig.tight_layout(); fig.savefig(DATA/"bench.png", dpi=DPI); plt.close(fig)
print("saved bench.png")

# sample spectrogram HQ
import torch, torchaudio, sys
sys.path.insert(0, str(DATA))
from train_jet import JetCNN, wav_to_logmel
f = "yt_flybys_5s/jet_0261.wav"
w, sr = torchaudio.load(str(DATA/f)); w = w.squeeze(0)
m = JetCNN(); m.load_state_dict(torch.load(DATA/"esc50_jet.pt", map_location="cpu")); m.eval()
with torch.no_grad():
    p = torch.sigmoid(m(wav_to_logmel(w).unsqueeze(0).unsqueeze(0))).item()
mel = wav_to_logmel(w)
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"hspace": 0.35})
ax[0].plot(w.numpy(), lw=0.4); ax[0].set_title(f"{f}  waveform  jet_prob={p:.3f}  rms={w.pow(2).mean().sqrt():.3f}")
ax[0].set_xlim(0, len(w)); ax[0].set_xlabel("samples @16kHz"); ax[0].set_ylabel("amp")
im = ax[1].imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma")
ax[1].set_title("log-mel 64 bands × 313 frames (5 s @16 kHz)"); ax[1].set_xlabel("time"); ax[1].set_ylabel("mel")
fig.colorbar(im, ax=ax[1], shrink=0.9)
fig.tight_layout(); fig.savefig(DATA/"sample_jet_0261.png", dpi=DPI); plt.close(fig)
print("saved sample_jet_0261.png")
