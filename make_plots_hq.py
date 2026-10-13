"""Regenerate all plots — modern theme (clean geometry, cohesive palette)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA = Path(__file__).parent
DPI = 220

# ---- Theme ----
BLUE   = "#2563EB"   # train / primary
EMERALD = "#059669"  # test
AMBER  = "#F59E0B"   # secondary series
RED    = "#DC2626"   # best marker
SLATE_BG = "#F8FAFC"
GRID_C = "#E2E8F0"
SPINE_C = "#CBD5E1"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": SLATE_BG,
    "axes.edgecolor": SPINE_C,
    "axes.grid": True,
    "grid.color": GRID_C,
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": SPINE_C,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def read_hist(name):
    ep, loss, tr, te = [], [], [], []
    with open(DATA / name) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
            tr.append(float(r["train_acc"])); te.append(float(r["test_acc"]))
    return ep, loss, tr, te


def acc_plot(csvname, out, title, subtitle):
    ep, loss, tr, te = read_hist(csvname)
    best = max(te); be = ep[te.index(best)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(ep, tr, te, color=BLUE, alpha=0.07, zorder=1)
    ax.plot(ep, tr, "o-", color=BLUE, lw=2.6, ms=7, mfc="white", mew=2,
            label="train acc", zorder=3)
    ax.plot(ep, te, "s-", color=EMERALD, lw=2.6, ms=7, mfc="white", mew=2,
            label="test acc", zorder=3)
    ax.axhline(best, color=RED, ls=(0, (5, 4)), lw=1.6, alpha=0.85, zorder=2)
    ax.plot(be, best, "*", color=RED, ms=16, mec="white", mew=1.2, zorder=4)
    ax.annotate(f"best {best:.3f} @ ep{be}",
                xy=(be, best), xytext=(12, 18), textcoords="offset points",
                fontsize=11, color=RED, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=RED, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("epoch", fontweight="bold")
    ax.set_ylabel("accuracy", fontweight="bold")
    ax.set_title(title, loc="left", pad=34)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=11, color="#64748B",
            va="bottom", ha="left")
    lo = max(0.0, min(min(tr), min(te)) - 0.08)
    lo = min(lo, 0.5)  # keep a consistent zoomed floor across curves
    ax.set_ylim(lo, 1.06)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.set_xlim(min(ep) - 0.4, max(ep) + 0.4)
    ax.legend(framealpha=0.95, loc="lower right")
    fig.tight_layout()
    fig.savefig(DATA / out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, f"best={best:.3f}@{be}")


acc_plot("history_esc50.csv", "acc_vs_time.png",
         "ESC-50 baseline — accuracy vs epoch",
         "JET vs NO-JET · tiny data, jagged without augmentation")
acc_plot("history_v1.csv", "acc_v1.png",
         "v1 — accuracy vs epoch",
         "ESC-50 + YT military jets · augmented · best checkpoint kept")
acc_plot("history_v2.csv", "acc_v2.png",
         "v2 — accuracy vs epoch",
         "ESC-50 + ~864 YT jets · augmented · current best")

# ---- benchmark grouped bars ----
import numpy as np
labels = ["synth", "esc50", "v1", "v2"]
esc = [0.909, 0.648, 0.886, 0.932]
ba = [0.00, 0.96, 0.39, 0.07]
b1 = [0.00, 1.00, 0.98, 0.98]
x = np.arange(len(labels)); w = 0.22
fig, ax = plt.subplots(figsize=(12, 6))
b0 = ax.bar(x - w, esc, w, label="ESC-fold5 acc", color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
b1_ = ax.bar(x, ba, w, label="BlueAngels prob", color=AMBER, edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x + w, b1, w, label="B-1 pass prob", color=EMERALD, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, fontweight="bold", fontsize=12)
ax.set_ylim(0, 1.15); ax.set_ylabel("score", fontweight="bold")
ax.set_title("Jet model benchmarks", loc="left", pad=34)
ax.text(0, 1.015, "ESC-50 accuracy vs real-world detection probability · boom excluded (needs transient head)",
        transform=ax.transAxes, fontsize=11, color="#64748B", va="bottom", ha="left")
ax.legend(framealpha=0.95, loc="upper left", ncols=3)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
for bars in (b0, b1_, b2):
    lbls = [f"{b.get_height():.2f}" if b.get_height() > 0.03 else "" for b in bars]
    ax.bar_label(bars, labels=lbls, fontsize=10, padding=4, fontweight="bold")
fig.tight_layout()
fig.savefig(DATA / "bench.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("saved bench.png")

# ---- sample spectrogram ----
import torch, torchaudio, sys
sys.path.insert(0, str(DATA))
from train_jet import JetCNN, wav_to_logmel
f = "yt_flybys_5s/jet_0261.wav"
w, sr = torchaudio.load(str(DATA / f)); w = w.squeeze(0)
m = JetCNN(); m.load_state_dict(torch.load(DATA / "esc50_jet.pt", map_location="cpu")); m.eval()
with torch.no_grad():
    p = torch.sigmoid(m(wav_to_logmel(w).unsqueeze(0).unsqueeze(0))).item()
mel = wav_to_logmel(w)
fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"hspace": 0.65})
ax[0].fill_between(range(len(w)), w.numpy(), color=BLUE, alpha=0.25, linewidth=0)
ax[0].plot(w.numpy(), color=BLUE, lw=0.5)
ax[0].set_title("Waveform", loc="left", pad=30)
ax[0].text(0, 1.01, f"{f}  ·  jet_prob={p:.3f}  ·  rms={w.pow(2).mean().sqrt():.3f}",
           transform=ax[0].transAxes, fontsize=11, color="#64748B", va="bottom", ha="left")
ax[0].set_xlim(0, len(w)); ax[0].set_xlabel("samples @16kHz", fontweight="bold"); ax[0].set_ylabel("amp", fontweight="bold")
im = ax[1].imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma")
ax[1].set_title("Log-mel spectrogram", loc="left", pad=30)
ax[1].text(0, 1.01, "64 bands × 313 frames (5 s @16 kHz) — broadband roar, bright low-mel band = flyby signature",
           transform=ax[1].transAxes, fontsize=11, color="#64748B", va="bottom", ha="left")
ax[1].set_xlabel("time frames", fontweight="bold"); ax[1].set_ylabel("mel band", fontweight="bold")
cbar = fig.colorbar(im, ax=ax[1], shrink=0.92, pad=0.015)
cbar.set_label("log energy", fontweight="bold")
fig.tight_layout()
fig.savefig(DATA / "sample_jet_0261.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("saved sample_jet_0261.png")
