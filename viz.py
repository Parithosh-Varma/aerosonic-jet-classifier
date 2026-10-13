"""AeroSonic shared figure system — one theme, one palette, reusable helpers.

Visual identity: aerospace research lab meets scientific journal.
Warm off-white publication background, graphite ink, muted teal primary,
amber reserved for highlights. Colorblind-accessible (Okabe-Ito derived).

Usage:
    from viz import apply_theme, fig, save, CLASS_COLORS, subtitle

    apply_theme()
    figure, ax = fig("single")
    ...
    save(figure, "figures/my_plot")   # writes .png (220 dpi) + .pdf

Headless-safe: forces the Agg backend on import.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
PAPER = "#FAF7F1"        # warm off-white axes background
FIGURE_BG = "#FFFFFF"    # figure background
GRID = "#E5DFD2"         # subtle warm gridline
SPINE = "#C9C1AE"        # thin axis spine
INK = "#2B2E33"          # deep graphite primary text
INK_SOFT = "#6B7280"     # secondary text / subtitles
TEAL = "#0F766E"         # muted teal — primary data color
GRAPHITE = "#3F4750"     # graphite — secondary series
AMBER = "#B45309"        # amber — highlights only (best points, baselines)
LIGHT_TEAL = "#CCEBE4"   # fills
LIGHT_BLUE = "#DCE7F2"   # fills

# Consistent class colors across every figure.
CLASS_COLORS = {
    "JET": TEAL,
    "JET-heldout": TEAL,
    "NO-JET": GRAPHITE,
    "NOJET": GRAPHITE,
    "BOOM": AMBER,
}

# Consistent series colors for train/test curves.
SERIES_COLORS = {
    "train": GRAPHITE,
    "test": TEAL,
    "test_ap": AMBER,
    "best": AMBER,
}

# Metric colors for benchmark bars (meaning, not class).
METRIC_COLORS = {
    "esc_acc": GRAPHITE,
    "blueangels": AMBER,
    "b1": TEAL,
}

# Sensible figure sizes (inches): single column, wide, multi-panel.
SIZES = {
    "single": (7.5, 4.6),
    "wide": (12, 5.6),
    "square": (6.4, 6.0),
    "tall": (12, 7.5),
}

TITLE_SIZE = 13
SUBTITLE_SIZE = 10
LABEL_SIZE = 11
TICK_SIZE = 10


def apply_theme() -> None:
    """Apply the AeroSonic publication theme to matplotlib rcParams."""
    plt.rcParams.update({
        "figure.facecolor": FIGURE_BG,
        "axes.facecolor": PAPER,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": SPINE,
        "savefig.facecolor": FIGURE_BG,
    })


def fig(kind: str = "single", **kwargs):
    """Create a themed figure. ``kind`` is one of single/wide/square/tall."""
    size = SIZES.get(kind, SIZES["single"])
    return plt.subplots(figsize=kwargs.pop("figsize", size), **kwargs)


def subtitle(ax, text: str, y: float = 1.015) -> None:
    """Add a small gray subtitle line just under the axes title."""
    ax.text(0, y, text, transform=ax.transAxes, fontsize=SUBTITLE_SIZE,
            color=INK_SOFT, va="bottom", ha="left")


def title_block(ax, title: str, sub: str = "", pad: int = 34) -> None:
    """Left-aligned bold title with an optional gray subtitle."""
    ax.set_title(title, loc="left", pad=pad)
    if sub:
        subtitle(ax, sub)


def save(figure, path, dpi: int = 220, pdf: bool = True) -> list:
    """Save ``figure`` as PNG (+PDF/SVG optional). Returns written paths.

    Creates parent directories as needed and closes the figure to avoid
    memory leaks in batch generation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    png = path.with_suffix(".png")
    figure.savefig(png, dpi=dpi, bbox_inches="tight")
    written.append(png)
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        figure.savefig(pdf_path, bbox_inches="tight")
        written.append(pdf_path)
    plt.close(figure)
    return written


def integer_x(ax) -> None:
    """Integer tick locator for epoch axes."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))


def clean_bar_labels(ax, bars, fmt="{:.2f}", skip_below: float = 0.03) -> None:
    """Value labels on bars; skips near-zero bars to avoid clutter."""
    labels = [
        fmt.format(b.get_height()) if b.get_height() > skip_below else ""
        for b in bars
    ]
    ax.bar_label(bars, labels=labels, fontsize=10, padding=4, fontweight="bold")


def check_1d(name: str, arr) -> np.ndarray:
    """Validate a 1-D numeric array; raises ValueError on empty/NaN."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name}: empty array, nothing to plot")
    if np.isnan(a).any():
        raise ValueError(f"{name}: contains NaN, refusing to plot silently")
    return a


def confusion_stats(y_true, y_score, threshold: float = 0.5) -> dict:
    """Binary confusion counts + precision/recall/F1 at ``threshold``.

    Returns a dict with tp/tn/fp/fn, accuracy, precision, recall, f1, n,
    n_pos and the raw arrays for further plotting. No exceptions on
    single-class input — precision/recall degrade to 0.0 honestly.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("confusion_stats: empty input")
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n, "precision": prec, "recall": rec,
        "f1": f1, "n": n, "n_pos": int((y_true == 1).sum()),
        "threshold": threshold,
    }


def roc_points(y_true, y_score):
    """ROC curve points + AUC. Returns None when undefined (single class)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y_true, y_score))}


def pr_points(y_true, y_score):
    """Precision-recall points + average precision. None if single class."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {
        "precision": prec, "recall": rec,
        "ap": float(average_precision_score(y_true, y_score)),
        "prevalence": float((y_true == 1).mean()),
    }


def hz_tick_map(n_mels: int = 64, f_max: float = 8000.0,
                targets=(0, 500, 1000, 2000, 4000, 8000)):
    """Map mel-bin indices to nearest Hz for axis ticks (HTK mel scale).

    torchaudio's MelSpectrogram defaults (mel_scale='htk', f_min=0,
    f_max=nyquist) space filterbank centers uniformly in HTK mel space,
    so the inverse mapping here is exact for the default pipeline.
    Returns (positions, labels).
    """
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    centers_mel = np.linspace(0, mel_max, n_mels)
    centers_hz = 700.0 * (10.0 ** (centers_mel / 2595.0) - 1.0)
    pos, labels = [], []
    for t in targets:
        i = int(np.argmin(np.abs(centers_hz - t)))
        pos.append(i)
        labels.append(f"{t / 1000:g}k" if t >= 1000 else str(t))
    return pos, labels
