"""Generate model-diagnostic figures from REAL held-out test outputs.

Test set: ESC-50 fold 5, exactly the published benchmark selection in
bench.py (8 airplane JET + 80 hard-negative NO-JET = 88 clips, folds 1-4
were training — fold 5 is strictly held out). Model: v2_jet.pt (best).

Outputs (PNG 220 dpi + PDF) in figures/ plus figures/metrics_v2.json.
Nothing here is hardcoded: every number comes from model forward passes.

Usage: python3 make_eval_figs.py [--model v2_jet.pt] [--threshold 0.5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import viz
from viz import CLASS_COLORS, SERIES_COLORS, METRIC_COLORS
from bench import feat5, load_model
from train_jet import SR

DATA = Path(__file__).parent
FIG = DATA / "figures"

# Exact published benchmark selection (mirrors bench.py esc_fold5).
NEG_CATS = {"helicopter", "engine", "train", "chainsaw", "wind",
            "thunderstorm", "siren", "car_horn", "fireworks", "rain"}


def load_fold5():
    """Return (file_paths, y_true) for the published n=88 fold-5 set."""
    rows = list(csv.DictReader(open(DATA / "esc50_tmp" / "meta" / "esc50.csv")))
    paths, ys = [], []
    for r in rows:
        if int(r["fold"]) != 5:
            continue
        if r["category"] == "airplane":
            ys.append(1)
        elif r["category"] in NEG_CATS:
            ys.append(0)
        else:
            continue
        paths.append(DATA / "esc50_tmp" / "audio" / r["filename"])
    return paths, np.array(ys, dtype=int)


def score_model(model, device, paths):
    model.eval()
    scores = []
    with torch.no_grad():
        for p in paths:
            scores.append(torch.sigmoid(model(feat5(p).to(device))).item())
    return np.array(scores)


def fig_confusion(stats, normalized=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    viz.apply_theme()
    figure, ax = viz.fig("square")
    cm = np.array([[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]])
    if normalized:
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = "{:.2f}"
    else:
        fmt = "{:d}"
    cmap = LinearSegmentedColormap.from_list("teal", ["#FFFFFF", viz.TEAL])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1 if normalized else cm.max())
    ax.set_aspect("equal")
    ax.set_xticks([0, 1], ["pred NO-JET", "pred JET"])
    ax.set_yticks([0, 1], ["true NO-JET", "true JET"])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            dark = v > (0.55 if normalized else cm.max() * 0.55)
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if dark else viz.INK)
    ax.set_xlabel("predicted label", fontweight="bold")
    ax.set_ylabel("true label", fontweight="bold")
    kind = "normalized" if normalized else "counts"
    viz.title_block(
        ax, f"v2 confusion matrix — ESC-50 fold 5 ({kind})",
        f"n={stats['n']} ({stats['n_pos']} JET) · threshold {stats['threshold']} · "
        f"acc {stats['accuracy']:.3f} · F1 {stats['f1']:.3f}")
    return figure


def fig_roc(y_true, y_score, stats):
    import matplotlib.pyplot as plt
    viz.apply_theme()
    pts = viz.roc_points(y_true, y_score)
    figure, ax = viz.fig("square")
    ax.plot([0, 1], [0, 1], ls=(0, (5, 4)), color=viz.INK_SOFT, lw=1.4,
            label="chance")
    if pts is None:
        ax.text(0.5, 0.5, "ROC undefined: single class in test set",
                ha="center", transform=ax.transAxes, color=viz.INK_SOFT)
    else:
        ax.plot(pts["fpr"], pts["tpr"], color=viz.TEAL, lw=2.6,
                label=f"v2 ROC (AUC {pts['auc']:.3f})")
        op_fpr = stats["fp"] / max(stats["fp"] + stats["tn"], 1)
        op_tpr = stats["recall"]
        ax.plot(op_fpr, op_tpr, "D", color=viz.AMBER, ms=9, mec="white",
                label=f"operating point @ {stats['threshold']} "
                      f"(FPR {op_fpr:.2f}, TPR {op_tpr:.2f})", zorder=4)
    ax.set_xlabel("false positive rate", fontweight="bold")
    ax.set_ylabel("true positive rate", fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    viz.title_block(ax, "v2 ROC — ESC-50 fold 5 (held out)",
                    f"n={stats['n']} · {stats['n_pos']} JET / {stats['n'] - stats['n_pos']} NO-JET")
    return figure


def fig_pr(y_true, y_score, stats):
    import matplotlib.pyplot as plt
    viz.apply_theme()
    pts = viz.pr_points(y_true, y_score)
    figure, ax = viz.fig("square")
    if pts is None:
        ax.text(0.5, 0.5, "PR undefined: single class in test set",
                ha="center", transform=ax.transAxes, color=viz.INK_SOFT)
    else:
        ax.axhline(pts["prevalence"], ls=(0, (5, 4)), color=viz.INK_SOFT,
                   lw=1.4, label=f"prevalence {pts['prevalence']:.2f}")
        ax.plot(pts["recall"], pts["precision"], color=viz.TEAL, lw=2.6,
                label=f"v2 PR (AP {pts['ap']:.3f})")
        ax.plot(stats["recall"], stats["precision"], "D", color=viz.AMBER,
                ms=9, mec="white",
                label=f"operating point @ {stats['threshold']}", zorder=4)
    ax.set_xlabel("recall", fontweight="bold")
    ax.set_ylabel("precision", fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left")
    viz.title_block(ax, "v2 precision–recall — ESC-50 fold 5 (held out)",
                    "Only 8 JET clips in test: precision is the honest axis")
    return figure


def fig_perclass(stats):
    import matplotlib.pyplot as plt
    viz.apply_theme()
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    # NO-JET as the positive class: tp_n = tn, fp_n = fn, fn_n = fp
    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f
    jet = prf(tp, fp, fn)
    nojet = prf(tn, fn, fp)
    labels = ["precision", "recall", "F1"]
    x = np.arange(3); w = 0.32
    figure, ax = viz.fig("single")
    b0 = ax.bar(x - w / 2, jet, w, label="JET", color=CLASS_COLORS["JET"],
                edgecolor="white", linewidth=1.2, zorder=3)
    b1 = ax.bar(x + w / 2, nojet, w, label="NO-JET",
                color=CLASS_COLORS["NO-JET"], edgecolor="white",
                linewidth=1.2, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel("score", fontweight="bold")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    viz.clean_bar_labels(ax, b0); viz.clean_bar_labels(ax, b1)
    ax.legend(framealpha=0.95, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncols=2)
    viz.title_block(ax, "v2 per-class scores — ESC-50 fold 5 (held out)",
                    f"JET n={stats['n_pos']} · NO-JET n={stats['n'] - stats['n_pos']} · threshold {stats['threshold']}")
    return figure


def fig_confidence(y_true, y_score, stats):
    import matplotlib.pyplot as plt
    viz.apply_theme()
    figure, ax = viz.fig("single")
    bins = np.linspace(0, 1, 21)
    ax.hist(y_score[y_true == 0], bins=bins, color=CLASS_COLORS["NO-JET"],
            alpha=0.7, label=f"true NO-JET (n={(y_true == 0).sum()})",
            edgecolor="white", linewidth=0.8, zorder=2)
    # Minority class as a stepped outline so 8 JET clips stay visible
    # against 80 NO-JET clips (redundant fill + edge encoding).
    ax.hist(y_score[y_true == 1], bins=bins, color=CLASS_COLORS["JET"],
            histtype="step", lw=2.6,
            label=f"true JET (n={(y_true == 1).sum()})", zorder=4)
    ax.hist(y_score[y_true == 1], bins=bins, color=CLASS_COLORS["JET"],
            alpha=0.25, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(stats["threshold"], color=viz.AMBER, ls=(0, (5, 4)), lw=1.8,
               label=f"threshold {stats['threshold']}")
    ax.set_xlabel("predicted P(JET)", fontweight="bold")
    ax.set_ylabel("clips", fontweight="bold")
    viz.integer_x(ax)
    ax.legend(framealpha=0.95)
    viz.title_block(ax, "v2 confidence distributions — ESC-50 fold 5",
                    "Overlap near the threshold shows where errors come from")
    return figure


def fig_class_balance():
    import matplotlib.pyplot as plt
    viz.apply_theme()
    counts = {}
    with open(DATA / "combined_manifest.csv") as f:
        for r in csv.DictReader(f):
            counts[(r["source"], r["label"])] = counts.get((r["source"], r["label"]), 0) + 1
    sources = sorted({s for s, _ in counts})
    labels = ["JET", "JET-heldout", "NOJET", "BOOM"]
    fills = {  # JET-heldout hatched: same hue, distinct in B/W print
        "JET": {"color": CLASS_COLORS["JET"], "edgecolor": "white"},
        "JET-heldout": {"color": "#6FAFA2", "hatch": "//",
                        "edgecolor": CLASS_COLORS["JET"]},
        "NOJET": {"color": CLASS_COLORS["NOJET"], "edgecolor": "white"},
        "BOOM": {"color": CLASS_COLORS["BOOM"], "edgecolor": "white"},
    }
    x = np.arange(len(sources)); w = 0.18
    figure, ax = viz.fig("wide")
    for i, lab in enumerate(labels):
        vals = [counts.get((s, lab), 0) for s in sources]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab,
                      linewidth=1.0, zorder=3, **fills[lab])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12,
                        f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, sources, rotation=12, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(counts.values()) * 2.5)
    ax.set_ylabel("clip files (log scale)", fontweight="bold")
    ax.legend(framealpha=0.95, ncols=4)
    total = sum(counts.values())
    viz.title_block(ax, "Training-pool class balance (file counts by source)",
                    f"{total} files in combined_manifest.csv · excludes held-out JET from training")
    return figure


def fig_loss():
    import matplotlib.pyplot as plt
    viz.apply_theme()
    figure, ax = viz.fig("single")
    specs = [("history_esc50.csv", "esc50 baseline", viz.GRAPHITE),
             ("history_v1.csv", "v1", viz.TEAL),
             ("history_v2.csv", "v2", viz.AMBER),
             ("history_v3.csv", "v3", "#7C3AED")]
    any_val = False
    for name, label, color in specs:
        p = DATA / name
        if not p.exists():
            print(f"  skip {name} (missing)")
            continue
        with open(p) as f:
            reader = csv.DictReader(f)
            has_val = "val_loss" in reader.fieldnames
            ep, loss, val = [], [], [] if has_val else None
            for r in reader:
                ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
                if has_val:
                    val.append(float(r["val_loss"]))
        ax.plot(ep, loss, "o-", color=color, lw=2.2, ms=5, mfc="white",
                mew=1.8, label=f"{label} train")
        if has_val:
            any_val = True
            ax.plot(ep, val, "s--", color=color, lw=1.6, ms=4, mfc="white",
                    mew=1.5, alpha=0.9, label=f"{label} held-out")
    ax.set_xlabel("epoch", fontweight="bold")
    ax.set_ylabel("loss (BCE)", fontweight="bold")
    viz.integer_x(ax)
    ax.legend(framealpha=0.95)
    viz.title_block(ax, "Training vs held-out loss",
                    "Solid = train · dashed = held-out (logged by runs "
                    "with val-loss support; older runs train-only)"
                    if any_val else
                    "Train loss only — no validation loss was logged (see limitations)")
    return figure


def fig_errors(paths, y_true, y_score, stats, threshold):
    """Side-by-side waveforms of the most informative errors.

    Shows up to 2 false negatives (lowest scores) and up to 2 false
    positives (highest scores). Panels are omitted gracefully when a
    category is empty — the title always states the true counts.
    """
    import matplotlib.pyplot as plt
    viz.apply_theme()
    order_fn = np.where((y_true == 1) & (y_score < threshold))[0]
    order_fn = order_fn[np.argsort(y_score[order_fn])][:2]
    order_fp = np.where((y_true == 0) & (y_score >= threshold))[0]
    order_fp = order_fp[np.argsort(-y_score[order_fp])][:2]
    picks = [("FN — missed JET", i) for i in order_fn] + \
            [("FP — false alarm", i) for i in order_fp]
    if not picks:
        figure, ax = viz.fig("single")
        ax.text(0.5, 0.5, f"No errors at threshold {threshold} (n={stats['n']})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=viz.INK_SOFT)
        ax.set_axis_off()
        viz.title_block(ax, "v2 error analysis — no errors to show", "")
        return figure
    figure, axes = plt.subplots(len(picks), 1,
                                figsize=(12, 3.4 * max(len(picks), 1)),
                                squeeze=False)
    axes = axes.ravel()
    for ax, (kind, i) in zip(axes, picks):
        w, sr = torchaudio.load(str(paths[i]))
        w = w.mean(0) if w.shape[0] > 1 else w.squeeze(0)
        t = np.arange(len(w)) / sr
        ax.fill_between(t, w.numpy(), color=viz.AMBER, alpha=0.2,
                        linewidth=0, rasterized=True)
        ax.plot(t, w.numpy(), color=viz.GRAPHITE, lw=0.6, rasterized=True)
        ax.set_xlim(0, t[-1])
        viz.title_block(ax, f"{kind}: {paths[i].name}",
                        f"P(JET)={y_score[i]:.3f} · threshold {threshold}",
                        pad=28)
        ax.set_xlabel("time (s)", fontweight="bold")
        ax.set_ylabel("amp", fontweight="bold")
    figure.suptitle(
        f"v2 error analysis — {stats['fn']} missed JET / {stats['fp']} false alarms "
        f"(fold 5, n={stats['n']})",
        fontsize=viz.TITLE_SIZE, fontweight="bold", x=0.08, ha="left")
    figure.tight_layout()
    return figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2_jet.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    mp = DATA / args.model
    if not mp.exists():
        print(f"model not found: {mp}")
        sys.exit(1)
    meta = DATA / "esc50_tmp" / "meta" / "esc50.csv"
    if not meta.exists():
        print("ESC-50 metadata not found; eval figures need esc50_tmp/. Exiting.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {mp} on {device} ...")
    model = load_model(args.model, device)
    paths, y_true = load_fold5()
    print(f"fold-5 test: n={len(paths)} pos={(y_true == 1).sum()} "
          f"neg={(y_true == 0).sum()}")
    y_score = score_model(model, device, paths)
    stats = viz.confusion_stats(y_true, y_score, args.threshold)
    print(f"v2 fold-5 @ {args.threshold}: acc={stats['accuracy']:.3f} "
          f"prec={stats['precision']:.3f} rec={stats['recall']:.3f} "
          f"F1={stats['f1']:.3f} TP={stats['tp']} TN={stats['tn']} "
          f"FP={stats['fp']} FN={stats['fn']}")

    FIG.mkdir(exist_ok=True)
    with open(FIG / "metrics_v2.json", "w") as f:
        json.dump({"model": args.model, "test": "esc50-fold5-n88",
                   **{k: (float(v) if isinstance(v, float) else v)
                      for k, v in stats.items()}}, f, indent=2)
    print("saved figures/metrics_v2.json")

    viz.save(fig_confusion(stats), FIG / "confusion_v2")
    print("saved figures/confusion_v2.png/.pdf")
    viz.save(fig_confusion(stats, normalized=True), FIG / "confusion_v2_norm")
    print("saved figures/confusion_v2_norm.png/.pdf")
    viz.save(fig_roc(y_true, y_score, stats), FIG / "roc_v2")
    print("saved figures/roc_v2.png/.pdf")
    viz.save(fig_pr(y_true, y_score, stats), FIG / "pr_v2")
    print("saved figures/pr_v2.png/.pdf")
    viz.save(fig_perclass(stats), FIG / "perclass_v2")
    print("saved figures/perclass_v2.png/.pdf")
    viz.save(fig_confidence(y_true, y_score, stats), FIG / "confidence_v2")
    print("saved figures/confidence_v2.png/.pdf")
    if (DATA / "combined_manifest.csv").exists():
        viz.save(fig_class_balance(), FIG / "class_balance")
        print("saved figures/class_balance.png/.pdf")
    viz.save(fig_loss(), FIG / "loss_curves")
    print("saved figures/loss_curves.png/.pdf")
    viz.save(fig_errors(paths, y_true, y_score, stats, args.threshold),
             FIG / "errors_v2")
    print("saved figures/errors_v2.png/.pdf")


if __name__ == "__main__":
    main()
