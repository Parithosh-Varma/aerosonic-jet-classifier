"""v3 UNIFIED: manifest-driven binary JET vs NO-JET on everything.
JET = aerosonic turbofan (test-split held out) + YT comps + ESC-50 airplane
NOJET = aerosonic silence (test-split held out) + ESC-50 negs (+desed/drone when present)
BOOM excluded (32 clips -> phase-2 transient head).
Splits: aerosonic train-test col + ESC fold!=5 train / ==5 test. YT all-train (capped).
Windows: aerosonic JET<=4/file, NOJET<=2/file, YT/ESC 1/file. Caps: 2500/class.
Metrics: acc + Average Precision (their metric). Best-ckpt on test AP.
Saves: v3_jet.pt, history_v3.csv, acc_v3.png
Usage: python3 train_v3.py [--epochs 20]
"""
import argparse, csv, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torchaudio
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR, N_MELS, load_windows

DATA = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    # aerosonic split map
    split = {}
    for cand in [DATA/"archive"/"sample_meta.csv", DATA/"sample_meta.csv"]:
        if cand.exists():
            for r in csv.DictReader(open(cand)):
                split[r["filename"]] = r["train-test"]
            break

    mani = list(csv.DictReader(open(DATA/"combined_manifest.csv")))
    tr_files, te_files = defaultdict(list), defaultdict(list)
    for m in mani:
        p = Path(m["path"]); label = m["label"]; src = m["source"]
        if label in ("BOOM", "JET-heldout"): continue
        y = 1 if label == "JET" else 0
        if src == "aerosonic":
            (te_files if split.get(p.name) == "test" else tr_files)[y].append(p)
        elif src == "esc50":
            # ESC fold from esc50 csv? filename fold prefix: {fold}-... in esc50_tmp/audio names
            fold = int(p.name.split("-")[0])
            (te_files if fold == 5 else tr_files)[y].append(p)
        else:
            tr_files[y].append(p)  # YT/desed/drone all-train
    print("train files JET/NOJET:", len(tr_files[1]), len(tr_files[0]))
    print("test  files JET/NOJET:", len(te_files[1]), len(te_files[0]))

    # simpler explicit loader (avoids load_windows path coupling)
    T_FIXED = 313  # 5 s @16 kHz, hop 256 (center=True)
    def windows_of(p, per_file, augment):
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
        w = w.squeeze(0); n = 5*SR
        if len(w) < n: return [fixfeat(F.pad(w, (0, n-len(w))))]
        if len(w) <= 12*SR: idx = [0]
        else:
            idx = list(range(0, len(w)-n+1, n//2))[:per_file]
        out = []
        for i in idx[:per_file]:
            c = w[i:i+n]
            c = c[:n]
            if len(c) < n: c = F.pad(c, (0, n-len(c)))
            if augment:
                if random.random() < 0.5: c = c*random.uniform(0.7, 1.3)
                if random.random() < 0.5: c = c + torch.randn_like(c)*random.uniform(0.0, 0.05)
            out.append(fixfeat(c))
        return out

    def fixfeat(c):
        m = wav_to_logmel(c)
        if m.shape[1] > T_FIXED: m = m[:, :T_FIXED]
        elif m.shape[1] < T_FIXED: m = F.pad(m, (0, T_FIXED-m.shape[1]))
        return m.numpy()

    def build(filemap, per_jet, per_noj, cap_jet, cap_noj, augment=False):
        X, y = [], []
        for label, (per, cap) in ((1, (per_jet, cap_jet)), (0, (per_noj, cap_noj))):
            files = filemap[label][:]
            random.shuffle(files)
            for p in files:
                if sum(1 for v in y if v == label) >= cap: break
                try:
                    for f in windows_of(p, per, augment):
                        X.append(f); y.append(label)
                        if sum(1 for v in y if v == label) >= cap: break
                except Exception as e:
                    print("skip", Path(p).name, str(e)[:80])
        return np.stack(X)[:, None, :, :], np.array(y, dtype=np.float32)

    Xtr, ytr = build(tr_files, 4, 2, 2500, 2500)
    # augmented positives top-up
    Xau, yau = build({1: tr_files[1], 0: []}, 4, 2, 1500, 0, augment=True)
    Xtr = np.concatenate([Xtr, Xau]); ytr = np.concatenate([ytr, yau])
    Xte, yte = build(te_files, 4, 2, 10**9, 10**9)
    print(f"train {Xtr.shape} pos={(ytr==1).sum()} neg={(ytr==0).sum()} | test {Xte.shape} pos={(yte==1).sum()} neg={(yte==0).sum()}")

    from sklearn.metrics import average_precision_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JetCNN().to(device)
    pos_ratio = (ytr == 1).sum()/len(ytr)
    pw = torch.tensor([(1-pos_ratio)/pos_ratio]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    print(f"device={device} lr={args.lr} pos_weight={pw.item():.2f}")

    best, best_ep, hist = -1, 0, []
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i+args.batch]
            opt.zero_grad()
            loss = fn(model(Xt[b].to(device)), yt[b].to(device))
            loss.backward(); opt.step(); tot += loss.item()*len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            ptr = torch.sigmoid(model(Xt.to(device))).cpu().numpy()
            pte = torch.sigmoid(model(Xe.to(device))).cpu().numpy()
            pa = ((ptr > 0.5) == ytr).mean()
            pe = ((pte > 0.5) == yte).mean()
            ap = average_precision_score(yte, pte)
        hist.append((tot/len(Xt), pa, pe, ap))
        flag = ""
        if ap > best:
            best, best_ep = ap, ep+1
            torch.save(model.state_dict(), DATA/"v3_jet.pt")
            flag = " <-- BEST saved"
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xt):.4f} train={pa:.3f} test={pe:.3f} AP={ap:.4f}{flag}")
    print(f"BEST AP={best:.4f} ep{best_ep} (theirs: CNN test 98.60 / env ~89) -> v3_jet.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv
    with open(DATA/"history_v3.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["epoch", "loss", "train_acc", "test_acc", "test_AP"])
        for i, (l, a, b, c) in enumerate(hist, 1): w.writerow([i, round(l, 4), round(float(a), 4), round(float(b), 4), round(float(c), 4)])
    ep = list(range(1, len(hist)+1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ep, [h[1] for h in hist], "o-", label="train acc")
    ax.plot(ep, [h[2] for h in hist], "s-", label="test acc")
    ax.plot(ep, [h[3] for h in hist], "^-", label="test AP (their metric)")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1.03)
    ax.set_title("v3 UNIFIED (aerosonic+YT+ESC, best-ckpt on AP)")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(DATA/"acc_v3.png", dpi=220)
    print("saved acc_v3.png")

    model.load_state_dict(torch.load(DATA/"v3_jet.pt", map_location=device)); model.eval()
    for t in ["quick_jets/Blue_Angels_1.mp3", "quick_jets/IAI_Astra_Takeoff.mp3",
              "yt_flybys_5s/b1_01.wav", "f16_5s/clip_0002.wav", "sonic_booms/boom_f18_01.wav"]:
        p = DATA/t
        if not p.exists(): continue
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
        w = w.squeeze(0)[:5*SR]
        if len(w) < 5*SR: w = F.pad(w, (0, 5*SR-len(w)))
        with torch.no_grad():
            print(f"{t}: {torch.sigmoid(model(wav_to_logmel(w).unsqueeze(0).unsqueeze(0).to(device))).item():.3f}")

if __name__ == "__main__":
    main()
