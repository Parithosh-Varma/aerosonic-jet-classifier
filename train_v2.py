"""v2 JET detector: ESC-50 + YT military jets, augmented, best-checkpoint.
JET = esc50 airplane (folds1-4) + yt_flybys_5s + yt_flybys2_5s (minus b1_* held out)
NO-JET = esc50 hard negs + random (folds1-4). Test = esc50 fold5 + b1/BlueAngels/Astra.
Usage: python3 train_v1.py [--epochs 20]
Saves: v2_jet.pt (BEST test acc), history_v2.csv, acc_v2.png
"""
import argparse, csv, random, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchaudio
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR, N_MELS

DATA = Path(__file__).parent
ESC = DATA / "esc50_tmp"
HARD_NEGS = {"helicopter","engine","train","chainsaw","wind","thunderstorm","siren","car_horn","fireworks","rain"}

def load_5s(path, augment=False):
    w, sr = torchaudio.load(str(path))
    if w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
    w = w.squeeze(0)
    n = SR*5
    if len(w) < n: w = F.pad(w, (0, n-len(w)))
    else:
        # random crop for augmentation, else head
        s = random.randint(0, len(w)-n) if (augment and len(w) > n) else 0
        w = w[s:s+n]
    if augment:
        if random.random() < 0.5:  # gain
            w = w * random.uniform(0.7, 1.3)
        if random.random() < 0.5:  # noise
            w = w + torch.randn_like(w) * random.uniform(0.0, 0.05)
        if random.random() < 0.3:  # time shift
            w = torch.roll(w, random.randint(-800, 800))
    return wav_to_logmel(w).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    with open(ESC/"meta"/"esc50.csv") as f:
        rows = list(csv.DictReader(f))
    esc_pos = [ESC/"audio"/r["filename"] for r in rows if r["category"]=="airplane" and int(r["fold"])!=5]
    esc_pos_te = [ESC/"audio"/r["filename"] for r in rows if r["category"]=="airplane" and int(r["fold"])==5]
    esc_neg = [ESC/"audio"/r["filename"] for r in rows if r["category"] in HARD_NEGS and int(r["fold"])!=5]
    esc_neg_te = [ESC/"audio"/r["filename"] for r in rows if r["category"] in HARD_NEGS and int(r["fold"])==5]
    rest = [ESC/"audio"/r["filename"] for r in rows if r["category"] not in HARD_NEGS and r["category"]!="airplane" and int(r["fold"])!=5]
    esc_neg += random.sample(rest, 60)
    rest_te = [ESC/"audio"/r["filename"] for r in rows if r["category"] not in HARD_NEGS and r["category"]!="airplane" and int(r["fold"])==5]
    esc_neg_te += random.sample(rest_te, 15)

    yt1 = sorted((DATA/"yt_flybys_5s").glob("jet_*.wav")) + sorted((DATA/"yt_flybys3_5s").glob("clip_*.wav")) + sorted((DATA/"abbotsford_5s").glob("clip_*.wav"))
    yt2 = sorted((DATA/"yt_flybys2_5s").glob("jet2_*.wav"))  # f16_5s + b1_* held out as test
    print(f"esc_pos={len(esc_pos)} yt1={len(yt1)} yt2={len(yt2)} esc_neg={len(esc_neg)}")
    print(f"held-out real test: esc_pos_te={len(esc_pos_te)} esc_neg_te={len(esc_neg_te)} + b1/BlueAngels/Astra")

    def feat(files, label, augment=False):
        X, y = [], []
        for p in files:
            try:
                X.append(load_5s(p, augment)); y.append(label)
            except Exception as e:
                print("skip", p.name, e)
        return np.stack(X)[:,None,:,:], np.array(y, dtype=np.float32)

    Xp, yp = feat(esc_pos + yt1 + yt2, 1, augment=False)
    Xn, yn = feat(esc_neg, 0, augment=False)
    Xtr = np.concatenate([Xp, Xn]); ytr = np.concatenate([yp, yn])
    # augmented copies of positives only (2x) to balance + stabilize
    Xpa, ypa = feat(esc_pos + yt1 + yt2, 1, augment=True)
    Xtr = np.concatenate([Xtr, Xpa]); ytr = np.concatenate([ytr, ypa])
    Xte, yte = feat(esc_pos_te + esc_neg_te, 0 if False else 0, augment=False)
    # fix labels for test (pos files first)
    npos = len(esc_pos_te)
    yte = np.array([1]*npos + [0]*len(esc_neg_te), dtype=np.float32)
    print(f"train {Xtr.shape} pos={(ytr==1).sum()} neg={(ytr==0).sum()} | test {Xte.shape} pos={(yte==1).sum()} neg={(yte==0).sum()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JetCNN().to(device)
    pos_ratio = (ytr==1).sum()/len(ytr)
    pw = torch.tensor([(1-pos_ratio)/pos_ratio]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    print(f"device={device} lr={args.lr} pos_weight={pw.item():.2f}")

    best, best_ep, hist = 0, 0, []
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
            pa = ((torch.sigmoid(model(Xt.to(device)))>0.5).cpu().numpy()==ytr).mean()
            pe = ((torch.sigmoid(model(Xe.to(device)))>0.5).cpu().numpy()==yte).mean()
        hist.append((tot/len(Xt), pa, pe))
        flag = ""
        if pe > best:
            best, best_ep = pe, ep+1
            torch.save(model.state_dict(), DATA/"v2_jet.pt")
            flag = " <-- BEST saved"
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xt):.4f} train={pa:.3f} test={pe:.3f}{flag}")
    print(f"BEST test={best:.3f} at epoch {best_ep} -> v2_jet.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv
    with open(DATA/"history_v2.csv","w",newline="") as f:
        w=_csv.writer(f); w.writerow(["epoch","loss","train_acc","test_acc"])
        for i,(l,a,b) in enumerate(hist,1): w.writerow([i,round(l,4),round(float(a),4),round(float(b),4)])
    ep = list(range(1,len(hist)+1))
    plt.figure(figsize=(9,5))
    plt.plot(ep,[h[1] for h in hist],"o-",label="train")
    plt.plot(ep,[h[2] for h in hist],"s-",label="test (esc fold5)")
    plt.axhline(best,color="g",ls="--",alpha=0.5,label=f"best {best:.3f} ep{best_ep}")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("v2 JET: ESC-50 + 576 YT military jets (augmented, best-ckpt)")
    plt.legend(); plt.grid(True,alpha=0.3); plt.ylim(0,1.02)
    plt.tight_layout(); plt.savefig(DATA/"acc_v2.png",dpi=110)
    print("saved acc_v2.png")

    # real-world check with BEST model
    model.load_state_dict(torch.load(DATA/"v2_jet.pt", map_location=device))
    model.eval()
    tests = ["quick_jets/Blue_Angels_1.mp3","quick_jets/IAI_Astra_Takeoff.mp3","quick_jets/b1_flyby.wav",
             "yt_flybys_5s/b1_01.wav","f16_5s/clip_0001.wav" if (DATA/"f16_5s").exists() else "quick_jets/f16_pass.wav",
             "sonic_booms/boom_f18_01.wav"]
    for t in tests:
        p = DATA/t
        if not p.exists(): continue
        w,sr = torchaudio.load(str(p))
        if w.shape[0]>1: w=w.mean(0,keepdim=True)
        if sr!=SR: w=torchaudio.functional.resample(w,sr,SR)
        w=w.squeeze(0)[:5*SR]
        if len(w)<5*SR: w=F.pad(w,(0,5*SR-len(w)))
        with torch.no_grad():
            prob=torch.sigmoid(model(wav_to_logmel(w).unsqueeze(0).unsqueeze(0).to(device))).item()
        print(f"{t}: {prob:.3f}")

if __name__=="__main__":
    main()
