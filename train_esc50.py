"""Train JET vs NO-JET on ESC-50 (fast prototype while AeroSonicDB downloads).
JET = airplane (40 clips). NO-JET = hard negatives: helicopter/engine/train/chainsaw/wind/thunderstorm + random.
Split: fold 5 = test, folds 1-4 = train (ESC-50 standard).
Usage: python3 train_esc50.py [--epochs 15]
"""
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchaudio
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR, N_MELS

DATA = Path(__file__).parent
ESC = DATA / "esc50_tmp"
HARD_NEGS = {"helicopter","engine","train","chainsaw","wind","thunderstorm","siren","car_horn","fireworks"}

def load_5s(path):
    w, sr = torchaudio.load(path)
    if w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
    w = w.squeeze(0)
    n = SR * 5
    if len(w) < n:
        w = torch.nn.functional.pad(w, (0, n - len(w)))
    else:
        w = w[:n]
    return wav_to_logmel(w).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    with open(ESC/"meta"/"esc50.csv") as f:
        rows = list(csv.DictReader(f))
    pos = [r for r in rows if r["category"]=="airplane"]
    neg_pool = [r for r in rows if r["category"] in HARD_NEGS]
    rest = [r for r in rows if r["category"] not in HARD_NEGS and r["category"]!="airplane"]
    # balanced-ish: all 40 pos + 80 hard negs + 40 random = 1:3, handled by pos_weight
    random.seed(0)
    neg_extra = random.sample(rest, 40)
    print(f"pos airplane={len(pos)}, hard negs={len(neg_pool)}, random negs={len(neg_extra)}")

    def build(subset, folds):
        X, y = [], []
        for r in subset:
            if int(r["fold"]) not in folds: continue
            try:
                X.append(load_5s(ESC/"audio"/r["filename"]))
                y.append(1 if r["category"]=="airplane" else 0)
            except Exception as e:
                print("skip", r["filename"], e)
        return np.stack(X)[:,None,:,:] if X else np.zeros((0,1,N_MELS,313)), np.array(y, dtype=np.float32)

    all_pos_neg = pos + neg_pool + neg_extra
    Xtr, ytr = build(all_pos_neg, [1,2,3,4])
    Xte, yte = build(all_pos_neg, [5])
    print(f"train {Xtr.shape} pos={(ytr==1).sum()} neg={(ytr==0).sum()}")
    print(f"test  {Xte.shape} pos={(yte==1).sum()} neg={(yte==0).sum()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JetCNN().to(device)
    pos_ratio = (ytr==1).sum()/max(len(ytr),1)
    pw = torch.tensor([(1-pos_ratio)/max(pos_ratio,1e-6)]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    print(f"device={device} pos_weight={pw.item():.2f}")
    hist = []
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt)); tot=0
        for i in range(0,len(perm),args.batch):
            b=perm[i:i+args.batch]
            opt.zero_grad()
            loss=fn(model(Xt[b].to(device)), yt[b].to(device))
            loss.backward(); opt.step(); tot+=loss.item()*len(b)
        model.eval()
        with torch.no_grad():
            pa=((torch.sigmoid(model(Xt.to(device)))>0.5).cpu().numpy()==ytr).mean()
            pe=((torch.sigmoid(model(Xe.to(device)))>0.5).cpu().numpy()==yte).mean()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xt):.4f} train_acc={pa:.3f} test_acc={pe:.3f}")
        hist.append((tot/len(Xt), pa, pe))

    torch.save(model.state_dict(), DATA/"esc50_jet.pt")
    print("saved:", DATA/"esc50_jet.pt")

    # accuracy vs epoch plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv
    with open(DATA/"history_esc50.csv","w",newline="") as f:
        w=_csv.writer(f); w.writerow(["epoch","loss","train_acc","test_acc"])
        for i,(l,a,b) in enumerate(hist,1): w.writerow([i,round(l,4),round(float(a),4),round(float(b),4)])
    ep=list(range(1,len(hist)+1))
    plt.figure(figsize=(9,5))
    plt.plot(ep,[h[1] for h in hist],"o-",label="train acc")
    plt.plot(ep,[h[2] for h in hist],"s-",label="test acc (fold 5)")
    plt.xlabel("epoch (time)"); plt.ylabel("accuracy"); plt.title("ESC-50 JET vs NO-JET: accuracy vs time")
    plt.legend(); plt.grid(True,alpha=0.3); plt.ylim(0,1.02)
    plt.tight_layout(); plt.savefig(DATA/"acc_vs_time.png",dpi=110)
    print("saved:", DATA/"acc_vs_time.png")

    # quick check on real fighter jets
    model.eval()
    import torch.nn.functional as F
    for f in ["quick_jets/Blue_Angels_1.mp3","quick_jets/IAI_Astra_Takeoff.mp3"]:
        p = DATA/f
        if not p.exists(): continue
        w,sr = torchaudio.load(str(p))
        if w.shape[0]>1: w=w.mean(0,keepdim=True)
        if sr!=SR: w=torchaudio.functional.resample(w,sr,SR)
        w=w.squeeze(0)[:5*SR]
        if len(w)<5*SR: w=F.pad(w,(0,5*SR-len(w)))
        with torch.no_grad():
            prob=torch.sigmoid(model(wav_to_logmel(w).unsqueeze(0).unsqueeze(0).to(device))).item()
        print(f"{f}: jet_prob={prob:.3f}")

if __name__=="__main__":
    main()
