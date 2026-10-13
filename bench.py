"""Benchmark all models on all available test data.
Models: jet_cnn_synth, esc50_jet, v1_jet, v2_jet
Tests:
  A) ESC-50 fold5 (8 jet + 95 neg) -> acc/prec/rec/F1
  B) Real-world clips (never trained): BlueAngels, Astra, B-1 flyby tracking, f16 clips, booms
Saves: benchmarks.csv (bench.png is rendered from it by make_plots_hq.py,
which owns the publication figure style via viz.py).
"""
import sys, csv, os
from pathlib import Path
import numpy as np, torch, torchaudio
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR

DATA = Path(__file__).parent
MODELS = ["jet_cnn_synth.pt", "esc50_jet.pt", "v1_jet.pt", "v2_jet.pt"]

def prob5(model, wav512, device):
    with torch.no_grad():
        return torch.sigmoid(model(wav512.to(device))).item()

def feat5(path):
    w, sr = torchaudio.load(str(path))
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
    w = w.squeeze(0)[:5*SR]
    if len(w) < 5*SR: w = F.pad(w, (0, 5*SR-len(w)))
    return wav_to_logmel(w).unsqueeze(0).unsqueeze(0)  # [1,1,64,T]

def load_model(name, device):
    m = JetCNN().to(device)
    m.load_state_dict(torch.load(DATA/name, map_location=device))
    m.eval()
    return m

def esc_fold5(model, device):
    import csv as _csv
    rows = list(_csv.DictReader(open(DATA/"esc50_tmp"/"meta"/"esc50.csv")))
    Xs, ys = [], []
    for r in rows:
        if int(r["fold"]) != 5: continue
        if r["category"] == "airplane": ys.append(1)
        elif r["category"] in {"helicopter","engine","train","chainsaw","wind","thunderstorm","siren","car_horn","fireworks","rain"}: ys.append(0)
        else: continue
        Xs.append(feat5(DATA/"esc50_tmp"/"audio"/r["filename"]))
    # add 15 random other fold5 as extra negs (same as train script)
    n0 = len(ys)
    X = torch.cat(Xs, dim=0).to(device)
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) > 0.5).cpu().numpy()
    y = np.array(ys)
    tp = ((pred==1)&(y==1)).sum(); tn = ((pred==0)&(y==0)).sum()
    fp = ((pred==1)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()
    acc = (tp+tn)/len(y); prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    return acc, prec, rec, f1, len(y)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    real_tests = {
        "BlueAngels F/A-18 (crowd)": ("quick_jets/Blue_Angels_1.mp3", 1),
        "Astra takeoff": ("quick_jets/IAI_Astra_Takeoff.mp3", 1),
        "B-1 jet pass (5-10s)": ("yt_flybys_5s/b1_01.wav", 1),
        "F-16 pass clip": ("f16_5s/clip_0002.wav", 1),
        "F-18 sonic boom": ("sonic_booms/boom_f18_01.wav", 1),
    }
    rows_out = []
    for name in MODELS:
        p = DATA/name
        if not p.exists():
            print(f"SKIP {name} (missing)"); continue
        m = load_model(name, device)
        acc, prec, rec, f1, n = esc_fold5(m, device)
        line = f"{name:18s} ESCfold5(n={n}): acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} | "
        det = {}
        for tlabel, (rel, expected) in real_tests.items():
            fp = DATA/rel
            if not fp.exists():
                det[tlabel] = "n/a"; continue
            pr = prob5(m, feat5(fp), device)
            det[tlabel] = pr
            line += f"{tlabel.split('(')[0].strip()}={pr:.2f} "
        print(line)
        rows_out.append((name, acc, prec, rec, f1, det))

    with open(DATA/"benchmarks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","esc_acc","esc_prec","esc_rec","esc_F1"] + list(real_tests.keys()))
        for name, acc, prec, rec, f1, det in rows_out:
            w.writerow([name, round(acc,3), round(prec,3), round(rec,3), round(f1,3)] +
                       [round(det.get(k, -1),3) if isinstance(det.get(k),float) else det.get(k) for k in real_tests])
    print("saved benchmarks.csv")
    print("note: run python3 make_plots_hq.py to render bench.png from benchmarks.csv")

if __name__ == "__main__":
    main()
