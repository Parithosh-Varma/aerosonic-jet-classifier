"""Test reliability of jet_cnn models.
- Synthetic test now (optimistic, same fake distribution as training)
- Real test automatically when audio/*.wav exists (held-out test split)
Reports: acc, precision, recall, F1, confusion matrix.
Usage: python3 test_jet.py [--model jet_cnn_synth.pt]
"""
import argparse, csv, sys
from pathlib import Path
import torch, torchaudio
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, find_wav, load_windows, WIN_SEC, SR, N_MELS

DATA = Path(__file__).parent

def metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = ((y_pred==1)&(y_true==1)).sum(); tn = ((y_pred==0)&(y_true==0)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum(); fn = ((y_pred==0)&(y_true==1)).sum()
    acc = (tp+tn)/max(len(y_true),1)
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    return acc, prec, rec, f1, (tp,tn,fp,fn)

def synth_test(model, device, n=100):
    import random
    Xs, ys = [], []
    for _ in range(n):
        t = torch.arange(WIN_SEC*SR).float()/SR
        # JET
        f0 = random.uniform(70,130)
        w = torch.sin(2*np.pi*f0*t)*0.6 + torch.sin(2*np.pi*f0*2*t)*0.3 + torch.randn_like(t)*0.4
        Xs.append(wav_to_logmel(w).numpy()); ys.append(1)
        # NO-JET
        w = torch.randn_like(t)*0.08
        Xs.append(wav_to_logmel(w).numpy()); ys.append(0)
    X = torch.tensor(np.stack(Xs)[:,None,:,:], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = (torch.sigmoid(model(X))>0.5).cpu().numpy()
    return metrics(ys, pred)

def real_test(model, device, split="test", max_per_class=200):
    import csv
    rows_jet, rows_noj = [], []
    with open(DATA/"sample_meta.csv") as f:
        for r in csv.DictReader(f):
            if r["train-test"] != split: continue
            if r["subclass"]=="3": rows_jet.append(r)
            elif r["subclass"]=="0": rows_noj.append(r)
    Xs, ys = [], []
    missing = 0
    for rows, label in [(rows_jet,1),(rows_noj,0)]:
        c = 0
        for r in rows:
            p = find_wav(r["filename"])
            if p is None: missing+=1; continue
            try:
                for w in load_windows(p)[:2]:  # max 2 windows/file for speed
                    Xs.append(wav_to_logmel(w).numpy()); ys.append(label); c+=1
                    if c>=max_per_class: break
            except Exception as e:
                print(f"skip {r['filename']}: {e}")
            if c>=max_per_class: break
    if not Xs:
        return None, missing
    X = torch.tensor(np.stack(Xs)[:,None,:,:], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = (torch.sigmoid(model(X))>0.5).cpu().numpy()
    return (metrics(ys,pred), len(ys)), missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jet_cnn_synth.pt")
    args = ap.parse_args()
    mp = DATA/args.model
    if not mp.exists():
        print(f"not found: {mp}"); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JetCNN().to(device)
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()
    print(f"loaded {mp} ({mp.stat().st_size/1024:.0f}KB)")

    acc, prec, rec, f1, (tp,tn,fp,fn) = synth_test(model, device)
    print(f"\n[SYNTHETIC test - optimistic, same fake data family]")
    print(f" acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} TP={tp} TN={tn} FP={fp} FN={fn}")
    print(" -> This only proves code runs. Says NOTHING about real jets.")

    res, missing = real_test(model, device)
    if res is None:
        print(f"\n[REAL test] no wavs yet (missing {missing} files, audio.zip still downloading).")
        print(" Re-run after unzip. Then expect synth model to FAIL on real data (acc ~50%).")
        print(" True reliability needs model trained on real audio: python3 train_jet.py")
    else:
        (acc,prec,rec,f1,cm), n = res
        tp,tn,fp,fn = cm
        print(f"\n[REAL test on held-out {split} n={n}]")
        print(f" acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} TP={tp} TN={tn} FP={fp} FN={fn}")
        print(" Reliability checklist: want prec+rec >0.85, test on env_audio continuous + your own mic.")

if __name__=="__main__":
    main()
