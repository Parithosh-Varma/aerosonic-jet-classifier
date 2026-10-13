"""Binary JET vs NO-JET detector for AeroSonicDB v1.1
JET = subclass 3 (turbofan), NO-JET = subclass 0 (silence/background)
Features: 16kHz mono -> 5s windows -> log-mel (64 mels) -> small CNN
Usage:
  python3 train_jet.py --dry-run   # verify pipeline without audio (works now)
  python3 train_jet.py             # train when audio/*.wav exists
"""
import argparse, csv, os, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

DATA = Path(__file__).parent
META = DATA / "sample_meta.csv"
AUDIO_DIRS = [DATA / "audio", DATA]  # unzip may flatten to DATA/

SR = 16000
WIN_SEC = 5
N_MELS = 64

def find_wav(filename):
    for d in AUDIO_DIRS:
        p = d / filename
        if p.exists():
            return p
    # also search one level deep (audio/train/... etc)
    for d in AUDIO_DIRS:
        if d.exists():
            for p in d.rglob(filename):
                return p
    return None

def load_windows(wav_path, win_sec=WIN_SEC, sr=SR):
    wav, orig_sr = torchaudio.load(wav_path)  # [ch, t]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    wav = wav.squeeze(0)
    win = win_sec * sr
    # center-crop or sliding windows
    if len(wav) < win:
        wav = torch.nn.functional.pad(wav, (0, win - len(wav)))
        return [wav]
    # sliding with 50% overlap for long jet flybys (avg 52s), single window for silence
    stride = win if len(wav) <= 12 * sr else win // 2
    return [wav[i:i+win] for i in range(0, len(wav)-win+1, stride)]

MEL = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=256, n_mels=N_MELS)
DB = torchaudio.transforms.AmplitudeToDB()

def wav_to_logmel(waveform):
    with torch.no_grad():
        m = MEL(waveform.unsqueeze(0) if waveform.dim()==1 else waveform)
        return DB(m).squeeze(0)  # [mels, time]

class JetCNN(nn.Module):
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*4*8, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        return self.fc(self.net(x)).squeeze(1)

def build_file_list():
    jets, nojs = [], []
    with open(META) as f:
        for row in csv.DictReader(f):
            if row["subclass"] == "3":
                jets.append(row)
            elif row["subclass"] == "0":
                nojs.append(row)
    return jets, nojs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no audio needed, verify splits + model forward")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=400, help="cap windows per class to limit imbalance")
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic jet-like vs silence data (no audio needed)")
    args = ap.parse_args()

    jets, nojs = build_file_list()
    print(f"JET files: {len(jets)} (397 train / 73 test expected)")
    print(f"NO-JET files: {len(nojs)} (953 train / 317 test expected)")
    print(f"train/test JET: {Counter([r['train-test'] for r in jets])}")
    print(f"train/test NOJ: {Counter([r['train-test'] for r in nojs])}")

    model = JetCNN()
    dummy = torch.randn(2, 1, N_MELS, 313)  # ~5s @16kHz, hop 256
    out = model(dummy)
    print(f"model OK: in {tuple(dummy.shape)} -> out {tuple(out.shape)}, params={sum(p.numel() for p in model.parameters())}")

    if args.dry_run:
        print("dry-run PASS. Run without --dry-run once audio/*.wav exists.")
        return

    if args.synthetic:
        print("synthetic mode: generating fake JET (low rumble+sweep) vs NO-JET (quiet noise)...")
        def synth(n, jet=True):
            X = []
            for _ in range(n):
                t = torch.arange(WIN_SEC*SR).float() / SR
                if jet:
                    # low-freq rumble 80-150Hz + broadband noise, louder
                    f0 = random.uniform(70, 130)
                    tone = torch.sin(2*np.pi*f0*t) * 0.6 + torch.sin(2*np.pi*f0*2*t) * 0.3
                    noise = torch.randn_like(t) * 0.4
                    w = tone + noise
                else:
                    w = torch.randn_like(t) * 0.08
                X.append(wav_to_logmel(w).numpy())
            return np.stack(X)
        Xj = synth(200, True); Xn = synth(200, False)
        Xtr = np.concatenate([Xj, Xn])[:, None, :, :]
        ytr = np.array([1]*200 + [0]*200, dtype=np.float32)
        Xte, yte = Xtr[:40], ytr[:40]
        print(f"synthetic: train {Xtr.shape}, test {Xte.shape}")
        # jump straight to training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = JetCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.float32)
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
            for i in range(0, len(perm), args.batch):
                b = perm[i:i+args.batch]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[b].to(device)), ytr_t[b].to(device))
                loss.backward(); opt.step(); tot += loss.item()*len(b)
            model.eval()
            with torch.no_grad():
                pred = (torch.sigmoid(model(Xtr_t.to(device))) > 0.5).cpu().numpy()
                print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xtr_t):.4f} acc={(pred==ytr).mean():.3f}")
        torch.save(model.state_dict(), DATA / "jet_cnn_synth.pt")
        print("saved:", DATA / "jet_cnn_synth.pt")
        return

    # --- gather features ---
    def collect(rows, label, split):
        X, y = [], []
        rows = [r for r in rows if r["train-test"] == split]
        random.shuffle(rows)
        for r in rows:
            p = find_wav(r["filename"])
            if p is None:
                continue
            try:
                for w in load_windows(p):
                    X.append(wav_to_logmel(w).numpy())
                    y.append(label)
                    if len(y) >= args.max_per_class:
                        break
            except Exception as e:
                print(f"skip {r['filename']}: {e}")
            if len(y) >= args.max_per_class:
                break
        return np.stack(X) if X else np.zeros((0, N_MELS, 313)), np.array(y)

    print("loading train windows...")
    Xj, yj = collect(jets, 1, "train")
    Xn, yn = collect(nojs, 0, "train")
    if len(yj) == 0 and len(yn) == 0:
        print("No wav files found yet (audio.zip still downloading). Re-run later or use --dry-run.")
        return
    Xtr = np.concatenate([Xj, Xn])[:, None, :, :]
    ytr = np.concatenate([yj, yn]).astype(np.float32)
    print(f"train windows: JET={len(yj)} NO-JET={len(yn)}")

    print("loading test windows...")
    Xj, yj = collect(jets, 1, "test")
    Xn, yn = collect(nojs, 0, "test")
    Xte = np.concatenate([Xj, Xn])[:, None, :, :] if len(yj)+len(yn) else Xtr[:0]
    yte = np.concatenate([yj, yn]).astype(np.float32) if len(yj)+len(yn) else ytr[:0]
    print(f"test windows: JET={len(yj)} NO-JET={len(yn)}")

    # --- train ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JetCNN().to(device)
    # class weight for imbalance
    pos = (ytr == 1).sum() / max(len(ytr), 1)
    pos_weight = torch.tensor([(1-pos) / max(pos, 1e-6)]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"device={device} pos_weight={pos_weight.item():.2f}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i+args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b].to(device)), ytr_t[b].to(device))
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        # train acc
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(Xtr_t.to(device))) > 0.5).cpu().numpy()
            acc = (pred == ytr).mean()
            auc_msg = ""
            if len(yte):
                pred_t = (torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32).to(device))) > 0.5).cpu().numpy()
                acc_t = (pred_t == yte).mean()
                auc_msg = f" test_acc={acc_t:.3f}"
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xtr_t):.4f} train_acc={acc:.3f}{auc_msg}")

    torch.save(model.state_dict(), DATA / "jet_cnn.pt")
    print("saved:", DATA / "jet_cnn.pt")

if __name__ == "__main__":
    main()
