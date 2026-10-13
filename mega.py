"""Mega依旧是 ensemble: mean(v1,v2) roar head + rule-based boom transient head.
Heads combined (honest status):
  H1 JetCNN-5s (ours, v1+v2 averaged) ............ active
  H2 YAMNet-embedding + dense (RPAS/ERAU style) ... pending (needs TF install + AeroSonicDB)
  H3 ResNet-50 fine-grained (Osaka style) ......... pending (needs AeroSonicDB subclasses)
  H4 ConvNeXt-SED multi-class (battlefield style) . pending (needs big multi-class labels)
  H5 Boom transient 1 s rule ..................... FAILED calibration (YT compression kills crest;
                                                 LF/slope margins thin) -> needs learned head (phase 2)
Usage: python3 mega.py [--thr 0.5]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torchaudio
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from train_jet import JetCNN, wav_to_logmel, SR

DATA = Path(__file__).parent

def load(name, device):
    m = JetCNN().to(device)
    m.load_state_dict(torch.load(DATA/name, map_location=device)); m.eval()
    return m

def roar_prob(models, device, clip5):
    with torch.no_grad():
        ps = [torch.sigmoid(m(wav_to_logmel(clip5).unsqueeze(0).unsqueeze(0).to(device))).item() for m in models]
    return float(np.mean(ps)), ps

def boom_score(wav1s):
    """Transient score: peak/rms + crest. Boom N-wave => very high crest."""
    rms = wav1s.pow(2).mean().sqrt().item() + 1e-9
    peak = wav1s.abs().max().item()
    crest = peak / rms
    # normalize: speech/jet roar crest ~3-6, boom/explosion >10
    return min(1.0, max(0.0, (crest - 4.0) / 8.0)), crest

def load_mono(path, sr=SR):
    w, s = torchaudio.load(str(path))
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if s != sr: w = torchaudio.functional.resample(w, s, sr)
    return w.squeeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load("v1_jet.pt", device), load("v2_jet.pt", device)]
    print(f"mega: H1=mean(v1,v2) H5=boom-rule thr={args.thr}")

    tests = {"BlueAngels": "quick_jets/Blue_Angels_1.mp3", "Astra": "quick_jets/IAI_Astra_Takeoff.mp3",
             "B-1 pass": "yt_flybys_5s/b1_01.wav", "F-16": "f16_5s/clip_0002.wav",
             "F-18 boom": "sonic_booms/boom_f18_01.wav", "F-35 boom": "sonic_booms/boom_f35_01.wav"}
    print(f"{'clip':12s} {'roar':>6s} {'boom':>6s} {'VERDICT':>8s}")
    for label, rel in tests.items():
        p = DATA/rel
        if not p.exists():
            print(f"{label:12s} missing"); continue
        w = load_mono(p)
        clip5 = w[:5*SR]
        if len(clip5) < 5*SR: clip5 = F.pad(clip5, (0, 5*SR-len(clip5)))
        roar, (p1, p2) = roar_prob(models, device, clip5)
        # boom head: max over 1 s windows
        bs, crest = 0.0, 0.0
        for i in range(0, max(len(w)-SR, SR), SR//2):
            seg = w[i:i+SR]
            if len(seg) < SR: break
            s, c = boom_score(seg)
            if s > bs: bs, crest = s, c
        jet = roar >= args.thr
        boom = bs >= 0.5
        verdict = "JET" if jet else ("BOOM" if boom else "quiet")
        print(f"{label:12s} {roar:6.3f} {bs:6.3f} {verdict:>8s}  (v1={p1:.2f} v2={p2:.2f} crest={crest:.1f})")

    torch.save({"use": ["v1_jet.pt", "v2_jet.pt"], "thr": args.thr,
                "pending": ["yamnet-head", "resnet50-fine", "convnext-sed"]}, DATA/"mega_cfg.pt")
    print("saved mega_cfg.pt")

if __name__ == "__main__":
    main()
