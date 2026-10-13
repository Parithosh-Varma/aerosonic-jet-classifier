# Aerosonic Jet Classifier — Jet Engine Low-Fly Detection

Binary JET vs NO-JET classifier: 16 kHz mono → 5 s windows → log-mel (64 bands) → JetCNN (286k params) → sigmoid.
Training: `train_esc50.py` (baseline) → `train_v1.py` → `train_v2.py` → `train_v3.py` (unified, best-ckpt on AP).
Benchmark: `bench.py` → `benchmarks.csv` + `bench.png`. Report: `make_pdf.py` → `benchmarks.pdf`.

## Results

| model | ESC acc | prec | rec | F1 | BlueAngels | B-1 pass | boom |
|---|---|---|---|---|---|---|---|
| synth | 0.909 | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 |
| esc50 | 0.648 | 0.21 | 1.00 | 0.340 | 0.96 | 1.00 | 0.00 |
| v1 | 0.886 | 0.40 | 0.50 | 0.444 | 0.39 | 0.98 | 0.00 |
| v2 | 0.932 | 1.00 | 0.25 | 0.400 | 0.07 | 0.98 | 0.00 |

v2 leads on ESC-50 but crowd+music airshow audio regressed. All models score ~0 on sonic booms — needs a separate transient head. Full report: [benchmarks.pdf](benchmarks.pdf).

### Benchmark chart
![benchmarks](bench.png)

### Learning curves
![esc50 baseline](acc_vs_time.png)
![v1](acc_v1.png)
![v2](acc_v2.png)

### Sample detection
![sample jet_0261](sample_jet_0261.png)

## Capture & Review Pipeline

Folders (all in this directory):
- `pending_review/` — recorder drops loud clips here; dashboard reads from here
- `training_dataset/jet/` — confirmed jet clips (your training positives)
- `training_dataset/no_jet/` — rejected clips (training negatives)
- `review_log.csv` — every Yes/No/Delete decision with timestamp
- `events.csv` — recorder loud-event log

## 1. Dashboard (use now)
```bash
cd "/Users/varma/Downloads/AI model/Data"
streamlit run dashboard.py
```
- Pick a clip in **Review**, press play, check the waveform.
- **Yes — jet engine** (or type `yes` in the box) → moves file to `training_dataset/jet/`.
- **No** (or type `no`) → moves to `training_dataset/no_jet/`.
- Safe test file already waiting: `pending_review/selftest_*.wav` (synth tone, no mic involved).

## 2. Recorder (DO NOT mic-test until the OG model is established)
```bash
python3 recorder.py --self-test        # safe now: synth tone, no mic
python3 recorder.py --once             # LIVE MIC: one 10s clip (only after OG model)
python3 recorder.py                    # LIVE MIC: continuous loop (only after OG model)
python3 recorder.py --threshold -30 --clip 10 --sr 16000
```
- macOS mic via ffmpeg/avfoundation, 16 kHz mono to match planned preprocessing.
- Only clips louder than `threshold_dbfs` (see `config.json`) are kept.
- Tune `threshold_dbfs` / `clip_seconds` in `config.json`.

## Notes
- `audio.zip` is still downloading (size growing, central directory incomplete) — left untouched.
- No extra pip packages needed: stdlib + ffmpeg for recorder; streamlit/matplotlib for dashboard (already installed).
