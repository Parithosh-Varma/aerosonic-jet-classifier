"""Build benchmarks.pdf with reportlab Platypus (per pdf skill)."""
from pathlib import Path
from datetime import date
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, PageBreak)
from reportlab.lib.enums import TA_CENTER
from pypdf import PdfReader

DATA = Path(__file__).parent
PDF = DATA / "benchmarks.pdf"
TODAY = date.today().isoformat()
NAVY, ACCENT, LIGHT = colors.HexColor("#1b2a4a"), colors.HexColor("#2e86c1"), colors.HexColor("#eaf2f8")

styles = getSampleStyleSheet()
title_s = ParagraphStyle("Title2", parent=styles["Title"], textColor=NAVY, fontSize=20)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=NAVY, fontSize=14, spaceBefore=6, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=NAVY, fontSize=11, spaceBefore=8, spaceAfter=4)
body = styles["Normal"]
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#5d6d7e"))
center = ParagraphStyle("Center", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9)
mono = ParagraphStyle("Mono", parent=styles["Normal"], fontName="Courier", fontSize=9)

def header(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 18*mm, A4[0], 18*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(A4[0]/2, A4[1] - 11*mm, "JET ENGINE LOW-FLY DETECTION  •  BENCHMARK REPORT")
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(A4[0] - 12*mm, 10*mm, f"{TODAY}  •  p. {doc.page}")
    canvas.restoreState()

def img(path, w=170*mm):
    im = RLImage(str(path), width=w, height=w*0.55)
    im.hAlign = "CENTER"
    return im

story = []
story.append(Paragraph("Jet Engine Low-Fly Detection — Benchmarks", title_s))
story.append(Paragraph("Binary JET vs NO-JET  •  ESC-50 fold-5 (n=88, 8 jets) + quarantined field clips", center))
story.append(Spacer(1, 4*mm))
story.append(Paragraph("Model verdicts", h1))
for name, label, why, col in [
    ("synth", "useless", "predicts all-negative; accuracy lies", "#cb4335"),
    ("esc50", "biased", "predicts all-jet; no best-checkpoint", "#cb4335"),
    ("v1", "prototype", "balanced-ish, crowd slipping", "#b7950b"),
    ("v2", "best so far", "top ESC, crowd regressed", "#1e8449"),
]:
    story.append(Paragraph(
        f"<font face='Courier' size=10><b>{name}</b></font> &nbsp; "
        f"<font color='{col}'><b>● {label}</b></font> &nbsp; {why}", body))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "Headline: v2 leads on ESC-50 (0.932 acc, prec 1.0) and nails clean flybys (B-1 0.98), "
    "but crowd+music airshow audio is regressing (BlueAngels 0.96 → 0.07).", body))
story.append(Paragraph(
    "<i>All four models score 0.00 on sonic booms — a separate 1 s transient detector is required "
    "(32 boom clips banked).</i>", body))
story.append(PageBreak())

story.append(Paragraph("Metric table — F1 is the honest column (only 8 jets / 88 clips)", h1))
tdata = [["model", "ESC acc", "prec", "rec", "F1", "BlueAng", "B-1", "boom"],
         ["synth", "0.909", "0.00", "0.00", "0.000", "0.00", "0.00", "0.00"],
         ["esc50", "0.648", "0.21", "1.00", "0.340", "0.96", "1.00", "0.00"],
         ["v1", "0.886", "0.40", "0.50", "0.444", "0.39", "0.98", "0.00"],
         ["v2", "0.932", "1.00", "0.25", "0.400", "0.07", "0.98", "0.00"]]
t = Table(tdata, colWidths=[24*mm] + [20*mm]*7)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 3*mm))
story.append(Paragraph("green-tinted bests: v2 ESC acc 0.932 / prec 1.00, esc50 BlueAngels 0.96, esc50/v1/v2 B-1 ≥0.98. "
                       "Raw numbers: benchmarks.csv", small))
story.append(PageBreak())

story.append(Paragraph("Benchmark chart", h1))
story.append(img(DATA/"bench.png"))
story.append(Paragraph("ESC-50 accuracy vs real-world detection probs. Boom excluded (transient head pending).", small))
story.append(PageBreak())

story.append(Paragraph("Learning curves", h1))
for cap, name in [("baseline (ESC-50 only): jagged, tiny data", "acc_vs_time.png"),
                  ("v1 (+YT jets, augment, best-ckpt): stabilised", "acc_v1.png"),
                  ("v2 (+~864 YT jets): best 0.942 ep10", "acc_v2.png")]:
    story.append(Paragraph(cap, h2))
    story.append(img(DATA/name, w=160*mm))
    story.append(Spacer(1, 2*mm))
story.append(PageBreak())

story.append(Paragraph("Sample detection — jet_0261.wav", h1))
story.append(Paragraph("rms 0.30 (loud) • jet_prob 0.917 • broadband roar, bright low-mel band = classic flyby signature", center))
story.append(img(DATA/"sample_jet_0261.png"))
story.append(Paragraph("Listen: <font face='Courier'>afplay yt_flybys_5s/jet_0261.wav</font>", mono))
story.append(PageBreak())

story.append(Paragraph("Data inventory", h1))
inv = [["split", "what", "n"],
       ["JET train", "ESC-50 airliners (folds 1–4)", "32"],
       ["JET train", "YT compilations 5 s (filtered ≥0.5)", "~780"],
       ["JET train", "Abbotsford crowd-jets (thr 0.3, hard kept)", "46"],
       ["JET test-only", "B-1 passes, F-16 clips, BlueAngels, Astra", "8 files"],
       ["NO-JET train", "ESC-50 hard negs (heli/engine/train/storm…)", "~460"],
       ["NO-JET test", "ESC-50 fold-5 negs", "~80"],
       ["BOOM (pending)", "2 s transient clips (F-18/F-35/comps)", "32"],
       ["BIG (incoming)", "AeroSonicDB v1.1: 470 turbofan + 1270 bg", "downloading"]]
t2 = Table(inv, colWidths=[30*mm, 110*mm, 30*mm])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t2)
story.append(Spacer(1, 4*mm))
story.append(Paragraph(
    "Pipeline: 16 kHz mono → 5 s → log-mel(64) → JetCNN-286k → sigmoid. "
    "Next: AeroSonicDB retrain on arrival + 1 s boom head.", body))

doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                        topMargin=24*mm, bottomMargin=14*mm, leftMargin=12*mm, rightMargin=12*mm,
                        title="Jet Low-Fly Detection — Benchmarks", author="OpenCode")
doc.build(story, onFirstPage=header, onLaterPages=header)

r = PdfReader(str(PDF))
print(f"saved {PDF.name} {PDF.stat().st_size/1024:.0f}KB pages={len(r.pages)} meta_title={r.metadata.title!r}")
