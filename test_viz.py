"""Tests for viz.py — style helpers, dimensions, edge cases, file output.

Run: python3 -m pytest test_viz.py -q   (or python3 test_viz.py)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import viz


class TestTheme(unittest.TestCase):
    def test_apply_theme_sets_publication_tokens(self):
        viz.apply_theme()
        import matplotlib.pyplot as plt
        self.assertEqual(plt.rcParams["axes.facecolor"], viz.PAPER)
        self.assertFalse(plt.rcParams["axes.spines.top"])
        self.assertFalse(plt.rcParams["axes.spines.right"])

    def test_class_colors_cover_binary_labels(self):
        for label in ("JET", "NO-JET", "BOOM"):
            self.assertIn(label, viz.CLASS_COLORS)

    def test_figure_dimensions(self):
        for kind, (w, h) in viz.SIZES.items():
            fig, _ = viz.fig(kind)
            self.assertAlmostEqual(fig.get_figwidth(), w)
            self.assertAlmostEqual(fig.get_figheight(), h)
            viz.save(fig, Path(tempfile.mkdtemp()) / kind, pdf=False)


class TestHelpers(unittest.TestCase):
    def test_save_creates_png_and_pdf(self):
        viz.apply_theme()
        with tempfile.TemporaryDirectory() as d:
            fig, ax = viz.fig("single")
            ax.plot([0, 1], [0, 1])
            written = viz.save(fig, Path(d) / "sub" / "plot")
            self.assertTrue((Path(d) / "sub" / "plot.png").exists())
            self.assertTrue((Path(d) / "sub" / "plot.pdf").exists())
            self.assertEqual(len(written), 2)

    def test_check_1d_rejects_empty_and_nan(self):
        with self.assertRaises(ValueError):
            viz.check_1d("x", [])
        with self.assertRaises(ValueError):
            viz.check_1d("x", [0.1, float("nan")])
        out = viz.check_1d("x", [0.1, 0.2])
        self.assertEqual(out.shape, (2,))

    def test_hz_tick_map_monotonic(self):
        pos, labels = viz.hz_tick_map(64, 8000.0)
        self.assertEqual(list(pos), sorted(pos))
        self.assertEqual(len(pos), len(labels))
        self.assertEqual(labels[-1], "8k")


class TestStats(unittest.TestCase):
    def test_confusion_stats_known_case(self):
        s = viz.confusion_stats([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
        self.assertEqual((s["tp"], s["tn"], s["fp"], s["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 0.5)

    def test_confusion_stats_single_class_no_crash(self):
        s = viz.confusion_stats([0, 0, 0], [0.1, 0.2, 0.3])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["precision"], 0.0)
        self.assertEqual(s["recall"], 0.0)

    def test_confusion_stats_empty_raises(self):
        with self.assertRaises(ValueError):
            viz.confusion_stats([], [])

    def test_roc_pr_single_class_returns_none(self):
        self.assertIsNone(viz.roc_points([0, 0], [0.1, 0.2]))
        self.assertIsNone(viz.pr_points([1, 1], [0.8, 0.9]))

    def test_roc_pr_perfect_separation(self):
        y = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(viz.roc_points(y, scores)["auc"], 1.0)
        self.assertAlmostEqual(viz.pr_points(y, scores)["ap"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
