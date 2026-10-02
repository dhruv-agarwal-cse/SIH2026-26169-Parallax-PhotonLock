"""Beacon detection and sub-pixel centroiding.

Pipeline (robust mode):
  1. Impulse-noise suppression  : 3x3 median (5x5 if impulse density > 15 %)
  2. Background / haze removal  : subtract local mean (box filter >> beacon size)
  3. Matched filtering          : Gaussian smoothing tuned to beacon size (max SNR)
  4. Adaptive threshold         : median + k * robust sigma (MAD), per frame
  5. Candidate extraction       : connected components (optionally inside a gate ROI)
  6. Sub-pixel centroid         : threshold-subtracted intensity-weighted centroid
  7. AI beacon verifier         : small neural network scores each candidate (beacon vs
                                  clutter / rain streak / decoy / residual noise)
Naive mode (baseline for ablation): global threshold at 70 % of frame max + largest blob.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

FEATURE_NAMES = ["area_n", "w_n", "h_n", "fill", "aspect", "peak_snr", "mean_snr",
                 "flux_snr", "log_peak", "ratio_peak_mean"]


class Detector:
    def __init__(self, target_size: int = 10, mode: str = "ai", model=None, k_sigma: float = 5.0):
        """mode: 'ai' (robust + NN verifier), 'classical' (robust + brightest), 'naive'."""
        self.s = float(target_size)
        self.mode = mode
        self.model = model
        self.k = k_sigma
        self.max_cands = 40

    # ------------------------------------------------------------------ candidates
    def candidates(self, frame: np.ndarray, roi=None):
        H, W = frame.shape
        x0, y0, x1, y1 = 0, 0, W, H
        if roi is not None:
            x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
            x1, y1 = min(W, int(roi[2])), min(H, int(roi[3]))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return [], 1.0
        sub = frame[y0:y1, x0:x1]
        if self.mode == "naive":
            return self._naive(sub, x0, y0)
        samp = sub[::4, ::4]
        imp = float(np.mean((samp == 0) | (samp == 255)))
        med = cv2.medianBlur(sub, 5 if imp > 0.15 else 3).astype(np.float32)
        bk = int(max(31, 4 * self.s + 1)) | 1
        resid = med - cv2.blur(med, (bk, bk))
        rs = cv2.GaussianBlur(resid, (0, 0), max(0.8, self.s / 4.0))
        smp = rs[::3, ::3]
        mv = float(np.median(smp))
        sigma = 1.4826 * float(np.median(np.abs(smp - mv))) + 1e-3
        T = mv + self.k * sigma
        mask = (rs > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return [], sigma
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[: self.max_cands] + 1
        cands = []
        hh, ww = rs.shape
        base = mv + 0.5 * self.k * sigma
        for i in order:
            x, y, w, h, area = stats[i]
            if area < 2:
                continue
            xa, ya = max(0, x - 3), max(0, y - 3)
            xb, yb = min(ww, x + w + 3), min(hh, y + h + 3)
            win = np.clip(rs[ya:yb, xa:xb] - base, 0, None)
            tot = float(win.sum())
            if tot <= 0:
                continue
            cx = float(win.sum(0) @ np.arange(xa, xb)) / tot
            cy = float(win.sum(1) @ np.arange(ya, yb)) / tot
            reg = rs[y:y + h, x:x + w]
            m = lab[y:y + h, x:x + w] == i
            vals = reg[m]
            peak, mean, flux = float(vals.max()), float(vals.mean()), float(vals.sum())
            s2 = self.s * self.s
            feat = [area / s2, w / self.s, h / self.s, area / float(w * h), max(w, h) / max(1, min(w, h)),
                    (peak - mv) / sigma, (mean - mv) / sigma, (flux - mv * area) / (sigma * s2),
                    np.log1p(max(peak - mv, 0)), peak / max(mean, 1e-3)]
            cands.append({"x": cx + x0, "y": cy + y0, "peak": peak, "area": int(area),
                          "flux": flux, "feat": np.array(feat, np.float32)})
        return cands, sigma

    def _naive(self, sub, x0, y0):
        f = sub.astype(np.float32)
        T = 0.7 * float(f.max())
        mask = (f > T).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            win = f[y:y + h, x:x + w] * (lab[y:y + h, x:x + w] == i)
            tot = float(win.sum())
            cx = float(win.sum(0) @ np.arange(x, x + w)) / tot
            cy = float(win.sum(1) @ np.arange(y, y + h)) / tot
            cands.append({"x": cx + x0, "y": cy + y0, "peak": float(win.max()), "area": int(area),
                          "flux": tot, "feat": None})
        return cands, 1.0

    # ------------------------------------------------------------------ scoring
    def score(self, cands):
        if not cands:
            return np.zeros(0)
        if self.mode == "ai" and self.model is not None:
            F = np.stack([c["feat"] for c in cands])
            return self.model.predict_proba(F)[:, 1]
        if self.mode == "naive":
            a = np.array([c["area"] for c in cands], float)
            return (a >= a.max()).astype(float)
        # classical: brightest candidate among those with beacon-like size and SNR > 8
        f = np.stack([c["feat"] for c in cands])
        ok = (f[:, 0] > 0.3) & (f[:, 0] < 4.0) & (f[:, 5] > 8.0) & (f[:, 4] < 2.5)
        p = np.where(ok, f[:, 5], -1.0)
        return ((p >= p.max()) & ok).astype(float)

    def select(self, cands, pred=None, gate_r=None, thr=0.5):
        """Pick the designated beacon. Returns (candidate or None, probability)."""
        if not cands:
            return None, 0.0
        p = self.score(cands)
        s = p.copy()
        if pred is not None and gate_r is not None and self.mode != "naive":
            d2 = np.array([(c["x"] - pred[0]) ** 2 + (c["y"] - pred[1]) ** 2 for c in cands])
            s = p * np.exp(-0.5 * d2 / (gate_r / 2.0) ** 2)
        i = int(np.argmax(s))
        if self.mode != "naive" and p[i] < thr:
            return None, float(p[i])
        return cands[i], float(p[i])


def load_model(path=None):
    import joblib
    path = path or os.path.join(os.path.dirname(__file__), "..", "models", "beacon_verifier.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None
