"""STEP 2 - Train the AI beacon verifier (run once, ~2-5 min on a laptop).

What happens:
  1. DATA   : the simulator generates many random scenes across every PS disturbance
              (Gaussian / salt & pepper / Poisson noise, haze / fog / rain / low light, jitter,
              turbulence, decoys, 6 target sizes). The vision front-end extracts every bright blob.
              Each blob is labelled automatically: 'beacon' if it lies within 3 px of the true
              beacon (the simulator knows the truth), otherwise 'clutter'. No hand labelling.
  2. TRAIN  : a small neural network (MLP 32-16) learns beacon vs clutter from 10 blob features
              (size, fill, aspect, peak/mean/flux SNR ...).
  3. TEST   : accuracy / precision / recall on 25 % held-out blobs never seen in training.
  4. VALIDATE: closed-loop tracking on hard unseen scenarios, Classical vs AI.
Output: models/beacon_verifier.joblib + models/training_report.{json,txt}

Usage:  python train.py            (default 150 scenes)
        python train.py --scenes 400   (bigger, better, slower)
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fsoctrack.runner import run_sim
from fsoctrack.sim import MOTIONS, WEATHERS, Config, Scene, TargetMotion, jitter_sample
from fsoctrack.vision import FEATURE_NAMES, Detector

ROOT = os.path.dirname(os.path.abspath(__file__))


def bar(i, n, t0, label):
    f = (i + 1) / n
    eta = (time.time() - t0) / f * (1 - f)
    sys.stdout.write(f"\r  {label} [{'#' * int(30 * f):30s}] {100 * f:5.1f}%  ETA {eta:4.0f}s")
    sys.stdout.flush()


def make_data(n_scenes, frames, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    t0 = time.time()
    for ep in range(n_scenes):
        size = int(rng.choice([5, 8, 10, 12, 15, 20]))
        noise = [n for n in ["gaussian", "saltpepper", "poisson"] if rng.random() < 0.6] or ["gaussian"]
        cfg = Config(motion=str(rng.choice(MOTIONS)), weather=str(rng.choice(WEATHERS)), noise=noise,
                     gauss_sigma=float(rng.uniform(3, 20)), sp_fraction=float(rng.uniform(0.02, 0.15)),
                     target_size=size, n_decoys=int(rng.integers(0, 5)), turbulence=float(rng.uniform(0, 0.5)),
                     jitter_px=float(rng.uniform(0, 20)), seed=int(rng.integers(1e9)))
        r2 = np.random.default_rng(cfg.seed)
        sc, mo = Scene(cfg, r2), TargetMotion(cfg, r2)
        det = Detector(size, mode="classical")
        for k in range(frames):
            tgt = mo.step()
            cam = tgt + r2.normal(0, 120, 2)
            vis = r2.random() > 0.15
            fr, gt = sc.render(cam, jitter_sample(cfg, r2), tgt, visible=vis)
            cands, _ = det.candidates(fr)
            for c in cands:
                X.append(c["feat"])
                y.append(int(vis and np.hypot(c["x"] - gt[0], c["y"] - gt[1]) < 3.0))
        bar(ep, n_scenes, t0, "generating scenes")
    print()
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    T0 = time.time()
    print("\n[1/4] DATA: simulating random scenes and auto-labelling blobs")
    X, y = make_data(a.scenes, a.frames, a.seed)
    print(f"  {len(y):,} blobs  |  beacon: {y.sum():,}  clutter: {(1 - y).sum():,}")

    print("\n[2/4] TRAIN: neural network (MLP 32-16, early stopping)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    mlp = MLPClassifier((32, 16), max_iter=800, random_state=0, early_stopping=True, n_iter_no_change=20)
    clf = make_pipeline(StandardScaler(), mlp)
    t1 = time.time()
    clf.fit(Xtr, ytr)
    lc = mlp.loss_curve_
    for i in range(0, len(lc), max(1, len(lc) // 8)):
        print(f"  epoch {i + 1:4d}  training loss {lc[i]:.4f}")
    print(f"  epoch {len(lc):4d}  training loss {lc[-1]:.4f}   ({time.time() - t1:.1f}s)")

    print("\n[3/4] TEST on held-out blobs")
    yp = clf.predict(Xte)
    acc = float((yp == yte).mean())
    p, r, f1, _ = precision_recall_fscore_support(yte, yp, average="binary")
    cm = confusion_matrix(yte, yp)
    print(f"  accuracy {100 * acc:.2f}%   beacon precision {100 * p:.2f}%   recall {100 * r:.2f}%   F1 {f1:.3f}")
    print(f"  confusion matrix [[clutter->clutter, clutter->beacon],[beacon->clutter, beacon->beacon]] = {cm.tolist()}")
    joblib.dump(clf, os.path.join(ROOT, "models", "beacon_verifier.joblib"))

    print("\n[4/4] VALIDATE: closed-loop tracking on unseen hard scenarios (10 s each)")
    scen = [("figure8 + S&P 10% + Poisson", dict(motion="figure8", noise=["gaussian", "saltpepper", "poisson"])),
            ("random + rain + 4 decoys", dict(motion="random", weather="rain", n_decoys=4)),
            ("circle + fog + all noise", dict(motion="circle", weather="fog", noise=["gaussian", "saltpepper", "poisson"]))]
    val = []
    for name, kw in scen:
        row = {"scenario": name}
        for mode in ["classical", "ai"]:
            s = run_sim(Config(duration_s=10, occlusion_every_s=4, seed=777, **kw), mode=mode, model=clf).summary()
            row[mode] = {"loss_pct": s.get("target_loss_pct"), "track_err_px": s.get("pointing_err_mean_px"),
                         "centroid_rmse_px": s.get("centroid_err_rmse_px")}
        val.append(row)
        print(f"  {name:32s} target loss  classical {row['classical']['loss_pct']}%  ->  AI {row['ai']['loss_pct']}%")

    rep = {"samples": int(len(y)), "positives": int(y.sum()), "scenes": a.scenes, "features": FEATURE_NAMES,
           "accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1), "confusion": cm.tolist(),
           "epochs": len(lc), "final_loss": float(lc[-1]), "validation": val, "train_time_s": round(time.time() - T0, 1)}
    json.dump(rep, open(os.path.join(ROOT, "models", "training_report.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "models", "training_report.txt"), "w") as f:
        f.write(json.dumps(rep, indent=1))
    print(f"\nDONE in {rep['train_time_s']} s. Model saved to models/beacon_verifier.joblib")
    print("Next:  python gui.py\n")


if __name__ == "__main__":
    main()
