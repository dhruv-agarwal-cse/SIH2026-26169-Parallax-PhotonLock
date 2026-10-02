"""Benchmark suite: motions x disturbance conditions, ablation, acquisition trials, plots."""
import json
import os
from multiprocessing import Pool

import numpy as np

from fsoctrack.runner import run_sim
from fsoctrack.sim import MOTIONS, Config
from fsoctrack.vision import load_model

ALL = ["gaussian", "saltpepper", "poisson"]
CONDS = {
    "Clean": dict(noise=["gaussian"], gauss_sigma=3, jitter_px=0, platform_mode="none"),
    "Gaussian s=20": dict(noise=["gaussian"], gauss_sigma=20),
    "Salt&Pepper 10%": dict(noise=["saltpepper"]),
    "Poisson": dict(noise=["poisson"]),
    "Haze": dict(weather="haze"),
    "Rain": dict(weather="rain"),
    "Low light": dict(weather="lowlight"),
    "Fog + all noise": dict(weather="fog", noise=ALL),
    "STRESS": dict(weather="fog", noise=ALL, gauss_sigma=20, jitter_px=20, platform_mode="random",
                   platform_px=20, pan_speed_dps=10, tilt_speed_dps=10),
}
KEYS = ["acquisition_time_s", "detection_time_s", "pointing_within_10px_pct", "lock_retention_pct", "centroid_err_max_px", "pointing_err_mean_px", "pointing_err_rmse_px", "centroid_err_rmse_px",
        "target_loss_pct", "reacquisition_max_s", "processing_fps"]


def job(args):
    motion, cname, mode, seed, extra = args
    kw = dict(CONDS.get(cname, {}))
    kw.update(extra)
    cfg = Config(motion=motion, duration_s=kw.pop("duration_s", 12), occlusion_every_s=4, occlusion_len_s=1.0, seed=seed, **kw)
    s = run_sim(cfg, mode=mode, model=load_model() if mode == "ai" else None).summary()
    return dict(motion=motion, cond=cname, mode=mode, seed=seed, **{k: s.get(k) for k in KEYS})


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    jobs = [(m, c, "ai", 1, {}) for m in MOTIONS for c in CONDS]
    abl = [(m, c, md, 2, x) for m in ["figure8", "random"] for c, x in
           [("Salt&Pepper 10%", {}), ("Fog + all noise", {}), ("Rain + decoys", dict(weather="rain", n_decoys=4))]
           for md in ["naive", "classical", "ai"]]
    acq = [("circle", "Clean", "ai", s, dict(cued=cu, duration_s=5, jitter_px=5, platform_mode="linear"))
           for s in range(20) for cu in (True, False)]
    with Pool(2) as p:
        R = p.map(job, jobs)
        A = p.map(job, abl)
        Q = p.map(job, acq)
    for q, a in zip(Q, acq):
        q["cued"] = a[4]["cued"]
    json.dump({"matrix": R, "ablation": A, "acquisition": Q}, open("results/benchmark.json", "w"), indent=1, default=float)
    print("done", len(R), len(A), len(Q))
