"""Per-frame logging and automatic performance report (PS 'Performance Log' deliverable)."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


class Recorder:
    def __init__(self, cfg, label="run"):
        self.cfg, self.label = cfg, label
        self.rows = []
        self.t_start = time.perf_counter()

    def add(self, **kw):
        self.rows.append(kw)

    def summary(self):
        cfg, R = self.cfg, self.rows
        fps = cfg.fps
        n = len(R)
        st = np.array([r["state"] for r in R])
        occl = np.array([r["occluded"] for r in R], bool)
        perr = np.array([r["point_err"] for r in R], float)
        cerr = np.array([r["cent_err"] if r["cent_err"] is not None else np.nan for r in R], float)
        proc = np.array([r["proc_ms"] for r in R], float)
        tele = np.array([bool(r.get("teleport", False)) for r in R], bool)
        locked = (st == "TRACK") & (np.nan_to_num(cerr, nan=1e9) < 5.0)   # TRACK on the real beacon
        # acquisition = first frame in TRACK with boresight within 10 px of the beacon (lock-on)
        trk = np.where((st == "TRACK") & (perr <= 10.0))[0]
        acq_idx = int(trk[0]) if len(trk) else None
        det = np.where(st == "TRACK")[0]
        out_detect = round((int(det[0]) + 1) / fps, 3) if len(det) else None
        out = {"label": self.label, "frames": n, "sim_duration_s": round(n / fps, 2)}
        out["acquisition_time_s"] = round((acq_idx + 1) / fps, 3) if acq_idx is not None else None
        out["detection_time_s"] = out_detect
        # Tracking metrics are scored from acquisition onwards. A run that never acquires (e.g. the
        # detector latched onto a decoy) is scored over every frame, so its loss shows as 100 %.
        if n > 0:
            post = np.arange(acq_idx if acq_idx is not None else 0, n)
            vis = post[~occl[post]]
            lk = locked[vis]
            pe = perr[vis]
            ce = cerr[vis][st[vis] == "TRACK"]
            ce = ce[~np.isnan(ce)]
            out.update({
                "pointing_err_mean_px": round(float(pe.mean()), 2) if len(pe) else None,
                "pointing_err_rmse_px": round(float(np.sqrt((pe ** 2).mean())), 2) if len(pe) else None,
                "pointing_err_max_px": round(float(pe.max()), 2) if len(pe) else None,
                "pointing_within_10px_pct": round(100.0 * float((pe <= 10).mean()), 2) if len(pe) else None,
                "centroid_err_mean_px": round(float(ce.mean()), 3) if len(ce) else None,
                "centroid_err_rmse_px": round(float(np.sqrt((ce ** 2).mean())), 3) if len(ce) else None,
                "centroid_err_max_px": round(float(ce.max()), 3) if len(ce) else None,
                "lock_retention_pct": round(100.0 * float(lk.mean()), 2) if len(vis) else None,
                "target_loss_pct": round(100.0 - 100.0 * float(lk.mean()), 2) if len(vis) else None,
            })
        # Re-acquisition: one event per loss of lock after acquisition. The clock starts when the beacon
        # is visible again (end of an occlusion), at a teleport, or when lock has been lost for 0.2 s,
        # and stops at the next frame locked on the real beacon. Occluded frames do not count.
        reacq, pending, events = [], None, 0
        if acq_idx is not None:
            in_loss, dur, forced = False, 0, False
            for k in range(acq_idx + 1, n):
                reappear = occl[k - 1] and not occl[k]
                if not in_loss and (tele[k] or reappear or (not locked[k] and not occl[k])):
                    in_loss, dur, forced = True, 0, bool(tele[k] or reappear)
                if in_loss:
                    if locked[k]:
                        if forced or dur >= int(0.2 * fps):
                            reacq.append(round((dur + 1) / fps, 3))   # +1: the re-lock frame itself
                        in_loss = False
                    elif not occl[k]:
                        dur += 1
            if in_loss and (forced or dur >= int(0.2 * fps)):
                pending = round((dur + 1) / fps, 3)
        out["reacquisition_times_s"] = reacq
        out["reacquisition_events"] = len(reacq) + (1 if pending is not None else 0)
        out["reacquisition_max_s"] = max(reacq) if reacq else None
        out["reacquisition_pending_s"] = pending          # current loss still unrecovered, else None
        out["proc_time_mean_ms"] = round(float(proc.mean()), 2)
        out["proc_time_p99_ms"] = round(float(np.percentile(proc, 99)), 2)
        out["processing_fps"] = round(1000.0 / max(proc.mean(), 1e-6), 1)
        out["wall_time_s"] = round(time.perf_counter() - self.t_start, 2)
        return out

    def write(self, out_dir, prefix=None):
        os.makedirs(out_dir, exist_ok=True)
        prefix = prefix or self.label
        s = self.summary()
        s["config"] = self.cfg.to_dict()
        with open(os.path.join(out_dir, f"{prefix}_performance.json"), "w") as f:
            json.dump(s, f, indent=2)
        with open(os.path.join(out_dir, f"{prefix}_frames.csv"), "w", newline="") as f:
            keys = ["k", "t", "state", "occluded", "teleport", "gt_x", "gt_y", "meas_x", "meas_y", "cent_err", "point_err", "proc_ms", "prob"]
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            for r in self.rows:
                wr.writerow(r)
        with open(os.path.join(out_dir, f"{prefix}_performance.txt"), "w") as f:
            f.write(format_report(s))
        return s


SPEC = [  # key, label, PS target, pass test
    ("acquisition_time_s", "Acquisition time (s)", "<= 2", lambda v: v is not None and v <= 2),
    ("pointing_err_mean_px", "Tracking error mean (px)", "<= 10", lambda v: v is not None and v <= 10),
    ("centroid_err_rmse_px", "Centroiding error RMSE (px)", "(minimise)", lambda v: v is not None),
    ("target_loss_pct", "Target loss (%)", "< 5", lambda v: v is not None and v < 5),
    ("reacquisition_max_s", "Re-acquisition time (s)", "<= 1", lambda v: v is None or v <= 1),   # None = no loss event yet
    ("processing_fps", "Processing speed (FPS)", ">= 20", lambda v: v >= 20),
]


def format_report(s):
    lines = ["FSOC COARSE-POINTING PERFORMANCE REPORT", "=" * 44,
             f"Run: {s['label']}   frames: {s['frames']}   simulated: {s['sim_duration_s']} s", ""]
    for k, lab, tgt, ok in SPEC:
        v = s.get(k)
        lines.append(f"{lab:32s} {str(v):>10s}   target {tgt:10s} {'PASS' if ok(v) else 'FAIL'}")
    lines.append("")
    if s.get("acquisition_time_s") is None:
        lines.append("NOTE: target was never acquired; tracking metrics are scored over the whole run.")
    if s.get("reacquisition_pending_s") is not None:
        lines.append(f"NOTE: lock still lost at end of run ({s['reacquisition_pending_s']} s and counting).")
    for k in ["pointing_err_rmse_px", "pointing_err_max_px", "pointing_within_10px_pct", "centroid_err_mean_px",
              "centroid_err_max_px", "lock_retention_pct", "reacquisition_times_s", "reacquisition_events",
              "proc_time_mean_ms", "proc_time_p99_ms"]:
        lines.append(f"{k:32s} {s.get(k)}")
    return "\n".join(lines) + "\n"
