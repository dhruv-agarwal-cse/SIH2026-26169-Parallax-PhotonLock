"""Step-by-step sessions used by the interactive GUI.

SimSession   : live virtual world + virtual pan/tilt camera; every parameter can be changed while
               it runs.
VideoSession : Benchmark-2 mode. An .mp4 replaces the camera feed (PTZ bypassed).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .metrics import Recorder
from .sim import PlatformMotion, Scene, TargetMotion, jitter_sample
from .tracking import CoarsePointer
from .vision import Detector


class SimSession:
    def __init__(self, cfg, model=None, mode="ai"):
        self.cfg, self.model = cfg, model
        self.mode = mode if (mode != "ai" or model is not None) else "classical"
        self.reset(new_seed=False)

    # ------------------------------------------------------------------ set-up
    def reset(self, new_seed=True):
        cfg = self.cfg
        if new_seed:
            cfg.seed = int(np.random.randint(0, 2 ** 31 - 1))
        self.rng = np.random.default_rng(cfg.seed)
        self.scene = Scene(cfg, self.rng)
        self.motion = TargetMotion(cfg, self.rng)
        self.plat = PlatformMotion(cfg, self.rng)
        self.det = Detector(cfg.target_size, mode=self.mode, model=self.model)
        self.offset = np.zeros(2)
        self.tgt = self.motion.step()
        cue = self.tgt + self.rng.normal(0, cfg.cue_sigma_px, 2) if cfg.cued else None
        self.ptr = CoarsePointer(cfg, self.det, cue=cue, cue_fn=self._cue)
        self.rec = Recorder(cfg, "live")
        self.drift = np.zeros(2)
        self.k = 0
        self.trail, self.err = [], []
        self.occl_until = -1
        self._sig = self._motion_sig()

    def _cue(self):
        """Telemetry cue after a loss (GPS/INS position of the remote terminal, ~1 deg error)."""
        if not self.cfg.cued:
            return self.ptr.kf.pos
        return self.tgt - self.drift + self.rng.normal(0, self.cfg.cue_sigma_px, 2)

    def _motion_sig(self):
        c = self.cfg
        return (c.motion, round(c.target_speed_dps, 3), round(c.fov_x_deg, 3))

    def apply(self):
        """Push live parameter changes from cfg into the running components."""
        cfg = self.cfg
        cfg.fov_y_deg = cfg.fov_x_deg * cfg.cam_h / cfg.cam_w
        self.det.s = float(cfg.target_size)
        self.det.mode = self.mode
        self.det.model = self.model
        self.ptr.rate = np.array([cfg.dps_to_ppf(cfg.pan_speed_dps), cfg.dps_to_ppf(cfg.tilt_speed_dps)])
        self.ptr.kf.r = max(1.0, (cfg.jitter_px / 2.0) ** 2 + 0.5)
        if self._motion_sig() != self._sig:  # new motion/speed: continue from the current position
            cur = self.tgt.copy()
            self.motion = TargetMotion(cfg, self.rng)
            self.offset = cur - self.motion.step()
            self._sig = self._motion_sig()
        if len(self.scene.decoys) != cfg.n_decoys:
            self.scene.decoys = []
            for _ in range(cfg.n_decoys):
                self.scene.decoys.append({
                    "p": np.array([self.rng.uniform(300, cfg.world_w - 300), self.rng.uniform(300, cfg.world_h - 300)]),
                    "v": self.rng.normal(0, 1.5, 2), "size": float(self.rng.choice([3, 4, 24, 30])),
                    "amp": self.rng.uniform(110, 230)})

    def teleport(self, xy):
        """Move the beacon instantly (tests loss + re-acquisition)."""
        xy = np.asarray(xy, float)
        self.offset = self.offset + (xy - self.tgt)
        self.tgt = xy

    def occlude(self, seconds=1.0):
        self.occl_until = self.k + int(seconds * self.cfg.fps)

    # ------------------------------------------------------------------ one frame
    def step(self):
        cfg = self.cfg
        if self.k > 0:
            self.tgt = self.motion.step() + self.offset
        every = int(cfg.occlusion_every_s * cfg.fps)
        occluded = self.k < self.occl_until or (every > 0 and self.k > every // 2 and self.k % every < int(cfg.occlusion_len_s * cfg.fps))
        jit = jitter_sample(cfg, self.rng)
        cam_true = self.ptr.cam_cmd + self.drift
        frame, gt = self.scene.render(cam_true, jit, self.tgt, visible=not occluded)
        t1 = time.perf_counter()
        out = self.ptr.step(frame)
        proc_ms = (time.perf_counter() - t1) * 1000.0
        perr = float(np.linalg.norm(self.tgt - cam_true))
        m = out["meas_img"]
        in_fov = (0 <= gt[0] < cfg.cam_w) and (0 <= gt[1] < cfg.cam_h)
        cerr = float(np.linalg.norm(m - gt)) if (m is not None and in_fov and not occluded) else None
        self.rec.add(k=self.k, t=round(self.k / cfg.fps, 4), state=out["state"], occluded=bool(occluded),
                     gt_x=round(float(gt[0]), 3), gt_y=round(float(gt[1]), 3),
                     meas_x=None if m is None else round(float(m[0]), 3), meas_y=None if m is None else round(float(m[1]), 3),
                     cent_err=cerr, point_err=perr, proc_ms=proc_ms, prob=round(out["prob"], 3))
        self.drift = self.drift + self.plat.step()
        self.trail.append(self.tgt.copy())
        self.trail = self.trail[-240:]
        self.err.append(perr)
        self.err = self.err[-300:]
        self.k += 1
        return dict(frame=frame, out=out, gt=gt, tgt=self.tgt.copy(), cam=cam_true.copy(), perr=perr, cerr=cerr,
                    proc_ms=proc_ms, occluded=occluded, in_fov=in_fov)


class VideoSession:
    """Benchmark-2: .mp4 in, centroid per frame out. Truth CSV (frame,x,y) is picked up automatically
    if a file named <video>_truth.csv sits next to the video."""

    def __init__(self, path, cfg, model=None, mode="ai"):
        self.path, self.cfg = path, cfg
        self.cap = cv2.VideoCapture(path)
        ok, fr = self.cap.read()
        if not ok:
            raise IOError(f"Cannot read {path}")
        self.next = self._gray(fr)
        cfg.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if mode == "ai" and model is None:
            mode = "classical"
        self.det = Detector(cfg.target_size, mode=mode, model=model)
        self.ptr = CoarsePointer(cfg, self.det, fixed_camera=True, frame_shape=self.next.shape)
        self.rec = Recorder(cfg, "video_" + os.path.splitext(os.path.basename(path))[0])
        self.truth = None
        tp = os.path.splitext(path)[0] + "_truth.csv"
        if os.path.exists(tp):
            t = np.genfromtxt(tp, delimiter=",", names=True)
            self.truth = np.stack([t["x"], t["y"]], 1)
        self.k, self.err, self.trail = 0, [], []

    @staticmethod
    def _gray(fr):
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr

    def step(self):
        if self.next is None:
            return None
        g = self.next
        t1 = time.perf_counter()
        out = self.ptr.step(g)
        proc_ms = (time.perf_counter() - t1) * 1000.0
        m = out["meas_img"]
        gt = self.truth[self.k] if self.truth is not None and self.k < len(self.truth) else None
        visible = gt is not None and not np.isnan(gt[0])
        cerr = float(np.linalg.norm(m - gt)) if (m is not None and visible) else None
        perr = float(np.linalg.norm(self.ptr.est - gt)) if (visible and out["state"] != "SEARCH") else 1e3
        self.rec.add(k=self.k, t=round(self.k / self.cfg.fps, 4), state=out["state"],
                     occluded=bool(self.truth is not None and not visible),
                     gt_x=None if gt is None else float(gt[0]), gt_y=None if gt is None else float(gt[1]),
                     meas_x=None if m is None else round(float(m[0]), 3), meas_y=None if m is None else round(float(m[1]), 3),
                     cent_err=cerr, point_err=perr, proc_ms=proc_ms, prob=round(out["prob"], 3))
        ok, fr = self.cap.read()
        self.next = self._gray(fr) if ok else None
        if m is not None:
            self.trail.append(m.copy())
            self.trail = self.trail[-240:]
        self.err.append(cerr if cerr is not None else 0.0)
        self.err = self.err[-300:]
        self.k += 1
        return dict(frame=g, out=out, gt=gt, cerr=cerr, perr=perr if perr < 1e3 else None, proc_ms=proc_ms)
