"""Tracking and pointing control.

- AdaptiveKalmanCA : constant-acceleration Kalman filter per axis with innovation-driven
                     (NIS) process-noise adaptation -> handles line / circle / figure-8 / random
                     manoeuvres without switching models by hand.
- CoarsePointer    : acquisition state machine + latency-compensated, rate-limited pan/tilt
                     controller.
      SEARCH  --(2 confirmed hits)-->  TRACK  --(miss)-->  COAST (predict only)
         ^                                                   |
         +------------------(miss > 0.5 s)-------------------+
"""
from __future__ import annotations

import math

import numpy as np

F = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]], float)
Q1 = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]], float)


class AdaptiveKalmanCA:
    def __init__(self, q=0.02, r=4.0):
        self.q, self.r = q, r
        self.x = np.zeros((3, 2))
        self.P = np.stack([np.diag([r, 50.0, 5.0])] * 2)
        self.nis_ema, self.qs = 2.0, 1.0

    def init(self, z, v=None):
        self.x = np.zeros((3, 2))
        self.x[0] = z
        if v is not None:
            self.x[1] = v
        self.P = np.stack([np.diag([self.r, 30.0, 3.0])] * 2)
        self.nis_ema, self.qs = 2.0, 1.0

    def predict(self):
        self.x = F @ self.x
        Q = Q1 * self.q * self.qs
        for a in range(2):
            self.P[a] = F @ self.P[a] @ F.T + Q

    def S(self):
        return np.array([self.P[a][0, 0] + self.r for a in range(2)])

    def update(self, z):
        S = self.S()
        nu = z - self.x[0]
        nis = float(np.sum(nu ** 2 / S))
        for a in range(2):
            K = self.P[a][:, 0] / S[a]
            self.x[:, a] += K * nu[a]
            self.P[a] = self.P[a] - np.outer(K, self.P[a][0, :])
        # manoeuvre adaptation: inflate Q when innovations exceed expectation (E[NIS]=2)
        self.nis_ema = 0.85 * self.nis_ema + 0.15 * nis
        self.qs = float(np.clip(self.nis_ema / 2.0, 1.0, 60.0) ** 1.5)
        return nis

    @property
    def pos(self):
        return self.x[0].copy()

    @property
    def vel(self):
        return self.x[1].copy()


def spiral_offsets(max_ring=6):
    yield (0, 0)
    for r in range(1, max_ring + 1):
        x, y = r, -r + 1
        pts = []
        for yy in range(-r + 1, r + 1):
            pts.append((r, yy))
        for xx in range(r - 1, -r - 1, -1):
            pts.append((xx, r))
        for yy in range(r - 1, -r - 1, -1):
            pts.append((-r, yy))
        for xx in range(-r + 1, r + 1):
            pts.append((xx, -r))
        for p in pts:
            yield p


class CoarsePointer:
    SEARCH, TRACK, COAST = "SEARCH", "TRACK", "COAST"

    def __init__(self, cfg, detector, cue=None, fixed_camera=False, frame_shape=None, cue_fn=None):
        self.cue_fn = cue_fn  # optional live telemetry cue (e.g. GPS of remote terminal)
        self.cfg, self.det = cfg, detector
        self.fixed = fixed_camera
        h, w = frame_shape if frame_shape is not None else (cfg.cam_h, cfg.cam_w)
        self.w, self.h = w, h
        self.c_img = np.array([w / 2.0 - 0.5, h / 2.0 - 0.5])
        self.cam_cmd = self.c_img.copy() if fixed_camera else np.array([cfg.world_w / 2.0, cfg.world_h / 2.0])
        self.rate = np.array([cfg.dps_to_ppf(cfg.pan_speed_dps), cfg.dps_to_ppf(cfg.tilt_speed_dps)])
        r = max(1.0, (cfg.jitter_px / 2.0) ** 2 + 0.5)
        self.kf = AdaptiveKalmanCA(q=0.02, r=r)
        self.state = self.SEARCH
        self.hits, self.last_z = 0, None
        self.missed = 0
        self.coast_max = int(2.0 * cfg.fps)  # keep following the prediction for up to 2 s of dropout
        self.dwell = 0
        self.last_seen = None
        self._new_search(cue if cue is not None else self.cam_cmd.copy())
        self.last = {}

    # -------------------------------------------------------------- search pattern
    def _new_search(self, center):
        self.search_center = np.array(center, float)
        self.search_iter = spiral_offsets(8)
        self.waypoint = self._next_wp()

    def _next_wp(self):
        try:
            ox, oy = next(self.search_iter)
        except StopIteration:
            self.search_iter = spiral_offsets(8)
            ox, oy = next(self.search_iter)
        wp = self.search_center + np.array([ox * 0.6 * self.cfg.cam_w, oy * 0.6 * self.cfg.cam_h])
        return wp  # commanded frame drifts with the platform, so no world clipping here

    def _move_to(self, goal):
        if self.fixed:
            return
        d = goal - self.cam_cmd
        self.cam_cmd = self.cam_cmd + np.clip(d, -self.rate, self.rate)

    # -------------------------------------------------------------- main step
    def step(self, frame):
        roi, pred_img, gate_r = None, None, None
        if self.state in (self.TRACK, self.COAST):
            pred_img = self.kf.pos - self.cam_cmd + self.c_img
            sig = float(np.sqrt(self.kf.S().max()))
            gate_r = float(np.clip(4.0 * sig + self.det.s, 30.0, 220.0)) * (1.0 + 0.15 * self.missed)
            roi = (pred_img[0] - gate_r, pred_img[1] - gate_r, pred_img[0] + gate_r, pred_img[1] + gate_r)
        cands, noise_sigma = self.det.candidates(frame, roi)
        # after a long dropout, demand higher AI confidence before re-locking (avoids locking onto clutter)
        best, prob = self.det.select(cands, pred_img, gate_r, thr=0.5 if self.missed <= 5 else 0.7)
        meas_img = np.array([best["x"], best["y"]]) if best is not None else None
        z = (self.cam_cmd + meas_img - self.c_img) if meas_img is not None else None

        if self.state == self.SEARCH:
            if z is not None:
                if self.hits > 0 and self.last_z is not None and np.linalg.norm(z - self.last_z) < 80:
                    self.hits += 1
                else:
                    self.hits = 1
                if self.hits >= 2:
                    self.kf.init(z, v=z - self.last_z)
                    self.state, self.missed = self.TRACK, 0
                self.last_z = z
            else:
                self.hits = 0
        else:
            if z is not None:
                self.last_seen = z.copy()
                self.kf.update(z)
                self.state, self.missed = self.TRACK, 0
            else:
                self.missed += 1
                self.state = self.COAST
                if self.missed > self.coast_max:
                    self.state, self.hits = self.SEARCH, 0
                    self._new_search(self.cue_fn() if self.cue_fn else (self.last_seen if self.last_seen is not None else self.kf.pos))

        # ---- command for next frame (latency compensation: aim at predicted position)
        self.est = self.kf.pos.copy()
        if self.state in (self.TRACK, self.COAST):
            if self.state == self.COAST:
                self.kf.x[2] *= 0.85  # fade acceleration while coasting: no runaway extrapolation
            self.kf.predict()
            self._move_to(self.kf.pos)
        else:
            if z is not None:  # candidate seen: centre it while confirming
                self._move_to(z)
            else:
                if np.linalg.norm(self.waypoint - self.cam_cmd) < 1.0:
                    self.dwell += 1
                    if self.dwell >= 6:  # stare ~0.2 s at each search position
                        self.waypoint, self.dwell = self._next_wp(), 0
                self._move_to(self.waypoint)
        self.last = {"state": self.state, "meas_img": meas_img, "prob": prob, "roi": roi,
                     "pred_img": pred_img, "gate_r": gate_r, "n_cands": len(cands),
                     "noise_sigma": noise_sigma, "cands": cands}
        return self.last
