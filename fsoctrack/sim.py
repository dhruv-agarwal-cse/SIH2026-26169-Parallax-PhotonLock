"""Virtual environment: configuration, target motion, platform motion, scene rendering,
atmospheric effects and sensor noise.

Coordinate conventions
- World ("screen") pixels: 2000 x 2000 by default. 1 world px == 1 camera px (IFOV).
- IFOV = FOV_x / cam_w = 4 deg / 640 = 0.00625 deg/px (~109 urad/px).
- Image index coordinates: pixel (i, j) centre sits at integer (i, j).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

MOTIONS = ["line", "circle", "figure8", "random", "spiral", "sine"]
WEATHERS = ["clear", "haze", "fog", "rain", "lowlight"]
PLATFORM_MODES = ["none", "linear", "circular", "random"]


@dataclass
class Config:
    # Screen / camera (PS items 1-6)
    world_w: int = 2000
    world_h: int = 2000
    cam_w: int = 640
    cam_h: int = 480
    fov_x_deg: float = 4.0
    fov_y_deg: float = 3.0
    fps: float = 30.0
    # Camera motion constraints (PS items 13-15)
    pan_speed_dps: float = 5.0
    tilt_speed_dps: float = 5.0
    # Target (PS items 7-12)
    target_size: int = 10
    target_speed_dps: float = 1.5
    motion: str = "circle"
    n_decoys: int = 0
    cued: bool = True            # telemetry (GPS/INS) cue for initial pointing
    cue_sigma_px: float = 150.0  # 1-sigma cue error (~0.94 deg)
    # Disturbances (PS items 21-25)
    noise: list = field(default_factory=lambda: ["gaussian"])  # gaussian, saltpepper, poisson
    gauss_sigma: float = 10.0
    sp_fraction: float = 0.10
    weather: str = "clear"
    turbulence: float = 0.25     # scintillation (log-normal sigma) + beam wander
    jitter_px: float = 5.0       # max +/- px per frame
    platform_mode: str = "linear"
    platform_px: float = 3.0     # max px/frame
    occlusion_every_s: float = 0.0
    occlusion_len_s: float = 0.5
    duration_s: float = 20.0
    seed: int = 0

    @property
    def ifov_deg(self) -> float:
        return self.fov_x_deg / self.cam_w

    def dps_to_ppf(self, dps: float) -> float:
        """deg/s -> px/frame."""
        return dps / self.fps / self.ifov_deg

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------------------
class TargetMotion:
    """Beacon trajectories in world pixels. step() returns the position for the next frame."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng, self.k = cfg, rng, 0
        self.speed = cfg.dps_to_ppf(cfg.target_speed_dps)  # px/frame
        W, H = cfg.world_w, cfg.world_h
        self.lo = np.array([150.0, 150.0])
        self.hi = np.array([W - 150.0, H - 150.0])
        self.c = np.array([rng.uniform(0.35 * W, 0.65 * W), rng.uniform(0.35 * H, 0.65 * H)])
        self.phi = rng.uniform(0, 2 * math.pi)
        ang = rng.uniform(0, 2 * math.pi)
        self.p = np.array([rng.uniform(250, W - 250), rng.uniform(250, H - 250)])
        self.vel = self.speed * np.array([math.cos(ang), math.sin(ang)])
        self.heading = ang
        m = cfg.motion
        if m == "circle":
            self.R = 0.22 * min(W, H)
            self.w = self.speed / self.R
        elif m == "figure8":
            self.A, self.B = 0.25 * W, 0.17 * H
            self.w = self.speed / math.sqrt(self.A ** 2 + 4 * self.B ** 2)
        elif m == "spiral":
            self.Rmax = 0.22 * min(W, H)
            self.w = self.speed / self.Rmax
        elif m == "sine":
            self.vel = np.array([0.8 * self.speed * (1 if rng.random() < 0.5 else -1), 0.0])
            self.A = 0.15 * H
            self.w = 0.6 * self.speed / self.A

    def _bounce(self):
        for i in range(2):
            if self.p[i] < self.lo[i] or self.p[i] > self.hi[i]:
                self.p[i] = np.clip(self.p[i], self.lo[i], self.hi[i])
                self.vel[i] *= -1
                if self.cfg.motion == "random":
                    self.heading = math.atan2(self.vel[1], self.vel[0])

    def step(self) -> np.ndarray:
        k, m = self.k, self.cfg.motion
        self.k += 1
        if m == "line":
            self.p = self.p + self.vel
            self._bounce()
            return self.p.copy()
        if m == "random":
            self.heading += self.rng.normal(0, 0.06)
            sp = self.speed * (0.75 + 0.25 * math.sin(0.02 * k))
            self.vel = sp * np.array([math.cos(self.heading), math.sin(self.heading)])
            self.p = self.p + self.vel
            self._bounce()
            return self.p.copy()
        if m == "circle":
            a = self.w * k + self.phi
            return self.c + self.R * np.array([math.cos(a), math.sin(a)])
        if m == "figure8":
            a = self.w * k + self.phi
            return self.c + np.array([self.A * math.sin(a), self.B * math.sin(2 * a)])
        if m == "spiral":
            a = self.w * k * 1.6 + self.phi
            r = self.Rmax * (0.25 + 0.75 * (0.5 - 0.5 * math.cos(0.08 * self.w * k)))
            return self.c + r * np.array([math.cos(a), math.sin(a)])
        if m == "sine":
            self.p = self.p + self.vel
            self._bounce()
            return np.array([self.p[0], self.c[1] + self.A * math.sin(self.w * k + self.phi)])
        raise ValueError(m)


class PlatformMotion:
    """Motion of the host platform: shifts the camera line of sight (px/frame, accumulated)."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng, self.k = cfg, rng, 0
        ang = rng.uniform(0, 2 * math.pi)
        self.dir = np.array([math.cos(ang), math.sin(ang)])
        self.v = np.zeros(2)

    def step(self) -> np.ndarray:
        P, m = self.cfg.platform_px, self.cfg.platform_mode
        self.k += 1
        if m == "none" or P <= 0:
            return np.zeros(2)
        if m == "linear":
            return P * self.dir
        if m == "circular":
            a = 2 * math.pi * self.k / (self.cfg.fps * 4.0)
            return P * np.array([math.cos(a), math.sin(a)])
        if m == "random":
            self.v = 0.9 * self.v + self.rng.normal(0, 0.35 * P, 2)
            return np.clip(self.v, -P, P)
        raise ValueError(m)


def jitter_sample(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    J = cfg.jitter_px
    if J <= 0:
        return np.zeros(2)
    return np.clip(rng.normal(0, J / 2.0, 2), -J, J)


# ----------------------------------------------------------------------------------------
WEATHER_PARAMS = {  # transmission t, airlight, extra PSF blur (px), gain
    "clear": (1.00, 0.0, 0.0, 1.00),
    "haze": (0.55, 90.0, 0.6, 1.00),
    "fog": (0.30, 130.0, 1.3, 1.00),
    "rain": (0.80, 40.0, 0.4, 1.00),
    "lowlight": (1.00, 0.0, 0.0, 0.25),
}


def draw_square(img: np.ndarray, cx: float, cy: float, size: float, amp: float, sigma: float):
    """Add an anti-aliased square of side `size`, centred at continuous (cx, cy) where pixel
    i spans [i, i+1). Its intensity centroid in index coords is (cx-0.5, cy-0.5)."""
    H, W = img.shape
    m = int(math.ceil(3 * sigma)) + 2
    x0, x1 = cx - size / 2, cx + size / 2
    y0, y1 = cy - size / 2, cy + size / 2
    ix0, ix1 = int(math.floor(x0)) - m, int(math.ceil(x1)) + m
    iy0, iy1 = int(math.floor(y0)) - m, int(math.ceil(y1)) + m
    if ix1 <= 0 or iy1 <= 0 or ix0 >= W or iy0 >= H:
        return
    xs = np.arange(ix0, ix1, dtype=np.float32)
    ys = np.arange(iy0, iy1, dtype=np.float32)
    cx_ = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    cy_ = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
    patch = np.outer(cy_, cx_).astype(np.float32) * amp
    if sigma > 0.05:
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
    a0, a1 = max(ix0, 0), min(ix1, W)
    b0, b1 = max(iy0, 0), min(iy1, H)
    img[b0:b1, a0:a1] += patch[b0 - iy0:b1 - iy0, a0 - ix0:a1 - ix0]


class Scene:
    PAD = 1200

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        Wp, Hp = cfg.world_w + 2 * self.PAD, cfg.world_h + 2 * self.PAD
        clouds = rng.random((Hp // 100, Wp // 100)).astype(np.float32)
        clouds = cv2.resize(clouds, (Wp, Hp), interpolation=cv2.INTER_CUBIC)
        bg = 22.0 + 14.0 * clouds
        # dim point-like clutter (stars / distant lights), 1-3 px
        n = int(Wp * Hp / 9000)
        xs, ys = rng.integers(0, Wp, n), rng.integers(0, Hp, n)
        stars = np.zeros((Hp, Wp), np.float32)
        stars[ys, xs] = rng.uniform(15, 70, n)
        stars = cv2.GaussianBlur(stars, (0, 0), 0.7)
        self.bg = np.clip(bg + stars * 2.0, 0, 255).astype(np.uint8)
        # decoy beacons (optional multi-target clutter) with non-target sizes
        self.decoys = []
        for _ in range(cfg.n_decoys):
            size = float(rng.choice([3, 4, 24, 30]))
            self.decoys.append({
                "p": np.array([rng.uniform(300, cfg.world_w - 300), rng.uniform(300, cfg.world_h - 300)]),
                "v": rng.normal(0, 1.5, 2), "size": size, "amp": rng.uniform(110, 230)})

    def render(self, cam_true, jitter, target, visible=True):
        """Return (uint8 frame, ground-truth beacon centroid in image index coords)."""
        cfg, rng = self.cfg, self.rng
        w, h = cfg.cam_w, cfg.cam_h
        origin = np.round(cam_true + jitter - np.array([w / 2.0, h / 2.0])).astype(int)
        ox = int(np.clip(origin[0] + self.PAD, 0, self.bg.shape[1] - w))
        oy = int(np.clip(origin[1] + self.PAD, 0, self.bg.shape[0] - h))
        origin = np.array([ox - self.PAD, oy - self.PAD], dtype=float)
        img = self.bg[oy:oy + h, ox:ox + w].astype(np.float32)
        t, air, blur_extra, gain = WEATHER_PARAMS[cfg.weather]
        psf = 0.8 + blur_extra + 0.6 * cfg.turbulence
        # beacon with scintillation and beam wander
        scint = float(np.exp(rng.normal(0, cfg.turbulence)))
        wander = rng.normal(0, 0.4 * cfg.turbulence, 2)
        tgt = target + wander
        rel = tgt - origin
        if visible:
            draw_square(img, rel[0], rel[1], cfg.target_size, 190.0 * scint, psf)
        for d in self.decoys:
            d["p"] = d["p"] + d["v"]
            r = d["p"] - origin
            draw_square(img, r[0], r[1], d["size"], d["amp"], psf)
        # atmosphere
        img = img * gain
        if t < 1.0:
            img = img * t + air * (1.0 - t)
        if cfg.weather == "rain":
            layer = np.zeros_like(img)
            for _ in range(140):
                x, y = rng.integers(0, w), rng.integers(0, h)
                L = rng.integers(10, 26)
                cv2.line(layer, (int(x), int(y)), (int(x + 0.3 * L), int(y + L)), float(rng.uniform(20, 45)), 1)
            img += cv2.GaussianBlur(layer, (0, 0), 0.6)
        # sensor noise
        if "poisson" in cfg.noise:
            img = rng.poisson(np.clip(img, 0, None)).astype(np.float32)
        if "gaussian" in cfg.noise and cfg.gauss_sigma > 0:
            g = np.empty_like(img)
            cv2.randn(g, 0, cfg.gauss_sigma)
            img += g
        img = np.clip(img, 0, 255).astype(np.uint8)
        if "saltpepper" in cfg.noise and cfg.sp_fraction > 0:
            r = rng.random(img.shape, dtype=np.float32)
            img[r < cfg.sp_fraction / 2] = 0
            img[(r >= cfg.sp_fraction / 2) & (r < cfg.sp_fraction)] = 255
        return img, rel - 0.5
