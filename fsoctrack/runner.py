"""Simulation loop, video-input (PTZ bypass) loop and the real-time GUI / HUD."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .metrics import Recorder
from .sim import MOTIONS, WEATHERS, PlatformMotion, Scene, TargetMotion, jitter_sample
from .tracking import CoarsePointer
from .vision import Detector

STATE_COL = {"SEARCH": (0, 170, 255), "TRACK": (80, 220, 80), "COAST": (0, 220, 255)}
CANVAS = (720, 1280)


def draw_hud(frame, out, cfg, world_info, stats, err_hist, title=""):
    """Compose a 1280x720 display: camera view + world map + live statistics."""
    H, W = CANVAS
    canvas = np.full((H, W, 3), 18, np.uint8)
    cam = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape
    c = (int(w / 2), int(h / 2))
    col = STATE_COL.get(out["state"], (255, 255, 255))
    cv2.line(cam, (c[0] - 14, c[1]), (c[0] + 14, c[1]), (255, 255, 0), 1)
    cv2.line(cam, (c[0], c[1] - 14), (c[0], c[1] + 14), (255, 255, 0), 1)
    if out.get("roi") is not None:
        x0, y0, x1, y1 = [int(v) for v in out["roi"]]
        cv2.rectangle(cam, (x0, y0), (x1, y1), (120, 120, 255), 1)
    if out.get("pred_img") is not None:
        p = out["pred_img"]
        cv2.drawMarker(cam, (int(p[0]), int(p[1])), (255, 120, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
    if out.get("meas_img") is not None:
        m = out["meas_img"]
        s = int(cfg.target_size)
        cv2.rectangle(cam, (int(m[0] - s), int(m[1] - s)), (int(m[0] + s), int(m[1] + s)), col, 2)
    canvas[70:70 + h, 20:20 + w] = cam
    cv2.rectangle(canvas, (19, 69), (20 + w, 70 + h), col, 2)
    cv2.putText(canvas, f"VIRTUAL CAMERA  {cfg.cam_w}x{cfg.cam_h}  FOV {cfg.fov_x_deg}x{cfg.fov_y_deg} deg", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, title or "AI Virtual Camera Tracking - FSOC Coarse Alignment (SIH26169)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    # world map
    if world_info is not None:
        S = 400
        sc = S / cfg.world_w
        ox, oy = 690, 70
        cv2.rectangle(canvas, (ox, oy), (ox + S, oy + S), (60, 60, 60), 1)
        trail = world_info["trail"]
        for i in range(1, len(trail)):
            a, b = trail[i - 1], trail[i]
            cv2.line(canvas, (int(ox + a[0] * sc), int(oy + a[1] * sc)), (int(ox + b[0] * sc), int(oy + b[1] * sc)), (90, 90, 200), 1)
        t = world_info["target"]
        cv2.circle(canvas, (int(ox + t[0] * sc), int(oy + t[1] * sc)), 4, (60, 60, 255), -1)
        cc = world_info["cam"]
        fw, fh = cfg.cam_w * sc / 2, cfg.cam_h * sc / 2
        cv2.rectangle(canvas, (int(ox + cc[0] * sc - fw), int(oy + cc[1] * sc - fh)),
                      (int(ox + cc[0] * sc + fw), int(oy + cc[1] * sc + fh)), col, 1)
        cv2.putText(canvas, f"SCREEN {cfg.world_w}x{cfg.world_h} px  (red = beacon, box = camera FOV)", (ox, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    # stats
    x0, y0 = 1105, 90
    lines = [("STATE", out["state"], col)] + [(k, v, (230, 230, 230)) for k, v in stats]
    for i, (k, v, cl) in enumerate(lines):
        cv2.putText(canvas, k, (x0, y0 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(v), (x0, y0 + i * 40 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cl, 2, cv2.LINE_AA)
    # error plot
    px0, py0, pw, ph = 690, 500, 580, 190
    cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (60, 60, 60), 1)
    ymax = 30.0
    y10 = int(py0 + ph - 10 / ymax * ph)
    cv2.line(canvas, (px0, y10), (px0 + pw, y10), (0, 0, 180), 1)
    cv2.putText(canvas, "10 px spec", (px0 + pw - 80, y10 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    cv2.putText(canvas, "pointing error (px, last 10 s)", (px0 + 4, py0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if len(err_hist) > 1:
        e = np.clip(np.array(err_hist[-300:]), 0, ymax)
        xs = px0 + np.linspace(0, pw, len(e))
        ys = py0 + ph - e / ymax * ph
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (80, 220, 80), 1, cv2.LINE_AA)
    scen = f"motion={cfg.motion}  noise={'+'.join(cfg.noise) or 'none'}  weather={cfg.weather}  jitter=+/-{cfg.jitter_px:g}px  platform={cfg.platform_mode}:{cfg.platform_px:g}px/f  pan={cfg.pan_speed_dps:g}deg/s"
    cv2.putText(canvas, scen, (20, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def run_sim(cfg, mode="ai", model=None, gui=False, writer=None, label=None, title="", keys=True,
            max_frames=None, verbose=False):
    rng = np.random.default_rng(cfg.seed)
    scene = Scene(cfg, rng)
    motion = TargetMotion(cfg, rng)
    plat = PlatformMotion(cfg, rng)
    det = Detector(cfg.target_size, mode=mode, model=model)
    t0 = motion.step()
    cue = t0 + rng.normal(0, cfg.cue_sigma_px, 2) if cfg.cued else None
    state = {"tgt": t0, "drift": np.zeros(2)}
    cue_fn = (lambda: state["tgt"] - state["drift"] + rng.normal(0, cfg.cue_sigma_px, 2)) if cfg.cued else None
    ptr = CoarsePointer(cfg, det, cue=cue, cue_fn=cue_fn)
    rec = Recorder(cfg, label or f"{cfg.motion}_{cfg.weather}_{mode}")
    drift = np.zeros(2)
    N = max_frames or int(cfg.duration_s * cfg.fps)
    trail, err_hist = [], []
    tgt = t0
    occ_every = int(cfg.occlusion_every_s * cfg.fps)
    occ_len = int(cfg.occlusion_len_s * cfg.fps)
    for k in range(N):
        if k > 0:
            tgt = motion.step()
        occluded = occ_every > 0 and k > occ_every // 2 and (k % occ_every) < occ_len
        jit = jitter_sample(cfg, rng)
        state["tgt"], state["drift"] = tgt, drift
        cam_true = ptr.cam_cmd + drift
        frame, gt = scene.render(cam_true, jit, tgt, visible=not occluded)
        t1 = time.perf_counter()
        out = ptr.step(frame)
        proc_ms = (time.perf_counter() - t1) * 1000.0
        perr = float(np.linalg.norm(tgt - cam_true))
        m = out["meas_img"]
        in_fov = (0 <= gt[0] < cfg.cam_w) and (0 <= gt[1] < cfg.cam_h)
        cerr = float(np.linalg.norm(m - gt)) if (m is not None and in_fov and not occluded) else None
        rec.add(k=k, t=round(k / cfg.fps, 4), state=out["state"], occluded=bool(occluded),
                gt_x=round(float(gt[0]), 3), gt_y=round(float(gt[1]), 3),
                meas_x=None if m is None else round(float(m[0]), 3), meas_y=None if m is None else round(float(m[1]), 3),
                cent_err=cerr, point_err=perr, proc_ms=proc_ms, prob=round(out["prob"], 3))
        drift = drift + plat.step()
        err_hist.append(perr)
        if gui or writer is not None:
            trail.append(tgt.copy())
            trail = trail[-200:]
            done = rec.rows
            nt = sum(1 for r in done if r["state"] == "TRACK")
            stats = [("POINTING ERR", f"{perr:6.2f} px"),
                     ("CENTROID ERR", "-" if cerr is None else f"{cerr:5.2f} px"),
                     ("AI BEACON PROB", f"{out['prob']:.2f}"),
                     ("PROC TIME / FPS", f"{proc_ms:4.1f} ms / {1000 / max(proc_ms, 1e-3):4.0f}"),
                     ("TRACK FRAMES", f"{100.0 * nt / len(done):5.1f} %"),
                     ("OCCLUSION", "YES" if occluded else "no"),
                     ("TIME", f"{k / cfg.fps:5.1f} s"),
                     ("DETECTOR", mode.upper()),
                     ("CANDIDATES", out["n_cands"])]
            canvas = draw_hud(frame, out, cfg, {"trail": trail, "target": tgt, "cam": cam_true}, stats, err_hist, title)
            if writer is not None:
                writer.write(canvas)
            if gui:
                cv2.imshow("FSOC Coarse Tracker", canvas)
                key = cv2.waitKey(max(1, int(1000 / cfg.fps - proc_ms))) & 0xFF
                if key == ord("q"):
                    break
    return rec


def run_video(path, cfg, mode="ai", model=None, truth=None, gui=False, writer=None):
    """Benchmark-2: take an .mp4 as the camera feed (PTZ bypassed). A digital gate/window follows
    the beacon. Returns Recorder; if truth (N x 2 array, index coords) is given, errors are logged."""
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    if not ok:
        raise IOError(path)
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
    cfg.fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    det = Detector(cfg.target_size, mode=mode, model=model)
    ptr = CoarsePointer(cfg, det, fixed_camera=True, frame_shape=g.shape)
    rec = Recorder(cfg, f"video_{mode}")
    k = 0
    err_hist = []
    while ok:
        t1 = time.perf_counter()
        out = ptr.step(g)
        proc_ms = (time.perf_counter() - t1) * 1000.0
        m = out["meas_img"]
        gt = truth[k] if truth is not None and k < len(truth) else None
        visible = gt is not None and not np.isnan(gt[0])
        cerr = float(np.linalg.norm(m - gt)) if (m is not None and visible) else None
        perr = float(np.linalg.norm(ptr.est - gt)) if (visible and out["state"] != "SEARCH") else (float("nan"))
        rec.add(k=k, t=round(k / cfg.fps, 4), state=out["state"], occluded=(truth is not None and not visible),
                gt_x=None if gt is None else float(gt[0]), gt_y=None if gt is None else float(gt[1]),
                meas_x=None if m is None else round(float(m[0]), 3), meas_y=None if m is None else round(float(m[1]), 3),
                cent_err=cerr, point_err=perr if not np.isnan(perr) else 1e3, proc_ms=proc_ms, prob=round(out["prob"], 3))
        if gui or writer is not None:
            err_hist.append(cerr if cerr is not None else 0)
            view = g
            if m is not None:  # digital PTZ window 640x480 around the beacon
                x0 = int(np.clip(m[0] - 320, 0, max(0, g.shape[1] - 640)))
                y0 = int(np.clip(m[1] - 240, 0, max(0, g.shape[0] - 480)))
                view = g[y0:y0 + 480, x0:x0 + 640]
                o2 = dict(out)
                o2["meas_img"] = m - [x0, y0]
                o2["roi"] = None
                o2["pred_img"] = None
            else:
                view = cv2.resize(g, (640, 480))
                o2 = dict(out, meas_img=None, roi=None, pred_img=None)
            stats = [("CENTROID ERR", "-" if cerr is None else f"{cerr:5.2f} px"), ("PROC", f"{proc_ms:4.1f} ms"),
                     ("FRAME", k), ("INPUT", "MP4 (PTZ bypass)")]
            canvas = draw_hud(view, o2, cfg, None, stats, err_hist, "Benchmark-2 mode: .mp4 input, PTZ bypassed")
            if writer is not None:
                writer.write(canvas)
            if gui:
                cv2.imshow("FSOC Coarse Tracker", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        ok, fr = cap.read()
        if ok:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        k += 1
    return rec
