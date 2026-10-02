"""FSOC coarse-alignment virtual camera tracker - headless command-line tools (the GUI is gui.py).

Examples
  python gui.py                                                           # interactive app
  python app.py sim --motion figure8 --noise gaussian saltpepper --weather fog  # headless run + report
  python app.py sim --motion random --jitter 20 --platform random --platform-px 20 --pan 10 --duration 30
  python app.py video --input bench.mp4 --truth bench_truth.csv          # Benchmark-2 (PTZ bypass)
  python app.py makevideo --out bench.mp4                                # generate a test .mp4 + truth
Every run writes an automatic performance log to results/ (JSON + TXT + per-frame CSV).
"""
import argparse
import os

import cv2
import numpy as np

from fsoctrack.metrics import format_report
from fsoctrack.runner import run_sim, run_video
from fsoctrack.sim import MOTIONS, PLATFORM_MODES, WEATHERS, Config
from fsoctrack.vision import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--motion", choices=MOTIONS, default="circle")
    s.add_argument("--noise", nargs="*", default=["gaussian"], choices=["gaussian", "saltpepper", "poisson"])
    s.add_argument("--gauss-sigma", type=float, default=10)
    s.add_argument("--sp", type=float, default=0.10)
    s.add_argument("--weather", choices=WEATHERS, default="clear")
    s.add_argument("--jitter", type=float, default=5)
    s.add_argument("--platform", choices=PLATFORM_MODES, default="linear")
    s.add_argument("--platform-px", type=float, default=3)
    s.add_argument("--pan", type=float, default=5)
    s.add_argument("--tilt", type=float, default=5)
    s.add_argument("--target-size", type=int, default=10)
    s.add_argument("--target-speed", type=float, default=1.5, help="deg/s")
    s.add_argument("--decoys", type=int, default=0)
    s.add_argument("--occlusion-every", type=float, default=0)
    s.add_argument("--blind", action="store_true", help="no telemetry cue: blind spiral search")
    s.add_argument("--duration", type=float, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--detector", choices=["ai", "classical", "naive"], default="ai")
    s.add_argument("--record", default=None, help="save the GUI to .mp4")
    v = sub.add_parser("video")
    v.add_argument("--input", required=True)
    v.add_argument("--truth", default=None, help="CSV with columns frame,x,y (optional)")
    v.add_argument("--target-size", type=int, default=10)
    v.add_argument("--detector", choices=["ai", "classical", "naive"], default="ai")
    v.add_argument("--record", default=None)
    mv = sub.add_parser("makevideo")
    mv.add_argument("--out", default="bench.mp4")
    mv.add_argument("--seconds", type=float, default=10)
    mv.add_argument("--size", type=int, default=1000)
    mv.add_argument("--motion", choices=MOTIONS, default="figure8")
    a = ap.parse_args()
    model = load_model()
    os.makedirs("results", exist_ok=True)

    if a.cmd == "sim":
        cfg = Config(motion=a.motion, noise=a.noise, gauss_sigma=a.gauss_sigma, sp_fraction=a.sp, weather=a.weather,
                     jitter_px=a.jitter, platform_mode=a.platform, platform_px=a.platform_px, pan_speed_dps=a.pan,
                     tilt_speed_dps=a.tilt, target_size=a.target_size, target_speed_dps=a.target_speed,
                     n_decoys=a.decoys, occlusion_every_s=a.occlusion_every, cued=not a.blind,
                     duration_s=a.duration, seed=a.seed)
        wr = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (1280, 720)) if a.record else None
        rec = run_sim(cfg, mode=a.detector, model=model, gui=False, writer=wr)
        if wr:
            wr.release()
        print(format_report(rec.write("results")))
    elif a.cmd == "video":
        from fsoctrack.sim import Config as C
        cfg = C(target_size=a.target_size)
        truth = None
        if a.truth:
            t = np.genfromtxt(a.truth, delimiter=",", names=True)
            truth = np.stack([t["x"], t["y"]], 1)
        wr = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720)) if a.record else None
        rec = run_video(a.input, cfg, mode=a.detector, model=model, truth=truth, gui=False, writer=wr)
        if wr:
            wr.release()
        print(format_report(rec.write("results", "video_" + os.path.splitext(os.path.basename(a.input))[0])))
    elif a.cmd == "makevideo":
        from tools_make_video import make_video
        make_video(a.out, a.seconds, a.size, a.motion)


if __name__ == "__main__":
    main()
