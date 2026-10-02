"""Generate a Benchmark-2 style test video: full screen, noise, moving beacon, + truth CSV."""
import cv2
import numpy as np

from fsoctrack.sim import Config, Scene, TargetMotion, draw_square


def make_video(out, seconds=10, size=2000, motion="figure8", seed=7):
    cfg = Config(world_w=size, world_h=size, motion=motion, target_speed_dps=2.0, seed=seed)
    rng = np.random.default_rng(seed)
    mo = TargetMotion(cfg, rng)
    bg = Scene(cfg, rng).bg[Scene.PAD:Scene.PAD + size, Scene.PAD:Scene.PAD + size].astype(np.float32)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (size, size), isColor=False)
    rows = []
    for k in range(int(seconds * 30)):
        p = mo.step()
        img = bg.copy()
        occl = 150 <= k < 162  # 0.4 s dropout to test re-acquisition
        if not occl:
            draw_square(img, p[0], p[1], 10, 170 * float(np.exp(rng.normal(0, 0.2))), 1.2)
        img = img * 0.6 + 60  # haze
        img += rng.normal(0, 12, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        r = rng.random(img.shape, dtype=np.float32)
        img[r < 0.05] = 0
        img[(r >= 0.05) & (r < 0.10)] = 255
        wr.write(img)
        rows.append((k, np.nan, np.nan) if occl else (k, p[0] - 0.5, p[1] - 0.5))
    wr.release()
    np.savetxt(out.replace(".mp4", "_truth.csv"), np.array(rows), delimiter=",", header="frame,x,y", comments="", fmt="%.3f")
    print("wrote", out)
