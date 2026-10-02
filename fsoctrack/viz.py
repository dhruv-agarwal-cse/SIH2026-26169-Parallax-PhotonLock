"""Drawing helpers for the GUI (return BGR numpy images)."""
import cv2
import numpy as np

COL = {"SEARCH": (0, 170, 255), "TRACK": (80, 220, 80), "COAST": (0, 220, 255)}
F = cv2.FONT_HERSHEY_SIMPLEX


def draw_camera(frame, out, target_size, occluded=False, offset=(0, 0)):
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape
    c = (w // 2, h // 2)
    col = COL.get(out["state"], (255, 255, 255))
    ox, oy = offset
    cv2.line(img, (c[0] - 16, c[1]), (c[0] + 16, c[1]), (255, 255, 0), 1)
    cv2.line(img, (c[0], c[1] - 16), (c[0], c[1] + 16), (255, 255, 0), 1)
    if out.get("roi") is not None:
        x0, y0, x1, y1 = [int(v) for v in out["roi"]]
        cv2.rectangle(img, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (120, 120, 255), 1)
    if out.get("pred_img") is not None:
        p = out["pred_img"]
        cv2.drawMarker(img, (int(p[0] - ox), int(p[1] - oy)), (255, 120, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
    if out.get("meas_img") is not None:
        m = out["meas_img"]
        s = int(target_size)
        cv2.rectangle(img, (int(m[0] - ox - s), int(m[1] - oy - s)), (int(m[0] - ox + s), int(m[1] - oy + s)), col, 2)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 3)
    cv2.putText(img, out["state"], (12, 30), F, 0.8, col, 2, cv2.LINE_AA)
    if occluded:
        cv2.putText(img, "BEACON OCCLUDED", (w - 230, 30), F, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    return img


def draw_world(cfg, trail, tgt, cam, size=380, bg=None):
    sc = size / cfg.world_w
    img = np.full((size, size, 3), 22, np.uint8) if bg is None else bg.copy()
    for g in range(1, 4):
        v = int(g * size / 4)
        cv2.line(img, (v, 0), (v, size), (40, 40, 40), 1)
        cv2.line(img, (0, v), (size, v), (40, 40, 40), 1)
    for i in range(1, len(trail)):
        a, b = trail[i - 1], trail[i]
        cv2.line(img, (int(a[0] * sc), int(a[1] * sc)), (int(b[0] * sc), int(b[1] * sc)), (90, 90, 210), 1, cv2.LINE_AA)
    if tgt is not None:
        cv2.circle(img, (int(tgt[0] * sc), int(tgt[1] * sc)), 4, (60, 60, 255), -1)
    if cam is not None:
        fw, fh = cfg.cam_w * sc / 2, cfg.cam_h * sc / 2
        cv2.rectangle(img, (int(cam[0] * sc - fw), int(cam[1] * sc - fh)), (int(cam[0] * sc + fw), int(cam[1] * sc + fh)),
                      (80, 220, 80), 1)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (80, 80, 80), 1)
    return img


def draw_plot(vals, w=640, h=150, ymax=30.0, spec=10.0, label="pointing error (px)"):
    img = np.full((h, w, 3), 22, np.uint8)
    ys = int(h - spec / ymax * h)
    cv2.line(img, (0, ys), (w, ys), (0, 0, 190), 1)
    cv2.putText(img, f"{spec:g} px spec", (w - 95, ys - 5), F, 0.42, (60, 60, 230), 1, cv2.LINE_AA)
    cv2.putText(img, label + " - last 10 s", (8, 18), F, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if len(vals) > 1:
        e = np.clip(np.nan_to_num(np.array(vals[-300:], float), nan=0.0), 0, ymax)
        xs = np.linspace(0, w - 1, len(e))
        pts = np.stack([xs, h - 1 - e / ymax * (h - 2)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (80, 220, 80), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    return img
