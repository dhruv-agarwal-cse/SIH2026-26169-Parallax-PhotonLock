import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
INK, MUTED, ACC, OK, WARN = "#0B1F33", "#5B6B7A", "#E4572E", "#1B998B", "#F2A541"

# ---- architecture diagram
fig, ax = plt.subplots(figsize=(12, 4.2), dpi=200); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, s, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none"))
    ax.text(x + w / 2, y + h - 0.22, t, ha="center", va="top", color="white", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.55, s, ha="center", va="top", color="white", fontsize=8.2, linespacing=1.25)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
box(0.1, 2.3, 2.1, 1.7, "VIRTUAL WORLD", "2000x2000 screen\n6 beacon motions\ndecoys, clutter", "#27476E")
box(2.5, 2.3, 2.1, 1.7, "DISTURBANCES", "Gauss / S&P / Poisson\nhaze fog rain lowlight\njitter, platform, scint.", "#27476E")
box(4.9, 2.3, 2.1, 1.7, "VIRTUAL PTZ CAM", "640x480, 4x3 deg\n30 Hz, rate-limited\npan/tilt 5-10 deg/s", "#27476E")
box(7.3, 2.3, 2.1, 1.7, "DETECT + CENTROID", "median -> bg removal\nmatched filter, MAD thr\nsub-pixel centroid", "#1B998B")
box(9.7, 2.3, 2.2, 1.7, "AI BEACON VERIFIER", "MLP on 10 blob features\nrejects stars, rain,\ndecoys, impulse noise", "#E4572E")
box(9.7, 0.2, 2.2, 1.7, "ADAPTIVE KALMAN", "constant-accel model\nNIS-driven manoeuvre\nadaptation + gating", "#1B998B")
box(7.3, 0.2, 2.1, 1.7, "STATE MACHINE", "SEARCH (spiral / cue)\nTRACK -> COAST\n-> re-acquire", "#1B998B")
box(4.9, 0.2, 2.1, 1.7, "LEAD CONTROLLER", "aims at predicted t+1\nlatency compensation\nslew-rate limited", "#1B998B")
box(2.5, 0.2, 2.1, 1.7, "PERFORMANCE LOG", "acq / RMSE / lock %\nre-acq / FPS / proc ms\nJSON + CSV + TXT", "#5B6B7A")
box(0.1, 0.2, 2.1, 1.7, "INPUTS", "live simulation\nOR .mp4 file\n(PTZ bypass mode)", "#5B6B7A")
for x in [2.2, 4.6, 7.0, 9.4]: arr(x, 3.15, x + 0.3, 3.15)
arr(10.8, 2.3, 10.8, 1.9); arr(9.7, 1.05, 9.4, 1.05); arr(7.3, 1.05, 7.0, 1.05); arr(4.9, 1.05, 4.6, 1.05); arr(2.5, 1.05, 2.2, 1.05)
ax.annotate("", (5.95, 2.3), (5.95, 1.9), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.6, ls="--"))
ax.text(6.05, 2.08, "pan/tilt cmd", color=ACC, fontsize=8)
plt.savefig("docs/fig_architecture.png", bbox_inches="tight", transparent=False, facecolor="white"); plt.close()

d = json.load(open("results/benchmark.json"))
R = d["matrix"]
conds = list(dict.fromkeys(r["cond"] for r in R))
# ---- per-condition bars
cm = [np.mean([r["pointing_err_mean_px"] for r in R if r["cond"] == c and r["pointing_err_mean_px"] is not None]) for c in conds]
ce = [np.mean([r["centroid_err_rmse_px"] for r in R if r["cond"] == c and r["centroid_err_rmse_px"] is not None]) for c in conds]
lo = [np.mean([r["target_loss_pct"] for r in R if r["cond"] == c and r["target_loss_pct"] is not None]) for c in conds]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
labels = [c.replace(" + ", "+\n").replace(" ", "\n", 1) if len(c) > 9 else c for c in conds]
for a, vals, t, lim, col in [(axs[0], cm, "Tracking error, mean (px)", 10, OK), (axs[1], ce, "Centroiding error, RMSE (px)", None, "#27476E"), (axs[2], lo, "Target loss (%)", 5, ACC)]:
    cols = [col if (lim is None or v <= lim) else WARN for v in vals]
    b = a.barh(range(len(conds)), vals, color=cols); a.set_yticks(range(len(conds))); a.set_yticklabels(conds, fontsize=8.5); a.invert_yaxis()
    a.set_title(t, fontsize=10.5, loc="left", color=INK, weight="bold")
    if lim: a.axvline(lim, color=ACC, ls="--", lw=1); a.text(lim, len(conds) - 0.4, f" spec {lim}", color=ACC, fontsize=8)
    for i, v in enumerate(vals): a.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    for s in ["top", "right"]: a.spines[s].set_visible(False)
    if lim: a.set_xscale("symlog", linthresh=lim * 2)
plt.tight_layout(); plt.savefig("docs/fig_conditions.png", facecolor="white"); plt.close()

# ---- ablation
A = d["ablation"]
cs = list(dict.fromkeys(r["cond"] for r in A))
fig, a = plt.subplots(figsize=(6.4, 3.4), dpi=200)
w = 0.26
for j, (md, col, lab) in enumerate([("naive", "#B0B8C1", "Naive threshold"), ("classical", "#27476E", "Robust classical"), ("ai", ACC, "Ours (robust + AI + Kalman)")]):
    v = [np.mean([100.0 if r["target_loss_pct"] is None else r["target_loss_pct"] for r in A if r["cond"] == c and r["mode"] == md]) for c in cs]
    a.bar(np.arange(len(cs)) + (j - 1) * w, v, w, color=col, label=lab)
    for i, x in enumerate(v): a.text(i + (j - 1) * w, x + 1, f"{x:.0f}", ha="center", fontsize=8)
a.set_xticks(range(len(cs))); a.set_xticklabels(cs, fontsize=9); a.set_ylabel("Target loss (%)")
a.axhline(5, color=ACC, ls="--", lw=1); a.legend(fontsize=8, frameon=False)
for s in ["top", "right"]: a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("docs/fig_ablation.png", facecolor="white"); plt.close()
print("figs ok")
