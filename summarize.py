import json, numpy as np
d = json.load(open("results/benchmark.json"))
R, A, Q = d["matrix"], d["ablation"], d["acquisition"]
conds = list(dict.fromkeys(r["cond"] for r in R))
def agg(rs, k, fn):
    v = [r[k] for r in rs if r.get(k) is not None]
    return fn(v) if v else None
rows = []
for c in conds:
    rs = [r for r in R if r["cond"] == c]
    rows.append(dict(cond=c, acq_max=agg(rs, "acquisition_time_s", max), perr=agg(rs, "pointing_err_mean_px", np.mean),
                     perr_worst=agg(rs, "pointing_err_mean_px", max), within10=agg(rs, "pointing_within_10px_pct", np.mean),
                     cent=agg(rs, "centroid_err_rmse_px", np.mean), loss=agg(rs, "target_loss_pct", np.mean),
                     loss_worst=agg(rs, "target_loss_pct", max), reacq=agg(rs, "reacquisition_max_s", max), fps=agg(rs, "processing_fps", min)))
spec = [r for r in R if r["cond"] != "STRESS"]
S = dict(
    n_runs=len(R), n_spec=len(spec),
    acq_max=agg(spec, "acquisition_time_s", max), acq_median=agg(spec, "acquisition_time_s", np.median),
    perr_mean=agg(spec, "pointing_err_mean_px", np.mean), perr_worst=agg(spec, "pointing_err_mean_px", max),
    within10=agg(spec, "pointing_within_10px_pct", np.mean),
    cent_mean=agg(spec, "centroid_err_rmse_px", np.mean), cent_worst=agg(spec, "centroid_err_rmse_px", max),
    loss_mean=agg(spec, "target_loss_pct", np.mean), loss_worst=agg(spec, "target_loss_pct", max),
    reacq_max=agg(spec, "reacquisition_max_s", max), fps_min=agg(spec, "processing_fps", min),
    fps_median=agg(spec, "processing_fps", np.median),
    pass_count=sum(1 for r in spec if (r["target_loss_pct"] or 0) < 5 and (r["pointing_err_mean_px"] or 99) <= 10 and (r["acquisition_time_s"] or 99) <= 2),
)
for cu in (True, False):
    a = np.array([q["acquisition_time_s"] if q["acquisition_time_s"] is not None else 99 for q in Q if q["cued"] == cu])
    S["acq_cued" if cu else "acq_blind"] = dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)), le2=float(np.mean(a <= 2) * 100))
abl = {}
for md in ["naive", "classical", "ai"]:
    rs = [r for r in A if r["mode"] == md]
    abl[md] = dict(loss=agg(rs, "target_loss_pct", np.mean), perr=agg(rs, "pointing_err_mean_px", np.median))
S["ablation"] = abl
S["rows"] = rows
S["stress"] = [r for r in rows if r["cond"] == "STRESS"][0]
json.dump(S, open("results/summary.json", "w"), indent=1, default=float)
f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
md = ["| Condition (x6 motions) | Acq. max (s) | Track err mean (px) | Within 10 px (%) | Centroid RMSE (px) | Loss mean / worst (%) | Re-acq max (s) | Min FPS |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['cond']} | {f(r['acq_max'])} | {f(r['perr'])} | {f(r['within10'],1)} | {f(r['cent'],3)} | {f(r['loss'],1)} / {f(r['loss_worst'],1)} | {f(r['reacq'])} | {f(r['fps'],0)} |")
open("results/RESULTS.md", "w").write("\n".join(md) + "\n")
print("\n".join(md)); print(json.dumps({k: v for k, v in S.items() if k not in ("rows",)}, indent=1, default=float))
