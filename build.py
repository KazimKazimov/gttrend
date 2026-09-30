"""
Reproduce the UST 10Y Trend Lab page's labels in Python.

    python build.py data/ust10_monthly.csv            # page defaults
    python build.py data/ust10_monthly.csv --fixed    # fixed 100bp rule, no minimum length

Input CSV: columns Date,Rate (monthly 10Y yield in percent).
Writes trend_labels.csv (one column per method, 1 = up, 0 = sideways, -1 = down)
and trend_segments.csv (ex-post segments for A, B, D, E, F, G).
"""
import sys
import numpy as np
import pandas as pd
from trend_algos import run_all, decorate, Rule, kappa, NAMES

# page defaults
SETTINGS = dict(thr=1.0, scaled=True, z=2.0, min_len=3, sigma="rolling", cap=12,
                rho=0.618, floor=1.0, kalman_lam=1000.0, l1_lam=5.0)

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/ust10_monthly.csv")
if "--fixed" in sys.argv:
    SETTINGS.update(scaled=False, min_len=0)

df = pd.read_csv(path, parse_dates=["Date"])
y, dates = df.Rate.values.astype(float), df.Date.dt.strftime("%Y-%m-%d").values
labels, segs = run_all(y, **SETTINGS)
R = Rule(y, SETTINGS["thr"], scaled=SETTINGS["scaled"], z=SETTINGS["z"], min_len=SETTINGS["min_len"],
         sigma=SETTINGS["sigma"], cap=SETTINGS["cap"])

pd.DataFrame({"Date": dates, "Rate": y, **labels}).to_csv("trend_labels.csv", index=False)
rows = [dict(method=k, **s) for k, v in segs.items() for s in decorate(v, y, dates, R)]
pd.DataFrame(rows).drop(columns=["imp"], errors="ignore").to_csv("trend_segments.csv", index=False)

print("Settings:", SETTINGS)
print(f"Latest ({dates[-1]}, {y[-1]:.2f}%):")
for k, lab in labels.items():
    run = len(lab) - 1
    while run > 0 and lab[run - 1] == lab[-1]:
        run -= 1
    n_tr = sum(s["label"] != 0 for s in segs[k]) if k in segs else None
    print(f"  {k:5s} {NAMES[int(lab[-1])]:9s} since {dates[run]}" + (f"   trends: {n_tr}" if n_tr is not None else ""))
print("Cohen's kappa, D vs others:", {k: round(float(kappa(labels["D"], v)), 2) for k, v in labels.items() if k != "D"})
