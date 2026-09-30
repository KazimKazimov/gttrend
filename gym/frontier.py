"""
Delay versus false alarms: the quickest-detection trade-off.

A probabilistic detector can be made calmer by requiring confidence before it switches: the call moves
to state k only when P(k) >= theta, otherwise it keeps the previous call (hysteresis). Sweeping theta
traces a frontier of median detection delay against false alarms per decade. A variable that really
helps shifts the whole frontier towards the origin, not just one point on it.

    python -m gym.frontier --settings default     # adds "frontier" to results/gym_results_default.json
"""
from __future__ import annotations
import argparse, json, pathlib, warnings
import numpy as np

from .data import Panel
from .env import TrendGym
from .detectors import RuleDetector, LogitDetector, GBMDetector, MSDetector
from .metrics import event_metrics, month_metrics, CLS
from .experiments import add_macro_momentum, clean, TIERS

ROOT = pathlib.Path(__file__).resolve().parents[1]
THETAS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def hysteresis_calls(p, theta):
    """theta = 0 gives plain argmax calls."""
    out = np.empty(len(p), dtype=int)
    q = np.nan_to_num(p, nan=1 / 3)
    am = CLS[np.argmax(q + np.array([0, 1e-9, 0]), axis=1)]
    out[0] = am[0]
    for t in range(1, len(p)):
        k = am[t]
        if theta <= 0 or (k != out[t - 1] and q[t, list(CLS).index(k)] >= theta):
            out[t] = k
        else:
            out[t] = out[t - 1]
    return out


def curve(gym, res, h=0, thetas=THETAS):
    pts = []
    tgt = res.t_index + h
    keep = tgt <= gym.e1
    truth = gym.truth[tgt[keep]]
    for th in thetas:
        c = hysteresis_calls(res.probs[h], th)
        e, _ = event_metrics(gym, res, h, c)
        ck = c[keep]
        rec = [np.mean(ck[truth == k] == k) for k in CLS if (truth == k).any()]
        pts.append(dict(theta=th, delay_med=e["delay_med"], delay_mean=e["delay_mean"], hit_rate=e["hit_rate"],
                        false_alarms_10y=e["false_alarms_10y"], flips_yr=e["flips_yr"], capture=e["capture"],
                        bal_acc=float(np.mean(rec))))
    return pts


def run(settings="default", tier="L"):
    warnings.simplefilter("ignore")
    panel = add_macro_momentum(Panel.load())
    g = TrendGym(panel, "D", settings, (0, 3), eval_start=TIERS[tier])
    ref = LogitDetector(["clev10"])
    start = ref.feature_start(g)
    dets = [
        ("Rule D", RuleDetector("D")), ("Rule A", RuleDetector("A")),
        ("MS own", MSDetector()), ("MS + UMich infl. exp. (transitions)", MSDetector(tvtp=["mich1"])),
        ("MS + CPI (transitions)", MSDetector(tvtp=["cpi"])), ("MS + fed funds (transitions)", MSDetector(tvtp=["ff"])),
        ("Logit own", LogitDetector()), ("Logit + Cleveland 10y infl. exp.", LogitDetector(["clev10"])),
        ("GBM own", GBMDetector()),
    ]
    out = {"tier": tier, "train_from": start.strftime("%Y-%m"), "curves": {}}
    for name, d in dets:
        if not isinstance(d, RuleDetector):
            d.train_from = start
        r = g.run(d)
        ths = (0.0,) if isinstance(d, RuleDetector) else THETAS
        out["curves"][name] = {str(h): curve(g, r, h, ths) for h in (0, 3)}
        print(f"  {name}: argmax delay {out['curves'][name]['0'][0]['delay_med']}, FA/10y {out['curves'][name]['0'][0]['false_alarms_10y']:.1f}")
    return clean(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="default")
    a = ap.parse_args()
    path = ROOT / "results" / f"gym_results_{a.settings}.json"
    res = json.loads(path.read_text())
    res["frontier"] = run(a.settings, "L")
    res["frontier_M"] = run(a.settings, "M")          # out-of-window check of the same thresholds
    path.write_text(json.dumps(res))
    print(f"added frontier to {path}")
