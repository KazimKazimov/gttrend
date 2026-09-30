"""
Which variables help detect 10Y trends in real time?

For every candidate variable X and model family, run the gym twice on the same months with the same
training rows: once with X (augmented) and once without (matched baseline). Report the paired
differences in accuracy, probability skill, turning-point delay, false alarms and P&L, with
Diebold-Mariano (Brier), block-bootstrap (balanced accuracy) and Wilcoxon (delays) tests.

Variables enter in the longest window where they have data: tier L (scored from 1990), M (1997), S (2009).

    python -m gym.experiments                 # full run, writes results/gym_results.json
    python -m gym.experiments --quick         # a few variables, for testing
"""
from __future__ import annotations
import argparse, json, pathlib, time, warnings
import numpy as np
import pandas as pd

from .data import Panel
from .env import TrendGym
from .detectors import (RuleDetector, CompositeRule, LogitDetector, GBMDetector, MSDetector, Climatology)
from .metrics import score, compare
from .leadlag import lead_lag
from . import labels as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

TIERS = {"L": "1990-01-01", "M": "1997-01-01", "S": "2009-01-01"}
VARS = {   # id: (tier, label, rate-like -> also test the blended rule machine)
    "y2": ("L", "2Y yield", True), "ff": ("L", "Fed funds", True), "y5": ("L", "5Y yield", True),
    "mich1": ("L", "UMich 1y inflation expectations", False), "clev10": ("L", "Cleveland Fed 10y expected inflation", True),
    "cpi": ("L", "CPI inflation (YoY)", False), "core_cpi": ("L", "Core CPI inflation (YoY)", False),
    "core_pce": ("L", "Core PCE inflation (YoY)", False), "unrate": ("L", "Unemployment rate", False),
    "payems": ("L", "Payroll growth (YoY)", False), "spx": ("L", "S&P 500", False), "gold": ("L", "Gold", False),
    "usd": ("L", "Dollar index", False), "mm": ("L", "Macro momentum score (AQR-style)", False),
    "baa10y": ("M", "Baa - 10Y credit spread", False), "vix": ("M", "VIX", False), "wti": ("M", "WTI oil", False),
    "be10": ("S", "10Y breakeven", True), "fwd5y5y": ("S", "5y5y forward inflation", True), "tips10": ("S", "10Y TIPS real yield", True),
}
BLOCKS = {  # id: (members, tier, label)
    "B_rates": (("y2", "ff", "y5"), "L", "Rates block: 2Y, fed funds, 5Y"),
    "B_infl": (("cpi", "core_cpi", "core_pce", "mich1", "clev10"), "L", "Inflation block: CPI, core CPI, core PCE, UMich, Cleveland"),
    "B_activity": (("unrate", "payems"), "L", "Activity block: unemployment, payrolls"),
    "B_markets": (("spx", "gold", "usd"), "L", "Markets block: S&P 500, gold, dollar"),
    "B_risk": (("baa10y", "vix", "wti"), "M", "Risk block: credit spread, VIX, oil"),
    "B_breakevens": (("be10", "fwd5y5y", "tips10"), "S", "Breakeven block: 10Y BE, 5y5y, TIPS"),
}
HORIZONS = (0, 3, 6)


def add_macro_momentum(panel: Panel, min_hist=60):
    """AQR-style macro momentum for yields (Brooks 2017): 1-year changes in growth, inflation and the 2Y,
    1-year equity return and 1-year dollar depreciation, each z-scored on an expanding window (causal),
    averaged. Positive = macro trends that historically push yields up."""
    f = panel.realtime()
    comp = {
        "growth": panel.level("payems").diff(12),
        "inflation": panel.level("core_cpi").diff(12),
        "policy": panel.level("y2").diff(12),
        "equity": panel.level("spx").diff(12),
        "fx": -panel.level("usd").diff(12),
    }
    zs = []
    for s in comp.values():
        m = s.expanding(min_hist).mean()
        sd = s.expanding(min_hist).std()
        zs.append((s - m) / sd)
    Z = pd.concat(zs, axis=1)
    mm = Z.mean(axis=1).where(Z.notna().sum(axis=1) >= 3)
    panel.raw["mm"] = mm
    panel.meta.loc["mm"] = dict(name="Macro momentum score (AQR-style)", group="derived", kind="rate", pub_lag=0,
                                unit="z", start=mm.first_valid_index().strftime("%Y-%m") if mm.notna().any() else "",
                                end=mm.last_valid_index().strftime("%Y-%m") if mm.notna().any() else "", n=int(mm.notna().sum()),
                                source="derived from payems, core_cpi, y2, spx, usd (real-time)", url="")
    panel._rt = None
    return panel


def matched_pair(gym, make, xv):
    aug = make(xv)
    start = feature_start(gym, aug)
    base = make(())
    base.name = base.name + " (matched)"
    aug.train_from = start
    base.train_from = start
    return aug, base, start


def feature_start(gym, det):
    if hasattr(det, "feature_start"):
        return det.feature_start(gym)
    if isinstance(det, MSDetector):
        Z, U, s0 = det._data(gym)
        return gym.dates[s0]
    return None


def summarize(c, h):
    a, b = c["a"], c["b"]
    keep = ["bal_acc", "acc", "bss", "brier", "auc_up", "auc_down", "hit_rate", "delay_med", "delay_mean", "within6",
            "capture", "false_alarms_10y", "precision", "flips_yr", "pnl_bp_yr", "sharpe", "n_events", "n_months"]
    return dict(h=h, aug={k: a.get(k) for k in keep}, base={k: b.get(k) for k in keep},
                d_bal_acc=c["d_bal_acc"], d_bss=c["d_bss"], d_delay_med=c["d_delay_med"], d_delay_mean=c["d_delay_mean"],
                d_capture=c["d_capture"], d_false_alarms_10y=c["d_false_alarms_10y"], d_flips_yr=c["d_flips_yr"],
                d_pnl_bp_yr=c["d_pnl_bp_yr"], d_auc_up=c["d_auc_up"], d_auc_down=c["d_auc_down"],
                p_dm=c["p_dm"], bal_acc_ci90=c["bal_acc_ci90"], p_boot=c["p_boot"], p_wilcoxon_delay=c["p_wilcoxon_delay"])


FAMILIES = {
    "Logit": lambda xv: LogitDetector(xv),
    "MS-obs": lambda xv: MSDetector(obs=xv),
    "MS-tvtp": lambda xv: MSDetector(tvtp=xv),
}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m")
    return o


def run(settings="default", quick=False, B=500, families=None, verbose=True):
    warnings.simplefilter("ignore")
    panel = add_macro_momentum(Panel.load())
    families = families or FAMILIES
    gyms = {t: TrendGym(panel, "D", settings, HORIZONS, eval_start=s) for t, s in TIERS.items()}
    out = dict(settings=settings, label_settings=L.SETTINGS[settings], horizons=HORIZONS,
               tiers={t: dict(eval_start=g.dates[g.e0].strftime("%Y-%m"), eval_end=g.dates[g.e1].strftime("%Y-%m"),
                              n_events=len(L.trend_events(g.truth, g.e0, g.e1))) for t, g in gyms.items()},
               addone=[], leaderboard=[], loo=[], paths={}, events={})
    vars_ = {k: v for k, v in VARS.items() if not quick or k in ("y2", "clev10", "vix", "be10", "mm")}
    cands = [(v, (v,), t, lab, rl) for v, (t, lab, rl) in vars_.items()]
    if not quick:
        cands += [(b, m, t, lab, False) for b, (m, t, lab) in BLOCKS.items()]
    t0 = time.time()

    # ---------------------------------------------------------------- add-one, matched baselines
    for cid, members, tier, label, ratelike in cands:
        g = gyms[tier]
        fams = dict(families)
        if ratelike:
            fams["Rule+X"] = None
        if len(members) > 1:
            fams.pop("MS-obs", None)                  # several observables at once over-fit the emission model
            fams["GBM"] = lambda xv: GBMDetector(xv)
        for fam, make in fams.items():
            if fam == "Rule+X":
                aug, base, start = CompositeRule(cid), RuleDetector(), None
            else:
                aug, base, start = matched_pair(g, make, members)
            ra, rb = g.run(aug), g.run(base)
            for h in HORIZONS:
                c = compare(g, ra, rb, h, B=B)
                out["addone"].append(dict(var=cid, members=list(members), label=label, tier=tier, family=fam,
                                          train_from=start.strftime("%Y-%m") if start is not None else None,
                                          **summarize(c, h)))
            if verbose:
                s = [r for r in out["addone"] if r["var"] == cid and r["family"] == fam and r["h"] == 0][0]
                print(f"  [{time.time()-t0:6.0f}s] {tier} {cid:12s} {fam:8s} bal_acc {s['aug']['bal_acc']:.3f} vs {s['base']['bal_acc']:.3f}"
                      f"  dBSS {s['d_bss']:+.3f} (p={s['p_dm']:.2f})  d_delay {s['d_delay_med']}", flush=True)

    # ---------------------------------------------------------------- leaderboard per tier
    for tier, g in gyms.items():
        tv = [v for v, (t, _, _) in VARS.items() if TIERS[t] <= TIERS[tier] and v in vars_]
        tv = [v for v in tv if v != "mm"]
        dets = [RuleDetector("D", name="Rule D (real time)"), RuleDetector("A", name="Rule A (real time)"), Climatology(),
                LogitDetector(name="Logit own"), MSDetector(name="MS own"), GBMDetector(name="GBM own"),
                LogitDetector(tv, name="Logit all X"), GBMDetector(tv, name="GBM all X"),
                MSDetector(tvtp=["mm"], name="MS tvtp:macro momentum"), LogitDetector(["mm"], name="Logit + macro momentum")]
        # matched own-series baselines for the all-X models
        start = feature_start(g, dets[6])
        for d in dets[3:]:
            d.train_from = start
        res = {}
        for d in dets:
            r = g.run(d)
            res[d.name] = r
            for h in HORIZONS:
                out["leaderboard"].append(dict(tier=tier, train_from=start.strftime("%Y-%m"), **score(g, r, h)))
        if verbose:
            print(f"  [{time.time()-t0:6.0f}s] leaderboard {tier} done ({len(dets)} models)")
        if tier == "L":
            keep = ["Rule D (real time)", "Logit own", "MS own", "Logit all X", "GBM all X", "MS tvtp:macro momentum"]
            out["paths"] = dict(dates=[d.strftime("%Y-%m") for d in res[keep[0]].dates],
                                y=[round(float(x), 3) for x in g.y[g.e0:g.e1 + 1]],
                                truth=[int(x) for x in g.truth[g.e0:g.e1 + 1]],
                                models={k: {str(h): np.round(res[k].probs[h], 3).tolist() for h in (0, 3)} for k in keep})
            out["events"] = [dict(start=g.dates[s].strftime("%Y-%m"), end=g.dates[e].strftime("%Y-%m"), dir=d,
                                  move_bp=int(round(100 * (g.y[e] - g.y[s - 1]))))
                             for s, e, d in L.trend_events(g.truth, g.e0, g.e1)]
            # leave-one-out from the all-X logit
            full = res["Logit all X"]
            for v in tv:
                d = LogitDetector([x for x in tv if x != v], name=f"all - {v}")
                d.train_from = start
                r = g.run(d)
                for h in HORIZONS:
                    c = compare(g, full, r, h, B=B)
                    out["loo"].append(dict(var=v, label=VARS[v][1], **summarize(c, h)))
            if verbose:
                print(f"  [{time.time()-t0:6.0f}s] leave-one-out done")

    # ---------------------------------------------------------------- lead-lag of turning points
    ylab = pd.Series(gyms["L"].truth, index=gyms["L"].dates)
    out["leadlag"] = [lead_lag(panel, v, ylab, start="1978-01-01", end=gyms["L"].dates[gyms["L"].e1])
                      for v in vars_ if v not in ("mm",)] if not quick else []
    out["runtime_sec"] = round(time.time() - t0)
    return clean(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="default")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.settings, a.quick, a.B)
    OUT.mkdir(exist_ok=True)
    path = pathlib.Path(a.out) if a.out else OUT / f"gym_results_{a.settings}{'_quick' if a.quick else ''}.json"
    path.write_text(json.dumps(res))
    print(f"wrote {path} ({path.stat().st_size/1e3:.0f} kB) in {res['runtime_sec']}s")
