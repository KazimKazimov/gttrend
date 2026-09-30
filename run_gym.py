"""
Test whether a variable helps detect 10Y trends in real time.

    python run_gym.py --vars y2 clev10                        # logit + Markov-switching, matched baselines
    python run_gym.py --vars mm --models ms-tvtp --horizons 0 3 6
    python run_gym.py --list                                  # variables in the panel

Your own series (e.g. from Bloomberg): a monthly CSV with a date column and one column per series, plus
a meta CSV with columns id, kind (rate | price | yoy), pub_lag (months), name.

    python run_gym.py --add-csv my_series.csv --add-meta my_meta.csv --vars MOVE CESI

For each variable and model family the script runs the gym twice on the same months with the same
training rows (with and without the variable) and prints the paired differences and tests.
"""
import argparse, sys, warnings
import pandas as pd

from gym import Panel, TrendGym, LogitDetector, GBMDetector, MSDetector, CompositeRule, RuleDetector, compare
from gym.experiments import add_macro_momentum, matched_pair

FAMILIES = {
    "logit": lambda xv: LogitDetector(xv),
    "gbm": lambda xv: GBMDetector(xv),
    "ms-obs": lambda xv: MSDetector(obs=xv),
    "ms-tvtp": lambda xv: MSDetector(tvtp=xv),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vars", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=["logit", "ms-tvtp"], choices=list(FAMILIES) + ["rule"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[0, 3])
    ap.add_argument("--eval-start", default="1990-01-01")
    ap.add_argument("--settings", default="default", choices=["default", "fixed100"])
    ap.add_argument("--add-csv")
    ap.add_argument("--add-meta")
    ap.add_argument("--B", type=int, default=500, help="bootstrap draws")
    ap.add_argument("--out", default="gym_addone.csv")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    P = add_macro_momentum(Panel.load())
    if a.add_csv:
        new = pd.read_csv(a.add_csv, parse_dates=[0], index_col=0)
        new.index = new.index.to_period("M").to_timestamp()
        meta = pd.read_csv(a.add_meta).set_index("id")
        for c in new.columns:
            P.raw[c] = new[c].reindex(P.raw.index)
            P.meta.loc[c, ["name", "group", "kind", "pub_lag"]] = [meta.loc[c].get("name", c), "user",
                                                                  meta.loc[c, "kind"], int(meta.loc[c, "pub_lag"])]
        P._rt = None
    if a.list:
        print(P.meta[["name", "group", "kind", "pub_lag", "start"]].to_string())
        return
    if not a.vars:
        sys.exit("give --vars (see --list)")

    g = TrendGym(P, "D", a.settings, tuple(a.horizons), eval_start=a.eval_start)
    print(f"Scoring {g.dates[g.e0]:%Y-%m} to {g.dates[g.e1]:%Y-%m}, target = method D ({a.settings}), horizons {a.horizons}")
    rows = []
    for v in a.vars:
        for m in a.models:
            if m == "rule":
                aug, base, start = CompositeRule(v), RuleDetector(), None
            else:
                aug, base, start = matched_pair(g, FAMILIES[m], (v,))
            ra, rb = g.run(aug), g.run(base)
            for h in a.horizons:
                c = compare(g, ra, rb, h, B=a.B)
                rows.append(dict(var=v, model=m, h=h, train_from=None if start is None else start.strftime("%Y-%m"),
                                 bal_acc=c["a"]["bal_acc"], bal_acc_base=c["b"]["bal_acc"], d_bal_acc=c["d_bal_acc"],
                                 bss=c["a"]["bss"], d_bss=c["d_bss"], p_dm=c["p_dm"],
                                 delay=c["a"]["delay_med"], d_delay=c["d_delay_med"], p_delay=c["p_wilcoxon_delay"],
                                 false_alarms_10y=c["a"]["false_alarms_10y"], d_false_alarms=c["d_false_alarms_10y"],
                                 d_pnl_bp_yr=c["d_pnl_bp_yr"]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}.  d_* = with variable minus matched baseline; p_dm: Diebold-Mariano on Brier loss; "
          f"p_delay: Wilcoxon on per-trend delays.")


if __name__ == "__main__":
    main()
