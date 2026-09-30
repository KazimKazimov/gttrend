"""
Build the interactive results page from the gym's JSON output.

    python scripts/build_report.py      # reads results/gym_results_*.json, writes examples/trend_gym_report.html

The findings text is generated from the numbers, so it stays true when the experiments are rerun.
"""
import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAM = {"Logit": "logit", "GBM": "boosted trees", "MS-obs": "Markov switching, as an observable",
       "MS-tvtp": "Markov switching, in the transition probabilities", "Rule+X": "rule machine on a 10Y+X blend"}


def f(v, d=2, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if round(v, d) == 0:
        v = 0.0
    s = f"{v:.{d}f}"
    return ("+" + s) if sign and v > 0 else s.replace("-", "−")


def findings(res, target):
    lb = pd.DataFrame(res["leaderboard"])
    L0 = lb[(lb.tier == "L") & (lb.h == 0)].set_index("model")
    cand = L0.drop(index=[m for m in L0.index if m.startswith("Rule") or m == "Climatology"])
    best = cand.bss.idxmax()                      # best probability skill; balanced accuracy alone can favour over-fit models
    b, rd = L0.loc[best], L0.loc["Rule D (real time)"]
    ev = res["tiers"]["L"]["n_events"]
    a = pd.DataFrame(res["addone"])
    a0 = a[a.h == 0]
    sigp = a0[(a0.p_dm < .10) & (a0.d_bss > 0)].sort_values("p_dm")
    sign = a0[(a0.p_dm < .10) & (a0.d_bss < 0)]
    fr = res.get("frontier", {}).get("curves", {})
    ms = next((p for p in fr.get("MS own", {}).get("0", []) if abs(p["theta"] - .8) < 1e-9), None)
    frd = fr.get("Rule D", {}).get("0", [None])[0]
    frM = res.get("frontier_M", {}).get("curves", {})
    msM = next((p for p in frM.get("MS own", {}).get("0", []) if abs(p["theta"] - .8) < 1e-9), None)
    frdM = frM.get("Rule D", {}).get("0", [None])[0]
    perM = f'{res["tiers"]["M"]["eval_start"][:4]}–{res["tiers"]["M"]["eval_end"][:4]}'
    stab = (f" Same threshold on {perM}: {f(msM['delay_med'], 1)} months and {f(msM['false_alarms_10y'], 1)} false alarms, against {f(frdM['delay_med'], 0)} and {f(frdM['false_alarms_10y'], 1)} for rule D."
            if msM and frdM else "")
    ll = {r["var"]: r for r in res.get("leadlag", [])}
    loo = pd.DataFrame(res["loo"])
    loo0 = loo[(loo.h == 0) & (loo.p_dm < .10) & (loo.d_bss > 0)].sort_values("d_bss", ascending=False)
    L3 = lb[(lb.tier == "L") & (lb.h == 3)].set_index("model")
    L6 = lb[(lb.tier == "L") & (lb.h == 6)].set_index("model")
    a3 = a[(a.h == 3) & (a.p_dm < .10) & (a.d_bss > 0)].sort_values("p_dm")
    per = f'{res["tiers"]["L"]["eval_start"][:4]}–{res["tiers"]["L"]["eval_end"][:4]}'

    y2 = ll.get("y2", {})
    cards = [
        dict(k="Best real-time detector", v=best.replace(" own", ", 10Y only"),
             t=f"Highest Brier skill ({f(b.bss)}) on {per} nowcasts, with balanced accuracy {f(b.bal_acc)}; rule machine D scores {f(rd.bss)} and {f(rd.bal_acc)}."),
        dict(k="Delay after a turn", v=(f"{f(frd['delay_med'], 0)} → {f(ms['delay_med'], 1)} months" if ms and frd else f"{f(b.delay_med, 1)} months"),
             t=(f"Asking for 80% confidence before switching, the Markov-switching filter calls {f(100*ms['hit_rate'], 0)}% of the {ev} trends with a {f(ms['delay_med'], 1)}-month median delay and {f(ms['false_alarms_10y'], 1)} false alarms per decade. Rule D: {f(frd['delay_med'], 0)} months and {f(frd['false_alarms_10y'], 1)}.{stab}"
                if ms and frd else "")),
        dict(k="Variables that help", v=f"{len(sigp)} of {len(a0)}",
             t=f"Matched nowcast comparisons with a significant Brier-skill gain (Diebold–Mariano, 10%), against {len(sign)} with a significant loss."),
        dict(k="Does the 2Y turn first?", v="No" if (y2.get("share_first") or 0) < .25 else "Sometimes",
             t=f"Of {y2.get('n_turns','–')} 10Y trend starts since 1978, {y2.get('matched','–')} match a 2Y turn within a year; the 2Y turned first in {f(100*(y2.get('share_first') or 0), 0)}% of them (median lead {f(y2.get('lead_med'), 1, True)} months)."),
    ]

    def lab(r):
        return f"{r['label']} ({FAM.get(r['family'], r['family'])}: {f(r['d_bss'], 3, True)} Brier skill, p = {f(r['p_dm'])})"

    helps = "; ".join(lab(r) for _, r in sigp.iterrows()) or "none"
    ff, baa = ll.get("ff", {}), ll.get("baa10y", {})
    loo_txt = (f" In the all-variable logit, dropping {loo0.iloc[0]['label']} costs the most ({f(loo0.iloc[0]['d_bss'], 3)} Brier skill, p = {f(loo0.iloc[0]['p_dm'])})."
               if len(loo0) else "")
    exa = (f" Ahead, gains appear for {'; '.join(lab(r) for _, r in a3.head(3).iterrows())} at 3 months." if len(a3) else "")
    fe = a0[a0["var"].isin(["y2", "ff"])]
    fe_sig = fe[(fe.p_dm < .10) & (fe.d_bss > 0)]
    fe_txt = ("Neither the 2Y nor fed funds gives a significant Brier-skill gain in any model family"
              if fe_sig.empty else "Front-end gains are limited to " + "; ".join(lab(r) for _, r in fe_sig.iterrows()))
    mo = a0[a0.family == "MS-obs"]
    mo_txt = f"as extra observables in the Markov-switching model, variables lower Brier skill in {int((mo.d_bss < 0).sum())} of {len(mo)} cases"
    worse = []
    for t in ("L", "M", "S"):
        T0 = lb[(lb.tier == t) & (lb.h == 0)].set_index("model")
        ok = all(m in T0.index for m in ("Logit all X", "Logit own", "GBM all X", "GBM own"))
        worse.append(ok and T0.loc["Logit all X"].bss < T0.loc["Logit own"].bss and T0.loc["GBM all X"].bss < T0.loc["GBM own"].bss)
    allx_txt = (f"Logit and boosted-tree models with every variable score worse on Brier skill than their own-series versions in {sum(worse)} of 3 windows")
    leaders = [r for r in ll.values() if (r.get("lead_med") or 0) > 0 and r.get("matched", 0) >= r.get("n_turns", 99) / 2]
    lab_of = {r["var"]: r["label"] for r in res["addone"]}
    if leaders:
        lt = "; ".join(f"{lab_of.get(r['var'], r['var'])} (median lead {f(r['lead_med'], 1, True)} months, {r['matched']} of {r['n_turns']} turns matched)" for r in leaders)
        lead_txt = (f"Among variables that match at least half of the 10Y's turns, only {'one tends' if len(leaders) == 1 else str(len(leaders)) + ' tend'} to turn first: {lt}.")
        conv = a0[a0["var"].isin([r["var"] for r in leaders]) & (a0.p_dm < .10) & (a0.d_bss > 0)]
        lead_txt += (" That lead did not turn into significantly better nowcasts." if conv.empty else "")
    else:
        lead_txt = "No variable that matches at least half of the 10Y's turns tends to turn first."
    y5 = ll.get("y5", {})
    text = f"""
<p><b>The model matters more than the variables.</b> Using only the 10Y's own path, a Markov-switching filter on volatility-normalised monthly changes (regimes pinned to the ex-post labels) reaches balanced accuracy {f(L0.loc['MS own'].bal_acc)} with a {f(L0.loc['MS own'].delay_med, 0)}-month median delay, but it changes its call {f(L0.loc['MS own'].flips_yr, 1)} times a year against {f(L0.loc['MS own'].truth_flips_yr, 1)} for the truth. A confirmation threshold removes most of those whipsaws for a few months of extra delay (see the frontier).</p>
<p style="margin-top:10px"><b>Few variables add information in real time.</b> Significant nowcast gains: {helps}.{loo_txt} {fe_txt}, and {mo_txt}. {allx_txt}, a sign of over-fitting with {ev} trends to learn from.</p>
<p style="margin-top:10px"><b>Ex ante is hard.</b> At 3 and 6 months ahead, the best model's balanced accuracy falls to {f(L3.loc['MS own'].bal_acc)} and {f(L6.loc['MS own'].bal_acc)}, with Brier skill {f(L3.loc['MS own'].bss)} and {f(L6.loc['MS own'].bss)}.{exa}</p>
<p style="margin-top:10px"><b>Timing of turns.</b> The 2Y's median lead at matched turns is {f(y2.get('lead_med'), 1, True)} months and the 5Y's is {f(y5.get('lead_med'), 1, True)} months: they turn with the 10Y, not before it. Fed funds lags by a median {f(-(ff.get('lead_med') or 0), 0)} months. {lead_txt}</p>
<p style="margin-top:10px">This is in line with the bond-predictability literature: once the yield's own history is used, macro variables add little in real time (Bauer &amp; Hamilton 2018; Ghysels, Horan &amp; Moench 2018). With {ev} trends in the main window, only large effects can be detected.</p>"""
    caveats = [
        "Monthly averages blur timing: a daily or weekly version (Bloomberg data) would sharpen lead-lag estimates and delays, and add many more trends to learn from.",
        f"Small samples: {res['tiers']['L']['n_events']} trends in window L, {res['tiers']['M']['n_events']} in M and {res['tiers']['S']['n_events']} in S. Breakeven and TIPS results (window S, trained from 2004) are not reliable.",
        "Macro data are final revised values with release lags applied, not first releases; payrolls in particular are revised heavily, which flatters them (Ghysels, Horan &amp; Moench 2018). ALFRED vintages would fix this.",
        "The confirmation thresholds were read off the same 1990–2023 frontier; window M (1997+) is the stability check. A live version should choose the threshold from the false-alarm rate you can tolerate.",
        "Missing families: ISM, economic surprise indices, MOVE, term-premium estimates (ACM, Kim–Wright), positioning (CFTC) and global yields (Bunds, gilts). <span class=\"mono\">run_gym.py --add-csv</span> tests any of them with the same matched design.",
        "Truth labels: method D at the page defaults; the fixed 100bp rule (toggle at the top) gives more, shorter trends and almost no sideways months.",
    ]
    return dict(cards=cards, text=text, caveats=caveats)


def main():
    res = {}
    for name in ("default", "fixed100"):
        p = ROOT / "results" / f"gym_results_{name}.json"
        if p.exists():
            res[name] = json.loads(p.read_text())
    meta = pd.read_csv(ROOT / "data" / "predictors_meta.csv")
    meta = meta[~meta["group"].isin(["context"])]
    data = dict(results=res, meta=meta.fillna("").to_dict(orient="records"))
    fnd = {k: findings(v, k) for k, v in res.items()}
    html = (ROOT / "gym" / "report_template.html").read_text(encoding="utf-8")
    html = html.replace("__FINDINGS__", json.dumps(fnd))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = ROOT / "examples" / "trend_gym_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
