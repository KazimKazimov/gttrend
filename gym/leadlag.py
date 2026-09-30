"""
Lead-lag of turning points (Harding & Pagan style): does variable X turn before the 10Y?

For each series we date ex-post trends with the same retracement machine (method D). The 10Y uses the
target settings; other series use a volatility-scaled rule (their own units). Then:

  * matched turns: for every 10Y trend start, the nearest X trend start in the same direction within
    +/- `window` months (for curve-like variables the same direction means X moves the same way);
    lead = 10Y start - X start (positive: X turned first)
  * concordance(k): share of months in which the 10Y's state equals X's state k months earlier
    (k > 0: X leads), and the lag with the highest concordance

`sign` flips the direction for variables expected to move opposite to yields (none by default: equities,
inflation, activity and breakevens usually rise with yields in trend phases; unemployment falls).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import labels as L
from .features import X_TREND

OPPOSITE = {"unrate", "baa10y", "vix", "usd_inverse"}


def xtruth(level: pd.Series, settings=X_TREND):
    s = level.dropna()
    lab = pd.Series(np.nan, index=level.index)
    if len(s) > 40:
        lab.loc[s.index] = L.truth(s.values, "D", settings)
    return lab


def turns(lab: pd.Series):
    """(date, direction) of each trend start in a label series."""
    v = lab.dropna()
    out = []
    for s, e, d in L.segments(v.values.astype(int)):
        if d != 0 and s > 0:
            out.append((v.index[s], d))
    return out


def lead_lag(panel, v, ylab: pd.Series, window=12, lags=range(-12, 13), start="1960-01-01", end=None):
    lvl = panel.level(v)
    sign = -1 if v in OPPOSITE else 1
    xl = xtruth(lvl) * sign
    xl = xl.reindex(ylab.index)
    yl = ylab.loc[start:end]
    xt = [(d, s) for d, s in turns(xl) if d >= pd.Timestamp(start)]
    yt = [(d, s) for d, s in turns(yl)]
    leads = []
    for d, s in yt:
        cands = [(abs((d - dx).days), (d.year - dx.year) * 12 + d.month - dx.month) for dx, sx in xt
                 if sx == s and abs((d.year - dx.year) * 12 + d.month - dx.month) <= window]
        leads.append(min(cands)[1] if cands else np.nan)
    leads = np.array(leads, float)
    conc = {}
    for k in lags:
        a = yl
        b = xl.shift(k).reindex(a.index)
        ok = a.notna() & b.notna()
        conc[k] = float((a[ok] == b[ok]).mean()) if ok.sum() > 60 else np.nan
    cs = pd.Series(conc)
    best = int(cs.idxmax()) if cs.notna().any() else None
    return dict(var=v, n_turns=len(yt), matched=int(np.sum(~np.isnan(leads))),
                match_rate=float(np.mean(~np.isnan(leads))) if len(leads) else np.nan,
                lead_med=float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else np.nan,
                lead_mean=float(np.nanmean(leads)) if np.any(~np.isnan(leads)) else np.nan,
                share_first=float(np.mean(leads[~np.isnan(leads)] > 0)) if np.any(~np.isnan(leads)) else np.nan,
                conc=cs.round(4).to_dict(), best_lag=best, conc_best=float(cs.max()) if cs.notna().any() else np.nan,
                conc_0=conc.get(0, np.nan), leads=[None if np.isnan(x) else int(x) for x in leads],
                turn_dates=[d.strftime("%Y-%m") for d, _ in yt])
