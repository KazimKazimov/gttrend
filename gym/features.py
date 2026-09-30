"""
Causal features. Every column at row t uses only rows <= t of the real-time frame, so a feature matrix
computed once on the full panel equals one computed on data truncated at any T (tests check this).

own features (the 10Y itself)
    d1 d3 d6 d12      change over k months, bp
    dd12 ru12         distance below the 12m high / above the 12m low, bp
    gap12             distance from the 12m average, bp
    vol12             std of monthly changes over 12m, bp
    rt_up rt_down     real-time state of the rule machine (method D, page defaults)
    rt_age            log(1 + months since the real-time state last changed)

X features, per variable v (level in feature units: %, YoY %, or 100*log)
    v_d1 v_d3 v_d6 v_d12    change over k months
    v_lvl                   level (only for yoy/rate kinds that are stationary-ish: inflation, unemployment, spreads)
    v_spr, v_spr_d12        spread 10Y - v and its 12m change (rate-like v: yields, fed funds, breakevens)
    v_up v_down v_age       v's own real-time trend state (rule machine with a volatility-scaled threshold)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import labels as L
from .data import Panel

LAGS = (1, 3, 6, 12)
SPREAD_VARS = {"y2", "ff", "y5", "y30", "be10", "fwd5y5y", "tips10", "clev10"}
LEVEL_VARS = {"cpi", "core_cpi", "core_pce", "unrate", "payems", "baa10y", "mich1", "clev10", "be10", "fwd5y5y", "vix"}
# trend state of X: pure volatility-scaled rule (floor 0), same z / cap / min length / retracement as the page
X_TREND = dict(thr=0.0, scaled=True, z=2.0, min_len=3, sigma="rolling", cap=12, rho=0.618, floor=1.0)


def _age(state):
    age = np.zeros(len(state))
    for t in range(1, len(state)):
        age[t] = age[t - 1] + 1 if state[t] == state[t - 1] else 0
    return np.log1p(age)


def own_features(y: pd.Series, settings=L.DEFAULT) -> pd.DataFrame:
    y = y.astype(float)
    v = y.values
    f = pd.DataFrame(index=y.index)
    for k in LAGS:
        f[f"d{k}"] = 100 * (y - y.shift(k))
    f["dd12"] = 100 * (y - y.rolling(12).max())
    f["ru12"] = 100 * (y - y.rolling(12).min())
    f["gap12"] = 100 * (y - y.rolling(12).mean())
    f["vol12"] = 100 * y.diff().rolling(12).std()
    vol36 = 100 * y.diff().rolling(36, min_periods=12).std()
    for k in (3, 12):
        f[f"z{k}"] = f[f"d{k}"] / (vol36 * np.sqrt(k))           # volatility-normalised changes
    f["ddz"] = f["dd12"] / vol36
    f["ruz"] = f["ru12"] / vol36
    rt = L.realtime(v, "D", settings)
    f["rt_up"] = (rt == 1).astype(float)
    f["rt_down"] = (rt == -1).astype(float)
    f["rt_age"] = _age(rt)
    return f


def causal_rule(v, settings):
    """trend_algos.Rule, but with a strictly causal volatility in the first 24 months (expanding window
    instead of the rule's fixed seed, which uses the first 24 changes)."""
    R = L._rule(v, settings)
    d = np.diff(v)
    for t in range(1, min(25, len(R.sig))):
        w = d[: min(t, len(d))]
        R.sig[t] = np.std(w, ddof=1) if len(w) >= 6 else np.nan
    return R


def trend_state(level: pd.Series, settings=X_TREND, warmup=12) -> pd.Series:
    """Real-time trend state of any series; NaN during its first `warmup` months."""
    s = level.dropna()
    out = pd.Series(np.nan, index=level.index)
    if len(s) <= warmup + 2:
        return out
    v = s.values.astype(float)
    R = causal_rule(v, settings)
    R.sig[np.isnan(R.sig)] = np.nanmax(R.sig[:warmup + 1]) if np.any(~np.isnan(R.sig[:warmup + 1])) else 1.0
    _, rt, _ = L.retrace_machine(v, R, settings["rho"], settings["floor"])
    rt = rt.astype(float)
    rt[:warmup] = np.nan
    out.loc[s.index] = rt
    return out


def x_features(panel: Panel, v: str, y: pd.Series, frame=None, trend=True) -> pd.DataFrame:
    lvl = panel.level(v, frame)
    f = pd.DataFrame(index=lvl.index)
    for k in LAGS:
        f[f"{v}_d{k}"] = lvl - lvl.shift(k)
    sd = lvl.diff().rolling(36, min_periods=12).std()
    for k in (3, 12):
        f[f"{v}_z{k}"] = f[f"{v}_d{k}"] / (sd * np.sqrt(k))        # volatility-normalised changes
    if v in LEVEL_VARS:
        f[f"{v}_lvl"] = lvl
    if v in SPREAD_VARS:
        spr = y.reindex(lvl.index) - lvl
        f[f"{v}_spr"] = spr
        f[f"{v}_spr_d12"] = spr - spr.shift(12)
    if trend:
        st = trend_state(lvl)
        f[f"{v}_up"] = (st == 1).astype(float).where(st.notna())
        f[f"{v}_down"] = (st == -1).astype(float).where(st.notna())
        age = pd.Series(np.nan, index=lvl.index)
        ok = st.dropna()
        if len(ok):
            age.loc[ok.index] = _age(ok.values)
        f[f"{v}_age"] = age
    return f


class FeatureStore:
    """Computes and caches own + X feature blocks for a panel, aligned to the target's dates."""

    def __init__(self, panel: Panel, settings=L.DEFAULT, frame=None):
        self.panel = panel
        self.settings = settings
        self.frame = panel.realtime() if frame is None else frame
        y = self.frame[panel.target].dropna()
        self.index = y.index
        self.y = y
        self._own = own_features(y, settings)
        self._x = {}

    def own(self):
        return self._own

    def x(self, v):
        if v not in self._x:
            self._x[v] = x_features(self.panel, v, self.y, self.frame).reindex(self.index)
        return self._x[v]

    COMPACT = ("_z3", "_z12", "_spr", "_up", "_down")

    def matrix(self, xvars=(), own=True, compact=False) -> pd.DataFrame:
        """own block + X blocks. compact=True keeps, per X, only its volatility-normalised 3m and 12m changes,
        its spread to the 10Y (rate-like X) and its trend state: fewer parameters for short samples."""
        blocks = [self._own] if own else []
        for v in xvars:
            b = self.x(v)
            if compact:
                b = b[[c for c in b.columns if c.endswith(self.COMPACT)]]
            blocks.append(b)
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=self.index)
