"""
Panel of monthly series with publication lags, and the real-time view of it.

A row labelled month m holds the value for month m (monthly average or monthly statistic).
`pub_lag` = months after the end of m before the value is public. At the end of month t (the decision
time), the latest known value of a series with lag L is the one for month t - L, so the real-time
frame is the raw frame with each column shifted down by its lag. Everything downstream (features,
models) reads only the real-time frame, row t and earlier.
"""
from __future__ import annotations
import pathlib
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class Panel:
    raw: pd.DataFrame                      # DatetimeIndex (month start), raw levels
    meta: pd.DataFrame                     # index = id; columns: name, group, kind, pub_lag, unit, ...
    target: str = "y10"
    _rt: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def load(cls, panel_path=None, meta_path=None, target="y10"):
        panel_path = panel_path or ROOT / "data" / "predictors_monthly.csv"
        meta_path = meta_path or ROOT / "data" / "predictors_meta.csv"
        raw = pd.read_csv(panel_path, parse_dates=["date"], index_col="date")
        meta = pd.read_csv(meta_path).set_index("id")
        return cls(raw, meta, target)

    @classmethod
    def from_frames(cls, raw: pd.DataFrame, meta: dict | pd.DataFrame, target="y10"):
        """Build from your own data. `meta` maps id -> dict(kind=..., pub_lag=..., group=..., name=...)."""
        if isinstance(meta, dict):
            meta = pd.DataFrame(meta).T
        meta = meta.copy()
        for c, d in [("group", "user"), ("name", None), ("pub_lag", 0), ("kind", "rate"), ("unit", "")]:
            if c not in meta:
                meta[c] = d
        meta["name"] = meta["name"].fillna(pd.Series(meta.index, index=meta.index))
        return cls(raw.sort_index(), meta, target)

    # ---------------------------------------------------------------- views
    def realtime(self) -> pd.DataFrame:
        """Raw frame with each column shifted by its publication lag (value known at end of month t)."""
        if self._rt is None:
            rt = {}
            for c in self.raw.columns:
                lag = int(self.meta.loc[c, "pub_lag"]) if c in self.meta.index else 0
                s = self.raw[c].shift(lag)
                # a missing release (e.g. Oct 2025 CPI, not collected during the shutdown) keeps the last
                # published value: forward-fill internal gaps of up to 2 months (causal)
                last = s.last_valid_index()
                if last is not None:
                    s.loc[:last] = s.loc[:last].ffill(limit=2)
                rt[c] = s
            self._rt = pd.DataFrame(rt, index=self.raw.index)
        return self._rt

    def level(self, col: str, frame: pd.DataFrame | None = None) -> pd.Series:
        """Series in feature units: rates in %, `yoy` kinds as YoY % change, prices as 100*log."""
        f = self.realtime() if frame is None else frame
        s = f[col].astype(float)
        kind = self.meta.loc[col, "kind"] if col in self.meta.index else "rate"
        if kind == "yoy":
            return 100.0 * (s / s.shift(12) - 1.0)
        if kind == "price":
            return 100.0 * np.log(s.where(s > 0))
        return s

    def info_set(self, t) -> pd.DataFrame:
        """Everything known at the end of month t (real-time frame, rows up to t)."""
        return self.realtime().loc[:t]

    @property
    def y(self) -> pd.Series:
        return self.raw[self.target].dropna()

    def groups(self, exclude=("target", "context")) -> dict:
        g = {}
        for vid, row in self.meta.iterrows():
            if row["group"] in exclude or vid == self.target or vid not in self.raw.columns:
                continue
            g.setdefault(row["group"], []).append(vid)
        return g

    def first_valid(self, col: str) -> pd.Timestamp:
        return self.realtime()[col].first_valid_index()
