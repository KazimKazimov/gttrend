"""
The trend-detection gym.

TrendGym walks forward month by month. At each refit date T (every `refit_every` months) a detector is
trained on data up to the end of month T, with labels computed from y[0..T] only (the "vintage" ex-post
labels a researcher would have had at T). The frozen model then produces, at the end of each month t of
the next block, probabilities for the trend state at t + h for every horizon h. Nothing after month t is
visible to it: features are causal, publication lags are applied, and labels are truncated at T.

Scores compare those forecasts with the final ex-post labels ("truth") on months whose labels are
settled (before the last, still-open segment).

TrendEnv wraps the same machinery in a gym-style reset/step loop for custom agents: each observation
carries only the data known at that month-end.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import labels as L
from .data import Panel
from .features import FeatureStore


@dataclass
class Result:
    name: str
    dates: pd.DatetimeIndex                 # forecast origins (end of month t)
    t_index: np.ndarray                     # gym index of each origin
    probs: dict                             # h -> (n, 3) array: P(down), P(side), P(up) at t + h
    meta: dict = field(default_factory=dict)

    def frame(self):
        cols = {}
        for h, p in self.probs.items():
            for j, c in enumerate(["down", "side", "up"]):
                cols[f"p_{c}_h{h}"] = p[:, j]
        return pd.DataFrame(cols, index=self.dates)


class TrendGym:
    def __init__(self, panel: Panel, label_method="D", settings="default", horizons=(0, 3),
                 eval_start="1997-01-01", eval_end=None, refit_every=12):
        self.panel = panel
        self.settings = L.SETTINGS[settings] if isinstance(settings, str) else settings
        self.settings_name = settings if isinstance(settings, str) else "custom"
        self.label_method = label_method
        self.horizons = tuple(horizons)
        self.store = FeatureStore(panel, self.settings)
        self.dates = self.store.index
        self.y = self.store.y.values.astype(float)
        self.y_eom = panel.raw["y10_eom"].reindex(self.dates).values if "y10_eom" in panel.raw else self.y
        self.truth = L.truth(self.y, label_method, self.settings)
        self.settled = L.settled_end(self.y, label_method, self.settings)      # first provisional month
        self.e0 = int(np.searchsorted(self.dates, pd.Timestamp(eval_start)))
        last = self.settled - 1
        if eval_end is not None:
            last = min(last, int(np.searchsorted(self.dates, pd.Timestamp(eval_end), side="right")) - 1)
        self.e1 = last
        self.refit_every = refit_every
        self._vint = {}

    # labels a model could have trained on at the end of month T
    def vintage(self, T):
        if T not in self._vint:
            self._vint[T] = L.vintage(self.y, T, self.label_method, self.settings)
        return self._vint[T]

    def refit_dates(self):
        T = self.e0 - 1
        out = []
        while T < self.e1:
            out.append(T)
            T += self.refit_every
        return out

    def run(self, det, verbose=False) -> Result:
        n = self.e1 - self.e0 + 1
        probs = {h: np.full((n, 3), np.nan) for h in self.horizons}
        for T in self.refit_dates():
            det.fit(self, T)
            t0, t1 = T + 1, min(T + self.refit_every, self.e1)
            p = det.predict(self, t0, t1)
            for h in self.horizons:
                probs[h][t0 - self.e0:t1 - self.e0 + 1] = p[h]
            if verbose:
                print(f"  {det.name}: fitted at {self.dates[T]:%Y-%m}, predicted {self.dates[t0]:%Y-%m}..{self.dates[t1]:%Y-%m}")
        idx = np.arange(self.e0, self.e1 + 1)
        return Result(det.name, self.dates[idx], idx, probs, dict(uses=getattr(det, "uses", ())))


# ------------------------------------------------------------------------------------------ gym-style env
@dataclass
class Observation:
    t: int                        # gym index of the current month-end
    date: pd.Timestamp
    data: pd.DataFrame            # real-time panel up to this month (publication lags applied)
    features: pd.DataFrame        # causal feature matrix up to this month (own + all X)
    gym: TrendGym

    def labels_known(self):
        """Ex-post labels computed from data up to now (what you could train on)."""
        return self.gym.vintage(self.t)


class TrendEnv:
    """reset() -> Observation; step({h: probs}) -> (Observation | None, reward, done, info).
    reward is always 0: the truth for month t is only settled later; call score() at the end."""

    def __init__(self, gym: TrendGym, xvars=None):
        self.gym = gym
        xvars = xvars if xvars is not None else [v for vs in gym.panel.groups().values() for v in vs]
        self._F = gym.store.matrix(xvars, True)
        self._rt = gym.panel.realtime()

    def _obs(self):
        g, t = self.gym, self.t
        d = g.dates[t]
        return Observation(t, d, self._rt.loc[:d], self._F.iloc[: t + 1], g)

    def reset(self):
        self.t = self.gym.e0
        self._probs = {h: [] for h in self.gym.horizons}
        return self._obs()

    def step(self, action):
        for h in self.gym.horizons:
            p = np.asarray(action[h], float)
            self._probs[h].append(p / p.sum())
        self.t += 1
        done = self.t > self.gym.e1
        return (None if done else self._obs()), 0.0, done, {}

    def result(self, name="agent"):
        g = self.gym
        idx = np.arange(g.e0, g.e0 + len(self._probs[g.horizons[0]]))
        return Result(name, g.dates[idx], idx, {h: np.array(v) for h, v in self._probs.items()})


class DetectorAgent:
    """Runs any Detector inside TrendEnv (refits every `refit_every` months, like TrendGym.run)."""

    def __init__(self, det, refit_every=12):
        self.det, self.k = det, refit_every

    def act(self, obs: Observation):
        g = obs.gym
        if not hasattr(self, "_next") or obs.t >= self._next:
            self.det.fit(g, obs.t - 1)
            self._next = obs.t + self.k
        p = self.det.predict(g, obs.t, obs.t)
        return {h: p[h][0] for h in g.horizons}
