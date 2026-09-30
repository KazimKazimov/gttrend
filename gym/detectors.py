"""
Detectors: anything that, at the end of month t and using only information known then, outputs
probabilities for the trend state (down, sideways, up) at t + h.

Interface (used by TrendGym.run):
    fit(gym, T)                      train on data up to month index T (labels = gym.vintage(T))
    predict(gym, t0, t1) -> {h: (n,3)} probabilities for forecast origins t0..t1 (model frozen since T)

Implemented
    RuleDetector        real-time state of a rule machine on the 10Y (the earlier page's method D/A)
    CompositeRule       rule machine run on a blend of the 10Y and a beta-scaled X (e.g. the 2Y)
    LogitDetector       multinomial logit on lagged own + X features (regularisation chosen by time-series CV)
    GBMDetector         gradient-boosted trees on the same features
    MSDetector          3-state Markov-switching model: X as extra observables and/or driving transitions
    Climatology         training-sample class frequencies (reference)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from . import labels as L
from .features import trend_state
from .msm import IOHMM

CLASSES = np.array([-1, 0, 1])          # probability columns: down, sideways, up


def onehot(state):
    s = np.asarray(state, float)
    out = np.zeros((len(s), 3))
    for j, c in enumerate(CLASSES):
        out[:, j] = (s == c)
    out[np.isnan(s)] = np.nan
    return out


def full_proba(model, X):
    """predict_proba mapped onto the 3 classes (a class missing from training gets probability 0)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    cls = list(model.classes_) if not hasattr(model, "steps") else list(model.steps[-1][1].classes_)
    for j, c in enumerate(cls):
        out[:, list(CLASSES).index(c)] = p[:, j]
    return out


class Detector:
    name = "detector"
    uses = ()                     # variables used (for bookkeeping / matched training)
    train_from = None             # optional date: drop training rows before it (matched comparisons)

    def fit(self, gym, T):
        pass

    def predict(self, gym, t0, t1):
        raise NotImplementedError

    def first_row(self, gym):
        return 0 if self.train_from is None else int(np.searchsorted(gym.dates, pd.Timestamp(self.train_from)))


class Climatology(Detector):
    name = "Climatology"

    def fit(self, gym, T):
        lab = gym.vintage(T)[self.first_row(gym):]
        self.p = np.array([(lab == c).mean() for c in CLASSES])

    def predict(self, gym, t0, t1):
        return {h: np.tile(self.p, (t1 - t0 + 1, 1)) for h in gym.horizons}


class RuleDetector(Detector):
    """Real-time state of the rule machine on the target; ex-ante forecasts assume the state persists."""

    def __init__(self, method="D", settings=None, name=None):
        self.method, self.settings = method, settings
        self.name = name or f"Rule-{method}"

    def fit(self, gym, T):
        if not hasattr(self, "_rt"):
            self._rt = onehot(L.realtime(gym.y, self.method, self.settings or gym.settings))

    def predict(self, gym, t0, t1):
        return {h: self._rt[t0:t1 + 1] for h in gym.horizons}


class CompositeRule(Detector):
    """Rule machine on c_t = (1-w) * y_t + w * beta * x_t, beta = cov(dy, dx) / var(dx) estimated at each refit.
    If x turns before the 10Y, the blend turns earlier than the 10Y alone."""

    def __init__(self, xvar, w=0.5, name=None, settings=None):
        self.xvar, self.w, self.settings = xvar, w, settings
        self.uses = (xvar,)
        self.name = name or f"Rule+{xvar}"

    def fit(self, gym, T):
        x = gym.store.frame[self.xvar].reindex(gym.dates).values.astype(float)
        dy, dx = np.diff(gym.y[: T + 1]), np.diff(x[: T + 1])
        ok = ~np.isnan(dx)
        ok[: max(0, self.first_row(gym) - 1)] = False
        self.beta = np.cov(dy[ok], dx[ok])[0, 1] / np.var(dx[ok], ddof=1) if ok.sum() > 24 else 0.0
        c = (1 - self.w) * gym.y + self.w * self.beta * x
        st = np.full(len(c), np.nan)
        v = ~np.isnan(c)
        i0 = int(np.argmax(v))
        st[i0:] = L.realtime(c[i0:], "D", self.settings or gym.settings)
        self._p = onehot(st)

    def predict(self, gym, t0, t1):
        return {h: self._p[t0:t1 + 1] for h in gym.horizons}


class _Supervised(Detector):
    """Shared training loop for label-then-learn classifiers."""

    min_rows = 48

    def __init__(self, xvars=(), own=True, compact=True, name=None):
        self.xvars, self.own, self.compact = tuple(xvars), own, compact
        self.uses = self.xvars
        self.name = name or (self.__class__.__name__.replace("Detector", "") + ("+" + "+".join(xvars) if xvars else "-own"))

    def feature_start(self, gym):
        """First date at which all features are available (used to match a baseline's training sample)."""
        F = gym.store.matrix(self.xvars, self.own, self.compact)
        ok = ~F.isna().any(axis=1)
        return F.index[ok.values.argmax()] if ok.any() else None

    def make_model(self, X, y):
        raise NotImplementedError

    def fit(self, gym, T):
        F = gym.store.matrix(self.xvars, self.own, self.compact)
        self.cols = list(F.columns)
        lab = gym.vintage(T)
        r0 = self.first_row(gym)
        self.models, self.fallback = {}, {}
        for h in gym.horizons:
            rows = np.arange(r0, T - h + 1)
            Xi, yi = F.values[rows], lab[rows + h]
            ok = ~np.isnan(Xi).any(1)
            Xi, yi = Xi[ok], yi[ok]
            self.fallback[h] = np.array([(yi == c).mean() if len(yi) else 1 / 3 for c in CLASSES])
            if len(yi) < self.min_rows or len(np.unique(yi)) < 2:
                self.models[h] = None
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[h] = self.make_model(Xi, yi)
        self._F = F.values

    def predict(self, gym, t0, t1):
        X = self._F[t0:t1 + 1]
        ok = ~np.isnan(X).any(1)
        out = {}
        for h in gym.horizons:
            p = np.tile(self.fallback[h], (len(X), 1))
            if self.models[h] is not None and ok.any():
                p[ok] = full_proba(self.models[h], X[ok])
            out[h] = p
        return out


class LogitDetector(_Supervised):
    Cs = (0.01, 0.03, 0.1, 0.3, 1.0)

    def make_model(self, X, y):
        best, bestC = np.inf, 0.1
        if len(y) >= 150:
            for C in self.Cs:
                losses = []
                for tr, va in TimeSeriesSplit(n_splits=3).split(X):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                    m.fit(X[tr], y[tr])
                    p = full_proba(m, X[va])
                    losses.append(log_loss(y[va], np.clip(p, 1e-6, 1), labels=CLASSES))
                if losses and np.mean(losses) < best:
                    best, bestC = np.mean(losses), C
        self.C_ = bestC
        m = make_pipeline(StandardScaler(), LogisticRegression(C=bestC, max_iter=3000))
        return m.fit(X, y)


class GBMDetector(_Supervised):
    def make_model(self, X, y):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                           l2_regularization=1.0, random_state=0)
        return m.fit(X, y)


class MSDetector(Detector):
    """Markov-switching detector.
    obs:   variables whose monthly change is an extra observable with a state-specific mean
    tvtp:  variables whose 12-month change (standardised) drives the transition probabilities"""

    def __init__(self, obs=(), tvtp=(), switching_cov=True, l2=5.0, estimation="supervised", normalize=True, name=None):
        self.obs, self.tvtp, self.switching_cov, self.l2 = tuple(obs), tuple(tvtp), switching_cov, l2
        self.normalize = normalize              # divide each monthly change by its trailing 36m volatility (to t-1)
        self.estimation = estimation            # "supervised": states pinned to the ex-post labels; "em": free EM
        self.uses = tuple(dict.fromkeys(self.obs + self.tvtp))
        tag = ("+obs:" + ",".join(obs) if obs else "") + ("+tvtp:" + ",".join(tvtp) if tvtp else "")
        self.name = name or ("MS" + tag if tag else "MS")

    def _data(self, gym):
        if not hasattr(self, "_Z"):
            P, idx = gym.panel, gym.dates
            cols = [pd.Series(100 * np.diff(gym.y, prepend=np.nan), index=idx)]
            for v in self.obs:
                lv = P.level(v, gym.store.frame).reindex(idx)
                cols.append(lv.diff() * (100 if P.meta.loc[v, "kind"] == "rate" else 1))
            if self.normalize:
                cols = [c / c.rolling(36, min_periods=12).std().shift(1) for c in cols]
            self._Z = pd.concat(cols, axis=1).values
            U = [(P.level(v, gym.store.frame).reindex(idx) if v == "mm" else P.level(v, gym.store.frame).reindex(idx).diff(12))
                 for v in self.tvtp]
            self._U = pd.concat(U, axis=1).values if U else None
            valid = ~np.isnan(self._Z).any(1)
            if self._U is not None:
                valid &= ~np.isnan(self._U).any(1)
            # the filter needs an unbroken sample: start after the last missing row
            bad = np.where(~valid)[0]
            self._s0 = int(bad.max()) + 1 if len(bad) else 0
        return self._Z, self._U, self._s0

    def fit(self, gym, T):
        Z, U, s0 = self._data(gym)
        s0 = max(s0, self.first_row(gym))
        self.s0 = s0
        Zt = Z[s0:T + 1]
        self.zs = Zt.std(0)
        Ut = None
        if U is not None:
            self.um, self.us = U[s0:T + 1].mean(0), U[s0:T + 1].std(0)
            Ut = (U[s0:T + 1] - self.um) / self.us
        lab = gym.vintage(T)[s0:T + 1] + 1                          # -1,0,1 -> 0,1,2
        m = IOHMM(K=3, l2=self.l2, switching_cov=self.switching_cov)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.estimation == "supervised":
                m.fit_supervised(Zt / self.zs, Ut, lab)
            else:
                m.fit(Zt / self.zs, Ut, init_labels=lab)
        self.model = m

    def predict(self, gym, t0, t1):
        Z, U, _ = self._data(gym)
        Zt = Z[self.s0:t1 + 1] / self.zs
        Ut = None if U is None else (U[self.s0:t1 + 1] - self.um) / self.us
        alpha, A = self.model.filter(Zt, Ut)
        a = alpha[t0 - self.s0:]
        Aa = A[t0 - self.s0:]
        return {h: (a if h == 0 else self.model.ahead(a, Aa, h)) for h in gym.horizons}
