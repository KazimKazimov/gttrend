"""
Tests for the trend-detection gym.   Run:  python -m pytest -q tests/test_gym.py

The key property is "no look-ahead": a detector evaluated on data truncated at month T must produce
exactly the same forecasts, for every month up to T, as the same detector run on the full panel.
"""
import pathlib, sys, warnings
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gym import (Panel, TrendGym, TrendEnv, DetectorAgent, RuleDetector, CompositeRule, LogitDetector,
                 MSDetector, Climatology, score)
from gym import labels as L
from gym.features import FeatureStore
from gym.env import Result
from gym.detectors import onehot

warnings.simplefilter("ignore")
P = Panel.load()
CUT = "2008-12-01"


def truncated(panel, date):
    return Panel(panel.raw.loc[:date].copy(), panel.meta.copy(), panel.target)


def test_realtime_frame_applies_publication_lags():
    rt = P.realtime()
    assert rt.loc["2020-05-01", "unrate"] == P.raw.loc["2020-04-01", "unrate"]      # lag 1
    assert rt.loc["2020-05-01", "y2"] == P.raw.loc["2020-05-01", "y2"]              # lag 0


def test_features_are_causal():
    full = FeatureStore(P)
    cut = FeatureStore(truncated(P, CUT))
    xs = ["y2", "ff", "clev10", "core_cpi", "unrate", "vix", "usd"]
    a = full.matrix(xs, compact=False).loc[:CUT]
    b = cut.matrix(xs, compact=False)
    pd.testing.assert_frame_equal(a, b.reindex(a.index), check_exact=False, atol=1e-10)


def test_realtime_labels_are_causal():
    y = P.y.values
    T = 600
    np.testing.assert_array_equal(L.realtime(y[: T + 1]), L.realtime(y)[: T + 1])


@pytest.mark.parametrize("make", [
    lambda: RuleDetector(),
    lambda: CompositeRule("y2"),
    lambda: LogitDetector(["y2", "core_cpi"]),
    lambda: MSDetector(obs=["y2"], tvtp=["ff"]),
])
def test_no_look_ahead(make):
    full = TrendGym(P, horizons=(0, 3), eval_start="1995-01-01")
    cut = TrendGym(truncated(P, CUT), horizons=(0, 3), eval_start="1995-01-01")
    ra, rb = full.run(make()), cut.run(make())
    n = len(rb.dates)
    assert n > 100
    for h in (0, 3):
        np.testing.assert_allclose(ra.probs[h][:n], rb.probs[h], atol=1e-8)


def test_metrics_perfect_and_lagged_detectors():
    g = TrendGym(P, horizons=(0,), eval_start="1990-01-01")
    idx = np.arange(g.e0, g.e1 + 1)
    perfect = Result("perfect", g.dates[idx], idx, {0: onehot(g.truth[idx])})
    s = score(g, perfect, 0)
    assert s["acc"] == 1 and s["bal_acc"] == 1 and s["delay_med"] == 0 and s["false_alarms_10y"] == 0
    lagged = Result("lag3", g.dates[idx], idx, {0: onehot(g.truth[idx - 3])})
    s3 = score(g, lagged, 0)
    assert s3["delay_med"] == 3 and s3["hit_rate"] == 1


def test_env_matches_batch_runner():
    g = TrendGym(P, horizons=(0, 3), eval_start="2005-01-01", eval_end="2012-12-01")
    det = LogitDetector(["y2"])
    batch = g.run(det)
    env = TrendEnv(g)
    agent = DetectorAgent(LogitDetector(["y2"]), refit_every=g.refit_every)
    obs, done = env.reset(), False
    while not done:
        assert obs.data.index.max() == obs.date            # the agent never sees later rows
        obs, _, done, _ = env.step(agent.act(obs))
    res = env.result()
    for h in (0, 3):
        np.testing.assert_allclose(res.probs[h], batch.probs[h], atol=1e-8)


def test_vintage_labels_use_only_data_up_to_T():
    g = TrendGym(P, horizons=(0,), eval_start="2000-01-01")
    T = g.e0 + 50
    v = g.vintage(T)
    assert len(v) == T + 1
    np.testing.assert_array_equal(v, L.truth(g.y[: T + 1], "D", g.settings))
