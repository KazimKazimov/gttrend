"""
Tests for trend_algos.py.   Run:  python -m pytest -q

The parity test also runs the page's JavaScript (algos.js) under Node, if Node is installed,
and checks that all eleven methods give identical monthly labels in Python and in the page.
"""
import json, pathlib, shutil, subprocess
import numpy as np
import pandas as pd
import pytest

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trend_algos import (Rule, state_machine, retrace_machine, hier_merge, run_all,
                         segments_to_labels, UP, DOWN, SIDE)

Y = pd.read_csv(ROOT / "data" / "ust10_monthly.csv").Rate.values.astype(float)


def zigzag():
    """5 -> 3 (8m), -> 4 (3m), -> 1 (10m), then flat: a -2 / +1 / -3 path."""
    y = [5 - 2 * i / 8 for i in range(9)] + [3 + i / 3 for i in range(1, 4)] + [4 - 3 * i / 10 for i in range(1, 11)]
    return np.array(y + [1 + 0.1 * np.sin(i) for i in range(8)])


def trends(segs):
    return [s["label"] for s in segs if s["label"] != SIDE]


def test_plain_machine_splits_the_correction():
    segs, _, _ = state_machine(zigzag(), 1.0)
    assert trends(segs) == [DOWN, UP, DOWN]


def test_retracement_methods_keep_one_trend():
    y = zigzag()
    segs, _, _ = retrace_machine(y, 1.0)
    assert trends(segs) == [DOWN]
    assert trends(hier_merge(y, 1.0)) == [DOWN]


def test_scaled_rule_needs_more_for_slow_moves():
    R = Rule(Y, 1.0, scaled=True, z=2, cap=12)
    n = len(Y) - 1
    assert R.req(n - 3, n) == pytest.approx(1.0)        # 3 months: the 100bp floor binds
    assert R.req(n - 12, n) > 1.2                      # 12 months: needs more than 100bp
    assert R.req(n - 24, n) == pytest.approx(R.req(n - 12, n))   # capped at 12 months


def test_min_length_blocks_short_trends():
    R = Rule(Y, 0.5, min_len=6)
    segs, _, _ = state_machine(Y, R)
    assert all(s["end"] - s["start"] >= 6 for s in segs[:-1] if s["label"] != SIDE)


@pytest.mark.parametrize("cfg", [
    dict(thr=1.0, scaled=False, min_len=0),
    dict(thr=1.0, scaled=True, z=2.0, min_len=3),
    dict(thr=0.5, scaled=True, z=1.5, min_len=6, sigma="full", cap=6),
])
def test_ex_post_segments_obey_the_rule(cfg):
    labels, segs = run_all(Y, **cfg, l1_levels=Y)     # l1_levels=Y skips the slow real-time L1 fit
    R = Rule(Y, cfg["thr"], scaled=cfg.get("scaled", False), z=cfg.get("z", 2), min_len=cfg.get("min_len", 0),
             sigma=cfg.get("sigma", "rolling"), cap=cfg.get("cap", 12))
    for k, ss in segs.items():
        for s in ss:
            if not s.get("open"):
                assert R.lab(Y, s["start"], s["end"]) == s["label"], (k, s)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_python_matches_page_javascript():
    cfg = dict(thr=1.0, scaled=True, z=2.0, min_len=3, sigma="rolling", cap=12, rho=0.618, floor=1.0)
    js = f"""
    const TA = require({json.dumps(str(ROOT / 'algos.js'))});
    const fs = require('fs');
    const rows = fs.readFileSync({json.dumps(str(ROOT / 'data' / 'ust10_monthly.csv'))}, 'utf8').trim().split('\\n').slice(1);
    const dates = rows.map(r => r.split(',')[0]), y = rows.map(r => +r.split(',')[1]);
    const q = TA.compute(dates, y, 1, {{rho: 0.618, floor: 1, lam: 1000, l1lam: 5, scaled: true, z: 2, sigma: 'rolling', cap: 12, minLen: 3}});
    process.stdout.write(JSON.stringify(q.labels));
    """
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    js_labels = json.loads(out)
    py_labels, _ = run_all(Y, **cfg, kalman_lam=1000.0, l1_lam=5.0)
    for k, lab in py_labels.items():
        assert (np.array(js_labels[k]) == lab).all(), k
