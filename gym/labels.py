"""
Trend labels for the gym, built on trend_algos.

truth(y)          ex-post labels on the full sample: what the detector is scored against
vintage(y, T)     ex-post labels computed from y[0..T] only: what a model could train on at time T
realtime(y)       the rule machine's own real-time state (causal: rt[t] uses y[0..t] only)
settled_end(...)  first month of the still-open final segment; months from there on are not scored
"""
from __future__ import annotations
import sys, pathlib
from functools import lru_cache
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_algos import Rule, retrace_machine, state_machine, hier_merge, segments_to_labels, SIDE  # noqa: E402

# page defaults (the interactive page's starting settings) and the original fixed 100bp rule
DEFAULT = dict(thr=1.0, scaled=True, z=2.0, min_len=3, sigma="rolling", cap=12, rho=0.618, floor=1.0)
FIXED100 = dict(thr=1.0, scaled=False, z=2.0, min_len=0, sigma="rolling", cap=12, rho=0.618, floor=1.0)
SETTINGS = {"default": DEFAULT, "fixed100": FIXED100}


def _rule(y, s):
    return Rule(y, s["thr"], scaled=s["scaled"], z=s["z"], min_len=s["min_len"], sigma=s["sigma"], cap=s["cap"])


def label(y, method="D", settings=DEFAULT):
    """Return (ex-post labels, segments, real-time labels or None)."""
    y = np.asarray(y, float)
    R = _rule(y, settings)
    if method == "D":
        segs, rt, _ = retrace_machine(y, R, settings["rho"], settings["floor"])
    elif method == "A":
        segs, rt, _ = state_machine(y, R)
    elif method == "G":
        segs, rt = hier_merge(y, R, settings["rho"], settings["floor"]), None
    else:
        raise ValueError(f"unknown label method {method!r}")
    return segments_to_labels(len(y), segs), segs, rt


def truth(y, method="D", settings=DEFAULT):
    return label(y, method, settings)[0]


def realtime(y, method="D", settings=DEFAULT):
    rt = label(y, "A" if method == "G" else method, settings)[2]
    return rt


def vintage(y, T, method="D", settings=DEFAULT):
    """Ex-post labels as they would have been computed at the end of month T (index into y)."""
    return label(np.asarray(y, float)[: T + 1], method, settings)[0]


def segments(lab):
    """Contiguous runs of a label vector: list of (start, end, label), inclusive, end = last month."""
    out, s = [], 0
    for t in range(1, len(lab) + 1):
        if t == len(lab) or lab[t] != lab[s]:
            out.append((s, t - 1, int(lab[s])))
            s = t
    return out


def settled_end(y, method="D", settings=DEFAULT):
    """Index of the first month of the open (last) ex-post segment. Months >= this are provisional."""
    _, segs, _ = label(y, method, settings)
    return int(segs[-1]["start"]) + 1 if len(segs) > 1 else len(y)


def trend_events(lab, lo=0, hi=None):
    """True trend segments (label != 0) whose first month lies in [lo, hi]. Returns (start, end, dir)
    with start = first month of the trend (month after the turning point)."""
    hi = len(lab) - 1 if hi is None else hi
    ev = []
    for s, e, d in segments(lab):
        if d != SIDE and lo <= s <= hi:
            ev.append((s, e, d))
    return ev
