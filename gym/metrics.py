"""
Accuracy measures for real-time trend detection.

Month-level (is the state right, and are the probabilities good?)
    acc, bal_acc          share of months classified correctly / mean recall over the 3 states
    macro_f1, kappa       class-balanced F1 and Cohen's kappa
    brier, bss            multi-class Brier score (QPS) and its skill vs. the sample climatology
    logscore              mean -log p(true state)
    auc_up, auc_down      ROC area for "up" and "down" vs the rest (Berge & Jorda 2011)

Event-level (turning points of the ex-post truth)
    n_events              true up/down trends that start inside the scoring window
    hit_rate              share detected before the trend ended
    delay_med, delay_mean months from the trend's first month to the first correct call
                          (negative = anticipated, possible for h > 0)
    within3, within6      share detected within 3 / 6 months of the start
    capture               share of the trend's total move still ahead at detection (1 = at the start)
    false_alarms_10y      called trend episodes with no overlap with a true same-direction trend, per 10 years
    precision             share of called trend episodes that overlap a true same-direction trend
    flips_yr              changes of the called state per year (truth has truth_flips_yr)

Economic value (nowcasts only): long duration when the call is "down", short when "up", flat when
sideways; P&L in bp of 10Y yield from month-end to month-end.
    pnl_bp_yr, sharpe, hit

compare(a, b): paired differences with a Diebold-Mariano test on the Brier loss (HAC variance,
Harvey-Leybourne-Newbold correction), a moving-block bootstrap interval for the balanced-accuracy
difference, and a Wilcoxon signed-rank test on per-event delays.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

from . import labels as L

CLS = np.array([-1, 0, 1])


def _idx(lab):
    return np.searchsorted(CLS, lab)


def calls(p):
    """argmax, with ties resolved towards 'sideways'."""
    q = p.copy()
    q[:, 1] += 1e-9
    return CLS[np.argmax(q, axis=1)]


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return np.nan
    r = stats.rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def month_metrics(p, truth):
    ok = ~np.isnan(p).any(1)
    p, truth = p[ok], truth[ok]
    c = calls(p)
    o = np.zeros_like(p)
    o[np.arange(len(truth)), _idx(truth)] = 1
    brier_t = ((p - o) ** 2).sum(1)
    clim = o.mean(0)
    brier_clim = ((clim - o) ** 2).sum(1).mean()
    rec = [np.mean(c[truth == k] == k) for k in CLS if (truth == k).any()]
    f1 = []
    for k in CLS:
        tp = np.sum((c == k) & (truth == k))
        pp, ap = np.sum(c == k), np.sum(truth == k)
        if ap == 0:
            continue
        prec = tp / pp if pp else 0.0
        r = tp / ap
        f1.append(0 if prec + r == 0 else 2 * prec * r / (prec + r))
    po = np.mean(c == truth)
    pe = sum(np.mean(c == k) * np.mean(truth == k) for k in CLS)
    return dict(n_months=int(len(truth)), acc=po, bal_acc=float(np.mean(rec)), macro_f1=float(np.mean(f1)),
                kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan,
                brier=brier_t.mean(), bss=1 - brier_t.mean() / brier_clim,
                logscore=float(-np.log(np.clip(p[np.arange(len(truth)), _idx(truth)], 1e-6, 1)).mean()),
                auc_up=auc(p[:, 2], truth == 1), auc_down=auc(p[:, 0], truth == -1)), brier_t


def episodes(c):
    """Runs of a called trend state: list of (start, end, dir) over positions of c."""
    return [(s, e, d) for s, e, d in L.segments(c) if d != 0]


def event_metrics(gym, res, h, calls_h):
    """calls_h[i] is the call made at origin t = res.t_index[i] for month t + h."""
    e0, e1 = gym.e0, gym.e1
    truth = gym.truth
    tgt = res.t_index + h                                     # target month of each call
    valid = tgt <= e1
    call_at = {int(t): c for t, c, v in zip(tgt, calls_h, valid) if v}
    ev = L.trend_events(truth, e0, e1)
    delays, caps = [], []
    for s, e, d in ev:
        hit = None
        for m in range(max(s, e0 + h), e + 1):                # target months inside the trend; origin = m - h
            if call_at.get(m) == d:
                hit = m
                break
        if hit is None:
            delays.append(np.nan)
            caps.append(0.0)
            continue
        origin = hit - h
        delays.append(origin - s)
        y0, ye = gym.y[s - 1], gym.y[e]                       # level at the turning point and at the end
        yd = gym.y[max(origin, s - 1)]
        caps.append(float(np.clip((ye - yd) / (ye - y0), 0, 1)) if ye != y0 else 1.0)
    d = np.array(delays, float)
    # false alarms: called episodes (in target-month time) that never overlap a true same-direction trend
    months = sorted(call_at)
    cseq = np.array([call_at[m] for m in months])
    fa, eps = 0, episodes(cseq)
    for s, e, dd in eps:
        m_s, m_e = months[s], months[e]
        if not np.any(truth[m_s:m_e + 1] == dd):
            fa += 1
    years = len(months) / 12
    flips = np.sum(cseq[1:] != cseq[:-1]) / years if years else np.nan
    tr = truth[months[0]:months[-1] + 1]
    return dict(n_events=len(ev), hit_rate=float(np.mean(~np.isnan(d))) if len(d) else np.nan,
                delay_med=float(np.nanmedian(d)) if np.any(~np.isnan(d)) else np.nan,
                delay_mean=float(np.nanmean(d)) if np.any(~np.isnan(d)) else np.nan,
                within3=float(np.mean(np.nan_to_num(d, nan=99) <= 3)) if len(d) else np.nan,
                within6=float(np.mean(np.nan_to_num(d, nan=99) <= 6)) if len(d) else np.nan,
                capture=float(np.mean(caps)) if caps else np.nan,
                false_alarms_10y=10 * fa / years if years else np.nan,
                precision=(len(eps) - fa) / len(eps) if eps else np.nan,
                flips_yr=flips, truth_flips_yr=np.sum(tr[1:] != tr[:-1]) / years), d


def pnl_metrics(gym, res, calls0):
    """Trade the nowcast at month-end t, hold to month-end t+1."""
    t = res.t_index
    ok = t + 1 < len(gym.y_eom)
    t, c = t[ok], calls0[ok]
    pos = -c                                                  # down -> long duration (+1), up -> short (-1)
    dy = gym.y_eom[t + 1] - gym.y_eom[t]
    pnl = pos * (-dy) * 100
    good = ~np.isnan(pnl)
    pnl = pnl[good]
    active = pos[good] != 0
    sd = pnl.std(ddof=1)
    return dict(pnl_bp_yr=12 * pnl.mean(), sharpe=np.sqrt(12) * pnl.mean() / sd if sd > 0 else np.nan,
                hit=float(np.mean(pnl[active] > 0)) if active.any() else np.nan, exposure=float(active.mean()))


def score(gym, res, h=0, with_series=False):
    p = res.probs[h]
    tgt = res.t_index + h
    keep = tgt <= gym.e1
    truth = gym.truth[tgt[keep]]
    m, brier_t = month_metrics(p[keep], truth)
    c = calls(np.nan_to_num(p, nan=1 / 3))
    e, delays = event_metrics(gym, res, h, c)
    out = dict(model=res.name, h=h, **m, **e)
    if h == 0:
        out.update(pnl_metrics(gym, res, c))
    if with_series:
        return out, dict(brier_t=brier_t, correct=(calls(p[keep]) == truth), delays=delays, truth=truth,
                         calls=calls(p[keep]))
    return out


def dm_test(la, lb, h=0):
    """Diebold-Mariano on loss series la - lb (negative mean = a better), HAC (Bartlett) variance with
    lag h + 11, Harvey-Leybourne-Newbold small-sample correction, t distribution."""
    d = np.asarray(la) - np.asarray(lb)
    d = d[~np.isnan(d)]
    n = len(d)
    lag = min(h + 11, n - 1)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return d.mean(), np.nan
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * (h + 1) + (h + 1) * h / n) / n)
    stat = dm * hln
    return d.mean(), 2 * stats.t.sf(abs(stat), df=n - 1)


def block_bootstrap_diff(ca, cb, truth, B=1000, block=24, seed=0):
    """CI for the difference in balanced accuracy (a - b) by moving-block bootstrap over months."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    nb = int(np.ceil(n / block))

    def bal(c, tr):
        r = [np.mean(c[tr == k] == k) for k in CLS if (tr == k).any()]
        return np.mean(r)

    diffs = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        ix = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[b] = bal(ca[ix], truth[ix]) - bal(cb[ix], truth[ix])
    return np.percentile(diffs, [5, 95]), float(np.mean(diffs <= 0))


def compare(gym, ra, rb, h=0, B=1000):
    """Paired comparison of result a vs result b on the same months."""
    sa, xa = score(gym, ra, h, with_series=True)
    sb, xb = score(gym, rb, h, with_series=True)
    dmean, p_dm = dm_test(xa["brier_t"], xb["brier_t"], h)
    ci, p_boot = block_bootstrap_diff(xa["calls"], xb["calls"], xa["truth"], B=B)
    da, db = xa["delays"], xb["delays"]
    both = ~np.isnan(da) & ~np.isnan(db)
    p_w = np.nan
    if both.sum() >= 5 and np.any(da[both] != db[both]):
        p_w = stats.wilcoxon(da[both], db[both]).pvalue
    keys = ["bal_acc", "bss", "brier", "auc_up", "auc_down", "delay_med", "delay_mean", "hit_rate", "capture",
            "false_alarms_10y", "precision", "flips_yr", "pnl_bp_yr", "sharpe"]
    d = {f"d_{k}": (sa.get(k, np.nan) - sb.get(k, np.nan)) for k in keys}
    return dict(a=sa, b=sb, **d, dm_brier_diff=dmean, p_dm=p_dm, bal_acc_ci90=ci.tolist(), p_boot=p_boot,
                p_wilcoxon_delay=p_w)
