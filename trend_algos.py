"""
Trend classification for a yield series: UP / DOWN / SIDEWAYS  (mirror of the page's JavaScript).

The trend rule (Rule) decides what counts as a trend everywhere:
    a move from month a to month b is a trend if  |y[b] - y[a]| >= req(a, b)  and  b - a >= min_len
    fixed rule : req = thr
    scaled rule: req = max(thr, z * sigma[b] * sqrt(min(b - a, cap)))
sigma = volatility of monthly changes (trailing 36 months, or full sample). With the scaled rule,
100bp in 3 months can be a trend while 100bp over a year is not.

Ex-post (label history after the fact)          Real-time (data up to t only)
  A  breakout / reversal state machine             A_rt  same machine, no backdating
  B  optimal piecewise-linear fit (DP)             C     rolling 12m regression
  D  retracement-aware state machine               D_rt  same machine, no backdating
  E  Kalman local-linear trend (smoothed)          E_rt  D rule on the Kalman-filtered level
  F  L1 trend filter                               F_rt  D rule on a trailing-60m L1 endpoint
  G  hierarchical swing merging

Every function that takes `rule` also accepts a plain float threshold (fixed rule, no min length).
"""
from __future__ import annotations
import numpy as np

UP, SIDE, DOWN = 1, 0, -1
NAMES = {UP: "up", SIDE: "sideways", DOWN: "down"}
EPS = 1e-12


# ================================================================ trend rule
class Rule:
    def __init__(self, y, thr=1.0, scaled=False, z=2.0, min_len=0, sigma="rolling", win=36, cap=12):
        y = np.asarray(y, float); n = len(y)
        self.thr, self.scaled, self.z, self.min_len, self.cap, self.n = thr, scaled, z, min_len, cap, n
        d = np.diff(y)
        full, seed = np.std(d, ddof=1), np.std(d[:24], ddof=1)
        sig = np.empty(n + 1)
        for t in range(n + 1):
            if sigma == "full":
                sig[t] = full; continue
            e = min(t, n - 1); a = max(1, e - win + 1)
            sig[t] = np.std(d[a - 1:e], ddof=1) if e - a + 1 >= 24 else seed
        self.sig = sig

    def req(self, a, b):
        if not self.scaled:
            return self.thr
        return max(self.thr, self.z * self.sig[min(b, self.n)] * np.sqrt(min(max(b - a, 1), self.cap)))

    def ok(self, a, b):
        return b - a >= self.min_len

    def lab(self, y, a, b):
        if not self.ok(a, b):
            return SIDE
        mv, r = y[b] - y[a], self.req(a, b)
        return UP if mv >= r - EPS else DOWN if mv <= -r + EPS else SIDE

    def zscore(self, y, a, b):
        return abs(y[b] - y[a]) / (self.sig[min(b, self.n)] * np.sqrt(max(b - a, 1)))


def as_rule(y, rule):
    return rule if isinstance(rule, Rule) else Rule(y, float(rule))


def breakout(y, R, frm, t):
    """Best start s in [frm, t - min_len] whose move to t clears the rule, by margin move/req."""
    best, best_ratio = None, 1 - 1e-9
    for s in range(frm, t - max(R.min_len, 1) + 1):
        mv = y[t] - y[s]
        if mv == 0:
            continue
        ratio = abs(mv) / R.req(s, t)
        if ratio > best_ratio + 1e-12 or (best is None and ratio >= 1 - 1e-9):
            best, best_ratio = (s, int(np.sign(mv))), ratio
    return best


# ================================================================ helpers
def segments_to_labels(n, segs):
    lab = np.full(n, SIDE, dtype=int)
    for s in segs:
        lab[s["start"] + 1: s["end"] + 1] = s["label"]
    if segs:
        lab[0] = segs[0]["label"]
    return lab


def labels_to_segments(lab):
    segs, s = [], 0
    for t in range(1, len(lab) + 1):
        if t == len(lab) or lab[t] != lab[s]:
            segs.append(dict(start=s - 1 if s > 0 else 0, end=t - 1, label=int(lab[s]))); s = t
    return segs


def decorate(segs, y, dates, R=None):
    out = []
    for s in segs:
        a, b = s["start"], s["end"]
        row = {**s, "start_date": str(dates[a])[:10], "end_date": str(dates[b])[:10],
               "y0": float(y[a]), "y1": float(y[b]), "chg_bp": round(float(y[b] - y[a]) * 100), "months": int(b - a)}
        if R is not None:
            row["z"] = float(R.zscore(y, a, b)); row["req_bp"] = round(R.req(a, b) * 100)
        out.append(row)
    return out


def label_merge(y, segs, R):
    """Label each segment by the rule, merge same-label neighbours, repeat until stable."""
    while True:
        for s in segs:
            s["label"] = R.lab(y, s["start"], s["end"])
            s.setdefault("imp", abs(y[s["end"]] - y[s["start"]]))
        merged = [dict(segs[0])]
        for s in segs[1:]:
            if s["label"] == merged[-1]["label"]:
                merged[-1]["end"] = s["end"]; merged[-1]["imp"] = abs(y[s["end"]] - y[s["start"]])
            else:
                merged.append(dict(s))
        if len(merged) == len(segs):
            return merged
        segs = merged


def consistent(y, segs, R):
    """Machine ex-post segments must satisfy the rule over their full span (a burst that fades can
    fail it); closed segments are relabelled and merged, the open last segment keeps its state."""
    last, closed = segs[-1], segs[:-1]
    if not closed or all(R.lab(y, s["start"], s["end"]) == s["label"] for s in closed):
        return segs
    fixed = label_merge(y, [dict(s) for s in closed], R)
    if fixed[-1]["label"] == last["label"]:
        fixed[-1].update(end=last["end"], open=True, confirm=last["confirm"]); return fixed
    return fixed + [last]


# ================================================================ A: breakout / reversal state machine
def state_machine(y, rule=1.0, stall=6):
    """SIDEWAYS until some move into t clears the rule (trend backdated to its start). A trend ends at
    its extreme when the counter-move clears the rule (flip) or after `stall` months without a new extreme
    (sideways). Returns (segments, realtime_labels, final_state)."""
    y = np.asarray(y, float); R = as_rule(y, rule); n = len(y)
    segs, rt = [], np.full(n, SIDE, dtype=int)
    state, seg_start, ext, conf = SIDE, 0, 0, 0
    for t in range(1, n):
        if state == SIDE:
            b = breakout(y, R, seg_start, t)
            if b:
                if b[0] > seg_start:
                    segs.append(dict(start=seg_start, end=b[0], label=SIDE, confirm=t))
                state, seg_start, ext, conf = b[1], b[0], t, t
        else:
            d = state
            if d * (y[t] - y[ext]) >= 0:
                ext = t
            elif d * (y[ext] - y[t]) >= R.req(ext, t) - EPS and R.ok(ext, t):
                segs.append(dict(start=seg_start, end=ext, label=state, confirm=conf))
                state, seg_start, conf, ext = -d, ext, t, t
            elif t - ext >= stall:
                segs.append(dict(start=seg_start, end=ext, label=state, confirm=conf))
                state, seg_start = SIDE, ext
        rt[t] = state
    fin = dict(state=state, ext=ext, seg_start=seg_start)
    segs.append(dict(start=seg_start, end=n - 1, label=state, confirm=conf if state != SIDE else n - 1, open=True))
    return consistent(y, segs, R), rt, fin


# ================================================================ B: optimal piecewise-linear fit
def pwl_partition(y, penalty=None, min_len=6):
    """Optimal partition into linear pieces: min sum(SSE_k) + penalty * #pieces (O(n^2) DP)."""
    y = np.asarray(y, float); n = len(y); x = np.arange(n, dtype=float)
    c = lambda v: np.concatenate([[0], np.cumsum(v)])
    S1, Sx, Sxx, Sy, Sxy, Syy = c(np.ones(n)), c(x), c(x * x), c(y), c(x * y), c(y * y)

    def fit(i, j):
        m = S1[j] - S1[i]; sx = Sx[j] - Sx[i]; sxx = Sxx[j] - Sxx[i]
        sy = Sy[j] - Sy[i]; sxy = Sxy[j] - Sxy[i]; syy = Syy[j] - Syy[i]
        vx = sxx - sx * sx / m; cxy = sxy - sx * sy / m
        b = cxy / vx if vx > 0 else 0.0; a = (sy - b * sx) / m
        return max(syy - sy * sy / m - b * cxy, 0.0), a, b

    if penalty is None:
        penalty = 2.6 * (np.var(np.diff(y)) / 2) * np.log(n)
    F = np.full(n + 1, np.inf); F[0] = -penalty; arg = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        best, bi = np.inf, 0
        for i in range(0, j - min_len + 1):
            if np.isfinite(F[i]):
                cst = F[i] + fit(i, j)[0] + penalty
                if cst < best: best, bi = cst, i
        F[j], arg[j] = best, bi
    cuts, j = [], n
    while j > 0:
        i = arg[j]; cuts.append((i, j)); j = i
    cuts.reverse()
    pieces, fitted = [], np.empty(n)
    for i, j in cuts:
        _, a, b = fit(i, j); fitted[i:j] = a + b * np.arange(i, j)
        pieces.append(dict(start=i, end=j - 1, slope=b))
    return pieces, fitted, penalty


def pwl_segments(y, pieces, rule=1.0, snap=6):
    """B: breakpoints snapped to the local peak/trough, labelled by the rule, same-label pieces merged.
    The page scales the DP penalty with thr^2: penalty = base * thr**2 (base = the default penalty)."""
    y = np.asarray(y, float); R = as_rule(y, rule); n = len(y)
    bps = [0] + [p["end"] for p in pieces[:-1]] + [n - 1]
    for k in range(1, len(bps) - 1):
        b1, b2 = pieces[k - 1]["slope"], pieces[k]["slope"]
        lo_w, hi_w = max(bps[k - 1] + 1, bps[k] - snap), min(bps[k + 1] - 1, bps[k] + snap)
        if lo_w >= hi_w: continue
        w = y[lo_w:hi_w + 1]
        if b1 > 0 >= b2: bps[k] = lo_w + int(np.argmax(w))
        elif b1 < 0 <= b2: bps[k] = lo_w + int(np.argmin(w))
    return label_merge(y, [dict(start=bps[k], end=bps[k + 1]) for k in range(len(bps) - 1)], R)


# ================================================================ C: rolling regression (real time)
def rolling_regression(y, rule=1.0, L=12, min_run=3):
    y = np.asarray(y, float); R = as_rule(y, rule); n = len(y)
    min_run = max(min_run, R.min_len)
    lab = np.full(n, SIDE, dtype=int); xc = np.arange(L) - (L - 1) / 2; den = (xc ** 2).sum()
    for t in range(L - 1, n):
        w = y[t - L + 1:t + 1]; mv = (xc * (w - w.mean())).sum() / den * (L - 1); r = R.req(t - L + 1, t)
        lab[t] = UP if mv >= r else DOWN if mv <= -r else SIDE
    if min_run > 1:
        out = lab.copy(); s = 0
        for t in range(1, n + 1):
            if t == n or lab[t] != lab[s]:
                if t - s < min_run and s > 0: out[s:t] = out[s - 1]
                s = t
        lab = out
    return lab


# ================================================================ D: retracement-aware state machine
def retrace_machine(y, rule=1.0, rho=0.618, floor=1.0, stall=12, sig_from=None):
    """Like A, but a counter-move ends a trend only if it reaches max(floor*req, rho * last impulse).
    The impulse restarts at a swing point once a counter-move of >= req/2 is followed by a new extreme,
    so -2 / +1 / -3 is one 4pp decline. Returns (segments, realtime_labels, final_state)."""
    y = np.asarray(y, float); R = rule if isinstance(rule, Rule) else Rule(sig_from if sig_from is not None else y, float(rule))
    n = len(y); segs, rt = [], np.full(n, SIDE, dtype=int)
    state, seg_start, ext, conf, imp_top, rc = SIDE, 0, 0, 0, 0.0, 0
    for t in range(1, n):
        if state == SIDE:
            b = breakout(y, R, seg_start, t)
            if b:
                if b[0] > seg_start:
                    segs.append(dict(start=seg_start, end=b[0], label=SIDE, confirm=t))
                state, seg_start, ext, rc, conf, imp_top = b[1], b[0], t, t, t, y[b[0]]
        else:
            d = state
            if d * (y[t] - y[ext]) >= 0:
                if d * (y[ext] - y[rc]) >= R.req(ext, rc) / 2 - EPS:
                    imp_top = y[rc]
                ext = rc = t
            else:
                if d * (y[t] - y[rc]) < 0: rc = t
                counter, impulse = d * (y[ext] - y[t]), d * (y[ext] - imp_top)
                if counter >= max(floor * R.req(ext, t), rho * impulse) - EPS and R.ok(ext, t):
                    segs.append(dict(start=seg_start, end=ext, label=state, confirm=conf))
                    state, seg_start, conf, imp_top, ext, rc = -d, ext, t, y[ext], t, t
                elif t - ext >= stall:
                    segs.append(dict(start=seg_start, end=ext, label=state, confirm=conf))
                    if d * (y[ext] - y[rc]) >= floor * R.req(ext, rc) - EPS and R.ok(ext, rc):
                        state, seg_start, conf, imp_top = -d, ext, t, y[ext]
                        r2 = rc
                        for k in range(rc, t + 1):
                            if d * (y[k] - y[r2]) > 0: r2 = k
                        ext, rc = rc, r2
                    else:
                        state, seg_start = SIDE, ext
        rt[t] = state
    fin = dict(state=state, ext=ext, seg_start=seg_start, imp_top=imp_top)
    segs.append(dict(start=seg_start, end=n - 1, label=state, confirm=conf if state != SIDE else n - 1, open=True))
    return consistent(y, segs, R), rt, fin


# ================================================================ E: Kalman local-linear trend
def kalman_trend(y, lam=1000.0):
    """Level + slope, noise on slope only (= HP filter with lambda = 1/q). Filtered = real time, RTS-smoothed = ex-post."""
    y = np.asarray(y, float); n = len(y); q = 1.0 / lam
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.diag([0.0, q]); H = np.array([1.0, 0.0])
    x = np.array([y[0], 0.0]); P = np.eye(2) * 1e7
    xp, Pp, xf, Pf = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
    for t in range(n):
        if t > 0: x = F @ x; P = F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        K = P[:, 0] / (P[0, 0] + 1.0)
        x = x + K * (y[t] - x[0]); P = P - np.outer(K, H @ P)
        xf[t], Pf[t] = x, P
    xs = xf.copy()
    for t in range(n - 2, -1, -1):
        J = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1]); xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
    return dict(filt=xf[:, 0], fslope=xf[:, 1], smooth=xs[:, 0], sslope=xs[:, 1])


def absorb_retracements(y, segs, R, rho=0.618, floor=1.0):
    """Trend / counter / same-direction trend making a new extreme -> one trend, when the counter-move is
    below max(floor*req, rho * preceding impulse) or shorter than the minimum episode length."""
    segs = label_merge(y, [dict(s) for s in segs], R); changed = True
    while changed:
        changed = False
        for k in range(1, len(segs) - 1):
            a, b, c = segs[k - 1], segs[k], segs[k + 1]; d = a["label"]
            if d == SIDE or c["label"] != d or b["label"] == d: continue
            counter = -d * (y[b["end"]] - y[b["start"]])
            small = counter < max(floor * R.req(b["start"], b["end"]), rho * a["imp"]) - EPS or not R.ok(b["start"], b["end"])
            if d * (y[c["end"]] - y[a["end"]]) > 0 and small:
                segs[k - 1:k + 2] = [dict(start=a["start"], end=c["end"], label=d, imp=abs(y[c["end"]] - y[c["start"]]))]
                segs = label_merge(y, segs, R); changed = True; break
    return segs


def kalman_segments(y, kf, rule=1.0, rho=0.618, floor=1.0, snap=6):
    y = np.asarray(y, float); R = as_rule(y, rule); n = len(y); sl = kf["sslope"]; bps = [0]
    for t in range(1, n):
        if np.sign(sl[t]) != np.sign(sl[t - 1]) and sl[t] != 0: bps.append(t)
    bps.append(n - 1)
    for k in range(1, len(bps) - 1):
        lo_w, hi_w = max(bps[k - 1] + 1, bps[k] - snap), min(bps[k + 1] - 1, bps[k] + snap)
        if lo_w >= hi_w: continue
        w = y[lo_w:hi_w + 1]
        bps[k] = lo_w + int(np.argmax(w) if sl[bps[k] - 1] > 0 else np.argmin(w))
    u = sorted(set(bps))
    return absorb_retracements(y, [dict(start=u[k], end=u[k + 1]) for k in range(len(u) - 1)], R, rho, floor)


# ================================================================ F: L1 trend filtering
def l1_filter(y, lam=5.0, iters=3000, warm=None):
    """min 0.5||y-x||^2 + lam*||D2 x||_1 (Kim, Koh, Boyd & Gorinevsky 2009) via ADMM."""
    from scipy.linalg import cholesky_banded, cho_solve_banded
    y = np.asarray(y, float); n = len(y); m = n - 2; rho = max(lam, 1e-3)
    ab = np.zeros((3, n))
    for i in range(m):
        c = (1.0, -2.0, 1.0)
        for p in range(3):
            ab[2, i + p] += rho * c[p] * c[p]
            if p < 2: ab[1, i + p + 1] += rho * c[p] * c[p + 1]
            if p < 1: ab[0, i + p + 2] += rho * c[p] * c[p + 2]
    ab[2] += 1.0
    cb = cholesky_banded(ab); z = np.zeros(m); u = np.zeros(m)
    if warm is not None: z[:], u[:] = warm[0][:m], warm[1][:m]
    k = lam / rho
    for _ in range(iters):
        w = rho * (z - u); b = y.copy(); b[:m] += w; b[1:m + 1] -= 2 * w; b[2:] += w
        x = cho_solve_banded((cb, False), b)
        v = x[:-2] - 2 * x[1:-1] + x[2:] + u
        z = np.sign(v) * np.maximum(np.abs(v) - k, 0); u = v - z
    return x, (z, u)


def _slope_turns(x, tol=1e-4):
    bps, prev = [0], 0
    for t in range(1, len(x)):
        s = x[t] - x[t - 1]; sg = 0 if abs(s) < tol else int(np.sign(s))
        if sg != 0:
            if prev != 0 and sg != prev: bps.append(t - 1)
            prev = sg
    bps.append(len(x) - 1)
    return bps


def l1_segments(y, fit, rule=1.0, snap=6):
    y = np.asarray(y, float); R = as_rule(y, rule); bps = _slope_turns(fit)
    for k in range(1, len(bps) - 1):
        lo_w, hi_w = max(bps[k - 1] + 1, bps[k] - snap), min(bps[k + 1] - 1, bps[k] + snap)
        if lo_w >= hi_w: continue
        w = y[lo_w:hi_w + 1]
        bps[k] = lo_w + int(np.argmax(w) if fit[bps[k]] >= fit[bps[k - 1]] else np.argmin(w))
    u = sorted(set(bps))
    return label_merge(y, [dict(start=u[k], end=u[k + 1]) for k in range(len(u) - 1)], R)


def l1_realtime_levels(y, lam=5.0, W=60, iters=200):
    """Causal L1 level: at each month fit the trailing W months and keep the endpoint (warm-started)."""
    y = np.asarray(y, float); n = len(y); lev = y.copy(); warm = None
    for t in range(12, n):
        w = y[max(0, t - W + 1):t + 1]
        x, (z, u) = l1_filter(w, lam, iters if warm is not None else 3 * iters, warm if (warm is not None and len(w) == W) else None)
        lev[t] = x[-1]
        if len(w) == W: warm = (np.append(z[1:], 0.0), np.append(u[1:], 0.0))
    return lev


# ================================================================ G: hierarchical swing merging
def hier_merge(y, rule=1.0, rho=0.618, floor=1.0):
    """Start from every local turn; repeatedly delete the smallest interior swing b with
    |b| < max(floor*req(b), rho*min(|a|,|c|)) or shorter than min_len. First/last swings are kept."""
    y = np.asarray(y, float); R = as_rule(y, rule); n = len(y); piv = [0]
    for t in range(1, n - 1):
        a, b = y[t] - y[piv[-1]], y[t + 1] - y[t]
        if a * b < 0: piv.append(t)
        elif a == 0: piv[-1] = t
    piv.append(n - 1)
    P = [piv[0]]
    for p in piv[1:]:
        if len(P) >= 2 and (y[P[-1]] - y[P[-2]]) * (y[p] - y[P[-1]]) >= 0:
            P[-1] = p; continue
        P.append(p)
    piv = P
    while len(piv) > 3:
        best, best_size = -1, np.inf
        for k in range(1, len(piv) - 2):
            size = abs(y[piv[k + 1]] - y[piv[k]])
            nb = min(abs(y[piv[k]] - y[piv[k - 1]]), abs(y[piv[k + 2]] - y[piv[k + 1]]))
            small = size < max(floor * R.req(piv[k], piv[k + 1]), rho * nb) - EPS or not R.ok(piv[k], piv[k + 1])
            if small and size < best_size: best, best_size = k, size
        if best < 0: break
        del piv[best:best + 2]
    return label_merge(y, [dict(start=piv[k], end=piv[k + 1]) for k in range(len(piv) - 1)], R)


# ================================================================ run everything
def run_all(y, thr=1.0, scaled=False, z=2.0, min_len=0, sigma="rolling", cap=12,
            rho=0.618, floor=1.0, kalman_lam=1000.0, l1_lam=5.0, l1_levels=None):
    """All eleven labelings, same settings as the page. Returns (labels dict, ex-post segments dict)."""
    y = np.asarray(y, float); n = len(y)
    R = Rule(y, thr, scaled=scaled, z=z, min_len=min_len, sigma=sigma, cap=cap)
    sA, rtA, _ = state_machine(y, R)
    base = pwl_partition(y)[2]
    pieces, fitB, _ = pwl_partition(y, base * thr ** 2)
    sB = pwl_segments(y, pieces, R)
    labC = rolling_regression(y, R)
    sD, rtD, _ = retrace_machine(y, R, rho, floor)
    kf = kalman_trend(y, kalman_lam)
    sE = kalman_segments(y, kf, R, rho, floor)
    _, rtE, _ = retrace_machine(kf["filt"], R, rho, floor)
    fit, _ = l1_filter(y, l1_lam)
    sF = l1_segments(y, fit, R)
    lev = l1_levels if l1_levels is not None else l1_realtime_levels(y, l1_lam)
    _, rtF, _ = retrace_machine(lev, R, rho, floor)
    sG = hier_merge(y, R, rho, floor)
    segs = dict(A=sA, B=sB, D=sD, E=sE, F=sF, G=sG)
    labels = {k: segments_to_labels(n, v) for k, v in segs.items()}
    labels.update(A_rt=rtA, C=labC, D_rt=rtD, E_rt=rtE, F_rt=rtF)
    return labels, segs


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in (DOWN, SIDE, UP))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


if __name__ == "__main__":
    import pandas as pd, sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/ust10_monthly.csv", parse_dates=["Date"])
    labels, segs = run_all(df.Rate.values, thr=0.5, scaled=True, z=2, min_len=3)
    for s in decorate(segs["D"], df.Rate.values, df.Date.values)[-8:]:
        print(s["start_date"], s["end_date"], NAMES[s["label"]], s["chg_bp"], "bp", s["months"], "m")
