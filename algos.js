// JS port of trend_algos.py — runs in the page so the threshold can change live.
const TA = (() => {
  const UP = 1, SIDE = 0, DOWN = -1, EPS = 1e-12;

  // Trend rule. A move from a to b is a trend if |y[b]-y[a]| >= req(a,b) and b-a >= minLen, where
  //   fixed rule : req = thr
  //   scaled rule: req = max(thr, z * sigma[b] * sqrt(b-a))   (sigma = vol of monthly changes)
  function makeRule(y, thr, o = {}) {
    const n = y.length, scaled = !!o.scaled, z = o.z ?? 2, minLen = o.minLen ?? 0, win = o.sigWin ?? 36, cap = o.cap ?? 12;
    const d = []; for (let t = 1; t < n; t++) d.push(y[t] - y[t - 1]);
    const sd = a => { const m = a.reduce((p, q) => p + q, 0) / a.length; return Math.sqrt(a.reduce((p, q) => p + (q - m) ** 2, 0) / (a.length - 1)); };
    const sig = new Float64Array(n + 1), full = sd(d), seed = sd(d.slice(0, 24));
    for (let t = 0; t <= n; t++) {
      if (o.sigma === "full") { sig[t] = full; continue; }
      const e = Math.min(t, n - 1), a = Math.max(1, e - win + 1);
      sig[t] = e - a + 1 >= 24 ? sd(d.slice(a - 1, e)) : seed;
    }
    const req = (a, b) => scaled ? Math.max(thr, z * sig[Math.min(b, n)] * Math.sqrt(Math.min(Math.max(b - a, 1), cap))) : thr;
    const ok = (a, b) => b - a >= minLen;
    const lab = (a, b, yy = y) => { if (!ok(a, b)) return SIDE; const mv = yy[b] - yy[a], r = req(a, b);
      return mv >= r - EPS ? UP : mv <= -r + EPS ? DOWN : SIDE; };
    const zscore = (a, b) => Math.abs(y[b] - y[a]) / (sig[Math.min(b, n)] * Math.sqrt(Math.max(b - a, 1)));
    return { thr, scaled, z, minLen, cap, sig, req, ok, lab, zscore };
  }
  // breakout from a sideways box: best start s in [from, t-minLen] by margin (move / required move)
  function breakout(y, R, from, t) {
    let best = null, bestRatio = 1 - 1e-9;
    for (let s = from; s <= t - Math.max(R.minLen, 1); s++) {
      const mv = y[t] - y[s]; if (mv === 0) continue;
      const ratio = Math.abs(mv) / R.req(s, t);
      if (ratio > bestRatio + 1e-12 || (best === null && ratio >= 1 - 1e-9)) { if (ratio > bestRatio + 1e-12 || best === null) { best = { start: s, dir: Math.sign(mv) }; bestRatio = ratio; } }
    }
    return best;
  }

  function segsToLabels(n, segs) {
    const lab = new Array(n).fill(SIDE);
    for (const s of segs) for (let t = s.start + 1; t <= s.end; t++) lab[t] = s.label;
    if (segs.length) lab[0] = segs[0].label;
    return lab;
  }
  function labelsToSegs(lab) {
    const segs = []; let s = 0;
    for (let t = 1; t <= lab.length; t++)
      if (t === lab.length || lab[t] !== lab[s]) { segs.push({ start: s > 0 ? s - 1 : 0, end: t - 1, label: lab[s] }); s = t; }
    return segs;
  }
  function decorate(segs, y, dates) {
    return segs.map(s => ({ ...s, start_date: dates[s.start], end_date: dates[s.end], y0: y[s.start], y1: y[s.end],
      chg_bp: Math.round((y[s.end] - y[s.start]) * 100), months: s.end - s.start }));
  }

  // A: breakout / reversal state machine
  function stateMachine(y, R, stall = 6) {
    const n = y.length, segs = [], rt = new Array(n).fill(SIDE);
    let state = SIDE, segStart = 0, ext = 0, conf = 0;
    for (let t = 1; t < n; t++) {
      if (state === SIDE) {
        const b = breakout(y, R, segStart, t);
        if (b) {
          if (b.start > segStart) segs.push({ start: segStart, end: b.start, label: SIDE, confirm: t });
          state = b.dir; segStart = b.start; ext = t; conf = t;
        }
      } else {
        const d = state;
        if (d * (y[t] - y[ext]) >= 0) ext = t;
        else if (d * (y[ext] - y[t]) >= R.req(ext, t) - EPS && R.ok(ext, t)) {
          segs.push({ start: segStart, end: ext, label: state, confirm: conf });
          state = -d; segStart = ext; conf = t; ext = t;
        } else if (t - ext >= stall) {
          segs.push({ start: segStart, end: ext, label: state, confirm: conf });
          state = SIDE; segStart = ext;
        }
      }
      rt[t] = state;
    }
    const fin = { state, ext, segStart };
    segs.push({ start: segStart, end: n - 1, label: state, confirm: state !== SIDE ? conf : n - 1, open: true });
    return { segs, rt, fin };
  }

  // B: optimal piecewise-linear partition (threshold-independent), then threshold labelling
  function pwlPartition(y, penalty = null, minLen = 6) {
    const n = y.length, P = k => new Float64Array(n + 1);
    const S1 = P(), Sx = P(), Sxx = P(), Sy = P(), Sxy = P(), Syy = P();
    for (let i = 0; i < n; i++) {
      S1[i + 1] = S1[i] + 1; Sx[i + 1] = Sx[i] + i; Sxx[i + 1] = Sxx[i] + i * i;
      Sy[i + 1] = Sy[i] + y[i]; Sxy[i + 1] = Sxy[i] + i * y[i]; Syy[i + 1] = Syy[i] + y[i] * y[i];
    }
    const fit = (i, j) => {
      const m = S1[j] - S1[i], sx = Sx[j] - Sx[i], sxx = Sxx[j] - Sxx[i], sy = Sy[j] - Sy[i], sxy = Sxy[j] - Sxy[i], syy = Syy[j] - Syy[i];
      const vx = sxx - sx * sx / m, cxy = sxy - sx * sy / m, b = vx > 0 ? cxy / vx : 0, a = (sy - b * sx) / m;
      return [Math.max(syy - sy * sy / m - b * cxy, 0), a, b];
    };
    if (penalty == null) {
      const d = []; for (let i = 1; i < n; i++) d.push(y[i] - y[i - 1]);
      const mu = d.reduce((a, b) => a + b, 0) / d.length, v = d.reduce((a, b) => a + (b - mu) ** 2, 0) / d.length;
      penalty = 2.6 * (v / 2) * Math.log(n);
    }
    const F = new Float64Array(n + 1).fill(Infinity), arg = new Int32Array(n + 1); F[0] = -penalty;
    for (let j = minLen; j <= n; j++) {
      let best = Infinity, bi = 0;
      for (let i = 0; i <= j - minLen; i++) {
        if (!isFinite(F[i])) continue;
        const c = F[i] + fit(i, j)[0] + penalty;
        if (c < best) { best = c; bi = i; }
      }
      F[j] = best; arg[j] = bi;
    }
    const cuts = []; let j = n; while (j > 0) { const i = arg[j]; cuts.unshift([i, j]); j = i; }
    const fitted = new Array(n), pieces = [];
    for (const [i, j] of cuts) { const [, a, b] = fit(i, j); for (let x = i; x < j; x++) fitted[x] = a + b * x; pieces.push({ start: i, end: j - 1, slope: b }); }
    return { pieces, fitted, penalty };
  }
  function pwlSegments(y, part, R, snap = 6) {
    const n = y.length, pc = part.pieces;
    const bps = [0, ...pc.slice(0, -1).map(p => p.end), n - 1];
    for (let k = 1; k < bps.length - 1; k++) {
      const b1 = pc[k - 1].slope, b2 = pc[k].slope;
      const loW = Math.max(bps[k - 1] + 1, bps[k] - snap), hiW = Math.min(bps[k + 1] - 1, bps[k] + snap);
      if (loW >= hiW) continue;
      let best = loW;
      if (b1 > 0 && b2 <= 0) { for (let t = loW; t <= hiW; t++) if (y[t] > y[best]) best = t; bps[k] = best; }
      else if (b1 < 0 && b2 >= 0) { for (let t = loW; t <= hiW; t++) if (y[t] < y[best]) best = t; bps[k] = best; }
    }
    const segs = []; for (let k = 0; k < bps.length - 1; k++) segs.push({ start: bps[k], end: bps[k + 1] });
    return labelMerge(y, segs, R);
  }

  // C: rolling regression
  function rolling(y, R, L = 12, minRun = 3) {
    minRun = Math.max(minRun, R.minLen);
    const n = y.length; let lab = new Array(n).fill(SIDE);
    const xm = (L - 1) / 2; let den = 0; for (let x = 0; x < L; x++) den += (x - xm) ** 2;
    for (let t = L - 1; t < n; t++) {
      let wm = 0; for (let k = 0; k < L; k++) wm += y[t - L + 1 + k]; wm /= L;
      let num = 0; for (let k = 0; k < L; k++) num += (k - xm) * (y[t - L + 1 + k] - wm);
      const mv = num / den * (L - 1);
      const r = R.req(t - L + 1, t);
      lab[t] = mv >= r ? UP : mv <= -r ? DOWN : SIDE;
    }
    if (minRun > 1) {
      const out = lab.slice(); let s = 0;
      for (let t = 1; t <= n; t++) if (t === n || lab[t] !== lab[s]) { if (t - s < minRun && s > 0) for (let k = s; k < t; k++) out[k] = out[s - 1]; s = t; }
      lab = out;
    }
    return lab;
  }

  const mean = a => a.length ? a.reduce((p, q) => p + q, 0) / a.length : 0;
  function kappa(a, b) {
    const n = a.length; let po = 0; for (let i = 0; i < n; i++) po += a[i] === b[i];
    po /= n; let pe = 0;
    for (const c of [DOWN, SIDE, UP]) pe += (a.filter(v => v === c).length / n) * (b.filter(v => v === c).length / n);
    return pe < 1 ? (po - pe) / (1 - pe) : 1;
  }
  function summary(lab, segs) {
    const n = lab.length, share = c => lab.filter(v => v === c).length / n;
    return { share_up: share(UP), share_side: share(SIDE), share_down: share(DOWN), n_segments: segs.length,
      n_up: segs.filter(s => s.label === UP).length, n_down: segs.filter(s => s.label === DOWN).length,
      n_side: segs.filter(s => s.label === SIDE).length,
      avg_trend_months: mean(segs.filter(s => s.label !== SIDE).map(s => s.months)) };
  }
  function lags(ref, lab) {
    const out = [];
    for (const s of ref) { if (s.label === SIDE) continue; let hit = null;
      for (let t = s.start + 1; t <= s.end; t++) if (lab[t] === s.label) { hit = t - s.start; break; }
      out.push(hit); }
    const got = out.filter(v => v != null).sort((a, b) => a - b);
    const med = got.length ? (got.length % 2 ? got[(got.length - 1) / 2] : (got[got.length / 2 - 1] + got[got.length / 2]) / 2) : null;
    return { n: out.length, detected: got.length, median: med, mean: got.length ? mean(got) : null };
  }

  const partCache = {};
  // D: retracement-aware state machine.
  // A counter-move ends a trend only if it reaches max(floor*thr, rho * last impulse leg).
  // The impulse leg restarts at a swing point once a counter-move of >= thr/2 is followed by a new extreme.
  function retraceMachine(y, R, rho = 0.618, floor = 1, stall = 12) {
    const n = y.length, segs = [], rt = new Array(n).fill(SIDE);
    let state = SIDE, segStart = 0, ext = 0, conf = 0, impTop = 0, rc = 0, prevExt = 0;
    for (let t = 1; t < n; t++) {
      if (state === SIDE) {
        const b = breakout(y, R, segStart, t);
        if (b) {
          if (b.start > segStart) segs.push({ start: segStart, end: b.start, label: SIDE, confirm: t });
          state = b.dir; segStart = b.start; ext = t; rc = t; conf = t; impTop = y[b.start]; prevExt = b.start;
        }
      } else {
        const d = state;
        if (d * (y[t] - y[ext]) >= 0) {                       // new extreme
          if (d * (y[ext] - y[rc]) >= R.req(ext, rc) / 2 - EPS) impTop = y[rc];   // a real swing formed
          ext = t; rc = t;
        } else {
          if (d * (y[t] - y[rc]) < 0) rc = t;                   // deepest counter point so far
          const counter = d * (y[ext] - y[t]), impulse = d * (y[ext] - impTop);
          const need = Math.max(floor * R.req(ext, t), rho * impulse);
          if (counter >= need - EPS && R.ok(ext, t)) {          // reversal
            segs.push({ start: segStart, end: ext, label: state, confirm: conf });
            state = -d; segStart = ext; conf = t; impTop = y[ext]; ext = t; rc = t;
          } else if (t - ext >= stall) {                        // stalled trend
            segs.push({ start: segStart, end: ext, label: state, confirm: conf });
            if (d * (y[ext] - y[rc]) >= floor * R.req(ext, rc) - EPS && R.ok(ext, rc)) {   // counter-leg is itself a trend
              state = -d; segStart = ext; conf = t; impTop = y[ext];
              let r2 = rc;
              for (let k = rc; k <= t; k++) if (d * (y[k] - y[r2]) > 0) r2 = k;
              ext = rc; rc = r2;
            } else {                                            // -> sideways
              state = SIDE; segStart = ext;
            }
          }
        }
      }
      rt[t] = state;
    }
    const fin = { state, ext, segStart, impTop };
    segs.push({ start: segStart, end: n - 1, label: state, confirm: state !== SIDE ? conf : n - 1, open: true });
    return { segs, rt, fin };
  }

  // E: Kalman smooth local-linear trend (level + slope, slope noise only) == HP filter with lambda = 1/q.
  function kalmanTrend(y, lam) {
    const n = y.length, q = 1 / lam;
    let x0 = y[0], x1 = 0, P00 = 1e7, P01 = 0, P11 = 1e7;
    const xp = [], Pp = [], xf = [], Pf = [];
    for (let t = 0; t < n; t++) {
      if (t > 0) {                                      // predict: F=[[1,1],[0,1]], Q=diag(0,q)
        x0 = x0 + x1;
        const a = P00 + 2 * P01 + P11, b = P01 + P11, c = P11 + q;
        P00 = a; P01 = b; P11 = c;
      }
      xp.push([x0, x1]); Pp.push([P00, P01, P11]);
      const S = P00 + 1, k0 = P00 / S, k1 = P01 / S, v = y[t] - x0;
      x0 += k0 * v; x1 += k1 * v;
      const n00 = P00 - k0 * P00, n01 = P01 - k0 * P01, n11 = P11 - k1 * P01;
      P00 = n00; P01 = n01; P11 = n11;
      xf.push([x0, x1]); Pf.push([P00, P01, P11]);
    }
    const xs = new Array(n); xs[n - 1] = xf[n - 1].slice();
    for (let t = n - 2; t >= 0; t--) {                  // RTS: J = Pf F' Pp(t+1)^-1
      const [f00, f01, f11] = Pf[t], [p00, p01, p11] = Pp[t + 1];
      const m00 = f00 + f01, m01 = f01, m10 = f01 + f11, m11 = f11;   // Pf * F'
      const det = p00 * p11 - p01 * p01, i00 = p11 / det, i01 = -p01 / det, i11 = p00 / det;
      const J00 = m00 * i00 + m01 * i01, J01 = m00 * i01 + m01 * i11, J10 = m10 * i00 + m11 * i01, J11 = m10 * i01 + m11 * i11;
      const d0 = xs[t + 1][0] - xp[t + 1][0], d1 = xs[t + 1][1] - xp[t + 1][1];
      xs[t] = [xf[t][0] + J00 * d0 + J01 * d1, xf[t][1] + J10 * d0 + J11 * d1];
    }
    return { filt: xf.map(v => v[0]), fslope: xf.map(v => v[1]), smooth: xs.map(v => v[0]), sslope: xs.map(v => v[1]) };
  }

  function labelMerge(y, segs, R) {
    while (true) {
      for (const s of segs) { s.label = R.lab(s.start, s.end); if (s.imp == null) s.imp = Math.abs(y[s.end] - y[s.start]); }
      const merged = [{ ...segs[0] }];
      for (const s of segs.slice(1)) { const m = merged[merged.length - 1];
        if (s.label === m.label) { m.end = s.end; m.imp = Math.abs(y[s.end] - y[s.start]); } else merged.push({ ...s }); }
      if (merged.length === segs.length) return merged;
      segs = merged;
    }
  }
  // Ex-post retracement merge: trend / counter / same-direction trend making a new extreme
  // collapses into one trend when the counter-move is below max(floor*thr, rho * preceding impulse).
  function absorb(y, segs, R, rho, floor) {
    segs = labelMerge(y, segs.map(s => ({ ...s })), R);
    let changed = true;
    while (changed) {
      changed = false;
      for (let k = 1; k < segs.length - 1; k++) {
        const a = segs[k - 1], b = segs[k], c = segs[k + 1], d = a.label;
        if (d === SIDE || c.label !== d || b.label === d) continue;
        const counter = -d * (y[b.end] - y[b.start]);
        const small = counter < Math.max(floor * R.req(b.start, b.end), rho * a.imp) - EPS || !R.ok(b.start, b.end);
        if (d * (y[c.end] - y[a.end]) > 0 && small) {
          segs.splice(k - 1, 3, { start: a.start, end: c.end, label: d, imp: Math.abs(y[c.end] - y[c.start]) });
          segs = labelMerge(y, segs, R); changed = true; break;
        }
      }
    }
    return segs;
  }
  function kalmanSegments(y, kf, R, rho, floor, snap = 6) {
    const n = y.length, sl = kf.sslope, bps = [0];
    for (let t = 1; t < n; t++) if (Math.sign(sl[t]) !== Math.sign(sl[t - 1]) && sl[t] !== 0) bps.push(t);
    bps.push(n - 1);
    for (let k = 1; k < bps.length - 1; k++) {           // snap turning points to the actual extreme
      const loW = Math.max(bps[k - 1] + 1, bps[k] - snap), hiW = Math.min(bps[k + 1] - 1, bps[k] + snap);
      if (loW >= hiW) continue;
      const peak = sl[bps[k] - 1] > 0; let best = loW;
      for (let t = loW; t <= hiW; t++) if (peak ? y[t] > y[best] : y[t] < y[best]) best = t;
      bps[k] = best;
    }
    const u = [...new Set(bps)].sort((p, q) => p - q), segs = [];
    for (let k = 0; k < u.length - 1; k++) segs.push({ start: u[k], end: u[k + 1] });
    return absorb(y, segs, R, rho, floor);
  }


  // F: L1 trend filtering (Kim, Koh, Boyd & Gorinevsky 2009): min 0.5||y-x||^2 + lam*||D2 x||_1, solved by ADMM.
  function l1Filter(y, lam, iters = 3000, warm = null) {
    const n = y.length, m = n - 2, rho = Math.max(lam, 1e-3);
    // A = I + rho*D'D (pentadiagonal); banded Cholesky L with bands l0 (diag), l1, l2
    const a0 = new Float64Array(n), a1 = new Float64Array(n), a2 = new Float64Array(n);
    for (let i = 0; i < m; i++) {                 // row i of D: [1,-2,1] at i,i+1,i+2
      const c = [1, -2, 1];
      for (let p = 0; p < 3; p++) { a0[i + p] += rho * c[p] * c[p];
        if (p < 2) a1[i + p] += rho * c[p] * c[p + 1];
        if (p < 1) a2[i + p] += rho * c[p] * c[p + 2]; }
    }
    for (let i = 0; i < n; i++) a0[i] += 1;
    const l0 = new Float64Array(n), l1 = new Float64Array(n), l2 = new Float64Array(n);   // L[i][i], L[i+1][i], L[i+2][i]
    for (let i = 0; i < n; i++) {
      let d = a0[i] - (i >= 1 ? l1[i - 1] ** 2 : 0) - (i >= 2 ? l2[i - 2] ** 2 : 0);
      l0[i] = Math.sqrt(d);
      if (i + 1 < n) l1[i] = (a1[i] - (i >= 1 ? l1[i - 1] * l2[i - 1] : 0)) / l0[i];
      if (i + 2 < n) l2[i] = a2[i] / l0[i];
    }
    const solve = b => { const z = new Float64Array(n);
      for (let i = 0; i < n; i++) z[i] = (b[i] - (i >= 1 ? l1[i - 1] * z[i - 1] : 0) - (i >= 2 ? l2[i - 2] * z[i - 2] : 0)) / l0[i];
      const x = new Float64Array(n);
      for (let i = n - 1; i >= 0; i--) x[i] = (z[i] - (i + 1 < n ? l1[i] * x[i + 1] : 0) - (i + 2 < n ? l2[i] * x[i + 2] : 0)) / l0[i];
      return x; };
    let z = new Float64Array(m), u = new Float64Array(m), x = new Float64Array(n), b = new Float64Array(n);
    if (warm) { z.set(warm.z.subarray(0, m)); u.set(warm.u.subarray(0, m)); }
    const k = lam / rho;
    for (let it = 0; it < iters; it++) {
      b.set(y);
      for (let i = 0; i < m; i++) { const w = rho * (z[i] - u[i]); b[i] += w; b[i + 1] -= 2 * w; b[i + 2] += w; }
      x = solve(b);
      for (let i = 0; i < m; i++) { const dx = x[i] - 2 * x[i + 1] + x[i + 2], v = dx + u[i];
        z[i] = v > k ? v - k : v < -k ? v + k : 0; u[i] = v - z[i]; }
    }
    return { x: Array.from(x), z, u };
  }
  function slopeTurns(x, tol) {                 // indices where the fitted slope changes sign (ignoring flat pieces)
    const n = x.length, bps = [0]; let prev = 0;
    for (let t = 1; t < n; t++) { const s = x[t] - x[t - 1], sg = Math.abs(s) < tol ? 0 : Math.sign(s);
      if (sg !== 0) { if (prev !== 0 && sg !== prev) bps.push(t - 1); prev = sg; } }
    bps.push(n - 1); return bps;
  }
  function snapLabel(y, x, bps, R, snap = 6) {
    for (let k = 1; k < bps.length - 1; k++) {
      const loW = Math.max(bps[k - 1] + 1, bps[k] - snap), hiW = Math.min(bps[k + 1] - 1, bps[k] + snap);
      if (loW >= hiW) continue;
      const peak = x[bps[k]] >= x[bps[k - 1]]; let best = loW;
      for (let t = loW; t <= hiW; t++) if (peak ? y[t] > y[best] : y[t] < y[best]) best = t;
      bps[k] = best;
    }
    const u = [...new Set(bps)].sort((p, q) => p - q), segs = [];
    for (let k = 0; k < u.length - 1; k++) segs.push({ start: u[k], end: u[k + 1] });
    return labelMerge(y, segs, R);
  }
  function l1Segments(y, fit, R) { return snapLabel(y, fit, slopeTurns(fit, 1e-4), R); }
  // real time: at each t, L1 fit on the trailing W months; keep the endpoint level, then apply the D rule to it
  function l1Levels(y, lam, W = 60, iters = 200) {
    const n = y.length, lev = new Array(n); let warm = null;
    for (let t = 0; t < n; t++) {
      if (t < 12) { lev[t] = y[t]; continue; }
      const w = y.slice(Math.max(0, t - W + 1), t + 1);
      const r = l1Filter(w, lam, warm ? iters : 3 * iters, warm && w.length === W ? warm : null);
      lev[t] = r.x[r.x.length - 1];
      if (w.length === W) { const z = new Float64Array(W - 2), u = new Float64Array(W - 2);   // shift duals one step
        z.set(r.z.subarray(1)); u.set(r.u.subarray(1)); warm = { z, u }; }
    }
    return lev;
  }

  // G: hierarchical (bottom-up) merging of swings. Start from every local turn; repeatedly remove the smallest
  // swing b whose size is below max(floor*thr, rho*min(|a|,|c|)) by merging it with its neighbours.
  function hierMerge(y, R, rho = 0.618, floor = 1) {
    const n = y.length; let piv = [0];
    for (let t = 1; t < n - 1; t++) { const a = y[t] - y[piv[piv.length - 1]], b = y[t + 1] - y[t];
      if (a * b < 0) piv.push(t); else if (a === 0) piv[piv.length - 1] = t; }
    piv.push(n - 1);
    // enforce alternation (collapse equal-direction runs)
    const P = [piv[0]];
    for (let k = 1; k < piv.length; k++) {
      if (P.length >= 2) { const d1 = y[P[P.length - 1]] - y[P[P.length - 2]], d2 = y[piv[k]] - y[P[P.length - 1]];
        if (d1 * d2 >= 0) { P[P.length - 1] = piv[k]; continue; } }
      P.push(piv[k]);
    }
    piv = P;
    // only interior swings are merged; the first and last swings are kept (the last one is still unresolved)
    while (piv.length > 3) {
      let best = -1, bestSize = Infinity;
      for (let k = 1; k < piv.length - 2; k++) {
        const size = Math.abs(y[piv[k + 1]] - y[piv[k]]);
        const nb = Math.min(Math.abs(y[piv[k]] - y[piv[k - 1]]), Math.abs(y[piv[k + 2]] - y[piv[k + 1]]));
        const small = size < Math.max(floor * R.req(piv[k], piv[k + 1]), rho * nb) - EPS || !R.ok(piv[k], piv[k + 1]);
        if (small && size < bestSize) { best = k; bestSize = size; }
      }
      if (best < 0) break;
      piv.splice(best, 2);
    }
    const segs = []; for (let k = 0; k < piv.length - 1; k++) segs.push({ start: piv[k], end: piv[k + 1] });
    return labelMerge(y, segs, R);
  }

  // ex-post segments from the machines must satisfy the rule over their full span; the open last segment keeps its state
  function consistent(y, segs, R) {
    const last = segs[segs.length - 1], closed = segs.slice(0, -1);
    if (!closed.length) return segs;
    const bad = closed.some(s => R.lab(s.start, s.end) !== s.label);
    if (!bad) return segs;
    const fixed = labelMerge(y, closed.map(s => ({ ...s })), R);
    const tail = fixed[fixed.length - 1];
    if (tail.label === last.label) { tail.end = last.end; tail.open = true; tail.confirm = last.confirm; return fixed; }
    return [...fixed, last];
  }
  const kfCache = {}, l1Cache = {}, l1rtCache = {};
  function compute(dates, y, thr, opt = {}) {
    const rho = opt.rho ?? 0.618, floor = opt.floor ?? 1, lam = opt.lam ?? 1e5, l1lam = opt.l1lam ?? 5;
    const n = y.length;
    const R = makeRule(y, thr, { scaled: opt.scaled, z: opt.z, minLen: opt.minLen, sigma: opt.sigma, cap: opt.cap });
    const dec = segs => decorate(segs, y, dates).map(s => ({ ...s, z: R.zscore(s.start, s.end), req_bp: Math.round(R.req(s.start, s.end) * 100) }));
    const a = stateMachine(y, R);
    a.segs = consistent(y, a.segs, R);
    const segA = dec(a.segs).map(s => ({ ...s, confirm_date: dates[s.confirm] }));
    const labA = segsToLabels(n, a.segs);
    if (!partCache.base) partCache.base = pwlPartition(y).penalty;
    const part = partCache[thr] || (partCache[thr] = pwlPartition(y, partCache.base * thr * thr));
    const segBraw = pwlSegments(y, part, R);
    const segB = dec(segBraw), labB = segsToLabels(n, segBraw);
    const labC = rolling(y, R);
    const dm = retraceMachine(y, R, rho, floor);
    dm.segs = consistent(y, dm.segs, R);
    const segD = dec(dm.segs).map(s => ({ ...s, confirm_date: dates[s.confirm] }));
    const labD = segsToLabels(n, dm.segs);
    const kf = kfCache[lam] || (kfCache[lam] = kalmanTrend(y, lam));
    const segEraw = kalmanSegments(y, kf, R, rho, floor);
    const segE = dec(segEraw), labE = segsToLabels(n, segEraw);
    const er = retraceMachine(kf.filt, R, rho, floor);      // real-time: D rule on the filtered level
    const l1 = l1Cache[l1lam] || (l1Cache[l1lam] = l1Filter(y, l1lam).x);
    const segFraw = l1Segments(y, l1, R), segF = dec(segFraw), labF = segsToLabels(n, segFraw);
    const lev = l1rtCache[l1lam] || (l1rtCache[l1lam] = l1Levels(y, l1lam));
    const fr = { level: lev, rt: retraceMachine(lev, R, rho, floor).rt };
    const segGraw = hierMerge(y, R, rho, floor), segG = dec(segGraw), labG = segsToLabels(n, segGraw);
    const labels = { A: labA, B: labB, D: labD, E: labE, F: labF, G: labG, A_rt: a.rt, C: labC, D_rt: dm.rt, E_rt: er.rt, F_rt: fr.rt };
    const segs = { A: segA, B: segB, D: segD, E: segE, F: segF, G: segG };
    for (const k of ["A_rt", "C", "D_rt", "E_rt", "F_rt"]) segs[k] = dec(labelsToSegs(labels[k]));
    const keys = Object.keys(labels), agree = {}, kap = {}, summ = {};
    for (const p of keys) { agree[p] = {}; kap[p] = {};
      for (const q of keys) { let m = 0; for (let i = 0; i < n; i++) m += labels[p][i] === labels[q][i]; agree[p][q] = m / n; kap[p][q] = kappa(labels[p], labels[q]); }
      summ[p] = summary(labels[p], segs[p]); }
    const lag = { A_rt: lags(segA, a.rt), C: lags(segA, labC), D_rt: lags(segD, dm.rt), E_rt: lags(segE, er.rt), F_rt: lags(segF, fr.rt) };
    // what next month's print would have to be to flip each machine (month index n)
    const N = n;
    const sideTriggers = from => { let up = Infinity, dn = -Infinity;
      for (let s = from; s <= N - Math.max(R.minLen, 1); s++) { up = Math.min(up, y[s] + R.req(s, N)); dn = Math.max(dn, y[s] - R.req(s, N)); }
      return { up, dn }; };
    const rng = from => { let lo = Infinity, hi = -Infinity; for (let t = from; t < n; t++) { lo = Math.min(lo, y[t]); hi = Math.max(hi, y[t]); } return { lo, hi }; };
    const fa = a.fin, ra = rng(fa.segStart), ta = fa.state === SIDE ? sideTriggers(fa.segStart) : null;
    const box = { state: fa.state, since: dates[fa.segStart], lo: ra.lo, hi: ra.hi, last: y[n - 1], last_date: dates[n - 1],
      up_trigger: ta ? ta.up : null, down_trigger: ta ? ta.dn : null,
      flip: fa.state !== SIDE ? { ext: y[fa.ext], req: R.req(fa.ext, N), level: y[fa.ext] - fa.state * R.req(fa.ext, N) } : null };
    const fd = dm.fin, rd = rng(fd.segStart), td = fd.state === SIDE ? sideTriggers(fd.segStart) : null;
    let trig = null;
    if (fd.state !== SIDE) { const impulse = fd.state * (y[fd.ext] - fd.impTop), req = Math.max(floor * R.req(fd.ext, N), rho * impulse);
      trig = { ext: y[fd.ext], req, level: y[fd.ext] - fd.state * req, impulse }; }
    const boxD = { state: fd.state, since: dates[fd.segStart], lo: rd.lo, hi: rd.hi, trig, up_trigger: td ? td.up : null, down_trigger: td ? td.dn : null };
    const reqTable = [1, 3, 6, 12, 24].map(T => ({ T, bp: Math.round(R.req(n - 1 - T, n - 1) * 100) }));
    return { dates, y, labels, segs, agree, kappa: kap, lag, summary: summ, box, boxD, fitB: part.fitted, penalty: part.penalty,
             kf: { smooth: kf.smooth, filt: kf.filt }, l1: { fit: l1, rt: fr.level }, sigNow: R.sig[n - 1], reqTable };
  }
  return { compute, makeRule, stateMachine, retraceMachine, kalmanTrend, absorb, kalmanSegments, l1Filter, l1Segments, l1Levels, hierMerge };
})();
if (typeof module !== "undefined") module.exports = TA;
