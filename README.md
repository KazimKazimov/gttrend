# gttrend

Classify a yield series, month by month, as an **up** trend, **down** trend or **sideways**, using eleven
methods that share one definition of what a trend is. The first case is the US 10-year Treasury yield
(monthly, Apr 1953 – Jul 2026).

The same algorithms exist twice, with identical output: in Python (`trend_algos.py`) for research,
and in JavaScript (`algos.js`) inside a self-contained interactive page with every setting on a control.

## Quick start

```bash
pip install -r requirements.txt

python build.py data/ust10_monthly.csv            # labels + segments to CSV, current state printed
python make_page.py data/ust10_monthly.csv --open # interactive page -> trend_lab.html
```

`examples/ust10_trend_lab.html` is a pre-built page: open it in any browser. It needs no server and works offline.

## What counts as a trend

A move from month *a* to month *b* is a trend if

- |y<sub>b</sub> − y<sub>a</sub>| ≥ **required move**, and
- *b − a* ≥ **minimum episode length** (default 3 months).

The required move is either

- **fixed**: the minimum move (default 100bp), at any horizon; or
- **scaled** (default): max(minimum move, *z* · σ · √min(*b − a*, cap)), where σ is the volatility of monthly
  changes (trailing 36 months or full sample), *z* = 2 and cap = 12 months by default.

With the scaled rule and σ ≈ 20bp, 100bp in 3 months is a trend but 100bp over a year is not (a year
needs about 140bp). The cap stops multi-year trends from needing ever-larger moves.

## Methods

| | Ex-post (labels history after the fact) | Real-time (only data up to each month) |
|---|---|---|
| A | Breakout / reversal state machine, backdated to the turn | Same machine, no backdating |
| B | Optimal piecewise-linear fit (dynamic programming), breaks snapped to peaks/troughs | — |
| C | — | Rolling 12-month regression move vs. the required move |
| D | Retracement-aware state machine | Same machine, no backdating |
| E | Kalman local-linear trend, smoothed (= HP filter) | D rule on the Kalman-filtered level |
| F | L1 trend filtering (Kim, Koh, Boyd & Gorinevsky 2009) | D rule on a trailing-60-month L1 endpoint |
| G | Hierarchical swing merging | — (each merge depends on later swings) |

**Retracements (D, E, F real-time, G).** A counter-move ends a trend only if it reaches
max(floor × required move, ρ × the last impulse leg), with ρ = 61.8% and floor = 1 by default. So a path of
−2 / +1 / −3 is one 4pp decline, not three segments.

## Files

| File | What it is |
|---|---|
| `trend_algos.py` | All algorithms in Python. `run_all(y, **settings)` returns every method's labels and segments. |
| `build.py` | Runs `run_all` with the page defaults; writes `trend_labels.csv` and `trend_segments.csv`. `--fixed` uses the fixed 100bp rule. |
| `algos.js` | The same algorithms in JavaScript (the source inlined into the page; also usable from Node). |
| `trend_lab_template.html` | Page layout, chart and controls. `make_page.py` fills in the data and `algos.js`. |
| `make_page.py` | Builds the page from any CSV: `--series`, `--short`, `--title`, `--date-col`, `--value-col`, `--out`. |
| `data/ust10_monthly.csv` | US 10-year yield, monthly averages (FRED GS10, via the datasets/bond-yields-us-10y mirror). |
| `tests/test_trends.py` | Behaviour tests, plus a parity test that runs `algos.js` in Node and checks Python matches it month for month. |

## Using your own series

- Values in **percent** (the thresholds are 25–200bp).
- **Monthly** data. The windows (36-month σ, 12-month regression, stall rules) assume months; resample
  daily data first (`df.resample("MS").mean()`).
- Other column names: `--date-col` / `--value-col` in `make_page.py`; in Python pass the array to `run_all`.

```python
import pandas as pd
from trend_algos import run_all

df = pd.read_csv("data/ust10_monthly.csv")
labels, segs = run_all(df.Rate.values, thr=1.0, scaled=True, z=2, min_len=3)
labels["D"][-1]   # 1 up, 0 sideways, -1 down
```

## Keeping Python and the page in sync

Any change to an algorithm goes into both `trend_algos.py` and `algos.js`. Then run
`python -m pytest -q` (the parity test needs Node) and rebuild the example page with
`python make_page.py data/ust10_monthly.csv --out examples/ust10_trend_lab.html`.

## Trend-detection gym: do other variables help in real time?

`gym/` is a walk-forward evaluation harness for detectors that use outside variables (2Y, breakevens,
inflation, activity, credit, equities, dollar, ...) to detect the 10Y's trend state in real time
(nowcast) and ahead of time (3 and 6 months).

```bash
pip install -r requirements.txt
python scripts/fetch_public_data.py        # monthly predictor panel -> data/predictors_monthly.csv
python -m pytest -q tests/test_gym.py      # includes the no-look-ahead tests
python run_gym.py --vars y2 mich1          # one-off matched test of a variable
python -m gym.experiments                  # full study -> results/gym_results_default.json (~25 min)
python -m gym.frontier                     # delay vs false-alarm curves (added to the same file)
python scripts/build_report.py             # interactive page -> examples/trend_gym_report.html
```

**Guarantees.** At each month-end a detector sees only the real-time panel: market data through that
month, macro data lagged by its release delay (`pub_lag` in `data/predictors_meta.csv`). It is refit
every 12 months on *vintage* labels, computed from the 10Y up to the refit date only; it is scored
against the final ex-post labels on months whose label is settled. `tests/test_gym.py` reruns each
detector on data cut at 2008 and requires identical forecasts up to the cut.

**Detectors** (`gym/detectors.py`): `RuleDetector` (method D/A real-time state), `CompositeRule` (rule
machine on a 10Y + beta-scaled X blend), `LogitDetector` and `GBMDetector` (label-then-learn with lagged
features), `MSDetector` (3-state Markov switching; X as extra observables `obs=` or driving the
transition probabilities `tvtp=`), `Climatology`. Anything with `fit(gym, T)` / `predict(gym, t0, t1)`
works, and `TrendEnv` offers a gym-style `reset()` / `step()` loop for custom agents.

**Measures** (`gym/metrics.py`): balanced accuracy, Brier skill, log score, ROC AUC; per-trend delay,
hit rate, capture of the move, false alarms per decade, flips per year; P&L of a duration position that
follows the nowcast; Diebold-Mariano, block-bootstrap and Wilcoxon tests for matched comparisons.

**Your own series.** A monthly CSV (date + one column per series) and a meta CSV (`id, kind, pub_lag,
name`; kind is `rate`, `price` or `yoy`):

```bash
python run_gym.py --add-csv bbg_series.csv --add-meta bbg_meta.csv --vars MOVE CESI --models logit ms-tvtp
```
