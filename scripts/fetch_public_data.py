"""
Download public monthly/daily series and build the predictor panel used by the trend-detection gym.

    python scripts/fetch_public_data.py            # download into data/raw/, build data/predictors_monthly.csv
    python scripts/fetch_public_data.py --offline  # rebuild the panel from files already in data/raw/

FRED itself was not reachable from the build environment, so every series comes from a public GitHub
mirror that stores the original FRED (or CBOE/EIA/BLS/Shiller/H.10) file. Sources are listed in SOURCES
and repeated in data/predictors_meta.csv. With a FRED API key or Bloomberg, replace the files in
data/raw/ (same layout) or write your own panel with the same columns.

Output
  data/predictors_monthly.csv  one row per month (YYYY-MM-01), raw levels (percent, index levels, prices)
  data/predictors_meta.csv     id, name, group, kind, pub_lag, unit, start, source, url, notes

Timing convention: a row labelled month m holds the monthly average (or monthly value) for month m.
`pub_lag` is the number of months after m before the value is published, measured from the end of
month m (market data 0; CPI, payrolls, unemployment, PCE 1). The gym shifts each column by its lag.
"""
import argparse, io, json, pathlib, sys, urllib.request
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GH = "https://raw.githubusercontent.com"
TWD = f"{GH}/theodorewright11/macro_eco_dashboard/main/public/data/fred"
M5 = f"{GH}/soohucn-gif/macro5-dashboard/main/data"
DS = f"{GH}/datasets"

SOURCES = {
    "gs10_monthly.csv": f"{DS}/bond-yields-us-10y/main/data/monthly.csv",
    "DGS2.csv": f"{TWD}/DGS2.csv",
    "DGS10.csv": f"{TWD}/DGS10.csv",
    "FEDFUNDS.csv": f"{TWD}/FEDFUNDS.csv",
    "CPILFESL.csv": f"{TWD}/CPILFESL.csv",
    "PCEPILFE.csv": f"{TWD}/PCEPILFE.csv",
    "UNRATE.csv": f"{TWD}/UNRATE.csv",
    "PAYEMS.csv": f"{TWD}/PAYEMS.csv",
    "BAA10Y.csv": f"{TWD}/BAA10Y.csv",
    "WALCL.csv": f"{TWD}/WALCL.csv",
    "USREC.csv": f"{TWD}/USREC.csv",
    "real_rates.csv": f"{M5}/real_rates.csv",
    "inflation_expectations.csv": f"{M5}/inflation_expectations.csv",
    "vix-daily.csv": f"{DS}/finance-vix/main/data/vix-daily.csv",
    "wti-monthly.csv": f"{DS}/oil-prices/main/data/wti-monthly.csv",
    "cpiai.csv": f"{DS}/cpi-us/main/data/cpiai.csv",
    "sp500_shiller.csv": f"{DS}/s-and-p-500/main/data/data.csv",
    "fx_monthly.csv": f"{DS}/exchange-rates/main/data/monthly.csv",
    "gold_monthly.csv": f"{DS}/gold-prices/main/data/monthly.csv",
}

# id: (name, group, kind, pub_lag, unit, source note, raw file)
#   kind: rate   = percent level, features use changes in percentage points
#         price  = positive level, features use 100*log changes
#         yoy    = price/count index turned into a year-on-year % change, then treated as a rate
META = {
    "y10":     ("US 10-year Treasury yield (monthly avg, GS10)", "target", "rate", 0, "%", "FRED GS10 via datasets/bond-yields-us-10y; Aug 2026 from DGS10 daily", "gs10_monthly.csv"),
    "y10_eom": ("US 10-year Treasury yield, month-end (DGS10)", "context", "rate", 0, "%", "FRED DGS10", "DGS10.csv"),
    "y2":      ("US 2-year Treasury yield (monthly avg, DGS2)", "front_end", "rate", 0, "%", "FRED DGS2", "DGS2.csv"),
    "ff":      ("Effective fed funds rate (FEDFUNDS)", "policy", "rate", 0, "%", "FRED FEDFUNDS", "FEDFUNDS.csv"),
    "y5":      ("US 5-year Treasury yield (monthly avg, DGS5)", "curve", "rate", 0, "%", "FRED DGS5", "real_rates.csv"),
    "y30":     ("US 30-year Treasury yield (monthly avg, DGS30)", "curve", "rate", 0, "%", "FRED DGS30 (gap 2002-2006)", "real_rates.csv"),
    "be10":    ("10-year breakeven inflation (T10YIE)", "breakevens", "rate", 0, "%", "FRED T10YIE", "real_rates.csv"),
    "fwd5y5y": ("5y5y forward inflation expectation (T5YIFR)", "breakevens", "rate", 0, "%", "FRED T5YIFR", "real_rates.csv"),
    "tips10":  ("10-year TIPS real yield (DFII10)", "breakevens", "rate", 0, "%", "FRED DFII10", "real_rates.csv"),
    "clev10":  ("Cleveland Fed 10-year expected inflation", "infl_exp", "rate", 1, "%", "FRED EXPINF10YR", "inflation_expectations.csv"),
    "mich1":   ("UMich 1-year inflation expectations (MICH)", "infl_exp", "rate", 0, "%", "FRED MICH", "inflation_expectations.csv"),
    "cpi":     ("CPI-U all items, NSA (YoY %)", "infl_real", "yoy", 1, "% YoY", "BLS CUUR0000SA0 via datasets/cpi-us (not revised)", "cpiai.csv"),
    "core_cpi": ("Core CPI (CPILFESL, YoY %)", "infl_real", "yoy", 1, "% YoY", "FRED CPILFESL", "CPILFESL.csv"),
    "core_pce": ("Core PCE price index (PCEPILFE, YoY %)", "infl_real", "yoy", 1, "% YoY", "FRED PCEPILFE", "PCEPILFE.csv"),
    "unrate":  ("Unemployment rate (UNRATE)", "labor", "rate", 1, "%", "FRED UNRATE", "UNRATE.csv"),
    "payems":  ("Nonfarm payrolls (PAYEMS, YoY %)", "labor", "yoy", 1, "% YoY", "FRED PAYEMS (revised data)", "PAYEMS.csv"),
    "baa10y":  ("Moody's Baa yield minus 10-year Treasury (BAA10Y)", "credit", "rate", 0, "%", "FRED BAA10Y", "BAA10Y.csv"),
    "vix":     ("CBOE VIX (monthly avg of closes)", "risk", "price", 0, "index", "CBOE via datasets/finance-vix", "vix-daily.csv"),
    "spx":     ("S&P 500 (monthly avg, Shiller)", "risk", "price", 0, "index", "Shiller via datasets/s-and-p-500", "sp500_shiller.csv"),
    "wti":     ("WTI crude oil spot (monthly avg)", "commod", "price", 0, "$/bbl", "EIA via datasets/oil-prices", "wti-monthly.csv"),
    "gold":    ("Gold price (monthly avg)", "commod", "price", 0, "$/oz", "datasets/gold-prices", "gold_monthly.csv"),
    "usd":     ("US dollar vs major currencies (constructed, H.10 monthly avgs)", "dollar", "price", 0, "index", "FRED H.10 via datasets/exchange-rates; fixed weights EUR .39 CAD .30 JPY .13 GBP .11 CHF .04 AUD .02 SEK .01", "fx_monthly.csv"),
    "walcl":   ("Fed balance sheet total assets (WALCL, monthly avg)", "fed_bs", "price", 0, "$mn", "FRED WALCL", "WALCL.csv"),
    "usrec":   ("NBER recession indicator (USREC), context only: not known in real time", "context", "rate", 12, "0/1", "FRED USREC", "USREC.csv"),
}


def download(offline=False):
    RAW.mkdir(parents=True, exist_ok=True)
    for fn, url in SOURCES.items():
        p = RAW / fn
        if offline:
            if not p.exists():
                sys.exit(f"missing {p}; run without --offline")
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            p.write_bytes(r.read())
        print(f"  downloaded {fn:28s} {p.stat().st_size/1e3:8.1f} kB")


def mstart(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def fred_daily(fn, col=None, how="mean"):
    df = pd.read_csv(RAW / fn)
    col = col or df.columns[1]
    df = pd.DataFrame({"date": pd.to_datetime(df.iloc[:, 0]), "v": pd.to_numeric(df[col], errors="coerce")}).dropna()
    df["m"] = df.date.dt.to_period("M").dt.to_timestamp()
    g = df.sort_values("date").groupby("m")["v"]
    out = g.mean() if how == "mean" else g.last()
    # drop the current, incomplete month (fewer than 15 obs for daily data)
    n = g.count()
    if len(n) and n.iloc[-1] < 15 and df.date.diff().dt.days.median() < 5:
        out = out.iloc[:-1]
    return out


def fred_monthly(fn, col=None):
    df = pd.read_csv(RAW / fn)
    col = col or df.columns[1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=mstart(df.iloc[:, 0]))
    return s.dropna()


def build():
    cols = {}
    # target: GS10 monthly avg, extended with DGS10 daily averages for complete months after it ends
    gs10 = fred_monthly("gs10_monthly.csv", "Rate")
    d10 = fred_daily("DGS10.csv")
    ov = pd.concat([gs10, d10], axis=1, join="inner").dropna()
    diff = (ov.iloc[:, 0] - ov.iloc[:, 1]).abs()
    print(f"  GS10 vs DGS10 monthly avg: {len(ov)} common months, max |diff| {diff.max():.3f}, mean {diff.mean():.4f}")
    ext = d10[d10.index > gs10.index.max()]
    cols["y10"] = pd.concat([gs10, ext]).round(2)
    cols["y10_eom"] = fred_daily("DGS10.csv", how="last")
    cols["y2"] = fred_daily("DGS2.csv")
    cols["ff"] = fred_monthly("FEDFUNDS.csv")
    for k, c in [("y5", "dgs5"), ("y30", "dgs30"), ("be10", "t10yie"), ("fwd5y5y", "t5yifr"), ("tips10", "dfii10")]:
        cols[k] = fred_daily("real_rates.csv", c)
    ie = pd.read_csv(RAW / "inflation_expectations.csv")
    ie.index = mstart(ie.date)
    cols["clev10"] = pd.to_numeric(ie["cleveland_10y"], errors="coerce").dropna()
    cols["mich1"] = pd.to_numeric(ie["michigan_1y"], errors="coerce").dropna()
    cpi = pd.read_csv(RAW / "cpiai.csv")
    cols["cpi"] = pd.Series(cpi["Index"].values, index=mstart(cpi.Date)).astype(float)
    cols["core_cpi"] = fred_monthly("CPILFESL.csv")
    cols["core_pce"] = fred_monthly("PCEPILFE.csv")
    cols["unrate"] = fred_monthly("UNRATE.csv")
    cols["payems"] = fred_monthly("PAYEMS.csv")
    cols["baa10y"] = fred_daily("BAA10Y.csv")
    vix = pd.read_csv(RAW / "vix-daily.csv")
    vix = vix.rename(columns={"DATE": "date", "CLOSE": "close"})[["date", "close"]]
    vix.to_csv(RAW / "_vix_close.csv", index=False)
    cols["vix"] = fred_daily("_vix_close.csv", "close")
    sp = pd.read_csv(RAW / "sp500_shiller.csv")
    cols["spx"] = pd.Series(sp["SP500"].values, index=mstart(sp.Date)).astype(float)
    wti = pd.read_csv(RAW / "wti-monthly.csv")
    cols["wti"] = pd.Series(wti["Price"].values, index=mstart(wti.Date)).astype(float)
    gold = pd.read_csv(RAW / "gold_monthly.csv")
    cols["gold"] = pd.Series(gold["Price"].values, index=mstart(gold.Date)).astype(float)
    cols["usd"] = dollar_index()
    cols["walcl"] = fred_daily("WALCL.csv")
    cols["usrec"] = fred_monthly("USREC.csv")

    panel = pd.DataFrame(cols).sort_index()
    # last complete month: DGS10 must have an observation on (or after) that month's last business day
    d = pd.to_datetime(pd.read_csv(RAW / "DGS10.csv").iloc[:, 0])
    d = d[pd.to_numeric(pd.read_csv(RAW / "DGS10.csv").iloc[:, 1], errors="coerce").notna().values]
    last_obs = d.max()
    bme = last_obs + pd.offsets.BMonthEnd(0)
    last_full = (last_obs.to_period("M") - (0 if last_obs >= bme else 1)).to_timestamp()
    panel = panel.loc["1947-01-01":last_full]
    panel.index.name = "date"
    return panel


def dollar_index():
    fx = pd.read_csv(RAW / "fx_monthly.csv")
    fx["m"] = mstart(fx.Date)
    w = pd.pivot_table(fx, index="m", columns="Country", values="Exchange rate")
    eur = w["Euro"].copy()
    dem = w["Germany"] / 1.95583                      # DEM per USD -> EUR-equivalent per USD before 1999
    eur = eur.combine_first(dem)
    weights = {"EUR": .39, "Canada": .30, "Japan": .13, "United Kingdom": .11, "Switzerland": .04, "Australia": .02, "Sweden": .01}
    parts = {"EUR": eur}
    parts.update({k: w[k] for k in weights if k != "EUR"})
    logs = sum(weights[k] * np.log(parts[k] / parts[k].dropna().iloc[0]) for k in weights)   # foreign currency per USD
    return (100 * np.exp(logs)).dropna()


def write(panel):
    out = ROOT / "data" / "predictors_monthly.csv"
    panel.round(6).to_csv(out, date_format="%Y-%m-%d")
    rows = []
    for k, (name, group, kind, lag, unit, src, fn) in META.items():
        s = panel[k].dropna()
        rows.append(dict(id=k, name=name, group=group, kind=kind, pub_lag=lag, unit=unit,
                         start=s.index.min().strftime("%Y-%m"), end=s.index.max().strftime("%Y-%m"),
                         n=len(s), source=src, url=SOURCES.get(fn, "")))
    meta = pd.DataFrame(rows)
    meta.to_csv(ROOT / "data" / "predictors_meta.csv", index=False)
    print(f"  wrote {out.relative_to(ROOT)}: {panel.shape[0]} months x {panel.shape[1]} series, "
          f"{panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")
    print(meta[["id", "group", "kind", "pub_lag", "start", "end", "n"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    download(a.offline)
    write(build())
