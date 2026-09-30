"""
Build the interactive Trend Lab page from any monthly series.

    python make_page.py data/ust10_monthly.csv --open
    python make_page.py bund.csv --series "German 10-year Bund" --short "Bund" --title "Bund Trend Lab" --out bund.html

The CSV needs a date column and a value column (default: Date, Rate; values in percent).
The page is self-contained: the eleven algorithms in algos.js are inlined and run in the browser,
so every control works offline. Needs trend_lab_template.html and algos.js next to this script.
"""
import argparse, json, pathlib, webbrowser
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def build_html(dates, values, series="US Treasury 10-year", short="10-year", title="UST 10Y Trend Lab"):
    html = (HERE / "trend_lab_template.html").read_text(encoding="utf-8")
    algos = (HERE / "algos.js").read_text(encoding="utf-8")
    algos = algos.replace('if (typeof module !== "undefined") module.exports = TA;', "").rstrip() + "\n"
    data = {"dates": list(dates), "y": [round(float(v), 6) for v in values]}
    for k, v in {"__TITLE__": title, "__SERIES__": series, "__SHORT__": short}.items():
        html = html.replace(k, v)
    html = html.replace("/*__ALGOS__*/", algos)
    return html.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--value-col", default="Rate")
    ap.add_argument("--series", default="US Treasury 10-year", help="long name, shown above the title")
    ap.add_argument("--short", default="10-year", help='short name, used in "Is the <short> trending?"')
    ap.add_argument("--title", default="UST 10Y Trend Lab", help="browser tab title")
    ap.add_argument("--out", default="trend_lab.html")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, parse_dates=[a.date_col])[[a.date_col, a.value_col]].dropna().sort_values(a.date_col)
    if df[a.date_col].diff().dt.days.median() < 25:
        print("Note: the series looks higher-frequency than monthly. The page's settings (36-month vol, "
              "12-month windows, stall rules) assume monthly data; resample first, e.g. df.resample('MS').mean().")
    dates = df[a.date_col].dt.strftime("%Y-%m-%d").tolist()
    html = build_html(dates, df[a.value_col].values, a.series, a.short, a.title)
    out = pathlib.Path(a.out).resolve()
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(dates)} months, {dates[0]} to {dates[-1]})")
    if a.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
