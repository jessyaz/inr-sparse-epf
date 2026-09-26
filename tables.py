import argparse
from pathlib import Path

import pandas as pd

from data.loader import MARKETS

LABELS = {"lear": "LEAR", "dnn": "DNN", "epf_transformer": "EPF-Tr.",
          "transformer_imputed": "Tr. imp.", "transformer_masked": "Tr. masq.", "inr": "INR"}
ABLATIONS = {"full": "Full", "static": "Static cond.", "noexog": "No exog."}


def load(runs):
    files = sorted(Path(runs).glob("*/*/*/seed*/metrics.csv"))
    if not files:
        raise SystemExit(f"no metrics.csv under {runs}, run evaluate.py first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def table_nominal(df):
    d = df[(df.variant == "full") & (df.rate == 0)]
    g = d.groupby(["model", "market"]).MAE.agg(["mean", "std"])
    cell = g.apply(lambda r: f"{r['mean']:.2f}" if pd.isna(r["std"])
                   else f"{r['mean']:.2f} ±{r['std']:.2f}", axis=1)
    t = cell.unstack("market")
    t = t.reindex(index=[m for m in LABELS if m in t.index],
                  columns=[m for m in MARKETS if m in t.columns])
    return t.rename(index=LABELS)


def table_rho(df):
    mean = df.groupby(["model", "market", "variant", "rate"]).MAE.mean()
    rows = {}
    for (model, market), g in mean.groupby(level=["model", "market"]):
        g = g.droplevel(["model", "market"])
        if ("ablated", 0.0) not in g.index or ("full", 0.0) not in g.index:
            continue
        mae0, abl = g[("full", 0.0)], g[("ablated", 0.0)]
        for rate, mae in g.loc["full"].items():
            if rate > 0:
                rows[(model, market, rate)] = (mae - mae0) / (abl - mae0)
    t = pd.Series(rows).unstack([1, 2])
    t = t.reindex(index=[m for m in LABELS if m in t.index],
                  columns=[c for m in MARKETS for c in t.columns if c[0] == m])
    return t.rename(index=LABELS).round(2)


def table_ablation(df, market):
    d = df[(df.model == "inr") & (df.market == market) & df.variant.isin(ABLATIONS)]
    t = d.groupby(["variant", "rate"]).MAE.mean().unstack("rate")
    if "full" not in t.index:
        return None
    full = t.loc["full"]
    out = pd.DataFrame(index=[v for v in ABLATIONS if v in t.index], columns=t.columns, dtype=object)
    for v in out.index:
        for r in t.columns:
            gap = 100 * (t.loc[v, r] - full[r]) / full[r]
            out.loc[v, r] = f"{t.loc[v, r]:.2f}" if v == "full" else f"{t.loc[v, r]:.2f} ({gap:+.1f}%)"
    out.columns = [f"p={r:g}" for r in out.columns]
    return out.rename(index=ABLATIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    df = load(args.runs)

    print("Table 1. Nominal MAE at p = 0 (mean ± std over seeds)\n")
    print(table_nominal(df).fillna("--").to_string(), "\n")

    print("Table 2. History degradation ratio rho(p) under MCAR\n")
    print(table_rho(df).to_string(), "\n")

    for market in MARKETS:
        t = table_ablation(df, market)
        if t is not None and len(t) > 1:
            print(f"Table 3. Ablation of exogenous conditioning on {market} (MAE, gap to full)\n")
            print(t.to_string(), "\n")


if __name__ == "__main__":
    main()
