from __future__ import annotations

import argparse

import pandas as pd

from commodutil import dates, forwards, stats
from commodutil.forward import fly as flymod
from commodutil.forward import spreads as spreadsmod


def _try_fetch_contracts(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    try:
        from pyoilprice import oilprice as op
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "pyoilprice is required to fetch data for this example. "
            "Install/enable it or adapt the script to load a contracts dataframe from disk."
        ) from e

    contracts = op.contracts(symbol, start_year=start_year, end_year=end_year)
    return contracts[symbol]


def print_top_bottom(df: pd.DataFrame, *, title: str, n: int = 10) -> None:
    if df is None or df.empty:
        print(f"\n{title}: (no results)")
        return

    cols = ["prompt_year", "value", "mean", "std", "zscore", "percentile", "n_reference"]
    show = [c for c in cols if c in df.columns]

    print(f"\n{title}")
    print("-" * len(title))
    print("\nCheapest (lowest z-score):")
    print(df.sort_values("zscore", ascending=True)[show].head(n).to_string())
    print("\nRichest (highest z-score):")
    print(df.sort_values("zscore", ascending=False)[show].head(n).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan RBW structures for cheap/rich vs history (reindex-year z).")
    parser.add_argument("--symbol", default="Ice_ClearedOil:RBW", help="pyoilprice symbol to scan")
    parser.add_argument("--lookback-years", type=int, default=5, help="reference years to compare against")
    parser.add_argument("--history-years", type=int, default=10, help="how many years of contracts to request")
    parser.add_argument("--top", type=int, default=10, help="rows to show at top/bottom")
    args = parser.parse_args()

    start_year = dates.curyear - args.history_years
    end_year = dates.curyear + 1

    contracts = _try_fetch_contracts(args.symbol, start_year=start_year, end_year=end_year)

    # Monthly spreads (extended set; prefer 4-digit year labels for scan grouping)
    monthly = spreadsmod.all_monthly_spreads_extended(contracts, col_format=None)
    if monthly is None:
        monthly = forwards.all_monthly_spreads(contracts, col_format=None)
    monthly_tbl = stats.reindex_year_point_stats_table(monthly, lookback_years=args.lookback_years, min_columns=4)
    print_top_bottom(monthly_tbl, title="Monthly spreads (all)", n=args.top)

    # Flies (use forward.fly module directly so labels include 4-digit year by default)
    flies = flymod.all_fly_spreads(contracts, col_format=None)
    flies_tbl = stats.reindex_year_point_stats_table(flies, lookback_years=args.lookback_years, min_columns=4)
    print_top_bottom(flies_tbl, title="Monthly flies (all)", n=args.top)

    # Quarterly rolls (Q1Q2, Q2Q3, etc.)
    q = forwards.quarterly_contracts(contracts, col_format=None)
    qroll = forwards.all_quarterly_rolls(q, col_format=None)
    qroll_tbl = stats.reindex_year_point_stats_table(qroll, lookback_years=args.lookback_years, min_columns=4)
    print_top_bottom(qroll_tbl, title="Quarterly rolls (all)", n=args.top)


if __name__ == "__main__":
    main()
