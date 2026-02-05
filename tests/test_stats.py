import os
import unittest

import numpy as np
import pandas as pd

from commodutil import dates
from commodutil import forwards
from commodutil import stats
from commodutil.forward.util import convert_contract_to_date


class TestForwards(unittest.TestCase):
    def test_curve_zscore(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl.rename(
            columns={x: pd.to_datetime(convert_contract_to_date(x)) for x in cl.columns}
        )
        hist = contracts[["2020-01-01"]].dropna()
        fwd = contracts[["2020-01-01"]]

        res = stats.curve_seasonal_zscore(hist, fwd)
        self.assertAlmostEqual(res["zscore"]["2019-01-02"], 0.92, 2)

    def test_reindex_zscore(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl.rename(
            columns={x: pd.to_datetime(convert_contract_to_date(x)) for x in cl.columns}
        )

        q = forwards.quarterly_contracts(contracts)
        q = q[[x for x in q.columns if "Q1" in x]]

        res = stats.reindex_zscore(q, calc_year_start=2022)
        self.assertIsNotNone(res)

    def test_select_reindex_prompt_column(self):
        """
        If the current-year column ends within 10 days of the max x-date, prefer next year.
        Otherwise prefer current year.
        """
        idx = pd.date_range(f"{dates.curyear}-01-01", f"{dates.curyear}-01-31", freq="D")
        df = pd.DataFrame(
            {
                f"Spread {dates.curyear}": np.arange(len(idx), dtype=float),
                f"Spread {dates.curyear + 1}": np.arange(len(idx), dtype=float) + 100.0,
            },
            index=idx,
        )

        df1 = df.copy()
        df1.loc[idx[-20:], f"Spread {dates.curyear}"] = np.nan
        sel1 = stats.select_reindex_prompt_column(df1, within_days=10)
        self.assertEqual(sel1, f"Spread {dates.curyear}")

        df2 = df.copy()
        df2.loc[idx[-5:], f"Spread {dates.curyear}"] = np.nan
        sel2 = stats.select_reindex_prompt_column(df2, within_days=10)
        self.assertEqual(sel2, f"Spread {dates.curyear + 1}")

    def test_reindex_year_point_stats(self):
        """
        Ensure point-stats uses aligned as-of values across years after reindexing.
        """
        years = [dates.curyear - 2, dates.curyear - 1, dates.curyear, dates.curyear + 1]
        frames = []
        for y in years:
            # Extend y+1 so the overall max x-date is later than current-year column,
            # making the prompt selection stay on current year (delta>=10 days).
            end_day = "01-31" if y == dates.curyear + 1 else "01-10"
            idx = pd.date_range(f"{y}-01-01", f"{y}-{end_day}", freq="D")
            if y == dates.curyear - 2:
                vals = np.full(len(idx), 10.0)
            elif y == dates.curyear - 1:
                vals = np.full(len(idx), 12.0)
            elif y == dates.curyear:
                vals = np.full(len(idx), 15.0)
            else:
                vals = np.full(len(idx), 99.0)

            frames.append(pd.DataFrame({f"Spread {y}": vals}, index=idx))

        df = pd.concat(frames, axis=1, join="outer")

        asof = f"{dates.curyear}-01-10"
        res = stats.reindex_year_point_stats(df, asof=asof, lookback_years=2, within_days=10)

        self.assertEqual(res.current_year, dates.curyear)
        self.assertAlmostEqual(res.current_value, 15.0, 6)
        self.assertEqual(sorted(res.reference_years), [dates.curyear - 2, dates.curyear - 1])
        self.assertAlmostEqual(res.mean, 11.0, 6)
        self.assertAlmostEqual(res.std, np.sqrt(2.0), 6)
        self.assertAlmostEqual(res.zscore, 2.828427, 4)
        self.assertAlmostEqual(res.percentile, 1.0, 6)

    def test_reindex_year_point_stats_table_groups(self):
        idx = pd.date_range(f"{dates.curyear}-01-01", f"{dates.curyear}-01-10", freq="D")
        df = pd.DataFrame(
            {
                # Group A (3 years -> should be included with min_columns=3)
                "JunAug 2024": np.full(len(idx), 10.0),
                "JunAug 2025": np.full(len(idx), 11.0),
                "JunAug 2026": np.full(len(idx), 12.0),
                # Group B (2 years -> should be excluded with min_columns=3)
                "DecJan 2025": np.full(len(idx), 5.0),
                "DecJan 2026": np.full(len(idx), 6.0),
            },
            index=idx,
        )

        table = stats.reindex_year_point_stats_table(df, lookback_years=2, min_columns=3)
        self.assertIn("JunAug", table.index)
        self.assertNotIn("DecJan", table.index)
        self.assertIn("zscore", table.columns)
        self.assertEqual(int(table.loc["JunAug", "prompt_year"]), dates.curyear)

    def test_prompt_strip_point_stats_asof(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "M1": np.arange(1, 11, dtype=float),
                "M2": np.arange(11, 21, dtype=float),
            },
            index=idx,
        )

        res = stats.prompt_strip_point_stats(df, lookback_bdays=9, require_all_columns=True)
        self.assertEqual(res.attrs["asof"].date(), idx[-1].date())
        self.assertAlmostEqual(float(res.loc["M1", "value"]), 10.0, 6)
        # Reference is 1..9 (exclude current), so mean=5, std=sqrt(7.5), z=(10-5)/std
        self.assertAlmostEqual(float(res.loc["M1", "mean"]), 5.0, 6)
        self.assertAlmostEqual(float(res.loc["M1", "zscore"]), 1.825741858, 6)
        self.assertAlmostEqual(float(res.loc["M1", "percentile"]), 1.0, 6)

        df2 = df.copy()
        df2.loc[idx[-1], "M2"] = np.nan
        res2 = stats.prompt_strip_point_stats(df2, lookback_bdays=9, require_all_columns=True)
        self.assertEqual(res2.attrs["asof"].date(), idx[-2].date())


if __name__ == "__main__":
    unittest.main()
